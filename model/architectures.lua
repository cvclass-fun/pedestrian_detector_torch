--[[
    Experimental networks.
]]

local function CreateROIPooler(roipool_width,roipool_height, feat_stride)
-- w: feature grids width of the detection region
-- h: feature grids height of the detection region
-- stride: number of pixels that one grid pixel in the last convolutional layer represents on the image

  local roi = nn.Sequential()
  --roi:add(inn.ROIPooling(roipool_width,roipool_height):setSpatialScale(1/feat_stride)) -- works better than inn's
  roi:add(nn.ROIPooling(roipool_width,roipool_height):setSpatialScale(1/feat_stride)) -- works better than inn's
  
  roi:cuda()
  
  return roi
end

------------------------------------

local function CreateClassifierCore(backend, num_feats_last_conv, roipool_width, roipool_height)
  local backend = model_backend
  if backend == 'cudnn' then
    backend = cudnn
  else
    backend = nn
  end

  local classifier = nn.Sequential()
  classifier:add(nn.View(-1):setNumInputDims(3))
  classifier:add(nn.Linear(num_feats_last_conv * roipool_width * roipool_height, 4096))
  classifier:add(nn.BatchNormalization(4096, 1e-3))
  classifier:add(backend.ReLU(true))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, 4096))
  classifier:add(nn.BatchNormalization(4096, 1e-3))
  classifier:add(backend.ReLU(true))
  classifier:add(nn.Dropout(0.5))
  
  return classifier
end

----------------------------------

local function CreateCriterion(train_bbox_regressor)
  local criterion
  print('==> Criterion')  
  criterion = nn.ParallelCriterion():add(nn.CrossEntropyCriterion(), 1):add(nn.WeightedSmoothL1Criterion(), 1)

  return criterion
end

----------------------------------

local function SplitFeaturesNet(featuresNet, convLayerID)
  
  -- split features up to the sencond last max pool layer
  local features1 = nn.Sequential() -- before the second to last max pool layer
  -- the remaining layers untill the last maxpool layer
  local features2 = nn.Sequential()  -- between the features1 and the last max pool layer
  
  for i=1, featuresNet:size() do
    if i < convLayerID then
      features1:add(featuresNet.modules[i])
    else
      features2:add(featuresNet.modules[i])
    end    
  end

  return features1, features2
end

----------------------------------

local function model_arch1(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, nGPU, useGPU)
  
  assert(nGPU <= cutorch.getDeviceCount(), ('Assigned more gpus than detected ones: nGPUS: %d <= available:%d'):format(nGPU, cutorch.getDeviceCount()))
  
  -- verbose flag
  local verbose = verbose or true
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable.width, roipoolTable.height, feat_stride)

  
  -- create separate classifiers
  local class1 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable.width, roipoolTable.height)
  
  local classifier = nn.Sequential()
  classifier:add(nn.View(-1):setNumInputDims(3))
  local nfeats = num_feats_last_conv * (roipoolTable.width * roipoolTable.height)
  classifier:add(nn.Linear(nfeats, 4096))
  classifier:add(nn.BatchNormalization(4096, 1e-3))
  classifier:add(nn.ReLU(true))  
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, 4096))
  classifier:add(nn.BatchNormalization(4096, 1e-3))
  classifier:add(nn.ReLU(true))
  classifier:add(nn.Dropout(0.5))

  
  -- classifier output
  local cls = nn.Linear(4096,nClasses+1)
  local reg = nn.Linear(4096,(nClasses+1)*4)
  local cls_reg_layer = nn.ConcatTable():add(cls):add(reg)
  local cls_reg_module = nn.Sequential():add(cls_reg_layer)
  
  local roipooler_classifier =  nn.Sequential():add(roipool1):add(classifier)
  
  -- (3) group parts into a single model
  featuresNet:cuda()
  featuresNet = Framework.utils.model.makeDataParallel(featuresNet, nGPU, useGPU)
  
  local features = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity()))
  
  local model = nn.Sequential():add(features):add(roipooler_classifier):add(cls_reg_module)
  
  -- define a quick and easy lookup field for the regressor module
  model.regressor = reg
  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('Features network:')
    if featuresNet:size() > 40 then
      print('<network too big to print>')
    else
      print(features)
    end
    print('Roi pooling:')
    print(roipooler_classifier)
    print('Classifier network:')
    print(cls_reg_module)
    
    print('Criterion:')
    print(criterion)
  end
  
  return model, criterion
end

----------------------------------

local function model_arch2(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, convLayerID, convLayerNFeats, convLayerFeatStride, nGPU, useGPU)
  
  assert(nGPU <= cutorch.getDeviceCount(), ('Assigned more gpus than detected ones: nGPUS: %d <= available:%d'):format(nGPU, cutorch.getDeviceCount()))
  
  -- verbose flag
  local verbose = verbose or true
  
  -- split featureNet into two networks
  require 'nngraph'
  nngraph.setDebug(true)

  -- split features
  local features_pre, features_post = SplitFeaturesNet(featuresNet,convLayerID)
  features_pre:cuda()
  features_post:cuda()
  
  features_pre = Framework.utils.model.makeDataParallel(features_pre, nGPU, useGPU)
  features_post = Framework.utils.model.makeDataParallel(features_post, nGPU, useGPU)
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable.width, roipoolTable.height, convLayerFeatStride)
  local roipool2 = CreateROIPooler(roipoolTable.width, roipoolTable.height, feat_stride)
  -- create view layers (in case spp use is set to false)
  local view1 = nn.View(-1):setNumInputDims(3)
  local view2 = nn.View(-1):setNumInputDims(3)
  -- create join table
  local join_tensors = nn.JoinTable(1,1)
  -- create fully-connected layers
  local nfeats = num_feats_last_conv * (roipoolTable.width * roipoolTable.height) + convLayerNFeats * (roipoolTable.width * roipoolTable.height)
  local classifier = nn.Sequential()
  classifier:add(nn.Linear(nfeats, 4096))
  classifier:add(nn.BatchNormalization(4096, 1e-3))
  classifier:add(nn.ReLU(true))  
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, 4096))
  classifier:add(nn.BatchNormalization(4096, 1e-3))
  classifier:add(nn.ReLU(true))
  classifier:add(nn.Dropout(0.5))
  
  -- create classifier + regressor
  local cls = nn.Linear(4096,nClasses+1)
  local reg = nn.Linear(4096,(nClasses+1)*4)
  local cls_reg_layer =  nn.ConcatTable():add(cls):add(reg)
  
  -- link/combine all modules together using gModules
  local identity = nn.Identity()()
  features_pre = features_pre()
  local link_roipool1_feats = view1(roipool1({features_pre,identity}))
  local link_roipool2_feats = view2(roipool2({features_post(features_pre),identity}))

  local model = nn.gModule( -- inputs
                            {features_pre, identity},
                            -- output
                            {cls_reg_layer(classifier(join_tensors({link_roipool1_feats,link_roipool2_feats})))} 
                          )
  
  -- define a quick and easy lookup field for the regressor module
  model.regressor = reg

  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('network:')
    for i, module in ipairs(model.modules) do print(module) end
    local ok = pcall(require,'qt')
    if not ok then
      print('check image in the display server')
      graph.dot(model.fg, 'FRCNN architecture 2', 'network_graph')
      os.execute('inkscape -z -e network_graph.png network_graph.svg')
      local graph = image.load('network_graph.png')
      disp.image(graph)
      -- remove temporary files
      os.execute('rm -rf network_graph.png network_graph.svg network_graph.dot')
    else
      graph.dot(model.fg, 'FRCNN architecture 2')
    end
    
    print('Criterion:')
    print(criterion)
  end
  
  return model, criterion
end

----------------------------------

local function model_arch3(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, convLayerID, convLayerNFeats, convLayerFeatStride, nGPU, useGPU)
  
  assert(nGPU <= cutorch.getDeviceCount(), ('Assigned more gpus than detected ones: nGPUS: %d <= available:%d'):format(nGPU, cutorch.getDeviceCount()))
  
  -- verbose flag
  local verbose = verbose or true 
  
  -- split featureNet into two networks
  require 'nngraph'
  nngraph.setDebug(true)

  -- split features
  local features_pre, features_post = SplitFeaturesNet(featuresNet,convLayerID)
  
  features_pre = Framework.utils.model.makeDataParallel(features_pre, nGPU, useGPU)
  features_post = Framework.utils.model.makeDataParallel(features_post, nGPU, useGPU)

  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable.width, roipoolTable.height, convLayerFeatStride)
  local roipool2 = CreateROIPooler(roipoolTable.width, roipoolTable.height, feat_stride)

  -- create view layers (in case spp use is set to false)
  local view1 = nn.View(-1):setNumInputDims(3)
  local view2 = nn.View(-1):setNumInputDims(3)
  
  -- create join table
  local join_tensors = nn.JoinTable(1,1)
 
   -- create fully-connected layers
  --
  local function CreateClassifier(nfeats)
    local classifier = nn.Sequential()
    classifier:add(nn.Linear(nfeats, 4096))
    classifier:add(nn.ReLU(true))  
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, 4096))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    return classifier
  end

  local nfeats = num_feats_last_conv * (roipoolTable.width * roipoolTable.height)  
  local fully_connected1 = CreateClassifier(nfeats)
  local fully_connected2 = CreateClassifier(nfeats)
  
  -- create classifier + regressor
  local cls = nn.Linear(4096,nClasses+1)
  local reg = nn.Linear(4096,(nClasses+1)*4)
  local fc3 = nn.Sequential()
  fc3:add(nn.JoinTable(1,1))
  fc3:add(nn.Linear(4096*2,4096))
  fc3:add(nn.BatchNormalization(4096, 1e-3))
  fc3:add(nn.ReLU(true))
  fc3:add(nn.Dropout(0.5))
  
  local cls_reg_join = nn.Sequential():add(nn.ConcatTable():add(cls):add(reg))

  -- link/combine all modules together using gModules
  local identity = nn.Identity()()
  features_pre = features_pre()
  
  local link_fc1_roipool1_feats = fully_connected1(view1(roipool1({features_pre,identity})))
  local link_fc2_roipool2_feats = fully_connected1(view2(roipool2({features_post(features_pre),identity})))
  
  
  local link_fc1_fc2_fc3 = fc3({link_fc1_roipool1_feats, link_fc2_roipool2_feats})
  
  local model = nn.gModule( -- inputs
                            {features_pre, identity},
                            -- output
                            {cls_reg_join(link_fc1_fc2_fc3)} 
                          )
  
  -- define a quick and easy lookup field for the regressor module
  model.regressor = reg
  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('network:')
    for i, module in ipairs(model.modules) do print(module) end
    local ok = pcall(require,'qt')
    if not ok then
      print('check image in the display server')
      graph.dot(model.fg, 'FRCNN architecture 3', 'network_graph')
      os.execute('inkscape -z -e network_graph.png network_graph.svg')
      local graph = image.load('network_graph.png')
      disp.image(graph)
      -- remove temporary files
      os.execute('rm -rf network_graph.png network_graph.svg network_graph.dot')
    else
      graph.dot(model.fg, 'FRCNN architecture 3')
    end
    
    print('Criterion:')
    print(criterion)
  end
  
  return model, criterion
end

----------------------------------

local function model_arch4(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, convLayerID, convLayerNFeats, convLayerFeatStride, nGPU, useGPU)
  
  assert(nGPU <= cutorch.getDeviceCount(), ('Assigned more gpus than detected ones: nGPUS: %d <= available:%d'):format(nGPU, cutorch.getDeviceCount()))
  
  -- verbose flag
  local verbose = verbose or true
  
  -- split featureNet into two networks
  require 'nngraph'
  nngraph.setDebug(true)

  -- split features
  local features_pre, features_post = SplitFeaturesNet(featuresNet,convLayerID)
  
  features_pre = Framework.utils.model.makeDataParallel(features_pre, nGPU, useGPU) 
  features_post = Framework.utils.model.makeDataParallel(features_post, nGPU, useGPU) 
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable.width, roipoolTable.height, convLayerFeatStride)
  local roipool2 = CreateROIPooler(roipoolTable.width, roipoolTable.height, feat_stride)
  
  -- create view layers
  local view1 = nn.View(-1):setNumInputDims(3)
  local view2 = nn.View(-1):setNumInputDims(3)
 
  -- create fully-connected layers
  --
  local function CreateClassifier(nfeats)
    local classifier = nn.Sequential()
    classifier:add(nn.Linear(nfeats, 4096))
    classifier:add(nn.BatchNormalization(4096, 1e-3))
    classifier:add(nn.ReLU(true))  
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, 4096))
    classifier:add(nn.BatchNormalization(4096, 1e-3))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    return classifier
  end
  
  local nfeats = num_feats_last_conv * (roipoolTable.width * roipoolTable.height)
  local fully_connected1 = CreateClassifier(nfeats)
  local fully_connected2 = CreateClassifier(nfeats)
  
  -- create classifier + regressor
  local cls1 = nn.Sequential():add(nn.Linear(4096,nClasses+1))
  local reg1 = nn.Sequential():add(nn.Linear(4096,(nClasses+1)*4))
  local cls2 = nn.Sequential():add(nn.Linear(4096,nClasses+1))
  local reg2 = nn.Sequential():add(nn.Linear(4096,(nClasses+1)*4))
  local cls = nn.Linear((nClasses+1)*2,nClasses+1)
  local reg = nn.Linear(((nClasses+1)*4)*2,(nClasses+1)*4)
  local cls_join = nn.Sequential():add(nn.JoinTable(1,1)):add(cls)
  local reg_join = nn.Sequential():add(nn.JoinTable(1,1)):add(reg)

  -- link/combine all modules together using gModules
  local identity = nn.Identity()()
  features_pre = features_pre()
  local link_fc1_roipool1_feats = fully_connected1(view1(roipool1({features_pre,identity})))
  local link_fc2_roipool2_feats = fully_connected2(view2(roipool2({features_post(features_pre),identity})))
  
  
  local model = nn.gModule( -- inputs
                            {features_pre, identity},
                            -- output
                            {cls_join({cls1(link_fc1_roipool1_feats), cls2(link_fc2_roipool2_feats)}),
                             reg_join({reg1(link_fc1_roipool1_feats), reg2(link_fc2_roipool2_feats)})} 
                          )
  
  -- define a quick and easy lookup field for the regressor module
  model.regressor = reg
  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('network:')
    for i, module in ipairs(model.modules) do print(module) end
    local ok = pcall(require,'qt')
    if not ok then
      print('check image in the display server')
      graph.dot(model.fg, 'FRCNN architecture 4', 'network_graph')
      os.execute('inkscape -z -e network_graph.png network_graph.svg')
      local graph = image.load('network_graph.png')
      disp.image(graph)
      -- remove temporary files
      os.execute('rm -rf network_graph.png network_graph.svg network_graph.dot')
    else
      graph.dot(model.fg, 'FRCNN architecture 4')
    end
    
    print('Criterion:')
    print(criterion)
  end
  
  return model, criterion
end

-------------------------------------
-------------------------------------
-------------------------------------

return {
          architecture1 = model_arch1,
          architecture2 = model_arch2,
          architecture2V2 = model_arch2V2,
          architecture3 = model_arch3,
          architecture4 = model_arch4
        }