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

------------------------------------

function model_base(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend)

  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  
  -- create separate classifiers
  local class1 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[1].width, roipoolTable[1].height)

  -- classifier output
  local cls_reg_layer =  nn.ConcatTable()
  cls_reg_layer:add(nn.Linear(4096,nClasses+1))
  cls_reg_layer:add(nn.Linear(4096,(nClasses+1)*4))
  
  local roipooler_classifier = nn.Sequential():add(class_core_concated):add(nn.View(-1):setNumInputDims(2)):add(cls_reg_layer)
  
  -- (3) group parts into a single model
  local model = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity())):add(roipool1):add(class1):add(cls_reg_layer)
  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('Features network:')
    if featuresNet:size() > 25 then
      print('<network too big to print>')
    else
      print(featuresNet)
    end
    print('Roi pooling:')
    print(roipoolinglayer)
    print('Classifier network:')
    print(classifier)
    
    print('Criterion:')
    print(criterion)
  end
  
  return model, criterion
end

function model0(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose)
  -- verbose flag
  local verbose = verbose or false
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  --local roipool2 = CreateROIPooler(roipoolTable[2].width, roipoolTable[2].height, feat_stride)
  --local roipool3 = CreateROIPooler(roipoolTable[3].width, roipoolTable[3].height, feat_stride)
  
  -- create separate classifiers
  local class1 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[1].width, roipoolTable[1].height)
  --local class2 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[2].width, roipoolTable[2].height)
  --local class3 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[3].width, roipoolTable[3].height)
  
  local classifier = nn.Sequential()
  --classifier:add(inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}}))
  --local nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height)
  classifier:add(nn.View(-1):setNumInputDims(3))
  local nfeats = num_feats_last_conv * (roipoolTable[1].width * roipoolTable[1].height)
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
  local features = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity()))
  --local model = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity())):add(roipooler_classifier)
  
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
    if featuresNet:size() > 25 then
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


function model0_spp(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose)
  -- verbose flag
  local verbose = verbose or false
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  --local roipool2 = CreateROIPooler(roipoolTable[2].width, roipoolTable[2].height, feat_stride)
  --local roipool3 = CreateROIPooler(roipoolTable[3].width, roipoolTable[3].height, feat_stride)
  
  -- create separate classifiers
  local class1 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[1].width, roipoolTable[1].height)
  --local class2 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[2].width, roipoolTable[2].height)
  --local class3 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[3].width, roipoolTable[3].height)
  
  local classifier = nn.Sequential()
  classifier:add(inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}}))
  local nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height)
  --classifier:add(nn.View(-1):setNumInputDims(3))
  --local nfeats = num_feats_last_conv * (roipoolTable[1].width * roipoolTable[1].height)
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
  local features = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity()))
  --local model = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity())):add(roipooler_classifier)
  
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
    if featuresNet:size() > 25 then
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


-- - 3 roi dif, 3 network dif
function model1(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose)
  
  -- verbose flag
  local verbose = verbose or false
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  local roipool2 = CreateROIPooler(roipoolTable[2].width, roipoolTable[2].height, feat_stride)
  local roipool3 = CreateROIPooler(roipoolTable[3].width, roipoolTable[3].height, feat_stride)
  
  -- create separate classifiers
  local class1 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[1].width, roipoolTable[1].height)
  local class2 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[2].width, roipoolTable[2].height)
  local class3 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[3].width, roipoolTable[3].height)
  
  -- join classifiers into a single one
  class_core_concated = nn.ConcatTable()
  class_core_concated:add(nn.Sequential():add(roipool1):add(class1))
  class_core_concated:add(nn.Sequential():add(roipool2):add(class2))
  class_core_concated:add(nn.Sequential():add(roipool3):add(class3))
  
  -- classifier output
  local cls_reg_layer =  nn.ConcatTable()
  cls_reg_layer:add(nn.Linear(4096*3,nClasses+1))
  cls_reg_layer:add(nn.Linear(4096*3,(nClasses+1)*4))
  local cls_reg_module = nn.Sequential():add(cls_reg_layer)
  
  local roipooler_classifier = nn.Sequential():add(class_core_concated):add(nn.JoinTable(2))
  
  -- (3) group parts into a single model
  local features = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity()))
  --local model = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity())):add(roipooler_classifier)
  
  local model = nn.Sequential():add(features):add(roipooler_classifier):add(cls_reg_module)
  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('Features network:')
    if featuresNet:size() > 25 then
      print('<network too big to print>')
    else
      print(features)
    end
    print('Roi pooling:')
    print(roipoolinglayer)
    print('Classifier network:')
    print(roipooler_classifier)
    
    print('Criterion:')
    print(cls_reg_module)
  end
  
  return model, criterion
  
end

------------------------------------

-- 3 roipool dif , 1 network
function model2(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose)
  
  -- verbose flag
  local verbose = verbose or false

   -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  local roipool2 = CreateROIPooler(roipoolTable[2].width, roipoolTable[2].height, feat_stride)
  local roipool3 = CreateROIPooler(roipoolTable[3].width, roipoolTable[3].height, feat_stride)
  
  -- join classifiers into a single one
  class_core_concated = nn.ConcatTable()
  class_core_concated:add(nn.Sequential():add(roipool1):add(nn.View(-1):setNumInputDims(3)))
  class_core_concated:add(nn.Sequential():add(roipool2):add(nn.View(-1):setNumInputDims(3)))
  class_core_concated:add(nn.Sequential():add(roipool3):add(nn.View(-1):setNumInputDims(3)))
  
  local class = nn.Sequential()
  class:add(nn.Linear(num_feats_last_conv * (roipoolTable[1].width*roipoolTable[1].height + roipoolTable[2].width*roipoolTable[2].height + roipoolTable[3].width*roipoolTable[3].height) , 4096))
  class:add(nn.BatchNormalization(4096, 1e-3))
  class:add(nn.ReLU(true))
  --class:add(nn.Dropout(0.5))
  class:add(nn.Linear(4096, 4096))
  class:add(nn.BatchNormalization(4096, 1e-3))
  class:add(nn.ReLU(true))
  --class:add(nn.Dropout(0.5))
  
  -- classifier output
  local cls_reg_layer =  nn.ConcatTable()
  cls_reg_layer:add(nn.Linear(4096,nClasses+1))
  cls_reg_layer:add(nn.Linear(4096,(nClasses+1)*4))
  
  -- group features modules
  local features = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity()))
  local roipooler_classifier = nn.Sequential():add(class_core_concated):add(nn.JoinTable(1,1)):add(class)
  
  -- (3) group parts into a single model
  local model = nn.Sequential():add(features):add(roipooler_classifier):add(cls_reg_layer)
  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('Features network:')
    if featuresNet:size() > 25 then
      print('<network too big to print>')
    else
      print(features)
    end
    print('Roi pooling:')
    print(roipooler_classifier)
    print('Classifier network:')
    print(cls_reg_layer)
    
    print('Criterion:')
    print(criterion)
  end
  
  return model, criterion
end

------------------------------------

--1 roi pool, 3 redes (3 redes com o mesmo roipool)
function model3(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose)

  -- verbose flag
  local verbose = verbose or false
  
   -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  
  -- create separate classifiers
  local class1 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[1].width, roipoolTable[1].height)
  local class2 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[1].width, roipoolTable[1].height)
  local class3 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[1].width, roipoolTable[1].height)
  class1:remove(class1:size()) -- remove the dropout layer
  class2:remove(class2:size()) -- remove the dropout layer
  class3:remove(class3:size()) -- remove the dropout layer
  
  -- join classifiers into a single one
  class_core_concated = nn.ConcatTable()
  class_core_concated:add(nn.Sequential():add(class1))
  class_core_concated:add(nn.Sequential():add(class2))
  class_core_concated:add(nn.Sequential():add(class3))
  
  
  -- classifier output
  local cls_reg_layer =  nn.ConcatTable()
  cls_reg_layer:add(nn.Sequential():add(nn.Dropout(0.5)):add(nn.Linear(4096*3,nClasses+1)))
  cls_reg_layer:add(nn.Linear(4096*3,(nClasses+1)*4))
  
  local features = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity()))
  
  local roipooler_classifier = nn.Sequential():add(roipool1):add(class_core_concated):add(nn.JoinTable(1,1)):add(nn.Dropout(0.5))
  
  -- (3) group parts into a single model
  local model = nn.Sequential():add(features):add(roipooler_classifier):add(cls_reg_layer)
  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('Features network:')
    if featuresNet:size() > 25 then
      print('<network too big to print>')
    else
      print(featuresNet)
    end
    print('Roi pooling:')
    print(features)
    print('Classifier network:')
    print(roipooler_classifier)
    
    print('Criterion:')
    print(cls_reg_layer)
  end
  
  return model, criterion
end

------------------------------------


local function merge_outputs_network(nOutputs)
  
  -- filter tensors according to the classifier or regressor layers 
  -- cycle all tensors
  local cls = nn.ConcatTable()
  local reg = nn.ConcatTable()
  for i=1, nOutputs do    
    cls:add(nn.SelectTable((i-1)*2 + 1)) -- select odd tensors
    reg:add(nn.SelectTable((i-1)*2 + 2)) -- select p+air tensors
  end

  local separate_ouputs = nn.ConcatTable()
  separate_ouputs:add(nn.Sequential():add(cls):add(nn.JoinTable(1,1)))
  separate_ouputs:add(nn.Sequential():add(reg):add(nn.JoinTable(1,1)))
  
  local merge_network = nn.Sequential()
  merge_network:add(nn.FlattenTable())
  merge_network:add(separate_ouputs)
  
  return merge_network
end


--1 roipool, 2 redes treinadas independentemente (features fixed)
--depois junta-las e treinar umas epocas
function model4(model1_path, model2_path, nClasses, verbose)
  
  -- verbose flag
  local verbose = verbose or false
  
  -- load model4_1
  local model1 = torch.load(paths.concat(model1_path, 'model.t7'))
  local model_params = torch.load(paths.concat(model1_path, 'model_parameters.t7'))
  
  -- load model4_2
  local model2 = torch.load(paths.concat(model2_path, 'model.t7'))  
  
  -- do some net surgery to extract the classifiers from each network
  -- fetch features network
  local features = model1.modules[1]
  
  -- classifier
  local concat_class = nn.ConcatTable()
  concat_class:add(nn.Sequential():add(model1.modules[2]):add(model1.modules[3]))
  concat_class:add(nn.Sequential():add(model2.modules[2]):add(model2.modules[3]))
  
  -- merge outputs
  local numNets = 2
  local merge_network = merge_outputs_network(numNets)
  
  local classifier_experts = nn.Sequential():add(concat_class):add(merge_network)
  
  -- get classifier + regressor final layers
  local cls_reg_layer =  nn.ParallelTable()
  cls_reg_layer:add(nn.Linear(2*numNets,nClasses+1)) -- false = disable the bias
  cls_reg_layer:add(nn.Linear(8*numNets,(nClasses+1)*4)) -- false = disable the bias
  
 
  -- append a new linear classifier + regressor on top of the two networks outputs
  local model = nn.Sequential():add(features):add(classifier_experts):add(cls_reg_layer)
  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('Features network:')
    if features.modules[1]:size() > 25 then
      print('<network too big to print>')
    else
      print(features)
    end
    print('Experts sub-networks')
    print(classifier_experts)
    print('Classifier network:')
    print(cls_reg_layer)
    
    print('Criterion:')
    print(criterion)
  end
  
  return model, criterion, model_params
end

function model4_1(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose)
  return model0(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose)
end

function model4_2(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose)
  return model0(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose)
end


--1 roipool, 2 redes treinadas juntas (criar um modulo que separe as rois de acordo com um tamanho estabelecido)
local function model5()

  -- TODO
  
end


-------------

function model0b(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose)
  -- verbose flag
  local verbose = verbose or false
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  --local roipool2 = CreateROIPooler(roipoolTable[2].width, roipoolTable[2].height, feat_stride)
  --local roipool3 = CreateROIPooler(roipoolTable[3].width, roipoolTable[3].height, feat_stride)
  
  paths.dofile('/home/mf/Toolkits/Codigo/git/fast-rcnn-torch/models/ROIPoolingBilinear.lua')
  local roipool1 = nn.Sequential()
  roipool1:add(nn.ROIPoolingBilinear(roipoolTable[1].width,roipoolTable[1].height):setSpatialScale(1/feat_stride))
  roipool1:cuda()
  
  
  -- create separate classifiers
  local class1 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[1].width, roipoolTable[1].height)
  --local class2 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[2].width, roipoolTable[2].height)
  --local class3 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[3].width, roipoolTable[3].height)
  
  local classifier = nn.Sequential()
  classifier:add(inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}}))
  --classifier:add(nn.View(-1):setNumInputDims(3))
  local nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height)
  classifier:add(nn.Linear(nfeats, 4096))
  classifier:add(nn.BatchNormalization(4096, 1e-3))
  classifier:add(nn.ReLU(true))  
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, 4096))
  classifier:add(nn.BatchNormalization(4096, 1e-3))
  classifier:add(nn.ReLU(true))
  classifier:add(nn.Dropout(0.5))

  
  -- classifier output
  local cls_reg_layer =  nn.ConcatTable()
  --cls_reg_layer:add(nn.Sequential():add(nn.Dropout(0.5)):add(nn.Linear(4096,nClasses+1)))
  cls_reg_layer:add(nn.Linear(4096,nClasses+1))
  cls_reg_layer:add(nn.Linear(4096,(nClasses+1)*4))
  local cls_reg_module = nn.Sequential():add(cls_reg_layer)
  
  local roipooler_classifier =  nn.Sequential():add(roipool1):add(classifier)
  
  -- (3) group parts into a single model
  local features = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity()))
  --local model = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity())):add(roipooler_classifier)
  
  local model = nn.Sequential():add(features):add(roipooler_classifier):add(cls_reg_module)
  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('Features network:')
    if featuresNet:size() > 25 then
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




-------------------------------------------
-------------------------------------------

local function SplitFeaturesNet(featuresNet)
  
  -- split features up to the sencond last max pool layer
  local features1 = nn.Sequential() -- before the second to last max pool layer
  -- the remaining layers untill the last maxpool layer
  local features2 = nn.Sequential()  -- between the features1 and the last max pool layer
  
  if featuresNet:size() > 15 then
    for i=1, featuresNet:size() do
      if i < 24 then
        features1:add(featuresNet.modules[i])
      else
        features2:add(featuresNet.modules[i])
      end    
    end
  else
    for i=1, featuresNet:size() do
      if i < 7 then
        features1:add(featuresNet.modules[i])
      else
        features2:add(featuresNet.modules[i])
      end    
    end
  end

  return features1, features2
end

local function SplitFeaturesNetV2(featuresNet, convLayerID)
  
  -- split features up to the sencond last max pool layer
  local features1 = nn.Sequential() -- before the second to last max pool layer
  -- the remaining layers untill the last maxpool layer
  local features2 = nn.Sequential()  -- between the features1 and the last max pool layer
  
  if featuresNet:size() > 15 then
    for i=1, featuresNet:size() do
      if i < convLayerID then
        features1:add(featuresNet.modules[i])
      else
        features2:add(featuresNet.modules[i])
      end    
    end
  else
    for i=1, featuresNet:size() do
      if i < convLayerID then
        features1:add(featuresNet.modules[i])
      else
        features2:add(featuresNet.modules[i])
      end    
    end
  end

  return features1, features2
end


function model_arch2(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, flag_spp)
  -- verbose flag
  local verbose = verbose or false
  local flag_spp = flag_spp or false
  
  -- split featureNet into two networks
  local features_pre, features_post = SplitFeaturesNet(featuresNet)

  local feats_split = nn.ConcatTable()
  feats_split:add(nn.Identity())
  feats_split:add(features_post)
  local feats = nn.Sequential():add(features_pre):add(feats_split)
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride/2)
  local roipool2 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  
  -- create separate classifiers
  local class1 = CreateClassifierCore(backend, num_feats_last_conv, roipoolTable[1].width, roipoolTable[1].height)
  
  --roipool branches
  local branch1, branch2, nfeats
  if flag_spp then 
    nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height) *2
    branch1 = nn.Sequential():add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.SelectTable(3))):add(roipool1):add(inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}}))
    branch2 = nn.Sequential():add(nn.ConcatTable():add(nn.SelectTable(2)):add(nn.SelectTable(3))):add(roipool2):add(inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}}))
  else
    nfeats = num_feats_last_conv * roipoolTable[1].width * roipoolTable[1].height * 2
    branch1 = nn.Sequential():add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.SelectTable(3))):add(roipool1):add(nn.View(-1):setNumInputDims(3))
    branch2 = nn.Sequential():add(nn.ConcatTable():add(nn.SelectTable(2)):add(nn.SelectTable(3))):add(roipool2):add(nn.View(-1):setNumInputDims(3))
  end
  
 
  local branch_merge = nn.Concat():add(branch1):add(branch2)
  local roipool = nn.Sequential():add(branch_merge):add(nn.JoinTable(1))
  
  
  local classifier = nn.Sequential()
  local nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height) *2
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
  local cls_reg_layer =  nn.ConcatTable():add(cls):add(reg)
  local cls_reg_module = nn.Sequential():add(cls_reg_layer)
  
  local roipooler_classifier =  nn.Sequential():add(roipool):add(classifier)
  
  -- (3) group parts into a single model
  local features = nn.Sequential():add(nn.ParallelTable():add(feats):add(nn.Identity()))
  --local model = nn.Sequential():add(nn.ParallelTable():add(featuresNet):add(nn.Identity())):add(roipooler_classifier)
  
  local model = nn.Sequential():add(features):add(roipooler_classifier):add(cls_reg_module)
  
  -- define a quick and easy lookup field for the regressor module
  model.regressor = reg
  
  -- (4) define the loss criterion
  local criterion = CreateCriterion()
  criterion:cuda()
  
  -- (5) send model to the gpu
  model:cuda()
  
  if verbose then 
    print('network:')
    print(model)
    
    print('Criterion:')
    print(criterion)
  end
  
  return model, criterion
end


function model_arch2_nngraph(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, flag_spp)
  -- verbose flag
  local verbose = verbose or false
  local flag_spp = flag_spp or false
  
  -- split featureNet into two networks
  require 'nngraph'
  nngraph.setDebug(true)

  -- split features
  local features_pre, features_post = SplitFeaturesNet(featuresNet)
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride/2)
  local roipool2 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  -- create spatial pyramid poolers
  local spp1 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  local spp2 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  -- create view layers (in case spp use is set to false)
  local view1 = nn.View(-1):setNumInputDims(3)
  local view2 = nn.View(-1):setNumInputDims(3)
  -- create join table
  local join_tensors = nn.JoinTable(1,1)
  -- create fully-connected layers
  local nfeats
  if flag_spp then
    nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height) *2
  else
    nfeats = num_feats_last_conv * (roipoolTable[1].width * roipoolTable[1].height) *2
  end
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
  local link_spp1_roipool1_feats, link_spp2_roipool2_feats
  if flag_spp then
    link_spp1_roipool1_feats = spp1(roipool1({features_pre,identity}))
    link_spp2_roipool2_feats = spp2(roipool2({features_post(features_pre),identity}))
  else
    link_spp1_roipool1_feats = view1(roipool1({features_pre,identity}))
    link_spp2_roipool2_feats = view2(roipool2({features_post(features_pre),identity}))
  end

  local model = nn.gModule( -- inputs
                            {features_pre, identity},
                            -- output
                            {cls_reg_layer(classifier(join_tensors({link_spp1_roipool1_feats,link_spp2_roipool2_feats})))} 
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


function model_arch2_nngraphV2(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, flag_spp, convLayerID, convLayerNFeats, convLayerFeatStride)
  -- verbose flag
  local verbose = verbose or false
  local flag_spp = flag_spp or false
  
  -- split featureNet into two networks
  require 'nngraph'
  nngraph.setDebug(true)

  -- split features
  local features_pre, features_post = SplitFeaturesNetV2(featuresNet,convLayerID)
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, convLayerFeatStride)
  local roipool2 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  -- create spatial pyramid poolers
  local spp1 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  local spp2 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  -- create view layers (in case spp use is set to false)
  local view1 = nn.View(-1):setNumInputDims(3)
  local view2 = nn.View(-1):setNumInputDims(3)
  -- create join table
  local join_tensors = nn.JoinTable(1,1)
  -- create fully-connected layers
  local nfeats
  if flag_spp then
    nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height) + convLayerNFeats * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height) 
  else
    nfeats = num_feats_last_conv * (roipoolTable[1].width * roipoolTable[1].height) + convLayerNFeats * (roipoolTable[1].width * roipoolTable[1].height) 
  end
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
  local link_spp1_roipool1_feats, link_spp2_roipool2_feats
  if flag_spp then
    link_spp1_roipool1_feats = spp1(roipool1({features_pre,identity}))
    link_spp2_roipool2_feats = spp2(roipool2({features_post(features_pre),identity}))
  else
    link_spp1_roipool1_feats = view1(roipool1({features_pre,identity}))
    link_spp2_roipool2_feats = view2(roipool2({features_post(features_pre),identity}))
  end

  local model = nn.gModule( -- inputs
                            {features_pre, identity},
                            -- output
                            {cls_reg_layer(classifier(join_tensors({link_spp1_roipool1_feats,link_spp2_roipool2_feats})))} 
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


function model_arch3_nngraph(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, flag_spp)
  -- verbose flag
  local verbose = verbose or false 
  local flag_spp = flag_spp or false
  
  -- split featureNet into two networks
  require 'nngraph'
  nngraph.setDebug(true)

  -- split features
  local features_pre, features_post = SplitFeaturesNet(featuresNet)
  
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride/2)
  local roipool2 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  
  -- create spatial pyramid poolers
  local spp1 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  local spp2 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  
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
    classifier:add(nn.BatchNormalization(4096, 1e-3))
    classifier:add(nn.ReLU(true))  
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, 4096))
    classifier:add(nn.BatchNormalization(4096, 1e-3))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    return classifier
  end
  
  local nfeats 
  if flag_spp then
    nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height)
  else
    nfeats = num_feats_last_conv * (roipoolTable[1].width * roipoolTable[1].height)
  end
  local fully_connected1 = CreateClassifier(nfeats)
  local fully_connected2 = CreateClassifier(nfeats)
  
  -- create classifier + regressor
  local cls = nn.Linear(4096*2,nClasses+1)
  local reg = nn.Linear(4096*2,(nClasses+1)*4)
  local cls_join = nn.Sequential():add(nn.JoinTable(1,1)):add(cls)
  local reg_join = nn.Sequential():add(nn.JoinTable(1,1)):add(reg)

  -- link/combine all modules together using gModules
  local identity = nn.Identity()()
  features_pre = features_pre()
  
  local link_fc1_spp1_roipool1_feats, link_fc2_spp2_roipool2_feats
  if flag_spp then
    link_fc1_spp1_roipool1_feats = fully_connected1(spp1(roipool1({features_pre,identity})))
    link_fc2_spp2_roipool2_feats = fully_connected1(spp2(roipool2({features_post(features_pre),identity})))
  else
    link_fc1_spp1_roipool1_feats = fully_connected1(view1(roipool1({features_pre,identity})))
    link_fc2_spp2_roipool2_feats = fully_connected1(view2(roipool2({features_post(features_pre),identity})))
  end
  
  local model = nn.gModule( -- inputs
                            {features_pre, identity},
                            -- output
                            {cls_join({link_fc1_spp1_roipool1_feats, link_fc2_spp2_roipool2_feats}), 
                             reg_join({link_fc1_spp1_roipool1_feats, link_fc2_spp2_roipool2_feats})} 
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


function model_arch3_nngraphv2(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, flag_spp)
  -- verbose flag
  local verbose = verbose or false 
  local flag_spp = flag_spp or false
  
  -- split featureNet into two networks
  require 'nngraph'
  nngraph.setDebug(true)

  -- split features
  --local features_pre, features_post = SplitFeaturesNet(featuresNet)
  local features_pre, features_post = SplitFeaturesNetV2(featuresNet, 7)
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride/2)
  local roipool2 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  
  -- create spatial pyramid poolers
  local spp1 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  local spp2 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  
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
    --classifier:add(nn.BatchNormalization(4096, 1e-3))
    classifier:add(nn.ReLU(true))  
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, 4096))
    --classifier:add(nn.BatchNormalization(4096, 1e-3))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    return classifier
  end
  
  local nfeats 
  if flag_spp then
    nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height)
  else
    nfeats = num_feats_last_conv * (roipoolTable[1].width * roipoolTable[1].height)
  end
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
  
  local link_fc1_spp1_roipool1_feats, link_fc2_spp2_roipool2_feats
  if flag_spp then
    link_fc1_spp1_roipool1_feats = fully_connected1(spp1(roipool1({features_pre,identity})))
    link_fc2_spp2_roipool2_feats = fully_connected1(spp2(roipool2({features_post(features_pre),identity})))
  else
    link_fc1_spp1_roipool1_feats = fully_connected1(view1(roipool1({features_pre,identity})))
    link_fc2_spp2_roipool2_feats = fully_connected1(view2(roipool2({features_post(features_pre),identity})))
  end
  
  local link_fc1_fc2_fc3 = fc3({link_fc1_spp1_roipool1_feats, link_fc2_spp2_roipool2_feats})
  
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

function model_arch3_nngraphv3(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, flag_spp)
  -- verbose flag
  local verbose = verbose or false 
  local flag_spp = flag_spp or false
  
  -- split featureNet into two networks
  require 'nngraph'
  nngraph.setDebug(true)

  -- split features
  --local features_pre, features_post = SplitFeaturesNet(featuresNet)
  local features_pre, features_post = SplitFeaturesNetV2(featuresNet, 9)
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  local roipool2 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  
  -- create spatial pyramid poolers
  local spp1 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  local spp2 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  
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
    --classifier:add(nn.BatchNormalization(4096, 1e-3))
    classifier:add(nn.ReLU(true))  
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, 4096))
    --classifier:add(nn.BatchNormalization(4096, 1e-3))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    return classifier
  end
  
  local nfeats 
  if flag_spp then
    nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height)
  else
    nfeats = num_feats_last_conv * (roipoolTable[1].width * roipoolTable[1].height)
  end
  local fully_connected1 = CreateClassifier(nfeats)
  local fully_connected2 = CreateClassifier(nfeats)
  
    -- create classifier + regressor
  local cls = nn.Linear(4096*2,nClasses+1)
  local reg = nn.Linear(4096*2,(nClasses+1)*4)
  local cls_join = nn.Sequential():add(nn.JoinTable(1,1)):add(cls)
  local reg_join = nn.Sequential():add(nn.JoinTable(1,1)):add(reg)

  -- link/combine all modules together using gModules
  local identity = nn.Identity()()
  features_pre = features_pre()
  
  local link_fc1_spp1_roipool1_feats, link_fc2_spp2_roipool2_feats
  if flag_spp then
    link_fc1_spp1_roipool1_feats = fully_connected1(spp1(roipool1({features_pre,identity})))
    link_fc2_spp2_roipool2_feats = fully_connected1(spp2(roipool2({features_post(features_pre),identity})))
  else
    link_fc1_spp1_roipool1_feats = fully_connected1(view1(roipool1({features_pre,identity})))
    link_fc2_spp2_roipool2_feats = fully_connected1(view2(roipool2({features_post(features_pre),identity})))
  end
  
  local model = nn.gModule( -- inputs
                            {features_pre, identity},
                            -- output
                            {cls_join({link_fc1_spp1_roipool1_feats, link_fc2_spp2_roipool2_feats}), 
                             reg_join({link_fc1_spp1_roipool1_feats, link_fc2_spp2_roipool2_feats})} 
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


function model_arch4_nngraph(featuresNet, nClasses, num_feats_last_conv, feat_stride, roipoolTable, backend, verbose, flag_spp)
  -- verbose flag
  local verbose = verbose or false
  local flag_spp = flag_spp or false
  
  -- split featureNet into two networks
  require 'nngraph'
  nngraph.setDebug(true)

  -- split features
  local features_pre, features_post = SplitFeaturesNet(featuresNet)
  
  -- create roipoolers
  local roipool1 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride/2)
  local roipool2 = CreateROIPooler(roipoolTable[1].width, roipoolTable[1].height, feat_stride)
  
  -- create spatial pyramid poolers
  local spp1 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
  local spp2 = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{roipoolTable[1].width,roipoolTable[1].height}})
 
  -- create view layers (in case spp use is set to false)
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
  
  local nfeats 
  if flag_spp then nfeats = num_feats_last_conv * (1*1+2*2+3*3+roipoolTable[1].width * roipoolTable[1].height)
  else nfeats = num_feats_last_conv * (roipoolTable[1].width * roipoolTable[1].height) end
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
  local link_fc1_spp1_roipool1_feats, link_fc2_spp2_roipool2_feats
  if flag_spp then
    link_fc1_spp1_roipool1_feats = fully_connected1(spp1(roipool1({features_pre,identity})))
    link_fc2_spp2_roipool2_feats = fully_connected2(spp2(roipool2({features_post(features_pre),identity})))
  else
    link_fc1_spp1_roipool1_feats = fully_connected1(view1(roipool1({features_pre,identity})))
    link_fc2_spp2_roipool2_feats = fully_connected2(view2(roipool2({features_post(features_pre),identity})))
  end
  
  local model = nn.gModule( -- inputs
                            {features_pre, identity},
                            -- output
                            {cls_join({cls1(link_fc1_spp1_roipool1_feats), cls2(link_fc2_spp2_roipool2_feats)}),
                             reg_join({reg1(link_fc1_spp1_roipool1_feats), reg2(link_fc2_spp2_roipool2_feats)})} 
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