--[[
    Train a FRCNN model using the caltech dataset.
]]

-- (1) Load packages and options
print('==> (1/7) Load dependencies + options')
paths.dofile('/home/mf/Toolkits/Codigo/git/fast-rcnn-torch/FastRCNN.lua')
local opt = FastRCNN.Configs.parse(arg)
--opt.schedule = {{7,1e-3,5e-4},{3, 1e-4, 5e-4}}
--opt.batchSize = 128

-- do some initializations
cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)

-- (2) Load dataset
print('==> (2/7) Load dataset metadata')
local dataset = Framework.Dataset(opt, 'caltech','skip30overall')
--local dataset2 = Framework.Dataset(opt, 'caltech','FULLOVERALL')

-- (3) load/preprocess roi data
print('==> (3/7) Setup ROI boxes data filepath')
opt.train_rois_file = '/home/mf/Toolkits/Codigo/git/pedestrian-detector-torch/data/Caltech/proposals/EdgeBoxes_CaltechTrain_skip=30_BBs=50000_aspectRatio=0.6.mat'
opt.test_rois_file = '/home/mf/Toolkits/Codigo/git/pedestrian-detector-torch/data/Caltech/proposals/EdgeBoxes_CaltechTest_skip=30_BBs=50000_aspectRatio=0.6.mat'


-- (4) create model
print('==> (4/7) Load model + criterion')
local featuresNet, model_params, roipool_width, roipool_height = FastRCNN.model.LoadFeatures(opt.modelName, opt.save_dir)
roipool_width, roipool_height = 6, 6
local model, criterion = FastRCNN.model.CreateModel(featuresNet, dataset.data.train.className:size(1), model_params.num_feats, roipool_width, roipool_height, model_params.stride, opt.backend, opt.train_bbox_regressor, opt.verbose)
opt.model_params = model_params -- used when saving the training specs log file.

-- (5) create fastrcnn object
print('==> (5/7) Create fast rcnn object class (to be used to train the network)')
local frcnn = FastRCNN.NewObject({
                                  dataset = dataset,
                                  model = model,
                                  criterion = criterion,
                                  roi_box_train = opt.train_rois_file,
                                  roi_box_test = opt.test_rois_file,
                                  configs = opt,
                                  model_normalization_parameters = model_params,
                                })

-- (6) train fast rcnn model
print('==> (6/7) Start network training...')
frcnn:Train()

-- (7) Script completed
print('==> (7/7) Done.')


