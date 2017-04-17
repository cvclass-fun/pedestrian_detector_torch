--[[
    Test a FRCNN model using the caltech dataset.
]]

-- (1) Load packages and options
print('==> (1/7) Load dependencies + options')
paths.dofile('/home/mf/Toolkits/Codigo/git/fast-rcnn-torch/FastRCNN.lua')
local opt = FastRCNN.Configs.parse(arg)

-- do some initializations
cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)

-- (2) Load dataset
print('==> (2/7) Load dataset metadata')
local dataset = Framework.Dataset(opt, 'caltech','skip30reasonable')

-- (3) load/preprocess roi data
print('==> (3/7) Load ROI boxes data')
--opt.test_rois_file = '/media/HDD2/miguel_DATA/Full-FastRCNN-v2/proposals/EdgeBoxesCaltechSKIP30test_1k.mat'
--opt.test_rois_file = '/media/HDD2/miguel_DATA/proposals/EdgeBoxes/CaltechFULLtest_skip30_BBs1000.mat'
--opt.test_rois_file = '/media/HDD2/miguel_DATA/proposals/EdgeBoxes/CaltechFULLtest_skip30_BBs10000_ar.mat'
--opt.test_rois_file = '/home/mf/Toolkits/Codigo/git/pedestrian-detector-torch/data/Caltech/proposals/caltech_test_10x.mat'

-- (4) create model
local model_folder_path = opt.load_model_path
--model_folder_path = '/home/mf/Toolkits/Codigo/git/pedestrian-detector-torch/data/models/Tue_Jun__7_15:55:34_2016_alexnet/'
print('==> (4/7) Load model from disk: ' .. paths.concat(model_folder_path, 'model.t7'))
local modelPath = paths.concat(model_folder_path, 'model.t7')
local model_params = torch.load(paths.concat(model_folder_path, 'model_parameters.t7'))
local model = torch.load(modelPath)
model:cuda()


-- (5) create fastrcnn object
print('==> (5/7) Create fast rcnn object class (to be used to test the network)')
local frcnn = FastRCNN.NewObject({
                                  dataset = dataset,
                                  model = model,
                                  roi_box_test = opt.test_rois_file,
                                  configs = opt,
                                  model_normalization_parameters = model_params,
                                })


-- (6) train fast rcnn model
print('==> (6/7) Start network testing step: ')
frcnn:Test()

-- (7) Script completed
print('==> (7/7) Done.')


