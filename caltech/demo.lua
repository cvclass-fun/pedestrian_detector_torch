--[[
    Demo script to display the model's detections on a test image. 
]]

require 'paths'

-- (1) Load packages and options
print('==> (1/8) Load dependencies + options')
paths.dofile('/home/mf/Toolkits/Codigo/git/fast-rcnn-torch/FastRCNN.lua')
local opt = FastRCNN.Configs.parse(arg)

-- do some initializations
cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)

-- (2) Load dataset
print('==> (2/8) Load dataset metadata')
local dataset = Framework.Dataset(opt, 'caltech','skip30near')

-- (3) load/preprocess roi data
print('==> (3/8) Load ROI boxes data')
opt.test_rois_file = '/media/HDD2/miguel_DATA/proposals/EdgeBoxes/CaltechFULLtest_skip30_BBs1000.mat'

-- (4) create model
local model_folder_path = opt.load_model_path
--model_folder_path = '/media/HDD2/miguel_DATA/data_cache/FastRCNN/caltech_SKIP30_OVERALL/Fri_Jun__3_00:28:31_2016_alexnet/'
print('==> (4/8) Load model from disk: ' .. paths.concat(model_folder_path, 'model.t7'))
local modelPath = paths.concat(model_folder_path, 'model.t7')
local model_params = torch.load(paths.concat(model_folder_path, 'model_parameters.t7'))
local model = torch.load(modelPath)
model:cuda()


-- (5) create fastrcnn object
print('==> (5/8) Create fast rcnn object class (to be used to test the network)')
local frcnn = FastRCNN.NewObject({
                                  dataset = dataset,
                                  model = model,
                                  configs = opt,
                                  model_normalization_parameters = model_params,
                                })


-- (4) load test data
print('==> (6/8) Load test image + proposals boxes')
-- Paths
local imageID = 453
local image_path = ffi.string(dataset.data.test.fileName[imageID]:data())
print('Image path: ' .. image_path)
-- Loading proposals from file
local proposals = matio.load(opt.test_rois_file)['boxes'][imageID]:float()
proposals = proposals:index(2, torch.LongTensor{2,1,4,3}):float()

-- Loading the image
local im = image.load(image_path)

-- (5) Process image detection with the FRCNN
print('==> (7/8) Process image detections')
-- detect !
local scores, bboxes = frcnn:Detect(im, proposals)
-- visualization
local threshold = 0.5
-- classes from Pascal used for training the model
local cls = {'person'}
FastRCNN.utils.visualize_detections(im, bboxes, scores, threshold, cls)


-- (7) Script completed
print('==> (8/8) Done.')


