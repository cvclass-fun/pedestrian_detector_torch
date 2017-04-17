--[[
    Benchmark a FRCNN model on the caltech dataset.
]]

-- (1) Load packages and options
print('==> (1/8) Load dependencies + options')
paths.dofile('/home/mf/Toolkits/Codigo/git/fast-rcnn-torch/FastRCNN.lua')
local opt = FastRCNN.Configs.parse(arg)

-- do some initializations
cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)

-- (2) Load dataset
print('==> (2/8) Load dataset metadata')
local dataset = Framework.Dataset(opt, 'caltech','skip30overall')

-- (3) load/preprocess roi data
print('==> (3/8) Setup ROI boxes data')
--opt.test_rois_file = '/media/HDD2/miguel_DATA/proposals/EdgeBoxes/CaltechFULLtest_skip30_BBs10000.mat'

-- (4) create model
local model_folder_path = opt.load_model_path
--local model_folder_path = '/media/HDD2/miguel_DATA/data_cache/FastRCNN/caltech_SKIP30OVERALL/Fri_May_27_11:51:33_2016_alexnet/'
print('==> (4/8) Load model from disk: ' .. paths.concat(model_folder_path, 'model.t7'))
local modelPath = paths.concat(model_folder_path, 'model.t7')
local model_params = torch.load(paths.concat(model_folder_path, 'model_parameters.t7'))
local model = torch.load(modelPath)
model:cuda()

-- (5) create fastrcnn object
print('==> (5/8) Create fast rcnn object class (to be used to detect objects in images using the network)')
local frcnn = FastRCNN.NewObject({
                                  dataset = dataset,
                                  model = model,
                                  configs = opt,
                                  model_normalization_parameters = model_params,
                                })

-- (6)  fast rcnn model
print('==> (6/8) Compute caltech\'s pedestrian detection with our algorithm')
local ProcessDatasetDetections = paths.dofile('../utils/process_detections_dataset.lua')
local path = '../data/Caltech' --'/home/mf/Toolkits/Codigo/git/pedestrian-detector-torch/data/Caltech'
local flag_force_compute_detect = true -- forces to compute the image detections for the dataset
if not paths.dirp(paths.concat(path, 'algorithms', 'Ours')) or flag_force_compute_detect then
  os.execute('rm -rf ' ..  paths.concat(path, 'algorithms', 'Ours'))
  print('creating directory: ' .. paths.concat(path, 'algorithms', 'Ours'))
  os.execute('mkdir -p ' ..  paths.concat(path, 'algorithms', 'Ours'))
  os.execute('mkdir -p ' ..  paths.concat(path, 'plots'))
  ProcessDatasetDetections(frcnn, dataset.data.test, opt.test_rois_file, 0.5, paths.concat(path, 'algorithms', 'Ours'))
end

-- (6) benchmark the fast rcnn model
print('==> (7/8) Benchmark caltech algorithm')
local datasetPathDir = dataset.path_dataset ----'/media/HDD2/miguel_DATA/data_cache/datasets/Caltech/'
local experimentIDini = 1  --% Process experiment all experiments
local experimentIDend = 18 --% Process experiment all experiments 
local algPlotNum = 15   --% number of algorithms to print
local dataNamesID = 1   --% process 'Caltech-USA'
local algorithmsDir = paths.concat(path, 'algorithms')
local savePlotDir = paths.concat(path, 'plots')
local reset_eval_ours = true -- true- delete old files and process new one | false - use old processed files if available
if reset_eval_ours then
  print('Cleaning old cached results...')
  local dir = require 'pl.dir'
  local files = dir.getallfiles(paths.concat(path, 'plots'))
  for i=1, #files do
    if string.match(files[i]:sub(#files[i]-8), '-Ours.mat') then
      os.execute('rm -rf ' .. files[i])
    end
  end
  print('Done.')
end

-- benchmark algorithm
os.execute(('cd %s && matlab -nodisplay -nodesktop -r "try, Run_evaluate(%d, %d, %d, %d, \'%s\', \'%s\', \'%s\'), catch, exit, end, exit"'):format('/home/mf/Toolkits/Codigo/git/object_detection/Pedestrian/caltech', experimentIDini, experimentIDend, algPlotNum, dataNamesID, datasetPathDir, algorithmsDir, savePlotDir))

-- (7) Script completed
print('==> (8/8) Done.')


