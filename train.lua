--[[
    Train a Fast-RCNN detector network using the Caltech Pedestrian Dataset.
]]


require 'paths'
require 'torch'
--local fastrcnn = require 'fastrcnn'
local fastrcnn = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/init.lua')

torch.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

print('==> (1/5) Load options')
local opts = paths.dofile('options.lua')
local opt = opts.parse(arg)


--------------------------------------------------------------------------------
-- Load dataset data loader
--------------------------------------------------------------------------------
-- The fastrcnn.train() function receives a table with loading functions to fetch
-- the necessary data from a data structure. This way it is easy to use other
-- datasets with the fastrcnn package.

print('==> (2/5) Load dataset data loader')
local data_loader = paths.dofile('data.lua')
local data_gen = data_loader('train')


--------------------------------------------------------------------------------
-- Load/Process regions-of-interest (RoIs)
--------------------------------------------------------------------------------

print('==> (3/5) Load roi proposals data')
local loadRoiDataFn = fastrcnn.utils.load.matlab.single_file

local train_proposals_fname = '../data/caltech/proposals/acf_train.mat'
local test_proposals_fname = '../data/caltech/proposals/acf_test.mat'

-- check if the proposal files exist. If not, process the detections for the dataset
if not (paths.filep(train_proposals_fname) and paths.filep(test_proposals_fname)) then
    -- run matlab script
end

local rois = {
    train = loadRoiDataFn(train_proposals_fname),
    test =  loadRoiDataFn(test_proposals_fname)
}



--------------------------------------------------------------------------------
-- Setup model
--------------------------------------------------------------------------------

local model, model_parameters
if opt.loadModel == '' then
    print('==> (4/5) Setup model:')
    local load_model = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn-example/models/init.lua')
    model, model_parameters = load_model(opt.netType, opt.nGPU, 2)
else
    print('==> (4/5) Load model from file: ')
    _, model_parameters = model(opt.netType, opt.nGPU, 2)
    model = torch.load(opt.loadModel)
end


--------------------------------------------------------------------------------
-- Train a  Fast R-CNN detector
--------------------------------------------------------------------------------

print('==> (5/5) Train Fast-RCNN model')
fastrcnn.train(data_gen, rois, model, model_parameters, opt)

print('Script complete.')