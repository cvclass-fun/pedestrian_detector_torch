--[[
    Train a Fast-RCNN detector network using the Caltech Pedestrian Dataset.
]]


require 'paths'
require 'torch'
--local fastrcnn = require 'fastrcnn'
local fastrcnn = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/init.lua')

torch.setdefaulttensortype('torch.FloatTensor')
paths.dofile('projectdir.lua')


--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

print('==> (1/5) Load options')
local opts = paths.dofile('options.lua')
local opt = opts.parse(arg)


--------------------------------------------------------------------------------
-- Load dataset data loader
--------------------------------------------------------------------------------

print('==> (2/5) Load dataset data loader')
local data_loader = paths.dofile('data.lua')
local data_gen = data_loader(opt.dataset, 'train')


--------------------------------------------------------------------------------
-- Load regions-of-interest (RoIs)
--------------------------------------------------------------------------------

print('==> (3/5) Load roi proposals data')
local rois_loader = paths.dofile('rois.lua')
local rois = rois_loader(opt.dataset, opt.proposalAlg, 'train')


--------------------------------------------------------------------------------
-- Setup model
--------------------------------------------------------------------------------

local model, model_parameters
if opt.loadModel == '' then
    print('==> (4/5) Setup model:')
    local load_model = paths.dofile('model/init.lua')
    model, model_parameters = load_model(opt.netType, opt.clsType, opt.featID, opt.roi_size, opt.cls_size, opt.nGPU, 1)
else
    print('==> (4/5) Load model from file: ' .. opt.load)
    local model_data = torch.load(opt.load)
    model, model_parameters = model_data.model, model_data.params
end


--------------------------------------------------------------------------------
-- Train a  Fast R-CNN detector
--------------------------------------------------------------------------------

print('==> (5/5) Train Fast-RCNN model')
fastrcnn.train(data_gen, rois, model, model_parameters, opt)

print('Script complete.')