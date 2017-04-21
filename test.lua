--[[
    Test a pedestrian detector network's mAP accuracy.
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
local data_gen = data_loader(opt.dataset, 'test')


--------------------------------------------------------------------------------
-- Load regions-of-interest (RoIs)
--------------------------------------------------------------------------------

print('==> (3/5) Load roi proposals data')
local rois_loader = paths.dofile('rois.lua')
local rois = rois_loader(opt.dataset, opt.proposalAlg, 'test')


--------------------------------------------------------------------------------
-- Setup model
--------------------------------------------------------------------------------

print('==> (4/5) Load model: ' .. opt.load)
local model, model_parameters = unpack(torch.load(opt.load)


--------------------------------------------------------------------------------
-- Test detector mAP
--------------------------------------------------------------------------------

print('==> (5/5) Test Fast-RCNN model')
fastrcnn.test(data_gen, rois, model, model_parameters, opt)

print('Script complete.')