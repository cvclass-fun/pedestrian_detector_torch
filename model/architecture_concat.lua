--[[
    Concatenate multiple layers outputs of a network Basic Fast R-CNN architecture.
]]

require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'
--local utils = require 'fastrcnn.utils'
local utils = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/init.lua')

------------------------------------------------------------------------------------------------------------

local function setup_model(featuresNet, model_parameters, cls_params, nGPU, classes)
end

------------------------------------------------------------------------------------------------------------

return setup_model