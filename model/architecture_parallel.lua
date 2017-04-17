--[[
    Create multiple parallel classifiers from multiple layers outputs of a pre-trained network and combine all outputs into a single one.
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