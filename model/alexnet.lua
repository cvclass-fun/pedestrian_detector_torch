--[[
    Alexnet FRCNN model.
]]


require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'
--local utils = require 'fastrcnn.utils'
local utils = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/init.lua')


local function CreateModel()

    local net = torch.load(projectDir .. '/data/pretrained_models/model_alexnet.t7')
    local model_parameters = torch.load(projectDir .. 'data/pretrained_models/parameters_alexnet.t7')
    local features = net

    -- remove all unnecessary layers
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())

    return features, model_parameters
end

return CreateModel