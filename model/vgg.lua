--[[
    VGG (16-19) FRCNN model.
]]

require 'nn'
require 'cudnn'
require 'inn'
local utils = require 'fastrcnn.utils'

------------------------------------------------------------------------------------------------------------

local function CreateModel(netType)

    assert(netType)


    local function SelectFeatsDisableBackprop(net)
        local features = net
        features:remove(features:size()) -- remove logsoftmax layer
        features:remove(features:size()) -- remove 3rd linear layer
        features:remove(features:size()) -- remove 2nd dropout layer
        features:remove(features:size()) -- remove 2nd last relu layer
        features:remove(features:size()) -- remove 2nd linear layer
        features:remove(features:size()) -- remove 1st dropout layer
        features:remove(features:size()) -- remove 1st relu layer
        features:remove(features:size()) -- remove 1st linear layer
        features:remove(features:size()) -- remove view layer
        features:remove(features:size()) -- remove max pool
        utils.model.DisableFeatureBackprop(features, 10)
        return features
    end


    -- load features + model parameters (mean/std,stride/num feats (last conv)/colorspace format)
    local features
    if netType == 'vgg16' or netType == 'vgg' then
        local net = torch.load(projectDir .. '/data/pretrained_models/model_vgg16.t7'))
        local model_parameters = torch.load(projectDir .. '/data/pretrained_models/parameters_vgg16.t7'))
        features = SelectFeatsDisableBackprop(net)
    elseif netType == 'vgg19' then
        local net = torch.load(projectDir .. '/data/pretrained_models/model_vgg19.t7'))
        local model_parameters = torch.load(projectDir .. '/data/pretrained_models/parameters_vgg19.t7'))
        features = SelectFeatsDisableBackprop(net)
    else
        error('Undefined network type: '.. netType..'. Available networks: vgg16, vgg19.')
    end

    return features, model_parameters
end

return CreateModel