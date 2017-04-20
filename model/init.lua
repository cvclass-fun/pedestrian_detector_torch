--[[
    Models list
]]


require 'nn'
require 'cudnn'
require 'inn'
inn.utils = require 'inn.utils'
--local utils = require 'fastrcnn.utils'
local utils = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/init.lua')

------------------------------------------------------------------------------------------------------------

local function setup_feature_network(name)
    local model_loader
    local str = string.lower(name)
    if string.match(str, 'alexnet') then
        --model = require 'models.alexnet'
        model_loader = paths.dofile('alexnet.lua')
    elseif string.match(str, 'vgg') then
        --model = require 'models.vgg'
        model_loader = paths.dofile('vgg.lua')
    elseif string.match(str, 'zeiler') then
        --model = require 'models.zeiler'
        model_loader = paths.dofile('zeiler.lua')
    elseif string.match(str, 'resnet') then
        --model = require 'models.resnet'
        model_loader = paths.dofile('resnet.lua')
    elseif string.match(str, 'inception') then
        --model = require 'models.inceptionv3'
        model_loader = paths.dofile('inceptionv3.lua')
    else
        error('Undefined network type: ' .. name.. '. Available network types: alexnet, vgg, zeiler, resnet or inception.')
    end
    return model_loader(name)
end

------------------------------------------------------------------------------------------------------------

local function architecture_loader(name)
    local str = string.lower(name)
    if str == 'simple' then
        return paths.dofile('architecture_simple.lua')
    elseif str == 'concat' then
        return paths.dofile('architecture_concat.lua')
    elseif str == 'parallel' then
        return paths.dofile('architecture_parallel.lua')
    else
        error('Undefined architecture type: ' .. name.. '. Available architecture types: simple, concat or parallel.')
    end
end

------------------------------------------------------------------------------------------------------------

local function select_model(model_name, architecture_type, net_configs, nGPU, nClasses)
    assert(model_name)
    assert(architecture_type)
    assert(net_configs)
    assert(nGPU)
    assert(nClasses)

    -- setup feature network
    local featureNet, model_params = setup_feature_network(model_name)

    -- load architecture fun
    local create_architecture_fn = architecture_loader(architecture_type)

    -- setup the full model
    local model =  create_architecture_fn(featureNet, model_params, net_configs, nGPU, nClasses)

    return model, model_params
end

------------------------------------------------------------------------------------------------------------

return select_model