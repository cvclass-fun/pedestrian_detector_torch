--[[
    Models list
]]


require 'nn'
require 'cudnn'
require 'inn'
inn.utils = require 'inn.utils'
local utils = require 'fastrcnn.utils'

------------------------------------------------------------------------------------------------------------

local function setup_feature_network(name)
    local str = string.lower(name)
    if string.match(str, 'alexnet') then
        --model = require 'models.alexnet'
        return paths.dofile('alexnet.lua')
    elseif string.match(str, 'vgg') then
        --model = require 'models.vgg'
        return paths.dofile('vgg.lua')
    elseif string.match(str, 'zeiler') then
        --model = require 'models.zeiler'
        return paths.dofile('zeiler.lua')
    elseif string.match(str, 'resnet') then
        --model = require 'models.resnet'
        return paths.dofile('resnet.lua')
    elseif string.match(str, 'inception') then
        --model = require 'models.inceptionv3'
        return paths.dofile('inceptionv3.lua')
    else
        error('Undefined network type: ' .. name.. '. Available network types: alexnet, vgg, zeiler, resnet or inception.')
    end
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

local function select_model(model_name, architecture_type, cls_size, nGPU, nClasses)
    assert(model_name)
    assert(architecture_type)
    assert(nGPU)
    assert(nClasses)

    -- setup feature network
    local featureNet, model_params = setup_feature_network(model_name)

    -- load architecture fun
    local archFn = architecture_loader(architecture_type)

    -- setup classifier params
    local cls_params = {
        cls_size = cls_size
    }

    -- setup and return the model
    return archFn(featureNet, model_params, cls_params, nGPU, nClasses)
end

------------------------------------------------------------------------------------------------------------

return select_model