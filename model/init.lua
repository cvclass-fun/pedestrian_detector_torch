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

local function setup_feature_network(name, features_id, roi_pool_size, cls_size)
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
    return model_loader(name, features_id, roi_pool_size, cls_size)
end

------------------------------------------------------------------------------------------------------------

local function setup_classifier_network(name)
    local select_classifier = paths.dofile('classifier')
    local model_loader
    local str = string.lower(name)
    local classifier_loader = select_classifier(str)
    return classifier_loader(classifier_params, nClasses)
end

------------------------------------------------------------------------------------------------------------

local function select_model(model_name, architecture_type, features_id, roi_pool_size, cls_size, nGPU, nClasses)
    assert(model_name)
    assert(architecture_type)
    assert(net_configs)
    assert(nGPU)
    assert(nClasses)

    -- setup feature network
    local featureNet, model_params, classifier_params = setup_feature_network(model_name, features_id, roi_pool_size, cls_size)

    -- setup classifier network
    local classifierNet = setup_classifier_network(architecture_type, classifier_params, nClasses + 1)

    -- combine feature + classifier networks
    local model = nn.Sequential()
        :add(nn.ParallelTable()
            :add(utils.model.makeDataParallel(featuresNet, nGPU))
            :add(nn.Identity())
        )
        :add(classifierNet)

    return model, model_params
end

------------------------------------------------------------------------------------------------------------

return select_model