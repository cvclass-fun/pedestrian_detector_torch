--[[
    VGG (16-19) FRCNN model.
]]

require 'nn'
require 'cunn'
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
    local features, model_parameters
    if netType == 'vgg16' or netType == 'vgg' then
        local net = torch.load(projectDir .. '/data/pretrained_models/model_vgg16.t7')
        model_parameters = torch.load(projectDir .. '/data/pretrained_models/parameters_vgg16.t7')
        features = SelectFeatsDisableBackprop(net)
    elseif netType == 'vgg19' then
        local net = torch.load(projectDir .. '/data/pretrained_models/model_vgg19.t7')
        model_parameters = torch.load(projectDir .. '/data/pretrained_models/parameters_vgg19.t7')
        features = SelectFeatsDisableBackprop(net)
    else
        error('Undefined network type: '.. netType..'. Available networks: vgg16, vgg19.')
    end

    return features, model_parameters
end

------------------------------------------------------------------------------------------------------------

local function features_basic(name, roi_pool_size, cls_size)
    local featuresNet, model_parameters = CreateModel(name)

    -- classifier parameters (needed to config the classifier network with the correct parameters)
    local classifier_params = {
        {
            nfeats = model_parameters.num_feats,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        }
    }

    return featuresNet, model_parameters, classifier_params
end

------------------------------------------------------------------------------------------------------------

local function features_setup_2(name, roi_pool_size, cls_size)
    local features, model_parameters = CreateModel(name)

    local features_net1 = nn.Sequential()
    for i = 1, 19 do
        features_net1:add(features:get(i))
    end
    local features_net2 = nn.Sequential()
    for i = 20, 21 do
        features_net2:add(features:get(i))
    end

    local features_join = nn.Sequential()
        :add(features_net1)
        :add(nn.ConcatTable():add(nn.Identity()):add(features_net2))

    -- classifier parameters (needed to config the classifier network with the correct parameters)
    local classifier_params = {
        {
            nfeats = 512,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
        {
            nfeats = model_parameters.num_feats,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
    }

    return features_join, model_parameters, classifier_params
end

------------------------------------------------------------------------------------------------------------

local function features_setup_3(name, roi_pool_size, cls_size)
    local features, model_parameters = CreateModel(name)

    local features_net1 = nn.Sequential()
    for i = 1, 17 do
        features_net1:add(features:get(i))
    end
    local features_net2 = nn.Sequential()
    for i = 18, 21 do
        features_net2:add(features:get(i))
    end

    local features_join = nn.Sequential()
        :add(features_net1)
        :add(nn.ConcatTable():add(nn.Identity()):add(features_net2))

    -- classifier parameters (needed to config the classifier network with the correct parameters)
    local classifier_params = {
        {
            nfeats = 512,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
        {
            nfeats = model_parameters.num_feats,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
    }

    return features_join, model_parameters, classifier_params
end

------------------------------------------------------------------------------------------------------------

local function features_setup_4(name, roi_pool_size, cls_size)
    local features, model_parameters = CreateModel(name)

    local features_net1 = nn.Sequential()
    for i = 1, 14 do
        features_net1:add(features:get(i))
    end
    local features_net2 = nn.Sequential()
    for i = 15, 21 do
        features_net2:add(features:get(i))
    end

    local features_join = nn.Sequential()
        :add(features_net1)
        :add(nn.ConcatTable():add(nn.Identity()):add(features_net2))

    -- classifier parameters (needed to config the classifier network with the correct parameters)
    local classifier_params = {
        {
            nfeats = 512,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
        {
            nfeats = model_parameters.num_feats,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
    }

    return features_join, model_parameters, classifier_params
end

------------------------------------------------------------------------------------------------------------

local function features_setup_5(name, roi_pool_size, cls_size)
    local features, model_parameters = CreateModel(name)

    local features_net1 = nn.Sequential()
    for i = 1, 12 do
        features_net1:add(features:get(i))
    end
    local features_net2 = nn.Sequential()
    for i = 13, 21 do
        features_net2:add(features:get(i))
    end

    local features_join = nn.Sequential()
        :add(features_net1)
        :add(nn.ConcatTable():add(nn.Identity()):add(features_net2))

    -- classifier parameters (needed to config the classifier network with the correct parameters)
    local classifier_params = {
        {
            nfeats = 512,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
        {
            nfeats = model_parameters.num_feats,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
    }

    return features_join, model_parameters, classifier_params
end

------------------------------------------------------------------------------------------------------------

local function features_setup_6(name, roi_pool_size, cls_size)
    local features, model_parameters = CreateModel(name)

    local features_net1 = nn.Sequential()
    for i = 1, 10 do
        features_net1:add(features:get(i))
    end
    local features_net2 = nn.Sequential()
    for i = 11, 21 do
        features_net2:add(features:get(i))
    end

    local features_join = nn.Sequential()
        :add(features_net1)
        :add(nn.ConcatTable():add(nn.Identity()):add(features_net2))

    -- classifier parameters (needed to config the classifier network with the correct parameters)
    local classifier_params = {
        {
            nfeats = 512,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
        {
            nfeats = model_parameters.num_feats,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
    }

    return features_join, model_parameters, classifier_params
end

------------------------------------------------------------------------------------------------------------

local function features_setup_7(name, roi_pool_size, cls_size)
    local features, model_parameters = CreateModel(name)

    local features_net1 = nn.Sequential()
    for i = 1, 14 do
        features_net1:add(features:get(i))
    end
    local features_net2 = nn.Sequential()
    for i = 15, 19 do
        features_net2:add(features:get(i))
    end
    local features_net3 = nn.Sequential()
    for i = 20, 21 do
        features_net3:add(features:get(i))
    end

    local features_join = nn.Sequential()
        :add(features_net1)
        :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(nn.Sequential()
                :add(features_net2)
                :add(nn.ConcatTable()
                    :add(nn.Identity())
                    :add(features_net3))))

    -- classifier parameters (needed to config the classifier network with the correct parameters)
    local classifier_params = {
        {
            nfeats = 512,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
        {
            nfeats = 512,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
        {
            nfeats = model_parameters.num_feats,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
    }

    return features_join, model_parameters, classifier_params
end

------------------------------------------------------------------------------------------------------------

local function select_model(name, features_id, roi_pool_size, cls_size)
    assert(name)
    assert(features_id)
    assert(roi_pool_size)
    assert(cls_size)

    local roi_pool_size = (roi_pool_size and roi_pool_size>0) or 7
    if features_id == 1 then
        return features_basic(name, roi_pool_size, cls_size)
    elseif features_id == 2 then
        return features_setup_2(name, roi_pool_size, cls_size)
    elseif features_id == 3 then
        return features_setup_3(name, roi_pool_size, cls_size)
    elseif features_id == 4 then
        return features_setup_4(name, roi_pool_size, cls_size)
    elseif features_id == 5 then
        return features_setup_5(name, roi_pool_size, cls_size)
    elseif features_id == 6 then
        return features_setup_6(name, roi_pool_size, cls_size)
    elseif features_id == 7 then
        return features_setup_7(name, roi_pool_size, cls_size)
    else
        error(('Invalid featuresID: %d. Valid ids: 1, 2, 3, 4, 5, 6 or 7'):format())
    end
end

------------------------------------------------------------------------------------------------------------

return select_model