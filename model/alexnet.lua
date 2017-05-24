--[[
    Alexnet FRCNN model.
]]


require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'

------------------------------------------------------------------------------------------------------------

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

------------------------------------------------------------------------------------------------------------

local function features_basic(roi_pool_size, cls_size)
    local featuresNet, model_parameters = CreateModel()

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

local function features_setup_2(roi_pool_size, cls_size)
    local features, model_parameters = CreateModel()

    local features_net1 = nn.Sequential()
    for i = 1, 12 do
        features_net1:add(features:get(i))
    end
    local features_net2 = nn.Sequential()
    for i = 13, 14 do
        features_net2:add(features:get(i))
    end

    local features_join = nn.Sequential()
        :add(features_net1)
        :add(nn.ConcatTable():add(nn.Identity()):add(features_net2))

    -- classifier parameters (needed to config the classifier network with the correct parameters)
    local classifier_params = {
        {
            nfeats = 384,
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

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

local function features_setup_3(roi_pool_size, cls_size)
    local features, model_parameters = CreateModel()

    local features_net1 = nn.Sequential()
    for i = 1, 10 do
        features_net1:add(features:get(i))
    end
    local features_net2 = nn.Sequential()
    for i = 11, 14 do
        features_net2:add(features:get(i))
    end

    local features_join = nn.Sequential()
        :add(features_net1)
        :add(nn.ConcatTable():add(nn.Identity()):add(features_net2))

    -- classifier parameters (needed to config the classifier network with the correct parameters)
    local classifier_params = {
        {
            nfeats = 384,
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

local function features_setup_4(roi_pool_size, cls_size)
    local features, model_parameters = CreateModel()

    local features_net1 = nn.Sequential()
    for i = 1, 10 do
        features_net1:add(features:get(i))
    end
    local features_net2 = nn.Sequential()
    for i = 11, 12 do
        features_net2:add(features:get(i))
    end
    local features_net3 = nn.Sequential()
    for i = 13, 14 do
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
            nfeats = 384,
            roi_size = roi_pool_size,
            cls_size = cls_size,
            stride = model_parameters.stride
        },
        {
            nfeats = 384,
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

    local roi_pool_size = (roi_pool_size and roi_pool_size>0) or 6
    if features_id == 1 then
        return features_basic(roi_pool_size, cls_size)
    elseif features_id == 2 then
        return features_setup_2(roi_pool_size, cls_size)
    elseif features_id == 3 then
        return features_setup_3(roi_pool_size, cls_size)
    elseif features_id == 4 then
        return features_setup_4(roi_pool_size, cls_size)
    else
        error(('Invalid featuresID: %d. Valid ids: 1, 2, 3 or 4'):format())
    end
end

------------------------------------------------------------------------------------------------------------

return select_model