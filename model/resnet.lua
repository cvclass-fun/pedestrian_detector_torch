--[[
    Resnet (18, 32, 50, 101, 152, 200) FRCNN model.
]]


require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'
inn.utils = require 'inn.utils'
local utils = require 'fastrcnn.utils'

------------------------------------------------------------------------------------------------------------

local function CreateModel(netType)

    assert(netType)

    local available_nets = {
        ['resnet18'] = {512, 'resnet-18'},
        ['resnet32'] = {512, 'resnet-32'},
        ['resnet50'] = {2048, 'resnet-50'},
        ['resnet101'] = {2048, 'resnet-101'},
        ['resnet152'] = {2048, 'resnet-152'},
        ['resnet200'] = {2048,'resnet-200'}
    }

    local info = available_nets[string.lower(netType)]
    assert(info, 'Undefined network: '..netType..'. Available networks: resnet18, resnet32, resnet50, resnet101, resnet152, resnet200.')

    -- load features + model parameters (mean/std,stride/num feats (last conv)/colorspace format)
    local net = torch.load(projectDir .. '/data/pretrained_models/model_'..info[2]..'.t7')
    local model_parameters = torch.load(projectDir .. '/data/pretrained_models/parameters_'..info[2]..'.t7')
    net:cuda():evaluate()
    local features = net
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())

    local input = torch.randn(1, 3, 224, 224):cuda()
    utils.model.testSurgery(input, utils.model.DisableFeatureBackprop, features, 5)
    utils.model.testSurgery(input, inn.utils.foldBatchNorm, features:findModules'nn.NoBackprop'[1])
    utils.model.testSurgery(input, inn.utils.BNtoFixed, features, true)
    utils.model.testSurgery(input, inn.utils.BNtoFixed, net, true)

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

local function select_model(name, features_id, roi_pool_size, cls_size)
    assert(name)
    assert(features_id)
    assert(roi_pool_size)
    assert(cls_size)

    local roi_pool_size = (roi_pool_size and roi_pool_size>0) or 6
    if features_id == 1 then
        return features_basic(name, roi_pool_size, cls_size)
    else
        error(('Invalid featuresID: %d. Valid ids: 1.'):format())
    end
end

------------------------------------------------------------------------------------------------------------

return select_model