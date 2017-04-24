--[[
    Googlenet Inception v3 FRCNN model.
]]


require 'nn'
require 'cudnn'
require 'inn'
inn.utils = require 'inn.utils'
--local utils = require 'fastrcnn.utils'
local utils = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/init.lua')

------------------------------------------------------------------------------------------------------------

local function CreateModel()

    -- load features + model parameters (mean/std,stride/num feats (last conv)/colorspace format)
    local net = torch.load(projectDir .. '/data/pretrained_models/model_googlenet_inceptionv3_cunn.t7'):cuda()
    local model_parameters = torch.load(projectDir .. '/data/pretrained_models/parameters_googlenet_inceptionv3_cunn.t7')

    local input = torch.randn(1,3,299,299):cuda()
    local output1 = net:forward(input):clone()
    inn.utils.BNtoFixed(net, true)
    local output2 = net:forward(input):clone()
    assert((output1 - output2):abs():max() < 1e-5)

    local features = net
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())

    utils.model.testSurgery(input, utils.model.DisableFeatureBackprop, features, 16)
    utils.model.testSurgery(input, inn.utils.foldBatchNorm, features:findModules'nn.NoBackprop'[1])

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

local function select_model(name, features_id, roi_pool_size, cls_size)
    assert(name)
    assert(features_id)
    assert(roi_pool_size)
    assert(cls_size)

    local roi_pool_size = (roi_pool_size and roi_pool_size>0) or 6
    if features_id == 1 then
        return features_basic(roi_pool_size, cls_size)
    else
        error(('Invalid featuresID: %d. Valid ids: 1.'):format())
    end
end

------------------------------------------------------------------------------------------------------------

return select_model