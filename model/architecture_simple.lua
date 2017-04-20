--[[
    Basic Fast R-CNN architecture.
]]


require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'
--local utils = require 'fastrcnn.utils'
local utils = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/init.lua')

------------------------------------------------------------------------------------------------------------

local function setup_classifier_network(nfeats, roi_size, cls_size)
    local classifier = nn.Sequential()
        :add(nn.Linear(nfeats*roi_size*roi_size, cls_size))
        :add(nn.ReLU(true))
        :add(nn.Dropout(0.5))
        :add(nn.Linear(cls_size, cls_size))
        :add(nn.ReLU(true))
        :add(nn.Dropout(0.5))
    return classifier
end

------------------------------------------------------------------------------------------------------------

local function setup_model(featuresNet, model_parameters, cls_params, nGPU, nClasses)

    local roi_size = cls_params[1][1]
    local cls_size = cls_params[1][2]
    local pixel_stride = model_parameters.stride
    local nfeats = model_parameters.num_feats

    -- create model
    local model = nn.Sequential()
        :add(nn.ParallelTable()
            :add(utils.model.makeDataParallel(featuresNet, nGPU))
            :add(nn.Identity())
        )
        :add(inn.ROIPooling(roi_size, roi_size, 1/pixel_stride))
        :add(nn.View(-1):setNumInputDims(3))
        :add(setup_classifier_network(nfeats, roi_size, cls_size))
        :add(utils.model.CreateClassifierBBoxRegressor(cls_size, nClasses))

    model:cuda()

    return model
end

------------------------------------------------------------------------------------------------------------

return setup_model