--[[
    Classifier networks. Simple(vanilla Fast R-CNN), Concat and Parallel.
]]


require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'
--local utils = require 'fastrcnn.utils'
local utils = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/init.lua')

------------------------------------------------------------------------------------------------------------

local function basic_classifier_network(nfeats, roi_size, cls_size)
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

local function classifier_simple(cls_params, nClasses)
--[[Simple vanilla Fast R-CNN classifier.]]
    local nfeats = cls_params.nfeats
    local roi_size = cls_params.roi_size
    local cls_size = cls_params.cls_size
    local pixel_stride = cls_params.stride

    local classifier = nn.Sequential()
        :add(inn.ROIPooling(roi_size, roi_size, 1/pixel_stride))
        :add(nn.View(-1):setNumInputDims(3))
        :add(basic_classifier_network(nfeats, roi_size, cls_size))
        :add(utils.model.CreateClassifierBBoxRegressor(cls_size, nClasses))

    return classifier
end

------------------------------------------------------------------------------------------------------------

local function classifier_concat(cls_params, nClasses)
--[[Concatenate multiple layers outputs of a network Basic Fast R-CNN architecture.]]
    local prl = nn.ParallelTable()
    local nfeats = 0
    for i=1, #cls_params do
        nfeats = nfeats + cls_params[i].nfeats
        prl:add(inn.ROIPooling(params.roi_size, params.roi_size, 1/params.pixel_stride))
    end

    local join = nn.JoinTable(2)

    local classifier = nn.Sequential()
        :add(prl)
        :add(join)
        :add(nn.View(-1):setNumInputDims(3))
        :add(basic_classifier_network(nfeats, cls_params[1].roi_size, cls_params[1].cls_size))
        :add(utils.model.CreateClassifierBBoxRegressor(cls_params[1].cls_size, nClasses))

    return classifier
end

------------------------------------------------------------------------------------------------------------

local function classifier_parallel(cls_params, nClasses)
--[[Create multiple parallel classifiers from multiple layers outputs of a pre-trained network and combine all outputs into a single one.]]
    local prl = nn.ParallelTable()
    local select_cls = nn.Sequential()
    local select_reg = nn.Sequential()
    for i=1, #cls_params do
        prl:add(classifier_simple(cls_params[i], nClasses))
        select_cls:add(nn.Sequential():add(nn.SelectTable(i):add(nn.SelectTable(1))))
        select_reg:add(nn.Sequential():add(nn.SelectTable(i):add(nn.SelectTable(2))))
    end

    local final_cls = nn.Sequential()
        :add(select_cls)
        :add(nn.JoinTable(2))
        :add(nn.View(-1):setNumInputDims(3))
        :add(nn.Linear(nClasses*#cls_params), nClasses)

    local final_reg = nn.Sequential()
        :add(select_reg)
        :add(nn.JoinTable(2))
        :add(nn.View(-1):setNumInputDims(3))
        :add(nn.Linear(#cls_params*nClasses*4), nClasses*4)

    local classifier = nn.Sequential()
        :add(prl)
        :add(nn.ConcatTable():add(final_cls):add(final_reg))

    return classifier
end

------------------------------------------------------------------------------------------------------------

local function select_classifier(name)
    local str = string.lower(name)
    if str == 'simple' then
        return classifier_simple
    elseif str == 'concat' then
        return classifier_concat
    elseif str == 'parallel' then
        return classifier_parallel
    else
        error('Undefined architecture type: ' .. name.. '. Available architecture types: simple, concat or parallel.')
    end
end

------------------------------------------------------------------------------------------------------------

return select_classifier
