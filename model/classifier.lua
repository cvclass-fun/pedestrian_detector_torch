--[[
    Classifier networks. Simple(vanilla Fast R-CNN), Concat and Parallel.
]]


require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'
local utils = require 'fastrcnn.utils'

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
    local nfeats = cls_params[1].nfeats
    local roi_size = cls_params[1].roi_size
    local cls_size = cls_params[1].cls_size
    local pixel_stride = cls_params[1].stride

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
    if #cls_params == 1 then
        return classifier_simple(cls_params, nClasses)
    end

    local concat = nn.ConcatTable()
    local nfeats = 0
    for i=1, #cls_params do
        nfeats = nfeats + cls_params[i].nfeats
        concat:add(nn.Sequential()
            :add(nn.ConcatTable()
                :add(nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(i)))
                :add(nn.SelectTable(2)))
            :add(inn.ROIPooling(cls_params[i].roi_size, cls_params[i].roi_size, 1/cls_params[i].stride)))
    end

    local classifier = nn.Sequential()
        :add(concat)
        :add(nn.JoinTable(2))
        :add(nn.View(-1):setNumInputDims(3))
        :add(basic_classifier_network(nfeats, cls_params[1].roi_size, cls_params[1].cls_size))
        :add(utils.model.CreateClassifierBBoxRegressor(cls_params[1].cls_size, nClasses))

    return classifier
end

------------------------------------------------------------------------------------------------------------

local function classifier_parallel(cls_params, nClasses)
--[[Create multiple parallel classifiers from multiple layers outputs of a pre-trained
network and combine all outputs into a single one.]]
    if #cls_params == 1 then
        return classifier_simple(cls_params, nClasses)
    end

    local concat = nn.ConcatTable()
    local select_cls = nn.ConcatTable()
    local select_reg = nn.ConcatTable()
    for i=1, #cls_params do
        concat:add(nn.Sequential()
            :add(nn.ConcatTable()
                :add(nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(i)))
                :add(nn.SelectTable(2)))
            :add(classifier_simple({cls_params[i]}, nClasses)))
        select_cls:add(nn.Sequential():add(nn.SelectTable(i)):add(nn.SelectTable(1)))
        select_reg:add(nn.Sequential():add(nn.SelectTable(i)):add(nn.SelectTable(2)))
    end

    local nclass = nClasses + 1

    local final_cls = nn.Sequential()
        :add(select_cls)
        :add(nn.JoinTable(2))
        :add(nn.View(-1):setNumInputDims(1))
        :add(nn.Linear(nclass*#cls_params, nclass))

    local final_reg = nn.Sequential()
        :add(select_reg)
        :add(nn.JoinTable(2))
        :add(nn.View(-1):setNumInputDims(1))
        :add(nn.Linear(#cls_params*nclass*4, nclass*4))

    local classifier = nn.Sequential()
        :add(concat)
        :add(nn.ConcatTable()
            :add(final_cls)
            :add(final_reg))

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
