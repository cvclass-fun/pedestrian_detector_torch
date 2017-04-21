--[[
    Data loading method for the pascal voc 2007 dataset.
]]


local function get_db_loader(name)
    local dbc = require 'dbcollection.manager'

    local dbloader
    local str = string.lower(name)
    if str == 'caltech' then
        dbloader = dbc.load{name='caltech_pedestrian', task='detection'}
    elseif str == 'caltech_10x' then
        dbloader = dbc.load{name='caltech_pedestrian', task='detection_10x'}
    elseif str == 'caltech_30x' then
        dbloader = dbc.load{name='caltech_pedestrian', task='detection_30x'}
    elseif str == 'eth' then
        error('eth dataset not yet defined.')
    elseif str == 'inria' then
        error('inria dataset not yet defined.')
    elseif str == 'tudbrussels' then
        error('tudbrussels dataset not yet defined.')
    else
        error(('Undefined dataset: %s. Available options: caltech, eth, inria or tudbrussels'):format(name))
    end
    return dbloader
end

------------------------------------------------------------------------------------------------------------

local function fetch_data_set(name, set_name)

    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str
    local pad = require 'dbcollection.utils.pad'
    local unpad = pad.unpad_list

    -- get dataset loader
    local dbloader = get_db_loader(name)

    local loader = {}

    -- get image file path
    loader.getFilename = function(idx)
        local filename = ascii2str(dbloader:get(set_name, 'image_filenames', idx))[1]
        return paths.concat(dbloader.data_dir, filename)
    end

    -- get image ground truth boxes + class labels
    loader.getGTBoxes = function(idx)
        local objs_ids = unpad(dbloader:get(set_name, 'list_object_ids_per_image', idx):squeeze())
        if #objs_ids == 0 then
            return nil
        end
        local gt_boxes, gt_classes = {}, {}
        for _, id in ipairs(objs_ids) do
            local objID = dbloader:object(set_name, id + 1):squeeze()
            local bbox = dbloader:get(set_name, 'boxes', objID[3]):squeeze()
            local label = objID[2]
            if label == 1 or label == 2 then -- (1 - 'person' | 2 - 'person-fa'')
                table.insert(gt_boxes, bbox:totable())
                table.insert(gt_classes, 1)
            end
        end

        if #gt_boxes == 0 then
            return nil
        end

        gt_boxes = torch.FloatTensor(gt_boxes)
        return gt_boxes, gt_classes
    end

    -- number of samples
    local nfiles = dbloader:size(set_name, 'image_filenames')[1]
    loader.nfiles = nfiles

    -- classes
    local class_names = ascii2str(dbloader:get(set_name, 'classes', 1))
    loader.classLabel = class_names -- fetch the first two classes

    return loader
end

------------------------------------------------------------------------------------------------------------

local function loader_train(name)
    return {
        train = fetch_data_set(name, 'train'),
        test = fetch_data_set(name, 'test')
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_test(name)
    return {
        test = fetch_data_set(name, 'test')
    }
end

------------------------------------------------------------------------------------------------------------

local function data_loader(name, mode)
    assert(name)
    assert(mode)

    if mode == 'train' then
        return function() return loader_train(name) end
    elseif mode == 'test' then
        return function() return loader_test(name) end
    else
        error(('Undefined mode: %s. mode must be either \'train\' or \'test\''):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

return data_loader