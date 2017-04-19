--[[
    Process rois for a dataset.
]]


--local loadRoiDataFn = fastrcnn.utils.load.matlab.single_file
local load_files = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/load.lua')
local loadRoiDataFn = load_files.matlab.single_file

------------------------------------------------------------------------------------------------------------

local function get_data_dir(name)
    local dbc = require 'dbcollection.manager'
    local dbloader
    local str = string.lower(name)
    if str == 'caltech' then
        dbloader = dbc.load{name='caltech_pedestrian', task='detection'}
    elseif str == 'caltech_10x' then
        dbloader = dbc.load{name='caltech_pedestrian', task='detection_10x'}
    elseif str == 'caltech_30x' then
        dbloader = dbc.load{name='caltech_pedestrian', task='detection_30x'}
    elseif str == 'daimler' then
        error('Daimler dataset not yet defined.')
    elseif str == 'eth' then
        error('eth dataset not yet defined.')
    elseif str == 'inria' then
        error('inria dataset not yet defined.')
    elseif str == 'tudbrussels' then
        error('tudbrussels dataset not yet defined.')
    else
        error(('Undefined dataset: %s. Available options: caltech, daimler, eth, inria or tudbrussels'):format(name))
    end
    return dbloader.data_dir
end

------------------------------------------------------------------------------------------------------------

local function get_command(name, mode, data_dir, save_dir)
--[[Retrieve a command to process the sets using matlab from the terminal.]]

    local dset_dirname
    local str = string.lower(name)
    if str == 'caltech' then
        dset_fn = 'script_process_rois_acf_caltech'
        if mode == 'train' then
            dset_dirname = ('%s_skip=%d_thresh=%d_cal=%s'):format(name, 30, 1, 0.1)
        else
            dset_dirname = ('%s_skip=%d_thresh=%d_cal=%s'):format(name, 30, 1, 0.025)
        end
    elseif str == 'caltech_10x' then
        dset_fn = 'script_process_rois_acf_caltech'
        if mode == 'train' then
            dset_dirname = ('%s_skip=%d_thresh=%d_cal=%s'):format(name, 3, 1, 0.1)
        else
            dset_dirname = ('%s_skip=%d_thresh=%d_cal=%s'):format(name, 3, 1, 0.025)
        end
    elseif str == 'caltech_30x' then
        dset_fn = 'script_process_rois_acf_caltech'
        if mode == 'train' then
            dset_dirname = ('%s_skip=%d_thresh=%d_cal=%s'):format(name, 1, 1, 0.1)
        else
            dset_dirname = ('%s_skip=%d_thresh=%d_cal=%s'):format(name, 1, 1, 0.025)
        end
    elseif str == 'daimler' then
        error('Daimler dataset not yet defined.')
    elseif str == 'eth' then
        error('eth dataset not yet defined.')
        dset_fn = 'script_process_rois_acf_eth'
    elseif str == 'inria' then
        error('inria dataset not yet defined.')
        dset_fn = 'script_process_rois_acf_inria'
    elseif str == 'tudbrussels' then
        error('tudbrussels dataset not yet defined.')
        dset_fn = 'script_process_rois_acf_tudbrussels'
    else
        error(('Undefined dataset: %s. Available options: caltech, daimler, eth, inria or tudbrussels'):format(name))
    end

    local command = ('cd roi_proposals && matlab -nodisplay -nodesktop -r ' ..
                  '"try, %s(\'%s\', \'%s\'), catch, exit, end, exit"')
                  :format(dset_fn, data_dir, '.' .. save_dir)

    return command, paths.concat(save_dir, dset_dirname)
end

------------------------------------------------------------------------------------------------------------

local function process_rois(name, mode, save_dir)
    assert(name)

    -- get data directory
    local data_dir = get_data_dir(name)

    -- get command
    local command, dset_path = get_command(name, mode, data_dir, save_dir)

    if not paths.dirp(dset_path) then
        print('\n***WARNING: This may take some minutes to process.***')
        os.execute(command)
    end

    return dset_path
end

------------------------------------------------------------------------------------------------------------

local function load_rois_files(dset_path, name, mode)
--[[load all roi boxes of all files into a table]]

    local rois = {}

    -- get dataloader to cycle all files and load all roi's bbox proposals
    local data_loader = paths.dofile('data.lua')
    local data_gen = data_loader(name, mode)
    local loader = data_gen()

    for k, set in pairs(loader) do
        rois[k] = {}
        for i=1, set.nfiles do -- cycle all files
            local image_filename = set.getFilename(i)
            local tmp_str =  string.split(image_filename, '/')
            local set_name = tmp_str[#tmp_str-3]
            local video_name = tmp_str[#tmp_str-2]
            local fname = string.split(tmp_str[#tmp_str],'.jpg')[1]

            local rois_fname = paths.concat(dset_path, set_name, video_name, fname .. '.mat')

            local boxes = loadRoiDataFn(rois_fname)
            table.insert(rois[k], boxes)
        end
    end

    return rois
end

------------------------------------------------------------------------------------------------------------

local function load_rois(name, mode)
--[[Load rois bboxes of all files into memory]]

    assert(name, 'Undefined dataset name: ' .. name)
    assert(mode == 'train' or mode == 'test', ('Invalid mode: %s. Valid modes: train or test.'):format(mode))

    local save_dir = './data/proposals'

    local proposals_fname = ('%s/%s_acf_%s.t7'):format(save_dir, name, mode)

    -- check if the cache proposals .t7 file exists
    local rois
    if not paths.filep(proposals_fname) then
        -- process roi proposals
        local dset_path = process_rois(name, mode, save_dir)

        -- load rois into a table
        rois = load_rois_files(dset_path, name, mode)

        -- save rois cache file
        torch.save(proposals_fname, rois)
    else
        rois = torch.load(proposals_fname)
    end

    return rois
end

------------------------------------------------------------------------------------------------------------

return load_rois