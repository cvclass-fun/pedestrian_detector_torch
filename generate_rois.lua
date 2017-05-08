--[[
    Script to process roi proposals from the dataset.

    This will run the ACF and LDCF methods on the dataset's images and output .mat files for each
    image with all the detected regions (defined by a set of threshold/calibrations).

    After, all rois of the train/test sets are joined into a single .t7 file.

    Note: run this script to produce roi proposals. Note that pre-processed detections are available
          for download because processing roi proposals takes several days to complete.
]]


require 'paths'
require 'torch'
require 'xlua'
local matio = require 'matio'

torch.setdefaulttensortype('torch.FloatTensor')
paths.dofile('projectdir.lua')

------------------------------------------------------------------------------------------------------------

local function get_data_dir(name)
    local dbc = require 'dbcollection.manager'

    local dbloader
    local str = string.lower(name)
    if str == 'caltech' then
        dbloader = dbc.load{name='caltech_pedestrian', task='detection_d'}
    elseif str == 'caltech_10x' then
        dbloader = dbc.load{name='caltech_pedestrian', task='detection_10x_d'}
    elseif str == 'caltech_30x' then
        dbloader = dbc.load{name='caltech_pedestrian', task='detection_30x_d'}
    elseif str == 'eth' then
        error('eth dataset not yet defined.')
    elseif str == 'inria' then
        error('inria dataset not yet defined.')
    elseif str == 'tudbrussels' then
        error('tudbrussels dataset not yet defined.')
    else
        error(('Undefined dataset: %s. Available options: caltech, eth, inria or tudbrussels'):format(name))
    end
    return dbloader.data_dir
end

------------------------------------------------------------------------------------------------------------

local function get_command(name, data_dir, save_dir, alg_name)
--[[Retrieve a command to process the sets using matlab from the terminal.]]

    local alg_opts = {}
    local dset_fn
    local str = string.lower(name)
    if str == 'caltech' then
        dset_fn = 'script_process_rois_caltech'
        alg_opts = {
            train = { skip_step = 30, threshold = 1, calibration = 0.1 },
            test = { skip_step = 30, threshold = 1, calibration = 0.025 }
        }
    elseif str == 'caltech_10x' then
        dset_fn = 'script_process_rois_caltech'
        alg_opts = {
            train = { skip_step = 3, threshold = 1, calibration = 0.1 },
            test = { skip_step = 3, threshold = 1, calibration = 0.025 }
        }
        name = 'caltech'
    elseif str == 'caltech_30x' then
        dset_fn = 'script_process_rois_caltech'
        alg_opts = {
            train = { skip_step = 1, threshold = 1, calibration = 0.1 },
            test = { skip_step = 1, threshold = 1, calibration = 0.025 }
        }
        name = 'caltech'
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
        error(('Undefined dataset: %s. Available options: caltech, eth, inria or tudbrussels'):format(name))
    end


    local dset_dirname
    str = string.lower(alg_name)
    if str == 'acf' then
        dset_dirname = {
            train = paths.concat(save_dir, ('%s_acf_skip=%d_thresh=%d_cal=%s'):format(name, alg_opts['train']['skip_step'],
                        alg_opts['train']['threshold'], alg_opts['train']['calibration'])),
            test = paths.concat(save_dir, ('%s_acf_skip=%d_thresh=%d_cal=%s'):format(name, alg_opts['test']['skip_step'],
                        alg_opts['test']['threshold'], alg_opts['test']['calibration']))
        }
    elseif alg_name == 'ldcf' then
        dset_dirname = {
            train = paths.concat(save_dir, ('%s_ldcf_skip=%d'):format(name, alg_opts['train']['skip_step'])),
            test = paths.concat(save_dir, ('%s_ldcf_skip=%d'):format(name, alg_opts['test']['skip_step']))
        }
    elseif alg_name == 'edgeboxes' then
        dset_dirname = {
            train = paths.concat(save_dir, ('%s_edgeboxes_skip=%d'):format(name, alg_opts['train']['skip_step'])),
            test = paths.concat(save_dir, ('%s_edgeboxes_skip=%d'):format(name, alg_opts['test']['skip_step']))
        }
    else
        error(('Invalid algorithm: %s. Available algorithms: acf, ldcf or edgeboxes'):format(alg_name))
    end


    local command = ('cd roi_proposals && matlab -nodisplay -nodesktop -r ' ..
                  '"try, %s(\'%s\', \'%s\'), catch, exit, end, exit"')
                  :format(dset_fn, data_dir, save_dir)

    return command, dset_dirname
end

------------------------------------------------------------------------------------------------------------

local function process_rois(name, alg_name, save_dir)
    assert(name)

    -- get data directory
    local data_dir = get_data_dir(name)

    -- get command
    local command, dset_path = get_command(name, data_dir, save_dir, alg_name)

    if not paths.dirp(dset_path['train']) then
        print('\n***WARNING: This may take some minutes to process.***')
        os.execute(command)
    end

    return dset_path
end

------------------------------------------------------------------------------------------------------------

--[[ load all roi boxes of all files into a table ]]
local function load_rois_files(dset_path, name, mode)

    local rois = {}

    -- get dataloader to cycle all files and load all roi's bbox proposals
    local data_loader = paths.dofile('data.lua')
    local data_gen = data_loader(name, mode)
    local loader = data_gen()

    print('Loading roi proposals: ' .. mode .. ' mode.')

    for k, set in pairs(loader) do
        rois[k] = {}
        local nfiles = set.nfiles
        for i=1, nfiles do -- cycle all files
            if i%100==0 or i==nfiles then
                xlua.progress(i, nfiles)
            end
            local image_filename = set.getFilename(i)
            local tmp_str =  string.split(image_filename, '/')
            local set_name = tmp_str[#tmp_str-3]
            local video_name = tmp_str[#tmp_str-2]
            local fname = string.split(tmp_str[#tmp_str],'.jpg')[1]

            local rois_fname = paths.concat(dset_path, set_name, video_name, fname .. '.mat')

            local bb = matio.load(rois_fname)
            local boxes = bb["boxes"]:float()

            table.insert(rois[k], boxes)
        end
    end

    return rois
end

------------------------------------------------------------------------------------------------------------


--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

local opts = paths.dofile('options.lua')
local opt = opts.parse(arg)


--------------------------------------------------------------------------------
-- Process regions-of-interest (RoIs)
--------------------------------------------------------------------------------

local save_dir = projectDir .. 'data/proposals'

local dset_path = process_rois(opt.dataset, opt.proposalAlg, save_dir)

for mode, path in pairs(dset_path) do
    local proposals_fname = ('%s/%s_%s_%s.t7'):format(save_dir, opt.dataset, opt.proposalAlg, mode)

    -- load rois into a table
    local rois = load_rois_files(path, opt.dataset, mode)

    -- save rois cache file
    torch.save(proposals_fname, rois)
end

