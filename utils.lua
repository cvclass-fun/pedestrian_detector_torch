--[[
    Utility functions.
]]


require 'xlua'
require 'image'
local nms = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/nms.lua')

------------------------------------------------------------------------------------------------------------

local function select_dataset(name)
    local str = string.lower(name)
    if string.match(name, 'caltech') then
        return 'caltech'
    elseif str == 'eth' then
        return 'eth'
    elseif str == 'inria' then
        return 'inria'
    elseif str == 'tudbrussels' then
        return 'tudbrussels'
    else
        error(('Invalid dataset: %s. Available options: caltech, eth, inria or tudbrussels'):format(name))
    end
end

------------------------------------------------------------------------------------------------------------

local function filter_detections(scores, bboxes, nms_threshold, cl_names)
    --local nbest = 50

    -- select best scoring boxes without background
    local max_score, idx = scores[{{},{1, #cl_names}}]:max(2)

    local idx_thresh = max_score:gt(thresh)
    max_score = max_score[idx_thresh]
    idx = idx[idx_thresh]

    -- check if any box is left after applying the threshold
    if idx:numel() == 0 then
        -- no bboxes detected, return empty tensors
        return torch.FloatTensor(), torch.FloatTensor()
    end

    local r = torch.range(1,boxes:size(1)):long()
    local rr = r[idx_thresh]
    local boxes_thresh = boxes:index(1, rr)

    -- non-maximum suppression
    local keep = nms.dense(torch.cat(boxes_thresh:float(), max_score:float(),2), nms_threshold)

    boxes_thresh = boxes_thresh:index(1,keep)
    max_score = max_score:index(1,keep)
    idx = idx:index(1,keep)

    -- filter boxes
    if boxes_thresh:numel() > 0 then
        local detect_boxes = torch.FloatTensor(boxes_thresh:size(1), 4):fill(0)
        local range = torch.range(1,4)
        for i=1, boxes_thresh:size(1) do
            detect_boxes[i]:copy(boxes_thresh[i]:index(1,(range + (idx[i]-1)*4):long()))
        end
        -- sort the boxes by score
        local _, sorted_idx = max_score:sort(1, true)
        sorted_idx = sorted_idx:squeeze()
        return max_score:index(1, sorted_idx), detect_boxes:index(1, sorted_idx)
    else
        return torch.FloatTensor(), torch.FloatTensor()
    end

end

------------------------------------------------------------------------------------------------------------

local function save_boxes_to_file(scores, bboxes, filename, save_path)
--[[ Save boxes + scores into a file w.r.t. the set, video and file name. ]]

    -- process set id, file id
    local file
    local str = string.split(filepath, '/')
    local setID, volumeID, fileID = str[#str-3], str[#str-2], tonumber(str[#str]:sub(2, #str-5))+1
    local detections_filepath =  paths.concat(save_path, setID, volumeID .. '.txt')

    if paths.filep(detections_filepath) then
        file = io.open(detections_filepath, 'a')
    else
        -- create directory to store the set's video files
        os.execute('mkdir -p ' .. paths.concat(save_path, setID))
        -- create file
        file = io.open(detections_filepath, 'w')
    end

    if scores:numel() > 0 then
        for i=1, detection_scores:size(1) do
            -- convert coordinates to [x,y,w,h] format
            local x = detectio_boxes[i][1]
            local y = detectio_boxes[i][2]
            local w = detectio_boxes[i][3]-detectio_boxes[i][1]+1
            local h = detectio_boxes[i][4]-detectio_boxes[i][2]+1
            file:write(('%d,%0.2f,%0.2f,%0.2f,%0.2f,%0.5f\n'):format(fileID, x,y,w,h, detection_scores[i]))
        end
    end

    file:close()
end

------------------------------------------------------------------------------------------------------------

local function process_detections(data_loader, rois, imdetector, opt)

    local plot_name = opt.eval_plot_name
    local save_path = projectDir .. '/data/benchmark_algorithms/' .. select_dataset(opt.dataset) .. '/' .. plot_name

    if opt.eval_force then
        os.execute('rm -rf ' ..  save_path)
    end

    if not paths.dirp(save_path) then
        print('> Processing detections...')
        os.execute('mkdir -p' ..  save_path)

        -- cycle all files
        local nfiles = data_loader.test.nfiles
        for ifile=1, nfiles do
            xlua.progress(ifile, nfiles)
            local filename = data_loader.test.getFilename(ifile)
            local img = image.load(filename)
            local proposals = rois['test'][ifile]
            local scores, bboxes = imdetector:detect(img, proposals)
            local detection_scores, detection_boxes = filter_detections(scores, bboxes, opt.frcnn_test_nms_thresh,
                                                                        data_loader.test.classLabel)
            save_boxes_to_file(detection_scores, detection_boxes, filename, save_path)  -- save results to .txt
        end
        print('> Done.')
    end

    return save_path
end

------------------------------------------------------------------------------------------------------------

local function get_dataset_id(name)
    local str = string.lower(name)
    if string.match(name, 'caltech') then
        return 1 -- 'UsaTest'
    elseif str == 'eth' then
        return 5 -- 'ETH'
    elseif str == 'inria' then
        return 3 -- 'InriaTest'
    elseif str == 'tudbrussels' then
        return 4 -- 'TudBrussels'
    else
        error(('Invalid dataset: %s. Available options: caltech, eth, inria or tudbrussels'):format(name))
    end
end

------------------------------------------------------------------------------------------------------------

local function benchmark_algorithm(dataset_dir, opt)

    print('> Processing benchmark results...')

    local datasetPathDir = dataset_dir
    local experimentIDini = opt.eval_ini
    local experimentIDend = opt.eval_end
    local algPlotNum = opt.eval_num_plots   --% number of algorithms to print
    local dataNamesID = get_dataset_id(opt.dataset)
    local algorithmsDir = projectDir .. '/data/benchmark_algorithms/'
    local savePlotDir = paths.concat(opt.savedir, 'plots')
    local algorithmsNames = opt.eval_plot_name
    local flag_benchmark = 0 -- plots all algorithms into a single graph

    if not paths.dirp(savePlotDir) then
        os.execute('mkdir -p ' .. savePlotDir)
    end

    local reset_eval_ours = true -- true - delete old files and process new one
                                 -- false - use old processed files if available
    if reset_eval_ours then
        print('Cleaning old cached results...')
        local dir = require 'pl.dir'
        local files = dir.getallfiles(savePlotDir)
        for i=1, #files do
            if string.match(files[i]:sub(#files[i]-8), '-' .. algorithmsNames .. '.mat') then
                os.execute('rm -rf ' .. files[i])
            end
        end
        print('Done.')
    end

    -- benchmark algorithm
    os.execute(('cd benchmark && matlab -nodisplay -nodesktop -r "try, Run_evaluate(%d, %d,' ..
                ' %d, %d, \'%s\', \'%s\', \'%s\', \'%s\', %s), catch, exit, end, exit"')
                :format(experimentIDini, experimentIDend, algPlotNum, dataNamesID,
                datasetPathDir, algorithmsDir, savePlotDir, algorithmsNames, flag_benchmark))

    print('> Done.')
end

------------------------------------------------------------------------------------------------------------

local function benchmark(data_loader, rois, imdetector, opt)
--[[ Process detections on all images of the dataset and plot the benchmark of the
     person detector against other top-ranked methods.
]]

    -- process detections
    process_detections(data_loader, rois, imdetector,opt)

    -- benchmark method using the eval toolbox from piotr dollar
    benchmark_algorithm(data_loader.data_dir, opt)
end

------------------------------------------------------------------------------------------------------------

return {
    benchmark = benchmark
}