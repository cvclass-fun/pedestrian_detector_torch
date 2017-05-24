--[[
    Utility functions.
]]


require 'xlua'
require 'image'
local nms = require 'fastrcnn.utils.nms'

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

local function filter_detections(scores, boxes, nms_threshold, cl_names, rois)
    --local nbest = 50

    -- select best scoring boxes without background
    local max_score, idx = scores:max(2)
    local idx_nobg = idx:gt(1)
    max_score = max_score[idx_nobg]
    local score_rois = rois:select(2,5)
    score_rois = score_rois[idx_nobg]
    if max_score:numel() == 0 then
        return torch.FloatTensor(), torch.FloatTensor()
    end
    local boxes_thresh = boxes[{{},{1+4,-1}}]
    local ind_range = idx_nobg:squeeze(2):nonzero():squeeze(2)
    boxes_thresh = boxes_thresh:index(1, ind_range)
    idx = idx[idx_nobg]

    -- check if any box is left after applying the threshold
    if idx:numel() == 0 then
        -- no boxes detected, return empty tensors
        return torch.FloatTensor(), torch.FloatTensor()
    end

    -- non-maximum suppression
    local keep = nms.dense(torch.cat(boxes_thresh:float(), max_score:float(),2), nms_threshold)

    --return max_score:index(1, keep), boxes_thresh:index(1, keep)
    return score_rois:index(1, keep), boxes_thresh:index(1, keep)
end

------------------------------------------------------------------------------------------------------------

local function filter_detections2(scores, boxes, nms_threshold, cl_names)
    --local nbest = 50

    -- select best scoring boxes without background
    local max_score, maxID = scores[{{},{2,-1}}]:max(2)

    -- max id
    local idx = max_score:gt(0.3):squeeze(2):nonzero()

    if idx:numel()==0 then
        return torch.FloatTensor(), torch.FloatTensor()
    end

    idx=idx:select(2,1)
    boxes = boxes:index(1, idx)
    maxID = maxID:index(1, idx)
    max_score = max_score:index(1, idx)

    if idx:numel()==1 then
        return max_score:squeeze(2), boxes
    end

    -- select bbox
    local boxes_thresh = {}
    for i=1, boxes:size(1) do
        local label = maxID[i][1]
        table.insert(boxes_thresh, boxes[i]:narrow(1,(label-1)*4 + 1,4):totable())
    end
    boxes_thresh = torch.FloatTensor(boxes_thresh)

    local scored_boxes = torch.cat(boxes_thresh:float(), max_score:float(), 2)
    local keep = nms.dense(scored_boxes, nms_threshold or 0.5)

    boxes_thresh = boxes_thresh:index(1,keep)
    max_score = max_score:index(1,keep):squeeze(2)
    maxID = maxID:index(1,keep):squeeze(2)

    return max_score, boxes_thresh
end

------------------------------------------------------------------------------------------------------------

local function save_boxes_to_file(scores, boxes, filename, save_path)
--[[ Save boxes + scores into a file w.r.t. the set, video and file name. ]]

    -- process set id, file id
    local file
    local str = string.split(filename, '/')
    local setID, volumeID, fileID = str[#str-3], str[#str-2], tonumber(str[#str]:sub(2, -5))+1
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
        for i=1, scores:size(1) do
            -- convert coordinates to [x,y,w,h] format
            local x = boxes[i][1]
            local y = boxes[i][2]
            local w = boxes[i][3] - boxes[i][1] + 1
            local h = boxes[i][4] - boxes[i][2] + 1
            file:write(('%d,%0.2f,%0.2f,%0.2f,%0.2f,%0.5f\n'):format(fileID, x, y, w, h, scores[i]))
        end
    end

    file:close()
end

------------------------------------------------------------------------------------------------------------

local function clean_roi_proposals(proposals)
    if proposals:numel() > 0 then
        local keep = {}
        for i=1, proposals:size(1) do
            if proposals[i]:sum() > 0 then
                table.insert(keep, i)
            end
        end

        if next(keep) then
            return proposals:index(1, torch.LongTensor(keep))
        else
            return torch.FloatTensor()
        end
    else
        return proposals
    end
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
        print('Save detections to directory: ' .. save_path)
        os.execute('mkdir -p ' ..  save_path)

        local nfiles = data_loader.test.nfiles
        local nms_thresh = opt.frcnn_test_nms_thresh
        local classes = data_loader.test.classLabel

        -- cycle all files
        for ifile=1, nfiles do
            xlua.progress(ifile, nfiles)
            local filename = data_loader.test.getFilename(ifile)
            local img = image.load(filename)
            local proposals = rois['test'][ifile]
            local proposals_clean = clean_roi_proposals(proposals)
            if proposals_clean:numel() > 0 then
                local scores, bboxes = imdetector:detect(img, proposals_clean[{{},{1,4}}])
                -- clamp predictions within image
                bboxes:select(2,1):clamp(1, img:size(3))
                bboxes:select(2,2):clamp(1, img:size(2))
                local detection_scores, detection_boxes = filter_detections(scores, bboxes, nms_thresh, classes, proposals_clean)
                if detection_scores:numel() > 0 then
                    save_boxes_to_file(detection_scores, detection_boxes, filename, save_path) -- save results to .txt
                end
            end
        end
        print('> Done.')
    end

    return save_path
end

------------------------------------------------------------------------------------------------------------

local function process_detections2(data_loader, rois, imdetector, opt)

    local tds = require 'tds'
    local utils = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/init.lua')
    local plot_name = opt.eval_plot_name
    local save_path = projectDir .. '/data/benchmark_algorithms/' .. select_dataset(opt.dataset) .. '/' .. plot_name

    if opt.eval_force then
        os.execute('rm -rf ' ..  save_path)
    end

    if not paths.dirp(save_path) then
        print('> Processing detections...')
        print('Save detections to directory: ' .. save_path)
        os.execute('mkdir -p ' ..  save_path)

        local nfiles = data_loader.test.nfiles
        local nms_thresh = opt.frcnn_test_nms_thresh
        local classes = data_loader.test.classLabel
        local thresh = torch.ones(#classes):mul(-1.5)
        local max_boxes = 50

        -- cycle all files
        for ifile=1, nfiles do
            xlua.progress(ifile, nfiles)

            local all_output = {}
            local all_bbox_pred = {}
            local img_boxes = tds.hash()

            local filename = data_loader.test.getFilename(ifile)
            local img = image.load(filename)
            local proposals = rois['test'][ifile]
            local output, bbox_pred = imdetector:detect(img, proposals)
            -- clamp predictions within image
            bbox_pred:select(2,1):clamp(1, img:size(3))
            bbox_pred:select(2,2):clamp(1, img:size(2))

            table.insert(all_output, output)
            table.insert(all_bbox_pred, bbox_pred)

            output = utils.table.joinTable(all_output, 1)
            bbox_pred = utils.table.joinTable(all_bbox_pred, 1)

            for j = 1, #classes do
                local scores = output:select(2, j+1)
                local idx = torch.range(1, scores:numel()):long()
                local idx2 = scores:gt(thresh[j])
                idx = idx[idx2]
                local scored_boxes = torch.FloatTensor(idx:numel(), 5)
                if scored_boxes:numel() > 0 then
                    local bx = scored_boxes:narrow(2, 1, 4)
                    bx:copy(bbox_pred:narrow(2, j*4+1, 4):index(1, idx))
                    scored_boxes:select(2, 5):copy(scores[idx2])
                end

                -- apply non-maxmimum suppression
                img_boxes[j] = utils.nms.fast(scored_boxes, 0.5)
            end

            local detections = utils.box.keep_top_k(img_boxes, max_boxes)[1]

            if detections:numel() > 0 then
                save_boxes_to_file(detections[{{},{5}}]:squeeze(2), detections[{{},{1,4}}], filename, save_path) -- save results to .txt
            end
        end

        print('> Done.')
    end

    return save_path
end

------------------------------------------------------------------------------------------------------------

local function process_detections3(data_loader, rois, imdetector, opt)

    local plot_name = opt.eval_plot_name
    local save_path = projectDir .. '/data/benchmark_algorithms/' .. select_dataset(opt.dataset) .. '/' .. plot_name

    if opt.eval_force then
        os.execute('rm -rf ' ..  save_path)
    end

    if not paths.dirp(save_path) then
        print('> Processing detections...')
        print('Save detections to directory: ' .. save_path)
        os.execute('mkdir -p ' ..  save_path)

        local nfiles = data_loader.test.nfiles
        local nms_thresh = opt.frcnn_test_nms_thresh
        local classes = data_loader.test.classLabel

        -- cycle all files
        for ifile=1, nfiles do
            xlua.progress(ifile, nfiles)
            local filename = data_loader.test.getFilename(ifile)
            local img = image.load(filename)
            local proposals = rois['test'][ifile]
            local scores, bboxes = imdetector:detect(img, proposals)
            -- clamp predictions within image
            bboxes:select(2,1):clamp(1, img:size(3))
            bboxes:select(2,2):clamp(1, img:size(2))
            local detection_scores, detection_boxes = filter_detections2(scores, bboxes, nms_thresh, classes)
            if detection_scores:numel() > 0 then
                save_boxes_to_file(detection_scores, detection_boxes, filename, save_path) -- save results to .txt
            end
        end
        print('> Done.')
    end

    return save_path
end

------------------------------------------------------------------------------------------------------------

local function get_dataset_id(name)
    local str = string.lower(name)
    if string.match(name, 'caltech') then
        return 'caltech', 1 -- 'UsaTest'
    elseif str == 'eth' then
        return 'eth', 5 -- 'ETH'
    elseif str == 'inria' then
        return 'inria', 3 -- 'InriaTest'
    elseif str == 'tudbrussels' then
        return 'tudbrussels', 4 -- 'TudBrussels'
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
    local dataset_name, dataNamesID = get_dataset_id(opt.dataset)
    local algorithmsDir = paths.concat(projectDir .. '/data/benchmark_algorithms/', dataset_name)
    local savePlotDir = paths.concat(opt.savedir, 'benchmark_plots')
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
    local command = ('cd %s/benchmark && matlab -nodisplay -nodesktop -r "try, Run_evaluate(%d, %d,' ..
        ' %d, %d, \'%s\', \'%s\', \'%s\', \'%s\', %s), catch, exit, end, exit"')
        :format(projectDir, experimentIDini, experimentIDend, algPlotNum, dataNamesID,
                datasetPathDir, algorithmsDir, savePlotDir, algorithmsNames, flag_benchmark)

    os.execute(command)

    print('> Done.')
end

------------------------------------------------------------------------------------------------------------

local function benchmark(data_loader, rois, imdetector, data_dir, opt)
--[[ Process detections on all images of the dataset and plot the benchmark of the
     person detector against other top-ranked methods.
]]

    -- process detections
    process_detections(data_loader, rois, imdetector,opt)

    -- benchmark method using the eval toolbox from piotr dollar
    benchmark_algorithm(data_dir, opt)
end

------------------------------------------------------------------------------------------------------------

return {
    benchmark = benchmark
}