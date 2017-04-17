--[[
    Process object detection in a batch of images.
]]

-- Main function
local function process_dataset(frcnn, data, rois_filepath, threshold, nms_threshold, path)
  
  -- initializations
  local fileName = data.fileName
  local nFiles = fileName:size(1)
  local classNames = {}
  for i=1, data.className:size(1) do
    table.insert(classNames, ffi.string(data.className[i]:data()))
  end

  local roi_boxes = FastRCNN.utils.LoadMatlabFiles(rois_filepath)
  
  -- cycle all images and store detected windows to file
  for ifile=1, nFiles do
    xlua.progress(ifile, nFiles)
    
    if ifile == 1500 then
      aqui = 1
    end
    
    
    local image_path = ffi.string(fileName[ifile]:data())

    -- Loading proposals from file
    local proposals = roi_boxes[ifile]

    -- Loading the image
    local im = image.load(image_path)

    -- (5) Process image detection with the FRCNN
    
    -- detect !
    local scores, boxes = frcnn:Detect(im, proposals[{{},{1,4}}])
    if boxes:numel() == 0 then boxes = proposals[{{},{1,4}}] end
    -- visualization
    local threshold = 0.5
    local thresh = threshold
    -- classes from Pascal used for training the model
    local cls = {'person'}
    
    -- filter rois
     -- select best scoring boxes without background
    local max_score,idx = scores[{{},{1, #cls}}]:max(2)
    --local max_score,idx = scores[{{},{2, -1}}]:max(2) --bgr

    local idx_thresh = max_score:gt(thresh)
    max_score = max_score[idx_thresh]
    idx = idx[idx_thresh]

    local r = torch.range(1,boxes:size(1)):long()
    local rr = r[idx_thresh]
    
    -- add detection entry to the scores file
    -- process set id, file id
    local file
    local str = string.split(image_path, '/')
    local setID, volumeID, fileID = str[#str-3], str[#str-2], tonumber(str[#str]:sub(2, #str-5))+1
    local detections_filepath =  paths.concat(path, setID, volumeID .. '.txt')
    if paths.filep(detections_filepath) then
      file = io.open(detections_filepath, 'a')
    else
      -- create directory to store the set's video files
      os.execute('mkdir -p ' .. paths.concat(path, setID))
      -- create file
      file = io.open(detections_filepath, 'w')
    end
    
    if rr:numel() > 0 then

      local boxes_thresh = boxes:index(1,rr)
      local bb_thresh = proposals:index(1,rr)
      local bb_scores = bb_thresh[{{},{5}}]
      
      local keep = FastRCNN.utils.nms(torch.cat(boxes_thresh:float(),bb_scores:float(),2), nms_threshold)
      
      boxes_thresh = boxes_thresh:index(1,keep)
      local scores_thresh = proposals:index(1,keep)
      local bb_scores = scores_thresh[{{},{5}}]
    
      --[[
      -- add detection entry to the scores file
      -- process set id, file id
      local file
      local str = string.split(image_path, '/')
      local setID, volumeID, fileID = str[#str-3], str[#str-2], tonumber(str[#str]:sub(2, #str-5))+1
      local detections_filepath =  paths.concat(path, setID, volumeID .. '.txt')
      if paths.filep(detections_filepath) then
        file = io.open(detections_filepath, 'a')
      else
        -- create directory to store the set's video files
        os.execute('mkdir -p ' .. paths.concat(path, setID))
        -- create file
        file = io.open(detections_filepath, 'w')
      end
      --]]
        
      if boxes_thresh:numel() > 0 then
        for i=1, boxes_thresh:size(1) do
          -- convert coordinates to [x,y,w,h] format
          local x = boxes_thresh[i][1]
          local y = boxes_thresh[i][2]
          local w = boxes_thresh[i][3]-boxes_thresh[i][1]+1
          local h = boxes_thresh[i][4]-boxes_thresh[i][2]+1
          file:write(('%d,%0.2f,%0.2f,%0.2f,%0.2f,%0.5f\n'):format(fileID, x,y,w,h, bb_scores[i][1]))
        end
      end
    end
    
    file:close()
  end --for ifile
  
end

return process_dataset