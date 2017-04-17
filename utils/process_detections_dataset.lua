--[[
    Process object detection in a batch of images.
]]


local function get_detections_image(scores, boxes, thresh, nms_thresh, cl_names)

  -- select best scoring boxes without background
  local max_score,idx = scores[{{},{1, #cl_names}}]:max(2)

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
  local boxes_thresh = boxes:index(1,rr)
  
  local keep = FastRCNN.utils.nms(torch.cat(boxes_thresh:float(),max_score:float(),2), nms_thresh)
  
  boxes_thresh = boxes_thresh:index(1,keep)
  max_score = max_score:index(1,keep)
  idx = idx:index(1,keep)
  
  -- filter boxes
  if boxes_thresh:numel() > 0 then
    local detect_boxes = torch.FloatTensor(boxes_thresh:size(1),4):fill(0)
    local range = torch.range(1,4)
    for i=1, boxes_thresh:size(1) do
      detect_boxes[i]:copy(boxes_thresh[i]:index(1,(range + (idx[i]-1)*4):long()))
    end
    return max_score, detect_boxes
  else
    return torch.FloatTensor(), torch.FloatTensor()
  end

  
  --[[
  local num_boxes = boxes_thresh:size(1)
  local widths  = boxes_thresh[{{},3}] - boxes_thresh[{{},1}]
  local heights = boxes_thresh[{{},4}] - boxes_thresh[{{},2}]
  
  for i=1,num_boxes do
    local x,y = boxes_thresh[{i,1}],boxes_thresh[{i,2}]
    local width,height = widths[i], heights[i]
    --w:rectangle(x,y,width,height)
  end

  return detect_boxes
  --]]
end

------------------------------------------

-- Main function
local function process_dataset(frcnn, data, rois_filepath, threshold, nms_threshold, path)
  
  -- initializations
  local fileName = data.fileName
  local nFiles = fileName:size(1)
  local classNames = {}
  for i=1, data.className:size(1) do
    table.insert(classNames, ffi.string(data.className[i]:data()))
  end

  -- load roi proposal boxes from file
  local roi_boxes = FastRCNN.utils.LoadMatlabFiles(rois_filepath)
  
  -- cycle all images and store detected windows to file
  for ifile=1, nFiles do
    xlua.progress(ifile, nFiles)
    
    -- select proposals
    local proposals = roi_boxes[ifile]
    
    -- check if there are any proposals. If no then skip this file
    if proposals:numel()>0 then
      -- load image
      local filepath = ffi.string(fileName[ifile]:data())
      local img = image.load(filepath)
      
      -- process detection scores
      local scores, bboxes = frcnn:Detect(img, proposals[{{},{1,4}}])
      
      -- filter relevant boxes
      local detection_scores, detectio_boxes  = get_detections_image(scores, bboxes, threshold, nms_threshold, classNames)
      
      -- add detection entry to the scores file
      -- process set id, file id
      local file
      local str = string.split(filepath, '/')
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
      
      if detection_scores:numel() > 0 then
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
    
  end-- for ifile
end

return process_dataset