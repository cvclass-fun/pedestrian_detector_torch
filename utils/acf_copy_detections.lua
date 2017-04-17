--[[
    Process object detection in a batch of images.
]]

-- Main function
local function process_dataset(frcnn, data, rois_filepath, threshold, path)
  
  -- initializations
  local fileName = data.fileName
  local nFiles = fileName:size(1)
  local classNames = {}
  for i=1, data.className:size(1) do
    table.insert(classNames, ffi.string(data.className[i]:data()))
  end

  --local roi_boxes = FastRCNN.utils.LoadMatlabFiles(rois_filepath)
  local roi_boxes = FastRCNN.utils.LoadMatlabFiles(rois_filepath)
  
  -- cycle all images and store detected windows to file
  for ifile=1, nFiles do
    xlua.progress(ifile, nFiles)
    
    -- select proposals
    local proposals = roi_boxes[ifile]
    
    local prop_fmt = proposals:clone()
    prop_fmt[{{},{3}}] = (prop_fmt[{{},{3}}] - prop_fmt[{{},{1}}]):add(1)
    prop_fmt[{{},{4}}] = (prop_fmt[{{},{4}}] - prop_fmt[{{},{2}}]):add(1) 
    
    local filepath = ffi.string(fileName[ifile]:data())
    
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
      
    if proposals:numel() > 0 then
      for i=1, proposals:size(1) do
        -- convert coordinates to [x,y,w,h] format
        local x = proposals[i][1]
        local y = proposals[i][2]
        local w = proposals[i][3]-proposals[i][1]+1
        local h = proposals[i][4]-proposals[i][2]+1
        file:write(('%d,%0.2f,%0.2f,%0.2f,%0.2f,%0.5f\n'):format(fileID, x,y,w,h, proposals[i][5]))
        --file:write(('%d,%0.2f,%0.2f,%0.2f,%0.2f,%0.5f\n'):format(fileID, x,y,w,h, torch.random(1,100)))
      end
    end
    file:close()
  end
  
end

return process_dataset