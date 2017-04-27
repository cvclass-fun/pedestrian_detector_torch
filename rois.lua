--[[
    Load rois for a dataset.
]]


local function load_rois(name, alg_name, mode)
--[[Load rois bboxes of all files into memory]]

    assert(name, 'Undefined dataset name: ' .. name)
    assert(mode == 'train' or mode == 'test', ('Invalid mode: %s. Valid modes: train or test.'):format(mode))

    local save_dir = projectDir .. 'data/proposals'

    local proposals_fname = ('%s/%s_%s_%s.t7'):format(save_dir, name, alg_name, mode)

    -- check if the cache proposals .t7 file exists
    if paths.filep(proposals_fname) then
        return torch.load(proposals_fname)
    else
        error('file not found: ' .. proposals_fname )
    end
end

return load_rois