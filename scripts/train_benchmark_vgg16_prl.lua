--[[
    Train + benchmark a vgg16 model with the parallel classifier network.
]]


require 'paths'
require 'torch'


--------------------------------------------------------------------------------
-- Train + benchmark network
--------------------------------------------------------------------------------

local info = {
    -- experiment id
    expID = 'caltech_10x_vgg16_prl',

    -- dataset setup
    dataset = 'caltech_10x',
    proposalAlg = 'acf',

    -- model setup
    netType = 'vgg16',
    clsType = 'parallel',
    featID = 4,
    clear_buffers = 'true',

    -- train options
    optMethod = 'sgd',
    nThreads = 4,
    trainIters = 1000,
    snapshot = 10,
    schedule = "{{30,1e-3,5e-4},{10,1e-4,5e-4}}",
    testInter = 'false',
    snapshot = 10,
    nGPU = 2,

    -- FRCNN options
    frcnn_scales = 600,
    frcnn_max_size = 1000,
    frcnn_imgs_per_batch = 2,
    frcnn_rois_per_img = 128,
    frcnn_fg_fraction = 0.5,
    frcnn_bg_fraction = 0.75,
    frcnn_fg_thresh = 0.5,
    frcnn_bg_thresh_hi = 0.5,
    frcnn_bg_thresh_lo = 0.1,
    frcnn_hflip = 0.5,
    frcnn_roi_augment_offset = 0.3,

    -- FRCNN Test options
    frcnn_test_scales = 600,
    frcnn_test_max_size = 1000,
    frcnn_test_nms_thresh = 0.5,
    frcnn_test_mode = 'voc',
    eval_plot_name = 'OURS'
}

-- concatenate options fields to a string
local str_args = ''
for k, v in pairs(info) do
    str_args = str_args .. ('-%s %s '):format(k, v)
end

local str_cuda
if info.nGPU <= 1 then
    str_cuda = 'CUDA_VISIBLE_DEVICES=1'
else
    str_cuda = 'CUDA_VISIBLE_DEVICES=1,0'
end

-- display options
print('Input options: ' .. str_args)

-- train network
os.execute(('%s th train.lua %s'):format(str_cuda, str_args))

-- benchmark network
os.execute('CUDA_VISIBLE_DEVICES=0 th benchmark.lua ' .. str_args)