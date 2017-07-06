local configs = {
    -- experiment id
    expID = '',

    -- dataset setup
    dataset = 'caltech_10x',
    proposalAlg = 'acf',

    -- model setup
    netType = 'alexnet',
    clsType = '',
    featID = 4,
    cls_size = 4096,
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

return configs