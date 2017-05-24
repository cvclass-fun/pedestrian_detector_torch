--[[
    Download all data necessary for this repo in a single script.
]]

-- benchmark algorithms
paths.dofile('download_extract_algorithms.lua')

-- Imagenet pre-trained models
paths.dofile('download_pretrained_models.lua')

-- roi proposals
paths.dofile('download_proposals.lua')