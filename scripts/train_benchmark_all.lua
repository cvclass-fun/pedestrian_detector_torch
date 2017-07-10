--[[
    Run all train+benchmark scripts
]]

-- alexnet
paths.dofile('train_benchmark_alexnet_vanilla.lua')
paths.dofile('train_benchmark_alexnet_concat.lua')
paths.dofile('train_benchmark_alexnet_prl.lua')

-- vgg16
paths.dofile('train_benchmark_vgg16_vanilla.lua')
paths.dofile('train_benchmark_vgg16_concat.lua')
paths.dofile('train_benchmark_vgg16_prl.lua')