--[[
    Train + test + benchmark FRCNN model script for the caltech dataset.
]]

-- 1. Load dependencies + options
paths.dofile('/home/mf/Toolkits/Codigo/git/fast-rcnn-torch/FastRCNN.lua')
local opt = FastRCNN.Configs.parse(arg)

-- 2. Setup input options for the script
opt.save_dir_frcnn = nil
opt.train_rois_file = '../data/Caltech/proposals/caltech_train_10x.mat'
opt.test_rois_file = '../data/Caltech/proposals/caltech_test_10x.mat'
--opt.train_rois_file = '/media/HDD2/miguel_DATA/proposals/EdgeBoxes/CaltechFULLtrain_skip30_BBs10000_ar.mat'
--opt.test_rois_file = '/media/HDD2/miguel_DATA/proposals/EdgeBoxes/CaltechFULLtest_skip30_BBs10000_ar.mat'
opt.schedule = {{7,1e-3,5e-4},{3, 1e-4, 5e-4}}
opt.batchSize = 128
-- set model save path
--opt.load_model_path = '/media/HDD2/miguel_DATA/data_cache/FastRCNN/caltech_SKIP30_OVERALL/Mon_Jun__6_22:57:08_2016_alexnet'
if opt.load_model_path == '' then
  local date_time = os.date():gsub(' ','_')
  opt.load_model_path = paths.concat('../data/models', 'Caltech' .. date_time .. '_' .. opt.modelName)
  os.execute('mkdir -p ' .. opt.load_model_path)
end
-- set options
local options = ''
for name, val in pairs(opt) do
  local value = val
  if type(value) == 'table' then
    value = string.addons.remove_whitespace(table.addons.to_string(value, 1, {}, 0))
  elseif type(value) == 'boolean' then
    value = '\"' .. tostring(value) .. '\"'
  end
  options = options .. '-' .. name .. ' ' .. value .. ' '
end

-- 3. Train network
print('1. Train a pedestrian detector model: ')
os.execute(('th run_train.lua %s'):format(options))

-- 4. Test network
print('2. Test the trained model mAP accuracy: ')
os.execute(('th run_test.lua %s'):format(options))

-- 5. Benchmark network
print('3. Benchmark the detector using Caltech\'s pedestrian benchmark toolbox: ')
os.execute(('th run_benchmark.lua %s'):format(options))

-- 6. Benchmark network
print('6. Train + test + benchmark script complete!')