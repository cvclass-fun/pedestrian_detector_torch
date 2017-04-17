--[[
    Download and extract Caltech's pedestrian dataset algorithm files for evaluation/comparison.
]]

require 'paths'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Download caltech\'s algorithms files for benchmark.')
cmd:text()
cmd:text('Options:')
cmd:option('-save_dir', '../data', 'Download models to this folder.')

-- parse options
local opt = cmd:parse(arg or {})

-- create directory if needed
print('creating directory: ' .. opt.save_dir)
os.execute('mkdir -p ' .. opt.save_dir)

-- Caltech
print('Download caltech\'s algorithms files for benchmark...')
os.execute(('th download_extract_caltech_algorithms.lua -save_dir %s'):format(opt.save_dir))

-- INRIA
print('Download INRIA\'s algorithms files for benchmark...')
os.execute(('th download_extract_inria_algorithms.lua -save_dir %s'):format(opt.save_dir))

-- ETH
print('Download ETH\'s algorithms files for benchmark...')
os.execute(('th download_extract_eth_algorithms.lua -save_dir %s'):format(opt.save_dir))

-- Tud-Brussels
print('Download Tud-Brussels\'s algorithms files for benchmark...')
os.execute(('th download_extract_tudbrussels_algorithms.lua -save_dir %s'):format(opt.save_dir))

-- Daimler
print('Download Daimler\'s algorithms files for benchmark...')
os.execute(('th download_extract_daimler_algorithms.lua -save_dir %s'):format(opt.save_dir))

print('Script complete.')