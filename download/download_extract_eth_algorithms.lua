--[[
    Download and extract Caltech's pedestrian dataset algorithm files for evaluation/comparison.
]]

require 'paths'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Download INRIA\'s algorithms files for benchmark.')
cmd:text()
cmd:text('Options:')
cmd:option('-save_dir', '../data', 'Download benchmarking algorithms to this folder.')

-- parse options
local opt = cmd:parse(arg or {})

-- create directory if needed
opt.save_dir = paths.concat( opt.save_dir, 'ETH', 'algorithms')
print('creating directory: ' .. opt.save_dir)
os.execute('mkdir -p ' .. opt.save_dir)

-- Download algorithm's data
print('\nDownloading algorithms files...')
local url = 'http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/ETH/res/'
os.execute(('cd %s && wget -r -nH -nd -np -e robots=off -R index.html* %s'):format(opt.save_dir, url))

-- Extract files to destination folder
print('\nExtracting algorithms files...')
os.execute(("cd %s && unzip \'*.zip\'"):format(opt.save_dir))

-- Remove tar files
print('\nRemoving unnecessary files...')
os.execute(('cd %s && rm -rf *.zip'):format(opt.save_dir))

print('Done.')