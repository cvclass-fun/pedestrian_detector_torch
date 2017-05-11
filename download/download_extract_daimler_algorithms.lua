--[[
    Download and extract Caltech's pedestrian dataset algorithm files for evaluation/comparison.
]]

require 'paths'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Download Daimler\'s algorithms files for benchmark.')
cmd:text()
cmd:text('Options:')
cmd:option('-save_dir', './data/benchmark_algorithms/', 'Download benchmarking algorithms to this folder.')

-- parse options
local opt = cmd:parse(arg or {})

-- create directory if needed
opt.save_dir = paths.concat(opt.save_dir, 'daimler')
print('creating directory: ' .. opt.save_dir)
os.execute('mkdir -p ' .. opt.save_dir)

local urls = {
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/ConvNet.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/HOG.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/HikSvm.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/HogLbp.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/LatSvm-V1.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/LatSvm-V2.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/MLS.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/MultiFtr%2BCSS.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/MultiFtr%2BMotion.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/MultiFtr.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/RandForest.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/Shapelet.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/Daimler/res/VJ.zip'
}

-- Download algorithm's data
print('\nDownloading algorithms files...')
for k, url in pairs(urls) do
    os.execute(('cd %s && wget %s'):format(opt.save_dir, url))
end

-- Extract files to destination folder
print('\nExtracting algorithms files...')
os.execute(("cd %s && unzip \'*.zip\'"):format(opt.save_dir))

-- Remove tar files
print('\nRemoving unnecessary files...')
os.execute(('cd %s && rm -rf *.zip'):format(opt.save_dir))

print('Done.')