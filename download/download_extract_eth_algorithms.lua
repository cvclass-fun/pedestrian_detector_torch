--[[
    Download and extract Caltech's pedestrian dataset algorithm files for evaluation/comparison.
]]

require 'paths'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Download INRIA\'s algorithms files for benchmark.')
cmd:text()
cmd:text('Options:')
cmd:option('-save_dir', './data/benchmark_algorithms/', 'Download benchmarking algorithms to this folder.')

-- parse options
local opt = cmd:parse(arg or {})

-- create directory if needed
opt.save_dir = paths.concat( opt.save_dir, 'eth')
print('creating directory: ' .. opt.save_dir)
os.execute('mkdir -p ' .. opt.save_dir)

local urls = {
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/ACF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/ChnFtrs.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/ConvNet.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/CrossTalk.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/DBN-Isol.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/DBN-Mut.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/FPDW.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/FisherBoost.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/Franken.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/HOG.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/HikSvm.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/HogLbp.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/JointDeep.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/LDCF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/LatSvm-V1.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/LatSvm-V2.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/MF%2BMotion%2B2Ped.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/MLS.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/MultiFtr%2BCSS.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/MultiFtr%2BMotion.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/MultiFtr.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/MultiSDP.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/Pls.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/PoseInv.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/RPN%2BBF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/RandForest.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/Roerei.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/SDN.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/Shapelet.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/SpatialPooling.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/TA-CNN.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/VJ.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/VeryFast.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/ETH/res/pAUCBoost.zip'
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