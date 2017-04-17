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
opt.save_dir = paths.concat( opt.save_dir, 'inria')
print('creating directory: ' .. opt.save_dir)
os.execute('mkdir -p ' .. opt.save_dir)

local urls = {
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/ACF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/ChnFtrs.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/ConvNet.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/CrossTalk.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/F-DNN.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/FPDW.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/FeatSynth.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/FisherBoost.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/Franken.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/FtrMine.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/HOG.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/HikSvm.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/HogLbp.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/InformedHaar.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/LDCF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/LatSvm-V1.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/LatSvm-V2.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/MLS.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/MultiFtr%2BCSS.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/MultiFtr.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/NAMC.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/Pls.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/PoseInv.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/PoseInvSvm.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/RPN%2BBF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/RandForest.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/Roerei.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/SCCPriors.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/Shapelet-orig.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/Shapelet.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/SketchTokens.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/SpatialPooling.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/VJ-OpenCv.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/VJ.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/VeryFast.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/WordChannels.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/INRIA/res/pAUCBoost.zip'
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