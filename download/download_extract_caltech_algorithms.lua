--[[
    Download and extract Caltech's pedestrian dataset algorithm files for evaluation/comparison.
]]

require 'paths'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Download caltech\'s algorithms files for benchmark.')
cmd:text()
cmd:text('Options:')
cmd:option('-save_dir', './data/benchmark_algorithms/', 'Download benchmarking algorithms to this folder.')

-- parse options
local opt = cmd:parse(arg or {})

-- create directory if needed
opt.save_dir = paths.concat(opt.save_dir, 'caltech')
print('creating directory: ' .. opt.save_dir)
os.execute('mkdir -p ' .. opt.save_dir)

local urls = {
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/ACF%2B%2B.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/ACF%2BSDt.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/ACF-Caltech%2B.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/ACF-Caltech.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/ACF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/AFS%2BGeo.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/AFS.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/CCF%2BCF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/CCF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/Checkerboards%2B.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/Checkerboards.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/ChnFtrs.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/CompACT-Deep.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/ConvNet.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/CrossTalk.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/CrossTalk.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/DBN-Isol.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/DBN-Mut.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/DeepCascade%2B.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/DeepCascade.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/DeepParts.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/F-DNN%2BSS.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/F-DNN.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/FPDW.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/FastCF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/FeatSynth.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/Franken.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/FtrMine.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/HOG.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/HikSvm.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/HogLbp.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/InformedHaar.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/JointDeep.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/Katamari.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/LDCF%2B%2B.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/LDCF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/LFOV.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/LatSvm-V1.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/LatSvm-V2.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MF%2BMotion%2B2Ped.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MLS.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MOCO.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MRFC%2BSemantic.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MS-CNN.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MT-DPM%2BContext.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MT-DPM.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MultiFtr%2BCSS.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MultiFtr%2BMotion.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MultiFtr.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MultiResC%2B2Ped.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MultiResC.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/MultiSDP.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/NAMC.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/Pls.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/PoseInv.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/RPN%2BBF.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/RandForest.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/Roerei.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/SA-FastRCNN.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/SCCPriors.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/SCF%2BAlexNet.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/SDN.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/Shapelet.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/SpatialPooling%2B.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/SpatialPooling.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/TA-CNN.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/VJ.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/WordChannels.zip',
    'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/res/pAUCBoost.zip'
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