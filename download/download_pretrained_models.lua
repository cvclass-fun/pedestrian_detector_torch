--[[
    Download models and convert/store them to torch7 file format.
]]


require 'paths'
require 'torch'


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Download ImageNet pre-trained models.')
cmd:text()
cmd:text('Options:')
cmd:option('-save_dir', './data/pretrained_models/', 'Download models to this folder.')

-- parse options
local opt = cmd:parse(arg or {})

-- create directory if needed
print('creating directory: ' .. opt.save_dir)
os.execute('mkdir -p ' .. opt.save_dir)


-- alexnet
print('Download AlexNet model...')
os.execute(('th download/download_alexnet.lua -save_dir %s'):format(opt.save_dir))

-- zeiler
print('Download ZeilerNet model...')
os.execute(('th download/download_zeiler.lua -save_dir %s'):format(opt.save_dir))

-- VGG
print('Download VGG16/19 models...')
os.execute(('th download/download_vgg16_vgg19.lua -save_dir %s'):format(opt.save_dir))

-- resnet
print('Download resnet models...')
os.execute(('th download/download_resnet.lua -save_dir %s'):format(opt.save_dir))

-- googlenet
print('Download GoogleNet inception V3 model...')
os.execute(('th download/download_googlenet.lua -save_dir %s'):format(opt.save_dir))

print('Script complete.')