--[[
    Download pre-processed region-of-interest proposals.
]]


require 'paths'
require 'torch'


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 Fast-RCNN download roi proposals.')
cmd:text()
cmd:text(' ---------- General options ------------------------------------')
cmd:text()
cmd:option('-save_dir',   'data/proposals/', 'Experiment ID')
cmd:text()

local opt = cmd:parse(arg or {})
local savepath = opt.save_dir

-- create directory if needed
if not paths.dirp(savepath) then
    print('creating directory: ' .. savepath)
    os.execute('mkdir -p ' .. savepath)
end

print('==> Downloading Region-of-Interest proposals...')

local url = 'https://www.dropbox.com/s/q0r5phgewyt39ai/proposals.tar.gz'

-- file names
local filename = paths.concat(savepath, 'proposals.tar.gz')

-- download file
if not paths.filep(filename) then
  local command = ('wget -O %s %s'):format(filename, url1)
  os.execute(command)
end

-- Extract files
os.execute(('cd %s && tar -xvf %s'):format(savepath, filename))

print('Done.')