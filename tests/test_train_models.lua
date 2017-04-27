--[[
    Train + benchmark models.
]]

require 'paths'
require 'torch'

paths.dofile('../projectdir.lua')


local cuda_devices = 'CUDA_VISIBLE_DEVICES=0'


-----------------------------------------------------------
-- simple + alexnet
-----------------------------------------------------------

local expID = 'alexnet_simple'

os.execute(('%s th %strain.lua -expID %s -nThreads 2 -clsType \'simple\' -featID 1'):format(cuda_devices, projectDir, expID))
os.execute(('%s th %sbenchmark.lua -expID %s'):format(cuda_devices, projectDir, expID))


-----------------------------------------------------------
-- concat + alexnet
-----------------------------------------------------------

local expID = 'alexnet_concat'

os.execute(('%s th %strain.lua -expID %s -nThreads 2 -clsType \'concat\' -featID 2'):format(cuda_devices, projectDir, expID))
os.execute(('%s th %sbenchmark.lua -expID %s'):format(cuda_devices, projectDir, expID))


-----------------------------------------------------------
-- parallel + alexnet
-----------------------------------------------------------

local expID = 'alexnet_parallel'

os.execute(('%s th %strain.lua -expID %s -nThreads 2 -clsType \'parallel\' -featID 2'):format(cuda_devices, projectDir, expID))
os.execute(('%s th %sbenchmark.lua -expID %s'):format(cuda_devices, projectDir, expID))