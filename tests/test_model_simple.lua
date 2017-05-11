require 'paths'
require 'torch'
--local fastrcnn = require 'fastrcnn'
local fastrcnn = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/init.lua')

torch.setdefaulttensortype('torch.FloatTensor')
paths.dofile('../projectdir.lua')

local models = {'alexnet', 'vgg16', 'vgg19', 'zeiler'}

local sample_img = torch.CudaTensor(1,3,600,600):uniform()
local sample_proposals = torch.CudaTensor({{1,1,1,100,100}, {1,100,100,200,200}})

for i=1, #models do
    local sample_img = torch.CudaTensor(1,3,600,600):uniform()
    local sample_proposals = torch.CudaTensor({{1,1,1,100,100}, {1,100,100,200,200}})

    print(('Setup model: %s (%d/%d)'):format(models[i], i, #models))
    local load_model = paths.dofile('../model/init.lua')
    local model, model_parameters = load_model(models[i], 'simple', 1, -1, 4096, 1, 1)

    local res = model:forward{sample_img, sample_proposals}
    model = nil
    model_parameters = nil
    res = nil
    collectgarbage()
    collectgarbage()
end
