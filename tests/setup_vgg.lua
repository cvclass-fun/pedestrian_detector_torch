require 'paths'
require 'torch'
local fastrcnn = require 'fastrcnn'

torch.setdefaulttensortype('torch.FloatTensor')
paths.dofile('../projectdir.lua')

local models = {'vgg16', 'vgg19'}
local max_feat_id = 7

local sample_img = torch.CudaTensor(1,3,600,600):uniform()
local sample_proposals = torch.CudaTensor({{1,1,1,100,100}, {1,100,100,200,200}})

for _, model_id in pairs(models) do
    for _, clsType in pairs({'concat','parallel'}) do
        print('\n=======================================================')
        print(('Testing **%s** with the classifier type: %s'):format(model_id, clsType))
        print('=======================================================\n')

        for i=1, max_feat_id do
            local sample_img = torch.CudaTensor(1,3,600,600):uniform()
            local sample_proposals = torch.CudaTensor({{1,1,1,100,100}, {1,100,100,200,200}})

            print(('\nSetup model: %s (%d/%d)'):format(model_id, i, max_feat_id))
            local load_model = paths.dofile('../model/init.lua')
            local model, model_parameters = load_model(model_id, clsType, i, -1, 2048, 1, 1)

            print('Display model arch:')
            print(model)

            print('Run model...')
            local res = model:forward{sample_img, sample_proposals}
            print('Forward pass successful!')
            model = nil
            model_parameters = nil
            res = nil
            collectgarbage()
            collectgarbage()
        end
    end
end