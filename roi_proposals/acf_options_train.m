function [ opts ] = acf_options_train(dataset_name, root_path, modelName)
%% check input arg type
if ischar(dataset_name)
    dataset = dataset_name;
elseif iscell(dataset_name)
    dataset = dataset_name{1};
else
    error('unknown data type: %s',class(dataset_name));
end

%% select options depending on the dataset
switch dataset
    case 'caltech'
        fprintf('Load Caltech options data... ')
        opts = options_caltech();
    case 'inria'
        fprintf('Load INRIA options data... ')
        opts = options_inria();
    case 'eth'
        fprintf('Load ETH options data... ')
        opts = options_eth();
        fprintf('Done!\n')
    case 'tudbrussels'
        fprintf('Load Tud-Brussels options data... ')
        opts = options_tudbrussels();
        fprintf('Done!\n')
    otherwise, error('unknown data type: %s',dataset);
end

%% add data path
opts.posGtDir = '';
opts.posImgDir = '';
opts.negImgDir = '';
if isempty(modelName),
    opts.name = [root_path '/data/acf/models/' opts.name];
else
    opts.name = [root_path '/data/acf/models/' modelName];
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function opts = options_caltech()
%% set up opts for training detector (see acfTrain)
opts=acfTrain(); opts.modelDs=[50 20.5]; opts.modelDsPad=[64 32];
opts.pPyramid.pChns.pColor.smooth=0; opts.nWeak=[64 256 1024 4096 4096*2];
opts.pBoost.pTree.maxDepth=5; opts.pBoost.discrete=0;
opts.pBoost.pTree.fracFtrs=1/16; opts.nNeg=25000*4; opts.nAccNeg=50000*2;
opts.pPyramid.pChns.pGradHist.softBin=1; opts.pJitter=struct('flip',1);
opts.pPyramid.pChns.shrink=2; opts.name='ACF_Caltech_model';
pLoad={'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}};
opts.pLoad = [pLoad 'hRng',[50 inf], 'vRng',[1 1] ];
end

function opts = options_inria()
%% set up opts for training detector (see acfTrain)
opts=acfTrain(); opts.modelDs=[100 41]; opts.modelDsPad=[128 64];
opts.nWeak=[32 128 512 2048]; opts.pJitter=struct('flip',1);
opts.pBoost.pTree.fracFtrs=1/16;
opts.nWeak=[32 128 512 2048];
opts.pJitter=struct('flip',1);
opts.pBoost.pTree.fracFtrs=1/16;
opts.pLoad={'squarify',{3,.41}}; opts.name='ACF_INRIA_model';
end

function opts = options_eth()
%% set up opts for training detector (see acfTrain)
opts=acfTrain(); opts.modelDs=[100 41]; opts.modelDsPad=[128 64];
opts.nWeak=[32 128 512 2048]; opts.pJitter=struct('flip',1);
opts.pBoost.pTree.fracFtrs=1/16;
opts.nWeak=[32 128 512 2048];
opts.pJitter=struct('flip',1);
opts.pBoost.pTree.fracFtrs=1/16;
opts.pLoad={'squarify',{3,.41}}; opts.name='ACF_ETH_model';
end

function opts = options_tudbrussels()
%% set up opts for training detector (see acfTrain)
opts=acfTrain(); opts.modelDs=[100 41]; opts.modelDsPad=[128 64];
opts.nWeak=[32 128 512 2048]; opts.pJitter=struct('flip',1);
opts.pBoost.pTree.fracFtrs=1/16;
opts.nWeak=[32 128 512 2048];
opts.pJitter=struct('flip',1);
opts.pBoost.pTree.fracFtrs=1/16;
opts.pLoad={'squarify',{3,.41}}; opts.name='ACF_TudBrussels_model';
end
