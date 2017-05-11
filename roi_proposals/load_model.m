function [ model, opts ] = load_model(alg_name, dataset_name)
% Load the detector model from disk.
switch alg_name
    case 'acf'
        [ model, opts ] = load_model_acf(dataset_name);
    case 'ldcf'
        [ model, opts ] = load_model_ldcf(dataset_name);
    case 'edgeboxes'
        [ model, opts ] = load_model_edgeboxes(dataset_name);
    otherwise, error('Invalid algorithm: %s', alg_name);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ model, opts ] = load_model_acf(dataset_name)
%% set options
opts=acfTrain();
opts.filters=[5 4];

%% select model name
switch dataset_name
    case 'caltech'
        %opts.name='../data/acf/models/ACF_Caltech_model';
        opts.name='./toolbox/detector/models/AcfCaltech+';
    case 'inria'
        %opts.name='../data/acf/models/ACF_INRIA_model';
        opts.name='./toolbox/detector/models/AcfInria';
    case 'eth'
        %opts.name='../data/acf/models/ACF_ETH_model';
        opts.name='./toolbox/detector/models/AcfInria';
    case 'tudbrussels'
        %opts.name='../data/acf/models/ACF_TudBrussels_model';
        opts.name='./toolbox/detector/models/AcfInria';
    otherwise,  error('Invalid dataset name: %s', dataset_name);
end

%% load detector model
nm=[opts.name 'Detector.mat'];
t=load(nm); 
model=t.detector;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ model, opts ] = load_model_ldcf(dataset_name)
%% set options
opts=acfTrain();
opts.filters=[5 4];

%% select model name
switch dataset_name
    case 'caltech'
        %opts.name='../data/acf/models/ACF_Caltech_model';
        opts.name='./toolbox/detector/models/LdcfCaltech';
    case 'inria'
        %opts.name='../data/acf/models/ACF_INRIA_model';
        opts.name='./toolbox/detector/models/LdcfInria';
    case 'eth'
        %opts.name='../data/acf/models/ACF_ETH_model';
        opts.name='./toolbox/detector/models/LdcfInria';
    case 'tudbrussels'
        %opts.name='../data/acf/models/ACF_TudBrussels_model';
        opts.name='./toolbox/detector/models/LdcfInria';
    otherwise,  error('Invalid dataset name: %s', dataset_name);
end

%% load detector model
nm=[opts.name 'Detector.mat'];
t=load(nm); 
model=t.detector;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ model, opts ] = load_model_edgeboxes(varargin)
%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('./edges/models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = 0.65; %0.65;   % step size of sliding window search
opts.beta  = 0.75; %0.90;   % nms threshold for object proposals
%opts.eta=.9996;
opts.minScore = .001;       % min score of boxes to detect
opts.maxBoxes = 1e4;%5e4;   % max number of boxes to detect

%% process input args
if nargin > 0, if ~isempty(varargin{1}), opts.maxBoxes = varargin{1}; end; end
end