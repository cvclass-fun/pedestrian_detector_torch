function [ model, opts ] = edgeboxes_options(varargin)
%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
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
