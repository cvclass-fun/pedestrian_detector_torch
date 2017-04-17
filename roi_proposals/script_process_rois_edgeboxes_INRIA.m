function script_process_rois_edgeboxes_INRIA(varargin)
%% process Edge boxes for the INRIA dataset
% input arguments:
%     [1] - number of boxes per file
%     [2] - use aspect ratio value
%
fprintf('\n**************************************************************')
fprintf('\n**** Start INRIA Edgebox roi detection/extraction script. ****')
fprintf('\n**************************************************************')

%% setup toolboxes paths
[root_path] = add_paths_toolboxes();

%% initializations/parse input arguments
% number of boxes per file 
if nargin > 0
    if ~isempty(varargin{1}),
        max_number_boxes = varargin{1};
    else
        max_number_boxes = [];
    end
else
    max_number_boxes = [];
end

% aspect ratio threshold
if nargin > 1
    if ~isempty(varargin{2}),
        aspect_ratio_thresh = varargin{2};
    else
        aspect_ratio_thresh = inf;
    end
else
    aspect_ratio_thresh = inf;
end

%% load options
[model, opts] = edgeboxes_options(max_number_boxes);

%% configs
skip_step = 1;
savename_ext = strcat('_skip=',num2str(skip_step), '_BBs=', num2str(opts.maxBoxes), '_aspectRatio=', num2str(aspect_ratio_thresh), '.mat');
dataset_name = 'INRIA';
save_path = strcat(root_path, '/data/',dataset_name,'/proposals/');
dataset_path = strcat(root_path, '/data/',dataset_name,'/extracted_data/');

%% create directory
if(~exist(save_path,'dir')), mkdir(save_path); end

%% Train set
path_train = {strcat(dataset_path, 'set00/')};

%% process edgeboxes
fprintf('\nProcess Edgebox roi boxes for the training set:')
boxes = edgeboxes_process(path_train, skip_step, model, opts, strcat(dataset_name, '_Train'), aspect_ratio_thresh);

%% store to file
save_boxes(boxes, strcat(save_path, 'EdgeBoxes_',dataset_name,'Train', savename_ext))

%% Test set
path_test  = {strcat(dataset_path, 'set01/')};

%% process edgeboxes
fprintf('\nProcess Edgebox roi boxes for the testing set:')
boxes = edgeboxes_process(path_test, skip_step, model, opts, strcat(dataset_name, '_Test'), aspect_ratio_thresh);

%% store to file
save_boxes(boxes, strcat(save_path, 'EdgeBoxes_',dataset_name,'Test', savename_ext))

%% script complete
fprintf('\n--------------------------------------------')
fprintf('\nInria Edgeboxes processing script completed.')
fprintf('\n--------------------------------------------\n')
end