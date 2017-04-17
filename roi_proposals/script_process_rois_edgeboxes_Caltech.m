function script_process_rois_edgeboxes_Caltech(varargin)
%% process Edge boxes for the Caltech dataset
% input arguments:
%     [1] - skip_step
%     [2] - number of boxes per file
%     [3] - use aspect ratio value
%
fprintf('\n****************************************************************')
fprintf('\n**** Start Caltech Edgebox roi detection/extraction script. ****')
fprintf('\n****************************************************************')

%% setup toolboxes paths
[root_path] = add_paths_toolboxes();

%% initializations/parse input arguments
% skip_step
if nargin > 0
    if ~isempty(varargin{1}),
        skip_step = max(1, varargin{1});
    else
        skip_step = 30;
    end
else
    skip_step = 30;
end

% number of boxes per file 
if nargin > 1
    if ~isempty(varargin{2}),
        max_number_boxes = varargin{2};
    else
        max_number_boxes = [];
    end
else
    max_number_boxes = [];
end

% aspect ratio threshold
if nargin > 2
    if ~isempty(varargin{3}),
        aspect_ratio_thresh = varargin{3};
    else
        aspect_ratio_thresh = inf;
    end
else
    aspect_ratio_thresh = inf;
end

%% load options
[model, opts] = edgeboxes_options(max_number_boxes);

%% configs
savename_ext = strcat('_skip=',num2str(skip_step), '_BBs=', num2str(opts.maxBoxes), '_aspectRatio=', num2str(aspect_ratio_thresh), '.mat');
dataset_name = 'Caltech';
save_path = strcat(root_path, '/data/',dataset_name,'/proposals/');
dataset_path = strcat(root_path, '/data/',dataset_name,'/extracted_data/');

%% create directory
if(~exist(save_path,'dir')), mkdir(save_path); end

%% Train set
path_train = {strcat(dataset_path, 'set00/');
              strcat(dataset_path, 'set01/');
              strcat(dataset_path, 'set02/');
              strcat(dataset_path, 'set03/');
              strcat(dataset_path, 'set04/');
              strcat(dataset_path, 'set05/');
             };

%% process edgeboxes
fprintf('\nProcess Edgebox roi boxes for the training set:')
boxes = edgeboxes_process(path_train, skip_step, model, opts, 'Caltech Train', aspect_ratio_thresh);

%% store to file
save_boxes(boxes, strcat(save_path, 'EdgeBoxes_',dataset_name,'Train', savename_ext))

%% Test set
path_test  = {strcat(dataset_path, 'set06/');
              strcat(dataset_path, 'set07/');
              strcat(dataset_path, 'set08/');
              strcat(dataset_path, 'set09/');
              strcat(dataset_path, 'set10/');
             };

%% process edgeboxes
fprintf('\nProcess Edgebox roi boxes for the testing set:')
boxes = edgeboxes_process(path_test, skip_step, model, opts, 'Caltech Test', aspect_ratio_thresh);

%% store to file
save_boxes(boxes, strcat(save_path, 'EdgeBoxes_',dataset_name,'Test', savename_ext))

%% script complete
fprintf('\n----------------------------------------------')
fprintf('\nCaltech Edgeboxes processing script completed.')
fprintf('\n----------------------------------------------\n')
end