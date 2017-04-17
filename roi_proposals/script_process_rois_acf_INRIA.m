function script_process_rois_acf_INRIA(varargin)
%% process ACF ROI boxes for the INRIA dataset
% input arguments:
%     [1] - threshold for suppression of weak(er) boxes
%
fprintf('\n**********************************************************')
fprintf('\n**** Start INRIA ACF roi detection/extraction script. ****')
fprintf('\n**********************************************************')

%% setup toolboxes paths
[root_path] = add_paths_toolboxes();

%% initializations/parse input arguments
% acf threshold
if nargin > 0
    if ~isempty(varargin{1}),
        cascThr = varargin{1};
    else
        cascThr = -1;
    end
else
    cascThr = -1;
end

% acf calibration 
if nargin > 1
    if ~isempty(varargin{2}),
        cascCal = varargin{2};
    else
        cascCal = .01;
    end
else
    cascCal = .01;
end

%% load options
[model] = acf_options_process('inria');

%% configs
skip_step = 1;
savename_ext = strcat('_skip=',num2str(skip_step), '_thresh=', num2str(abs(cascThr)), '_cal=', num2str(abs(cascCal)), '.mat');
dataset_name = 'INRIA';
save_path = strcat(root_path, '/data/',dataset_name,'/proposals/');
dataset_path = strcat(root_path, '/data/',dataset_name,'/extracted_data/');

%% create directory
if(~exist(save_path,'dir')), mkdir(save_path); end

%% Train set
path_train = {strcat(dataset_path, 'set00/')};

%% process ACF roi boxes
fprintf('\nProcess ACF roi boxes for the training set:')
boxes = acf_process_detections(path_train, skip_step, model, strcat(dataset_name, ' Train'), cascThr, cascCal);

%% store to file
save_boxes(boxes, strcat(save_path, 'ACF_',dataset_name,'Train', savename_ext))

%% Test set
path_test  = {strcat(dataset_path, 'set01/')};

%% process ACF roi boxes
fprintf('\nProcess ACF roi boxes for the testing set:')
boxes = acf_process_detections(path_test, skip_step, model, strcat(dataset_name, ' Test'), cascThr, cascCal);

%% store to file
save_boxes(boxes, strcat(save_path, 'ACF_',dataset_name,'Test', savename_ext))

%% script complete
fprintf('\n--------------------------------------------')
fprintf('\nINRIA ACF boxes processing script completed.')
fprintf('\n--------------------------------------------\n')
end