function script_process_rois_ldcf_Caltech(varargin)
%% process ACF ROI boxes for the Caltech dataset
% input arguments:
%     [1] - skip_step
%     [2] - threshold for suppression of weak(er) boxes
%     [3] - Flag to skip processing the test set. If 1, skip the test set.
%           If 0 then compute both train and test sets.
%
fprintf('\n*************************************************************')
fprintf('\n**** Start Caltech LDCF roi detection/extraction script. ****')
fprintf('\n*************************************************************')

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

% flag_exist
if nargin > 3
    if ~isempty(varargin{4}),
        flag_exit = varargin{4};
    else
        flag_exit = 0;
    end
else
    flag_exit = 0;
end

%% load options
[model] = ldcf_options_process('caltech');

%% configs
savename_ext = strcat('_skip=',num2str(skip_step), '.mat');
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

%% process ACF roi boxes
fprintf('\nProcess LDCF roi boxes for the training set:')
boxes = ldcf_process_detections(path_train, skip_step, model, strcat(dataset_name, ' Train'));

%% store to file
save_boxes(boxes, strcat(save_path, 'LDCF_',dataset_name,'Train', savename_ext))

%% check if the flag exit is true
if flag_exit, 
    return % exit script
end

%% Test set
path_test  = {strcat(dataset_path, 'set06/');
              strcat(dataset_path, 'set07/');
              strcat(dataset_path, 'set08/');
              strcat(dataset_path, 'set09/');
              strcat(dataset_path, 'set10/');
             };

%% process ACF roi boxes
fprintf('\nProcess LDCF roi boxes for the testing set:')
boxes = ldcf_process_detections(path_test, skip_step, model, strcat(dataset_name, ' Test'));

%% store to file
save_boxes(boxes, strcat(save_path, 'LDCF_',dataset_name,'Test', savename_ext))

%% script complete
fprintf('\n-----------------------------------------------')
fprintf('\nCaltech LDCF boxes processing script completed.')
fprintf('\n-----------------------------------------------\n')
end
