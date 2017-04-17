function script_process_rois_ldcf_TudBrussels(varargin)
%% process ACF ROI boxes for the Tud-Brussels dataset
% input arguments:
%     [1] - threshold for suppression of weak(er) boxes
%
fprintf('\n******************************************************************')
fprintf('\n**** Start Tud-Brussels LDCF roi detection/extraction script. ****')
fprintf('\n******************************************************************')

%% setup toolboxes paths
[root_path] = add_paths_toolboxes();

%% load options
[model] = ldcf_options_process('tudbrussels');

%% configs
skip_step = 1;
savename_ext = strcat('_skip=',num2str(skip_step), '.mat');
dataset_name = 'Tud-Brussels';
save_path = strcat(root_path, '/data/',dataset_name,'/proposals/');
dataset_path = strcat(root_path, '/data/',dataset_name,'/extracted_data/');

%% create directory
if(~exist(save_path,'dir')), mkdir(save_path); end

%% Train set
path_train = {strcat(dataset_path, 'set00/')};

%% process ACF roi boxes
fprintf('\nProcess LDCF roi boxes for the training/testing set:')
boxes = ldcf_process_detections(path_train, skip_step, model, strcat(dataset_name, ' Train'));

%% store to file
save_boxes(boxes, strcat(save_path, 'LDCF_',dataset_name,'TrainTest', savename_ext))
% Note: both are the same because the same data can be used for
% tes/benchmarking or for augmenting an already existing dataset with this
% extra data.

%% script complete
fprintf('\n----------------------------------------------------')
fprintf('\nTud-Brussels LDCF boxes processing script completed.')
fprintf('\n----------------------------------------------------\n')
end