function [detector] = script_train_acf_detector_Caltech( varargin )
% Trains the ACF detector on the INRIA dataset.
%
fprintf('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++')
fprintf('\n++++ Start Caltech ACF detector training script. ++++')
fprintf('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++')

if nargin > 0
    modelName = varargin{1};
else
    modelName = '';
end

%% setup toolboxes paths
fprintf('\n(1/6) Setting toolbox dependencies...')
[root_path] = add_paths_toolboxes();
fprintf('Done!')

%% define datasets
fprintf('\n(2/6) Select dataset(s) to train...')
dataset_name = {'caltech', 'tudbrussels', 'inriaTrain', 'inriaTest'};
fprintf(dataset_name{1})

%% load options
fprintf('\n(3/6) Setting ACF options...')
opts = acf_options_train( dataset_name, root_path, modelName);
fprintf('Done!')

%% prepare training data
fprintf('\n(4/6) Fetch all file names of the selected dataset/s and make symbolic links...')
acf_prepare_training_data( dataset_name, root_path );

%% set paths
if exist([root_path '/data/acf/train_data/annotations/'], 'dir'), opts.posGtDir = [root_path '/data/acf/train_data/annotations/']; end
if exist([root_path '/data/acf/train_data/images/'], 'dir'), opts.posImgDir = [root_path '/data/acf/train_data/images/']; end
if exist([root_path '/data/acf/train_data/neg/'], 'dir'), opts.negImgDir = [root_path '/data/acf/train_data/neg/']; end

%% train detector (see acfTrain)
fprintf('\n(5/6) Start ACF detector train:\n')
time = tic;
detector = acfTrain( opts );
toc(time)
fprintf('\nACF detector training complete!')

%% remove symbolic links folders
fprintf('\n(6/6) Removing symbolic links data...')
status = system(['rm -rf ' root_path '/data/acf/train_data']);
fprintf('Done!')

%% script complete
fprintf('\n----------------------------------------------')
fprintf('\nCaltech ACF detector training script complete!')
fprintf('\n----------------------------------------------\n')
end