function process_rois(alg_name, dataset_name, set_name, data_dir, save_dir, skip_step, cascThr, cascCal)
%% process ACF ROI boxes for a dataset

fprintf('\n> Process %s (%s) %s roi detection/extraction script', dataset_name, set_name, alg_name)

%% setup toolboxes paths
[root_path] = add_paths_toolboxes();

%% load model
[model, opts] = load_model(alg_name, dataset_name);
switch alg_name
    case 'acf'
        pModify=struct('cascThr', cascThr, 'cascCal', cascCal);
        model=acfModify(model, pModify);
        detector = @(img) acfDetect(img, model);
        savename_dir = strcat(dataset_name, '_acf_skip=',num2str(skip_step), '_thresh=', ...
                              num2str(abs(cascThr)), '_cal=', num2str(abs(cascCal)));
    case 'ldcf'
        detector = @(img) acfDetect(img, model);
        savename_dir = strcat(dataset_name, '_ldcf_skip=',num2str(skip_step));
    case 'edgeboxes'
        detector = @(img) edgeBoxes(img, model, opts);
        savename_dir = strcat(dataset_name, '_edgeboxes_skip=',num2str(skip_step));
    otherwise, error('Invalid algorithm: %s', alg_name);
end

%% configs
save_path = fullfile(save_dir, savename_dir);

%% create directory
if(~exist(save_path,'dir')), mkdir(save_path); end

%% process ACF roi boxes
process_roi_detections(data_dir, save_path, skip_step, detector);

%% script complete
fprintf('Script completed.\n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function process_roi_detections(paths, save_path, skip_step, detector)
% Process ACF detections on a set of images using multiple processes for
% faster execution.
%
%% get all images filenames
fprintf('\n==> Fetching filenames... ')
filenames = get_files_subdirs(paths, skip_step);
fprintf(sprintf('%d videos have been selected.', size(filenames,1)))

%% Process edgeboxes
fprintf('\n==> Compute boxes: \n')
process_images(filenames, save_path, detector);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function filenames = get_files_subdirs(path, skip_step)
%% initializations
filenames = {};

%% cycle all paths 
for i=1:1:size(path,1)
    tmp = strsplit(path{i}, '/');
    set_name = tmp{end-1};
    % Get a list of all files and folders in this folder.
    files = dir(path{i});
    % Get a logical vector that tells which is a directory.
    dirFlags = [files.isdir];
    % Extract only those that are directories.
    subFolders = files(dirFlags);
    % Print folder names to command window.
    for k = 3 : length(subFolders) %3 means it will skip the . and .. names
        [filenames_set] = get_files_dir([path{i} subFolders(k).name '/'], skip_step);

        % add filenames to the full list
        filenames{end+1,1} = {set_name, subFolders(k).name, filenames_set};
    end
end
end


function [filenames] = get_files_dir(path, skip_step)
%% get all images
filenames = {};

% get all files contained under this path folder + subfolders
fileList = getAllFiles(path);

% Delete any entry that is not in JPG or PNG format
for ifile = 1:1:size(fileList,1)
    if isempty(strfind(fileList{ifile}, '.jpg')) && isempty(strfind(fileList{ifile}, '.png')) && isempty(strfind(fileList{ifile}, '.JPEG'))
        fileList{ifile,1} = [];
    end
end

% remove empty filenames
filenames = fileList(~cellfun('isempty',fileList));

%% select only filenames to be processed
fname = {};
for ifile=skip_step:skip_step:size(filenames,1)
    fname{end+1,1} = filenames{ifile};
end
filenames = fname;

end

function parsave(fname, boxes)
    save(fname, 'boxes')
end

function process_images(fname, save_path, detector)
%% setup progress bar
% Initialize progress bar with optinal parameters:
progressbar = textprogressbar(size(fname,1), 'barlength', 20, ...
                         'updatestep', 1, ...
                         'startmsg', sprintf('Processing rois: '),...
                         'endmsg', ' Done!', ...
                         'showbar', true, ...
                         'showremtime', true, ...
                         'showactualnum', true, ...
                         'barsymbol', '+', ...
                         'emptybarsymbol', '-');

%% Process boxes
for i=1:1:size(fname, 1)
    %% fetch filename batch
    data = fname{i,1};
    set_name = data{1};
    video_name = data{2};
    image_fnames = data{3};
    nfiles = size(image_fnames,1);
    
    % save boxes to file
    save_dir = fullfile(save_path, set_name, video_name);
    if(~exist(save_dir,'dir')), mkdir(save_dir); end
    
    
    %% process all images
    parfor ifile=1:nfiles
        filename = image_fnames{ifile};
        
        % load image
        img = imread(filename);
        
        % process detections
        bbs=detector(img);
        
        % validate boxes
        if isempty(bbs)
            bbs = [0,0,0,0,0];
        end
        
        % convert bbos from [x,y,w,h] to [xmin,ymin,xmax,ymax]
        boxes = [bbs(:,1), bbs(:,2), bbs(:,1) + bbs(:,3)-1, bbs(:,2) + bbs(:,4)-1];
        
        
        % save boxes to file
        [~,name,~] = fileparts(filename);
        bb_fname = fullfile(save_dir, [name '.mat']);
        
        parsave(bb_fname, boxes)
    end
    
    %% progress bar update
    progressbar(i)
end

end