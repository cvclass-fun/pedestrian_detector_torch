function [ boxes ] = acf_process_detections(paths, skip_step, model, setType, thresh, cal)
% Process ACF detections on a set of images using multiple processes for
% faster execution.
%
%% get all images filenames
fprintf('\n==> Fetching filenames...')
filenames = get_files_subdirs(paths, skip_step);
fprintf(sprintf('%d files have been selected.', size(filenames,1)))

%% Process edgeboxes
fprintf('\n==> Compute ACF boxes:\n')
boxes = process_acf_images(filenames, model, setType, thresh, cal);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function filenames = get_files_subdirs(path, skip_step)
%% initializations
filenames = {};

%% cycle all paths 
for i=1:1:size(path,1)
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
        filenames = vertcat(filenames, filenames_set);
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

function [boxes] = process_acf_images(fname, model, setType, thresh, cal)
%% setup detector
pModify=struct('cascThr',thresh,'cascCal',cal);
detector=acfModify(model,pModify);

%% divide filenames into N batches 
% this will allow to use the progressbar when doing large computations
% with the use of a parfor
nFiles = size(fname,1);
batchsize_min = 100;
if nFiles / batchsize_min < batchsize_min
    batchSize = min(batchsize_min, nFiles);
else
    batchSize = floor(nFiles / batchsize_min);
end
fname_split = {};
temp_fname = {};
runningIndex = 1;
for ifile=1:1:nFiles
    % insert filename into the temporary storage
    temp_fname{end+1,1} = fname{ifile,1};
    
    % save fnames to the main storage
    if rem(ifile,batchSize) == 0
        % save criteria is true
        fname_split{runningIndex,1} = temp_fname;
        temp_fname = {};
        % increment counter
        runningIndex = runningIndex + 1;
    end

end
if ~isempty(temp_fname), fname_split{runningIndex,1} = temp_fname; end


%% setup progress bar
% Initialize progress bar with optinal parameters:
progressbar = textprogressbar(size(fname_split,1), 'barlength', 20, ...
                         'updatestep', 1, ...
                         'startmsg', sprintf('Processing ACF boxes %s: files=%d, thresh=%0.2f, cal=%0.3f batchSize=%d ', setType, nFiles, thresh, cal, batchSize),...
                         'endmsg', ' Done!', ...
                         'showbar', true, ...
                         'showremtime', true, ...
                         'showactualnum', true, ...
                         'barsymbol', '+', ...
                         'emptybarsymbol', '-');
                     
%% Process ACF boxes
boxes = cell(1, nFiles);
offset = 0;
for i=1:1:size(fname_split,1)
    %% fetch filename batch
    fname = fname_split{i,1};
    
    %% process batch
    offset = offset + size(fname,1);
    parfor ifile=1:size(fname,1)
        %% process edgeboxes
        img = imread(fname{ifile});
        bbs=acfDetect(img, detector);

        %% visualize the results
        %figure(1); imshow(img); bbApply('draw',bbs);

        %% validate boxes
        % check if the buffer is empty. If so, set a box with the size of the
        % whole image.
        if isempty(bbs)
            bbs = [0,0,0,0,0];
        end

        %% store edgeboxes to file
        bbs_correct_format = [bbs(:,1), bbs(:,2), bbs(:,1) + bbs(:,3)-1, bbs(:,2) + bbs(:,4)-1];
        boxes{1, offset + ifile} = [max(1,bbs_correct_format(:,2)), max(1,bbs_correct_format(:,1)), min(size(img,1),bbs_correct_format(:,4)), min(size(img,2),bbs_correct_format(:,3)), bbs(:,5)];
    end
    
    %% increment offset
    offset = offset + size(fname,1);
    
    %% progress bar update
    progressbar(i)
end
boxes = boxes(~cellfun('isempty',boxes));
end