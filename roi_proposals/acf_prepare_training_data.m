function acf_prepare_training_data( dataset_name, root_path  )
% Prepares the sample data for training the acf detector.
%
%% check input arg type
if ischar(dataset_name)
    dataset = {dataset_name};
elseif iscell(dataset_name)
    dataset = dataset_name;
else
    error('unknown data type: %s',class(dataset_name));
end

%% go to extracted_data folder
pth = [root_path '/data/acf/train_data/'];
% create dirs (if they don't exist)
%if(~exist([pth 'images/'],'dir')), mkdir([pth 'images/']); end
%if(~exist([pth 'annotations/'],'dir')), mkdir([pth 'annotations/']); end
%if(~exist([pth 'neg/'],'dir')), mkdir([pth 'neg/']); end
for i=1:1:length(dataset)
    fprintf(sprintf('\npreparing set %d/%d data...', i, length(dataset)))
    switch dataset{i}
        case {'caltechTrain', 'caltech'}
            fprintf('\n==> Process Caltech train data:\n ')
            % add set00 samples
            fprintf('\n==>> set00\n ')
            fetch_filenames_dataset_to_folder('caltechTrain', [root_path '/data/Caltech/extracted_data/set00'], pth, 1)
            % add set01 samples
            fprintf('\n==>> set01\n ')
            fetch_filenames_dataset_to_folder('caltechTrain', [root_path '/data/Caltech/extracted_data/set01'], pth, 1)
            % add set02 samples
            fprintf('\n==>> set02\n ')
            fetch_filenames_dataset_to_folder('caltechTrain', [root_path '/data/Caltech/extracted_data/set02'], pth, 1)
            % add set03 samples
            fprintf('\n==>> set03\n ')
            fetch_filenames_dataset_to_folder('caltechTrain', [root_path '/data/Caltech/extracted_data/set03'], pth, 1)
            % add set04 samples
            fprintf('\n==>> set04\n ')
            fetch_filenames_dataset_to_folder('caltechTrain', [root_path '/data/Caltech/extracted_data/set04'], pth, 1)
            % add set05 samples
            fprintf('\n==>> set05\n ')
            fetch_filenames_dataset_to_folder('caltechTrain', [root_path '/data/Caltech/extracted_data/set05'], pth, 1)
            fprintf('Done!')
        case 'caltechTest'
            fprintf('\n==> Process Caltech test data...\n ')
            % add set06 samples
            fprintf('\n==>> set06\n ')
            fetch_filenames_dataset_to_folder('caltechTest', [root_path '/data/Caltech/extracted_data/set06'], pth, 1)
            % add set07 samples
            fprintf('\n==>> set07\n ')
            fetch_filenames_dataset_to_folder('caltechTest', [root_path '/data/Caltech/extracted_data/set07'], pth, 1)
            % add set08 samples
            fprintf('\n==>> set08\n ')
            fetch_filenames_dataset_to_folder('caltechTest', [root_path '/data/Caltech/extracted_data/set08'], pth, 1)
            % add set09 samples
            fprintf('\n==>> set09\n ')
            fetch_filenames_dataset_to_folder('caltechTest', [root_path '/data/Caltech/extracted_data/set09'], pth, 1)
            % add set10 samples
            fprintf('\n==>> set10\n ')
            fetch_filenames_dataset_to_folder('caltechTest', [root_path '/data/Caltech/extracted_data/set10'], pth, 1)
            fprintf('Done!')
        case {'inriaTrain', 'inria'}
            fprintf('\n==> Process INRIA train data: \n ')
            % add positive samples
            fprintf('\n==>> set00\n ')
            fetch_filenames_dataset_to_folder('inriaTrain', [root_path '/data/INRIA/extracted_data/set00/V000'], pth, 1)
            % add negative samples
            fetch_filenames_dataset_to_folder('inriaTrain', [root_path '/data/INRIA/extracted_data/set00/V001'], pth, 0)
            fprintf('Done!')
        case 'inriaTest'
            fprintf('\n==> Process INRIA test data...\n ')
            fetch_filenames_dataset_to_folder('inriaTest', [root_path '/data/INRIA/extracted_data/set01'], pth,1)
            fprintf('Done!')
        case {'ethTest', 'eth'}
            fprintf('\n==> Process ETH test data: \n ')
            % add positive samples
            fprintf('\n==>> set00\n ')
            fetch_filenames_dataset_to_folder('eth', [root_path '/data/ETH/extracted_data/'], pth, 1)
            fprintf('Done!')
        case {'tudbrusselsTest', 'tudbrussels'}
            fprintf('\n==> Process Tud-Brussels test data...\n ')
            % add positive samples
            fprintf('\n==>> set00\n ')
            fetch_filenames_dataset_to_folder('tudbrussels', [root_path '/data/Tud-Brussels/extracted_data/'], pth, 1)
            fprintf('Done!')
        otherwise, error('unknown data type: %s',dataset{i});
    end
end

end


function fetch_filenames_dataset_to_folder(dataset_name, dataset_path, savepath, saveType)
%% get all image + annotations filenames from path
fprintf('\nFetching image/annotation filenames... ')
img_filenames = get_files_images_dir({dataset_path});
annotations_filenames = get_files_annotations_dir({dataset_path});
fprintf('Done!.')
%% make symbolic links for all files to the correct folder
if saveType
    if(~exist([savepath 'images/'],'dir')), mkdir([savepath 'images/']); end
    if(~exist([savepath 'annotations/'],'dir')), mkdir([savepath 'annotations/']); end

    make_links_positive_samples(img_filenames, annotations_filenames, dataset_name, savepath)
else
    if(~exist([savepath 'neg/'],'dir')), mkdir([savepath 'neg/']); end
    
    make_links_negative_samples(img_filenames, dataset_name, savepath)
end

end

function make_links_positive_samples(img_filenames, annotations_filenames, dataset_name, savepath)
%% setup progress bar
% Initialize progress bar with optinal parameters:
progressbar = textprogressbar(size(img_filenames,1), 'barlength', 20, ...
                         'updatestep', 50, ...
                         'startmsg', sprintf(' Processing positive data... '),...
                         'endmsg', ' Done!', ...
                         'showbar', true, ...
                         'showremtime', true, ...
                         'showactualnum', true, ...
                         'barsymbol', '+', ...
                         'emptybarsymbol', '-');
                     
%% make symbolic links for all files to the correct folder
for ifile=1:1:size(img_filenames,1)
    
    str = strsplit(img_filenames{ifile}, '/');
    ncells = length(str);
    fname = str{ncells}; %filename 
    set_vid_pathname = [dataset_name '_' str{ncells-3} '_' str{ncells-2}]; 
    new_img_filepath = [set_vid_pathname '_' fname];
    new_annotation_filepath = [set_vid_pathname '_' fname(1,1:end-3) 'txt'];
    
    % make symbolic links
    status = system(sprintf('ln -s %s %s', img_filenames{ifile}, [savepath 'images/' new_img_filepath]));    
    status = system(sprintf('ln -s %s %s', annotations_filenames{ifile}, [savepath 'annotations/' new_annotation_filepath]));
    
    % progress bar update
    progressbar(ifile)
end
%progressbar(size(img_filenames,1))
end

function make_links_negative_samples(img_filenames, dataset_name, savepath)
%% setup progress bar
% Initialize progress bar with optinal parameters:
progressbar = textprogressbar(size(img_filenames,1), 'barlength', 20, ...
                         'updatestep', 50, ...
                         'startmsg', sprintf(' Processing negative data: '),...
                         'endmsg', ' Done!', ...
                         'showbar', true, ...
                         'showremtime', true, ...
                         'showactualnum', true, ...
                         'barsymbol', '+', ...
                         'emptybarsymbol', '-');
                     
%% make symbolic links for all files to the correct folder
for ifile=1:1:size(img_filenames,1)

    str = strsplit(img_filenames{ifile}, '/');
    ncells = length(str);
    fname = str{ncells}; %filename 
    set_vid_pathname = [dataset_name '_' str{ncells-3} '_' str{ncells-2}]; 
    new_img_filepath = [set_vid_pathname '_' fname];

    % make symbolic links    
    status = system(sprintf('ln -s %s %s', img_filenames{ifile}, [savepath 'neg/' new_img_filepath]));
    
    % progress bar update
    progressbar(ifile)
end
%progressbar(size(img_filenames,1))
end


function [filenames] = get_files_images_dir(path)
%% get all images
filenames = {};
for ipath = 1:1:size(path,1)
    fileList = getAllFiles(path{ipath});
    
    % Delete any entry that is not in JPG or PNG format
    for ifile = 1:1:size(fileList,1)            
        if isempty(strfind(fileList{ifile}, '.jpg')) && isempty(strfind(fileList{ifile}, '.png')) && isempty(strfind(fileList{ifile}, '.JPEG'))
            fileList{ifile,1} = [];
        end
    end
    
    % remove empty filenames
    fileList = fileList(~cellfun('isempty',fileList));
    
    % add filenames to the full list
    filenames = vertcat(filenames,fileList);
end

% remove empty filenames
filenames = filenames(~cellfun('isempty',filenames));
end

function [filenames] = get_files_annotations_dir(path)
%% get all images
filenames = {};
for ipath = 1:1:size(path,1)
    fileList = getAllFiles(path{ipath});
    
    % Delete any entry that is not in JPG or PNG format
    for ifile = 1:1:size(fileList,1)            
        if isempty(strfind(fileList{ifile}, '.txt'))
            fileList{ifile,1} = [];
        end
    end
    
    % remove empty filenames
    fileList = fileList(~cellfun('isempty',fileList));
    
    % add filenames to the full list
    filenames = vertcat(filenames,fileList);
end

% remove empty filenames
filenames = filenames(~cellfun('isempty',filenames));
end