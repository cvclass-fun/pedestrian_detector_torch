function uppercaseFilenamesAlgs( path )
% Convert all filename strings to upper case if some are erroneously named
% from source.

%% search all files
dataPath = {path};   
for ipath = 1:1:size(dataPath,1)
    fileList = getAllFiles(dataPath{ipath});
    
    % Delete any entry that is not in JPG or PNG format
    for ifile = 1:1:size(fileList,1)            
        if isempty(strfind(fileList{ifile}, '.txt')) 
            fileList{ifile,1} = [];
        end
    end
end
    
% remove empty filenames
filenames = fileList(~cellfun('isempty',fileList)); 

%% convert all files to uppercase
fprintf('Converting all algorithms filenames to upper case (dbEval requires all files to be uppercase.)\n');
fprintf('Total files: %d\n', size(filenames,1));
for ifile = 1:size(filenames,1)
    fname_ext = strsplit(filenames{ifile},'/');
    fname = strsplit(fname_ext{end},'.txt');
    fpath = strsplit(filenames{ifile},fname_ext(end));
    fpath = fpath{1};
    
    if ~strcmp(filenames{ifile}, [fpath upper(fname{1}) '.txt']) 
        movefile(filenames{ifile}, [fpath upper(fname{1}) '.txt'])
    end    
end
fprintf('Renaming process complete.\n');

end

