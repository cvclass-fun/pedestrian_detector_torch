function save_boxes( boxes, savepath)
%% initializations
max_files_per_var = 1e3; % maximum files stored per field

%% check if the number of files is smaller than the threshold
if size(boxes,2) <= max_files_per_var
    fprintf('\nSaving roi boxes to file... ')
    save(savepath, 'boxes')
    fprintf('Done!')
    return
end

%% Initialize progress bar with optinal parameters:
progressbar = textprogressbar(size(boxes,2), 'barlength', 20, ...
                         'updatestep', max_files_per_var, ...
                         'startmsg', 'saving roi boxes to file: ', ...
                         'endmsg', ' Done!', ...
                         'showbar', true, ...
                         'showremtime', true, ...
                         'showactualnum', true, ...
                         'barsymbol', '=', ...
                         'emptybarsymbol', '.');

%% Save boxes to file (separate into several containers if the number of files is too big)
idx = 1;
tmp_boxes = {};
container_counter=1;
nfiles = size(boxes,2);
for ifile=1:1:nfiles
    if rem(ifile, max_files_per_var) == 0
        % save to file
        if container_counter == 1 
            eval(strcat('boxes',num2str(container_counter),'=tmp_boxes; save(savepath, ''boxes',num2str(container_counter),''')'))
        else
            eval(strcat('boxes',num2str(container_counter),'=tmp_boxes; save(savepath, ''boxes',num2str(container_counter),''', ''-append'')'))
        end
        
        
        % increment container counter
        container_counter = container_counter + 1;
        
        % reset index counter
        idx = 1;
        tmp_boxes = {};
    else
        %increment counter
        idx = idx + 1;
    end
    
    % add data
    tmp_boxes{1,idx} = boxes{1,ifile};
    
    %% update progress bar
    progressbar(ifile)
end

if ~isempty(tmp_boxes)
    eval(strcat('boxes',num2str(container_counter),'=tmp_boxes; save(savepath, ''boxes',num2str(container_counter),''', ''-append'')'))
end

end