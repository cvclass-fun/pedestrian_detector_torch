function script_process_rois_all(data_dir, save_dir)
%% initializations
algs = {'acf', 'ldcf', 'edgeboxes'};

%% Process rois
for i=1:1:size(algs,2)
    algorithm = algs{i};

    %% caltech
    script_process_rois_caltech(data_dir, save_dir, algorithm);

    %% inria
    script_process_rois_inria(data_dir, save_dir, algorithm);

    %% eth
    script_process_rois_eth(data_dir, save_dir, algorithm);

    %% tudbrussels
    script_process_rois_tudbrussels(data_dir, save_dir, algorithm);
end

end
