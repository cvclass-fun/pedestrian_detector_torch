function script_process_rois_all(data_dir, save_dir)
%% Process rois 

%% caltech
script_process_rois_caltech(data_dir, save_dir);

%% inria
script_process_rois_inria(data_dir, save_dir);

%% eth
script_process_rois_eth(data_dir, save_dir);

%% tudbrussels
script_process_rois_tudbrussels(data_dir, save_dir);
end
