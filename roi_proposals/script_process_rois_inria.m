function script_process_rois_inria(data_dir, save_dir, alg_name)
%% process ROI boxes for the inria dataset

fprintf('\n************************************************************')
fprintf('\n**** Start INRIA %s roi detection/extraction script. ****', upper(alg_name))
fprintf('\n************************************************************')

%% inits
dataset_name = 'inria';
skip_step = [1];   % skip image step
cascThr = -1;           % acf opts
cascCal = [0.01, 0.1]; % acf opts

path_train = {fullfile(data_dir, 'extracted_data', 'set00/')};

path_test  = {fullfile(data_dir, 'extracted_data', 'set01/')};

%% process roi data
switch alg_name
    case 'acf'
        for istep=1:1:size(skip_step, 2)
        for ical=1:1:size(cascCal,2)
            fprintf('\n %s roi proposals settings: step=%d, threshold=%d, calibration=%0.3f', ...
                    alg_name, skip_step(istep), cascThr, cascCal(ical))

            process_rois(alg_name, dataset_name, 'train', path_train, save_dir, skip_step(istep), cascThr, cascCal(ical));
            process_rois(alg_name, dataset_name, 'test', path_test, save_dir, skip_step(istep), cascThr, cascCal(ical));
        end
        end
    case 'ldcf'
        for istep=1:1:size(skip_step, 2)
            fprintf('\n %s roi proposals settings: step=%d', alg_name, skip_step(istep))
            process_rois(alg_name, dataset_name, 'train', path_train, save_dir, skip_step(istep));
            process_rois(alg_name, dataset_name, 'test', path_test, save_dir, skip_step(istep));
        end
    case 'edgeboxes'
        for istep=1:1:size(skip_step, 2)
            fprintf('\n %s roi proposals settings: step=%d', alg_name, skip_step(istep))        
            process_rois(alg_name, dataset_name, 'train', path_train, save_dir, skip_step(istep));
            process_rois(alg_name, dataset_name, 'test', path_test, save_dir, skip_step(istep));
        end
    otherwise, error('Invalid algorithm: %s', alg_name);
end

%% script complete
fprintf('\n----------------------------------------------')
fprintf('\nINRIA %s boxes processing script completed.', upper(alg_name))
fprintf('\n----------------------------------------------\n')
end
