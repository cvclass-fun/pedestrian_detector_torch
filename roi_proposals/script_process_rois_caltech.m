function script_process_rois_caltech(data_dir, save_dir, alg_name)
%% process ROI boxes for the Caltech dataset

fprintf('\n************************************************************')
fprintf('\n**** Start Caltech %s roi detection/extraction script. ****', upper(alg_name))
fprintf('\n************************************************************')

%% inits
dataset_name = 'caltech';
skip_step = [30,3,1];   % skip image step
cascThr = -1;           % acf opts
cascCal = [0.025, 0.1]; % acf opts

path_train = {fullfile(data_dir, 'extracted_data', 'set00/');
              fullfile(data_dir, 'extracted_data', 'set01/');
              fullfile(data_dir, 'extracted_data', 'set02/');
              fullfile(data_dir, 'extracted_data', 'set03/');
              fullfile(data_dir, 'extracted_data', 'set04/');
              fullfile(data_dir, 'extracted_data', 'set05/');
             };

path_test  = {fullfile(data_dir, 'extracted_data', 'set06/');
              fullfile(data_dir, 'extracted_data', 'set07/');
              fullfile(data_dir, 'extracted_data', 'set08/');
              fullfile(data_dir, 'extracted_data', 'set09/');
              fullfile(data_dir, 'extracted_data', 'set10/');
             };

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
fprintf('\nCaltech %s boxes processing script completed.', upper(alg_name))
fprintf('\n----------------------------------------------\n')
end
