function script_process_rois_eth(data_dir, save_dir)
%% process ROI boxes for the ETH dataset

algs = {'ldcf',  'acf', 'edgeboxes'};
dataset_name = 'eth';
skip_step = [1];   % skip image step
cascThr = -1;           % acf opts
cascCal = [0.025, 0.1]; % acf opts

path_train = {fullfile(data_dir, 'extracted_data', 'set00/');
              fullfile(data_dir, 'extracted_data', 'set01/');
              fullfile(data_dir, 'extracted_data', 'set02/');
             };

%% process roi data
for i=1:1:size(algs,2)
    alg_name = algs{i};
    
    display_message(alg_name, 1)
    
    %% Process algporithm
    switch alg_name
        case 'acf'
            for istep=1:1:size(skip_step, 2)
            for ical=1:1:size(cascCal,2)
                fprintf('\n %s roi proposals settings: step=%d, threshold=%d, calibration=%0.3f', ...
                        alg_name, skip_step(istep), cascThr, cascCal(ical))

                process_rois(alg_name, dataset_name, 'train', path_train, save_dir, skip_step(istep), cascThr, cascCal(ical));
            end
            end
        case 'ldcf'
            for istep=1:1:size(skip_step, 2)
                fprintf('\n %s roi proposals settings: step=%d', alg_name, skip_step(istep))
                process_rois(alg_name, dataset_name, 'train', path_train, save_dir, skip_step(istep));
            end
        case 'edgeboxes'
            for istep=1:1:size(skip_step, 2)
                fprintf('\n %s roi proposals settings: step=%d', alg_name, skip_step(istep))        
                process_rois(alg_name, dataset_name, 'train', path_train, save_dir, skip_step(istep));
            end
        otherwise, error('Invalid algorithm: %s', alg_name);
    end
end

%% script complete
display_message(alg_name, 0)
end

function display_message(alg_name, flag)
    if flag,
        fprintf('\n************************************************************')
        fprintf('\n**** Start ETH %s roi detection/extraction script. ****', upper(alg_name))
        fprintf('\n************************************************************')
    else
        fprintf('\n----------------------------------------------')
        fprintf('\nETH %s boxes processing script completed.', upper(alg_name))
        fprintf('\n----------------------------------------------\n')
    end
end