%% Caltech
fprintf('\nProcessing Caltech''s ACF boxes proposals...\n');
% skip files step 30x, threshold = -1, calibration = .025 
script_process_rois_acf_Caltech(30, -1, .025) %used for evaluation
% skip files step 30x, threshold = -100
script_process_rois_acf_Caltech(30, -1, .1) %used for training
% skip files step 3x, threshold = -1
% script_process_rois_acf_Caltech(3, -1, 1)
% % skip files step 3x, threshold = -100
% script_process_rois_acf_Caltech(3, -100, 1)
% % skip files step 1x (doesn't skip any file), threshold = -1
% script_process_rois_acf_Caltech(1, -1, 1)
% % skip files step 1x (doesn't skip any file), threshold = -100
% script_process_rois_acf_Caltech(1, -100, 1)
fprintf('\nProcessing Caltech''s ACF boxes proposals complete!\n');

%% INRIA
fprintf('\nProcessing INRIA''s ACF boxes proposals...\n');
% threshold = -1
script_process_rois_acf_INRIA(-1, .01) %used for evaluation
% threshold = -100
script_process_rois_acf_INRIA(-1, .1) %used for training
fprintf('\nProcessing INRIA''s ACF boxes proposals complete!\n');

% %% ETH
% fprintf('\mProcessing ETH''s ACF boxes proposals...\n');
% % threshold = -1
% script_process_rois_acf_ETH(-1)
% % threshold = -100
% script_process_rois_acf_ETH(-100)
% fprintf('\mProcessing ETH''s ACF boxes proposals complete!\n');
% 
% %% Tud-Brussels
% fprintf('\mProcessing Tud-Brussels''s ACF boxes proposals...\n');
% % threshold = -1
% script_process_rois_acf_TudBrussels(-1)
% % threshold = -100
% script_process_rois_acf_TudBrussels(-100)
% fprintf('\mProcessing Tud-Brussels''s ACF boxes proposals complete!\n');
