%% Caltech
aspect_ratio = 0.6;
fprintf('\nProcessing Caltech''s edge boxes proposals...\n');
% skip files step 30x, extract 10k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_Caltech(30, 1e4, aspect_ratio)
% skip files step 30x, extract 1k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_Caltech(30, 1e3, aspect_ratio)
% skip files step 3x, extract 10k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_Caltech(3, 1e4, aspect_ratio)
% skip files step 3x, extract 1k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_Caltech(3, 1e3, aspect_ratio)
% skip files step 1x (doesn't skip any file), extract 10k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_Caltech(1, 1e4, aspect_ratio)
% skip files step 1x (doesn't skip any file), extract 1k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_Caltech(1, 1e3, aspect_ratio)
fprintf('\nProcessing Caltech''s edge boxes proposals complete!\n');

%% INRIA
fprintf('\nProcessing INRIA''s edge boxes proposals...\n');
% extract 10k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_INRIA(1e4, aspect_ratio)
% extract 1k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_INRIA(1e3, aspect_ratio)
fprintf('\nProcessing INRIA''s edge boxes proposals complete!\n');

%% ETH
fprintf('\nProcessing ETH''s edge boxes proposals...\n');
% extract 10k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_ETH(1e4, aspect_ratio)
% extract 1k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_ETH(1e3, aspect_ratio)
fprintf('\nProcessing ETH''s edge boxes proposals complete!\n');

%% Tud-Brussels
fprintf('\nProcessing Tud-Brussels''s edge boxes proposals...\n');
% extract 10k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_TudBrussels(1e4, aspect_ratio)
% extract 1k boxes, aspect ratioh h/w <= aspect_ratio
script_process_rois_edgeboxes_TudBrussels(1e3, aspect_ratio)
fprintf('\nProcessing Tud-Brussels''s edge boxes proposals complete!\n');