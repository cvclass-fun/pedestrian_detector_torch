function script_train_acf_detector_all( )
% Trains the ACF detector for all datasets

%% Train caltech detector
script_train_acf_detector_Caltech()

%% Train inria detector
script_train_acf_detector_INRIA()

%% Train eth detector
script_train_acf_detector_ETH()

%% Train tudbrussels detector
script_train_acf_detector_TudBrussels()

end