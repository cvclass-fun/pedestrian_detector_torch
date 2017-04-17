function [ model ] = acf_options_process(dataset_name)
%% set options
opts=acfTrain();
opts.filters=[5 4];
if ~isempty(dataset_name)
    if ~isempty(dataset_name)
        dataset = dataset_name;
    else
        dataset = 'caltech';
    end
else
    dataset = 'caltech';
end

%% select model name
switch dataset
    case 'caltech'
        opts.name='../data/acf/models/ACF_Caltech_model';
    case 'inria'
        opts.name='../data/acf/models/ACF_INRIA_model';
    case 'eth'
        opts.name='../data/acf/models/ACF_ETH_model';
    case 'tudbrussels'
        opts.name='../data/acf/models/ACF_TudBrussels_model';
    otherwise,  error('unknown data type: %s', dataset);
end

%% load detector model
nm=[opts.name 'Detector.mat'];
t=load(nm); 
model=t.detector;
end

