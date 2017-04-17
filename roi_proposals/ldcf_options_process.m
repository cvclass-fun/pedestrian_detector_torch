function [ model ] = ldcf_options_process(dataset_name)
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
        opts.name='../data/acf/models/LdcfCaltech';
        if ~exists([opts.name 'Detector.mat'], 'file'),
            copyfile('toolbox/detector/models/LdcfCaltechDetector.mat', [opts.name 'Detector.mat']);
        end
    case {'inria', 'eth', 'tudbrussels'}
        opts.name='../data/acf/models/LdcfInria';
        if ~exists([opts.name 'Detector.mat'], 'file'),
            copyfile('toolbox/detector/models/LdcfInriaDetector.mat', [opts.name 'Detector.mat']);
        end
    otherwise,  error('unknown data type: %s', dataset);
end

%% load detector model
nm=[opts.name 'Detector.mat'];
t=load(nm); 
model=t.detector;
end

