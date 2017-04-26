function Run_evaluate(experimentIDini, experimentIDend, algPlotNum, dataNamesID, datasetDir, algorithmsDir, savePlotDir, algorithmsNames, flag_benchmark)
%% Evaluation script
%
% To evaluate the method, just setup the correct paths and run the script.
% Don't forget to add all the folders and subfolders to the env path.
%
% Options:
%   experimentID - Experiment id. The different experiments are designated
%                  by a specific number. The available experiments are:
%                     [1] 'Reasonable',     [50 inf],  [.65 inf], 0,   .5,  1.25
%                     [2] 'All',            [20 inf],  [.2 inf],  0,   .5,  1.25
%                     [3] 'Scale=large',    [100 inf], [inf inf], 0,   .5,  1.25
%                     [4] 'Scale=near',     [80 inf],  [inf inf], 0,   .5,  1.25
%                     [5] 'Scale=medium',   [30 80],   [inf inf], 0,   .5,  1.25
%                     [6] 'Scale=far',      [20 30],   [inf inf], 0,   .5,  1.25
%                     [7] 'Occ=none',       [50 inf],  [inf inf], 0,   .5,  1.25
%                     [8] 'Occ=partial',    [50 inf],  [.65 1],   0,   .5,  1.25
%                     [9] 'Occ=heavy',      [50 inf],  [.2 .65],  0,   .5,  1.25
%                     [10] 'Ar=all',        [50 inf],  [inf inf], 0,   .5,  1.25
%                     [11] 'Ar=typical',    [50 inf],  [inf inf],  .1, .5,  1.25
%                     [12] 'Ar=atypical',   [50 inf],  [inf inf], -.1, .5,  1.25
%                     [13] 'Overlap=25',    [50 inf],  [.65 inf], 0,   .25, 1.25
%                     [14] 'Overlap=50',    [50 inf],  [.65 inf], 0,   .50, 1.25
%                     [15] 'Overlap=75',    [50 inf],  [.65 inf], 0,   .75, 1.25
%                     [16] 'Expand=100',    [50 inf],  [.65 inf], 0,   .5,  1.00
%                     [17] 'Expand=125',    [50 inf],  [.65 inf], 0,   .5,  1.25
%                     [18] 'Expand=150',    [50 inf],  [.65 inf], 0,   .5,  1.50
%   algPlotNum - Total number of algorithms to be plotted.
%   dataNamesID - Select a dataset for testing by its id. The available
%                 datasets are:
%                     [1] 'UsaTest'
%                     [2] 'UsaTrain'
%                     [3] 'InriaTest'
%                     [4] 'TudBrussels'
%                     [5] 'ETH'
%                     [6] 'Daimler'
%                     [7] 'Japan'
%   algorithmsDir - path to the algorithms results. If empty, it will
%                   consider the path is inside the dataset path under the
%                   subfolder 'algorithms'.
%   savePlotDir - Folder path to store plot results.
%
%   algorithmsNames - String containing the names of algorithms to plot.
%                     The names of one or several algortihms can be passed
%                     by separating each algorithm name by a comma ','.
%                     Examples:
%                               1 algorithm: 'Ours'
%                               3 algorithms: 'ACF,LDCF,CCF'
%
%   flag_benchmark - Defines the plot output.
%                    If > 0, plots all each algorithm in their own graph.
%                    If == 0 then plots all agorithms plus the inputed ones into a single graph.
%                    If < 0 then plots only the introduced algorithms
%                    names.
%

%% Add eval code paths
addpath(genpath('./'))

%% Set all filenames to upper case (FastCF files begin with lower case so it needs to be fixed)
uppercaseFilenamesAlgs(algorithmsDir)

%% Evaluate/compare method
for experimentID=experimentIDini:1:experimentIDend
    fprintf('\nStart Experiment nº%d (%d/%d):\n', experimentID, experimentID-experimentIDini+1, size(experimentIDini:1:experimentIDend ,2));
    dbEval(experimentID, algPlotNum, dataNamesID, datasetDir, algorithmsDir, savePlotDir, algorithmsNames, flag_benchmark)
    fprintf('\nEvaluation procedure complete.\n');
end
