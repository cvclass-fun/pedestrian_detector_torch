function [root_path] = add_paths_toolboxes()
% Setup/add paths to necessary toolboxes for processing.
clear all;
%% get path
str = strsplit(pwd, '/');
root_path = '';
for i=2:size(str,2)-1
    root_path = strcat(root_path, '/', str{i});
end

%% add paths
addpath(genpath([root_path '/roi_proposals']));
addpath(genpath([root_path '/benchmark']));

end