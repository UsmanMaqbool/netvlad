function paths= localPaths()
    
    % --- dependencies
    
    % refer to README.md for the information on dependencies
    paths.libReljaMatlab= 'depends/relja_matlab/';
    paths.libMatConvNet= '/cluster/home/mbhutta/docker_ws/matconvnet/'; % should contain matlab/
    
    % If you have installed yael_matlab (**highly recommended for speed**),
    % provide the path below. Otherwise, provide the path as 'yael_dummy/':
    % this folder contains my substitutes for the used yael functions,
    % which are **much slower**, and only included for demonstration purposes
    % so do consider installing yael_matlab, or make your own faster
    % version (especially of the yael_nn function)
    paths.libYaelMatlab= 'yael_dummy/';
    
    % --- dataset specifications
    
    paths.dsetSpecDir= 'datasets-specs/';
    
    % --- dataset locations
    paths.dsetRootPitts= '/home/leo/docker_ws/datasets/Pittsburgh-all/Pittsburgh/'; % should contain images/ and queries/
    paths.dsetRootTokyo247= '/cluster/scratch/mbhutta/Test_247_Tokyo_GSV/'; % should contain images/ and query/
    %paths.dsetRootTokyoTM= '/home/leo/docker_ws/datasets/tokyoTimeMachine/'; % should contain images/
    paths.dsetRootTokyoTM= '/home/leo/docker_ws/datasets/tinyTimeMachine/'; % should contain images/
%   paths.dsetRootOxford= '/mnt/0287D1936157598A/docker_ws/datasets/NetvLad/OxfordBuildings/'; % should contain images/ and groundtruth/, and be writable
    paths.dsetRootParis= '/mnt/0287D1936157598A/docker_ws/datasets/NetvLad//Paris/'; % should contain images/ (with subfolders defense, eiffel, etc), groundtruth/ and corrupt.txt, and be writable
    paths.dsetRootHolidays= '/mnt/0287D1936157598A/docker_ws/datasets/NetvLad/Holidays/'; % should contain jpg/ for the original holidays, or jpg_rotated/ for rotated Holidays, and be writable
    
    % --- our networks
    % models used in our paper, download them from our research page
    % paths.ourCNNs= '~/Data/models/';
    paths.ourCNNs= '/cluster/home/mbhutta/docker_ws/netvlad/models/';
    % --- pretrained networks
    % off-the-shelf networks trained on other tasks, available from the MatConvNet
    % website: http://www.vlfeat.org/matconvnet/pretrained/
    paths.pretrainedCNNs= '/home/leo/docker_ws/netvlad/netvlad-original/pretrained/';
    
    % --- initialization data (off-the-shelf descriptors, clusters)
    % Not necessary: these can be computed automatically, but it is recommended
    % in order to use the same initialization as we used in our work
    paths.initData= '/home/leo/docker_ws/netvlad/netvlad-original/initdata/';
    
    % --- output directory
    paths.outPrefix= '/home/leo/docker_ws/netvlad/netvlad-original/output/';
end
