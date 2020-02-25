clc;
clear all;
setup;
netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white';
paths= localPaths();
load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
net= relja_simplenn_tidy(net);

im= vl_imreadjpeg({which('football.jpg')}); 
im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
feats_netvlad= computeRepresentation(net, im); % add `'useGPU', false` if you want to use the CPU
%feats_HybridNet= computeRepresentation(net_HybridNet, im); % add `'useGPU', false` if you want to use the CPU
protofile = '/home/leo/docker_ws/lm-distri-descriptor/matlab-DLfeature_PlaceRecog_icra2017/HybridNet/deploy.prototxt';
datafile = '/home/leo/docker_ws/lm-distri-descriptor/matlab-DLfeature_PlaceRecog_icra2017/HybridNet/HybridNet.caffemodel';
% Import network
net_HybridNet = importCaffeNetwork(protofile,datafile) ;

