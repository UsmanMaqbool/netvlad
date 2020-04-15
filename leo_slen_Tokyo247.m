clc;
clear all;
setup;
system('xinput set-prop 16 "Synaptics Two-Finger Scrolling" 1 0');
system('xinput set-prop 21 "Synaptics Two-Finger Scrolling" 1 0');
addpath(genpath(pwd));

%%
netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white';
% netID= 'caffe_tokyoTM_conv5_vlad_preL2_intra_white';


paths= localPaths();

load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );

%%
net= relja_simplenn_tidy(net); % potentially upgrate the network to the latest version of NetVLAD / MatConvNet

%%
%dbTest= dbTokyoTimeMachine('val');
dbTest= dbTokyo247();

dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);  % just to create the files in the out folder
qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);    % just to create the files in the out folder

%To create new output/*bin files on the datasets
%serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 1); % adjust batchSize depending on your GPU / network size
%serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);

%Test the features by loading the bin files
[recall, ~, ~, opts]= leo_slen_testFromFn(dbTest, dbFeatFn, qFeatFn);
save_results = strcat(paths.outPrefix,'plots/vd16_tokyoTM_conv5_3_vlad_preL2_intra_white_','t10.mat');
recallNs = opts.recallNs;
save(char(save_results), 'recall','recallNs');

x = load(char('/home/leo/docker_ws/datasets/netvlad-original-output/plots/vd16_tokyoTM_conv5_3_vlad_preL2_intra_white_t2.mat'));

ori = load(char('/home/leo/docker_ws/datasets/netvlad-original-output/plots/vd16_tokyoTM_conv5_3_vlad_preL2_intra_white_real.mat'));



plot(opts.recallNs, recall, 'ro-',x.recallNs, x.recall, 'go-',ori.recallNs, ori.recall, 'bo-'); grid on; xlabel('N'); ylabel('Recall@N'); title('Tokyo247 HYBRID Edge Image', 'Interpreter', 'none');

