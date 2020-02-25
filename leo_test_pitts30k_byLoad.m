%% clc;
clear all;
setup;
addpath(genpath('/home/leo/docker_ws/netvlad/netvlad-original'));
%%
netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';
% netID= 'caffe_tokyoTM_conv5_vlad_preL2_intra_white';


paths= localPaths();

load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );

%%
net= relja_simplenn_tidy(net); % potentially upgrate the network to the latest version of NetVLAD / MatConvNet

%%
%dbTest= dbTokyoTimeMachine('val');
dbTest= dbPitts('30k','test');

dbFeatFn = sprintf('output/dbPitt30k_GSVFns_grey_vt_3_1.mat') ; %load features
qFeatFn = sprintf('output/qPitt30k_GSVFns_grey_vt_3_1.mat')  ; %load features 

%dbFeatFn = sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name)  % just to create the files in the out folder
%qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name)    % just to create the files in the out folder

%To create new output/*bin files on the datasets
%serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 1); % adjust batchSize depending on your GPU / network size
%serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);

%Test the features by loading the bin files
%[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
[recall, ~, ~, opts]= testFromFn_leo(dbTest, dbFeatFn, qFeatFn);
plot(opts.recallNs, recall, 'ro-'); grid on; xlabel('N'); ylabel('Recall@N'); title('Tokyo247 HYBRID Edge Image', 'Interpreter', 'none');

