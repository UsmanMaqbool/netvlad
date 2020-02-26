%  Author: Relja Arandjelovic (relja@relja.info)
clear all;
clc;


setup;

% dbTrain= dbTokyoTimeMachine('train');
% dbVal= dbTokyoTimeMachine('val');


dbTrain= dbPitts('30k','train');
dbVal= dbPitts('30k','val');


%%Run the training:

sessionID= trainWeakly(dbTrain, dbVal, ...
    'netID', 'vd16', 'layerName', 'conv5_3', 'backPropToLayer', 'conv5_1', ...
    'method', 'vlad_preL2_intra', ...
    'learningRate', 0.0001, ... 
    'doDraw', true);

% Get the best network
% This can be done even if training is not finished, it will find the best network so far
[~, bestNet]= pickBestNet(sessionID);


% ---------- Tiny dummy training example
%dbTrain= dbTiny('train'); dbVal= dbTiny('val');
% 
% session_caffee = trainWeakly(dbTrain, dbVal, ...
%     'netID', 'caffe', 'layerName', 'conv5', ...
%     'method', 'max', 'backPropToLayer', 'conv5', ...
%     'margin', 0.1, ...
%     'batchSize', 25, 'learningRate', 0.01, 'lrDownFreq', 3, 'momentum', 0.9, 'weightDecay', 0.1, 'compFeatsFrequency', 10, ...
%     'nNegChoice', 30, 'nNegCap', 10, 'nNegCache', 10, ...
%     'nEpoch', 5, ...
%     'epochTestFrequency', 1, 'test0', true, ...
%     'nTestSample', inf, 'nTestRankSample', 40, ...
%     'saveFrequency', 15, 'doDraw', true, ...
%     'useGPU', true, 'numThreads', 12, ...
%     'info', 'tiny test');
%session_vd16 = trainWeakly(dbTrain, dbVal, ...
%     'netID', 'vgg16', 'layerName', 'conv5', ...
%     'method', 'max', 'backPropToLayer', 'conv5', ...
%     'margin', 0.1, ...
%     'batchSize', 25, 'learningRate', 0.01, 'lrDownFreq', 3, 'momentum', 0.9, 'weightDecay', 0.1, 'compFeatsFrequency', 10, ...
%     'nNegChoice', 30, 'nNegCap', 10, 'nNegCache', 10, ...
%     'nEpoch', 5, ...
%     'epochTestFrequency', 1, 'test0', true, ...
%     'nTestSample', inf, 'nTestRankSample', 40, ...
%     'saveFrequency', 15, 'doDraw', true, ...
%     'useGPU', false, 'numThreads', 12, ...
%     'info', 'tiny test');
