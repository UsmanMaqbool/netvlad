clear all;
clc;
%% 
setup;

paths= localPaths();
dbTrain= dbTiny('train'); dbVal= dbTiny('val');
lr= 0.0001;

% --- Train the VGG-16 network + NetVLAD, tuning down to conv5_1
sessionID= trainWeakly(dbTrain, dbVal, ...
    'netID', 'vd16', 'layerName', 'conv5_3', 'backPropToLayer', 'conv5_1', ...
    'method', 'vlad_preL2_intra', ...
    'learningRate', lr, ...
    'doDraw', true);

% Get the best network
% This can be done even if training is not finished, it will find the best network so far
[~, bestNet]= pickBestNet(sessionID);

% Either use the above network as the image representation extractor (do: finalNet= bestNet), or do whitening (recommended):
finalNet= addPCA(bestNet, dbTrain, 'doWhite', true, 'pcaDim', 4096);
