clc;
clear all;


unzip('MerchData.zip');


imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');


numImagesTrain = numel(imdsTrain.Labels);
idx = randperm(numImagesTrain,16);

for i = 1:16
    I{i} = readimage(imdsTrain,idx(i));
end

figure
imshow(imtile(I))
protofile = '/home/leo/docker_ws/lm-distri-descriptor/matlab-DLfeature_PlaceRecog_icra2017/HybridNet/deploy.prototxt';
datafile = '/home/leo/docker_ws/lm-distri-descriptor/matlab-DLfeature_PlaceRecog_icra2017/HybridNet/HybridNet.caffemodel';
% Import network
net = importCaffeNetwork(protofile,datafile) ;
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;


mdl = fitcecoc(featuresTrain,YTrain);


YPred = predict(mdl,featuresTest);

idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    
    imshow(I)
    title(label)
end


im = imread("football.jpg");
protofile = '/home/leo/docker_ws/lm-distri-descriptor/matlab-DLfeature_PlaceRecog_icra2017/HybridNet/deploy.prototxt';
datafile = '/home/leo/docker_ws/lm-distri-descriptor/matlab-DLfeature_PlaceRecog_icra2017/HybridNet/HybridNet.caffemodel';
% Import network
net = importCaffeNetwork(protofile,datafile) ;

%augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);

layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
