%% Load data
load('./data/cifar10Data.mat');
[ny,nx,nz] = size(testImages(:,:,:,1));
%% Train the network. 
options = trainingOptions('sgdm','MaxEpochs',50,...
	'InitialLearnRate',0.005,...
    'shuffle', 'every-epoch',...
    'ValidationFrequency', 250,...
    'ValidationPatience', Inf,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.7,...
    'LearnRateDropPeriod',5,...
    'ValidationData',{testImages,testLabels});  

channels = [32,64,1024];
outDim = 10;


%% original case L=3 reduce to 2
layers = ourArchitecture(ny,nx,nz,outDim);
[ourNetOrig, ourInfoOrig] = trainNetwork(trainingImages,trainingLabels,layers,options);

%% case L=3
L=3
fac = 0.82;
L3channels = round(fac*channels);
totalParamsL3 = totalNumberOfParams(L,cat(2,L3channels,outDim));
layers = ourArchitectureVariableL2(ny,nx,nz,outDim,L,L3channels);
[ourNet3, ourInfo3] = trainNetwork(trainingImages,trainingLabels,layers,options);

%% case L=4
L=4;
fac = 0.71;
L4channels = round(fac*channels);
totalParamsL4 = totalNumberOfParams(L,cat(2,L4channels,outDim));
layers = ourArchitectureVariableL2(ny,nx,nz,outDim,L,L4channels);
[ourNet4, ourInfo4] = trainNetwork(trainingImages,trainingLabels,layers,options);
%% case L=5
L=5;
fac = 0.63;
L5channels = round(fac*channels);
totalParamsL5 = totalNumberOfParams(L,cat(2,L5channels,outDim));
layers = ourArchitectureVariableL2(ny,nx,nz,outDim,L,L5channels);
[ourNet5, ourInfo5] = trainNetwork(trainingImages,trainingLabels,layers,options);

%% case L=6
L=6;
fac = 0.575;
L6channels = round(fac*channels);
totalParamsL6 = totalNumberOfParams(L,cat(2,L6channels,outDim));
layers = ourArchitectureVariableL2(ny,nx,nz,outDim,L,L6channels);
[ourNet6, ourInfo6] = trainNetwork(trainingImages,trainingLabels,layers,options);

%% case L=7
L=7;
fac = 0.53;
L7channels = round(fac*channels);
totalParamsL7 = totalNumberOfParams(L,cat(2,L7channels,outDim));
layers = ourArchitectureVariableL2(ny,nx,nz,outDim,L,L7channels);
[ourNet7, ourInfo7] = trainNetwork(trainingImages,trainingLabels,layers,options);

%% case L=10
L=10;
fac = 0.445;
L10channels = round(fac*channels);
totalParamsL10 = totalNumberOfParams(L,cat(2,L10channels,outDim));
layers = ourArchitectureVariableL2(ny,nx,nz,outDim,L,L10channels);
[ourNet10, ourInfo10] = trainNetwork(trainingImages,trainingLabels,layers,options);

%% save results
save('cifar10_results_Lcomparison_differentIntervals',...
    'ourNetOrig','ourInfoOrig',...
    'ourNet3', 'ourInfo3',...
    'ourNet4', 'ourInfo4',...
    'ourNet5', 'ourInfo5',...
    'ourNet6', 'ourInfo6',...
    'ourNet7', 'ourInfo7',...
    'ourNet10', 'ourInfo10',...
    'options');
