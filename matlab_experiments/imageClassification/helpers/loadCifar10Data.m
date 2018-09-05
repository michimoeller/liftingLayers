function [trainingImages,trainingLabels,validationImages,validationLabels] = loadCifar10Data()
if exist('../data/cifar10Data.mat','file')
    load('../data/cifar10Data.mat');
else
    datadir = '../data'; 
    if ~exist('../data/', 'dir')
        mkdir('../data');
    end
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
    helperCIFAR10Data.download(url,datadir);
    [trainingImages,trainingLabels,validationImages,validationLabels] = helperCIFAR10Data.load(datadir);
    save('../data/cifar10Data.mat', 'trainingImages','trainingLabels','validationImages','validationLabels');
end