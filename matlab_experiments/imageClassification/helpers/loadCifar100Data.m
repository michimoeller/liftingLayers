function [trainingImages,trainingLabels,validationImages,validationLabels] = loadCifar100Data()
if exist('../data/cifar100Data.mat','file')
    load('../data/cifar100Data.mat');
else
    datadir = '../data'; 
    if ~exist('../data/', 'dir')
        mkdir('../data');
    end
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz';
    helperCIFAR100Data.download(url,datadir);
    [trainingImages,trainingLabels,validationImages,validationLabels] = helperCIFAR100Data.load(datadir);
    save('../data/cifar100Data.mat', 'trainingImages','trainingLabels','validationImages','validationLabels');
end