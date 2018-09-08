% This is helper class to download and import the CIFAR-100 dataset. The
% dataset is downloaded from:
%
%  https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz
%
% References
% ----------
% Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of
% features from tiny images." (2009).

classdef helperCIFAR100Data
    
    methods(Static)
        
        %------------------------------------------------------------------
        function download(url, destination)
            if nargin == 1
                url = 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz';
            end        
            
            unpackedData = fullfile(destination, 'cifar-100-matlab');
            if ~exist(unpackedData, 'dir')
                fprintf('Downloading CIFAR-100 dataset...');     
                untar(url, destination); 
                fprintf('done.\n\n');
            end
        end
        
        %------------------------------------------------------------------
        % Return CIFAR-100 Training and Test data.
        function [XTrain, TTrain, XTest, TTest] = load(dataLocation)         
            
            location = fullfile(dataLocation, 'cifar-100-matlab');
            
            load([location, '\test.mat']);
            data = reshape(data,[10000,32,32,3]);
            data = permute(data, [2,3,4,1]);
            XTest = single(rot90(data,3))/255;
            TTest = categorical(fine_labels);
            
            load([location, '\train.mat']);
            data = reshape(data,[50000,32,32,3]);
            data = permute(data, [2,3,4,1]);
            XTrain = single(rot90(data,3))/255;
            TTrain = categorical(fine_labels);
        end
    end
end
