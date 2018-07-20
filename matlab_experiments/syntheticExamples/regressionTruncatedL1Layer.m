classdef regressionTruncatedL1Layer < nnet.layer.RegressionLayer
               
    methods
        function layer = regressionTruncatedL1Layer(name)
            % Create an exampleRegressionMAELayer

            % Set layer name
            if nargin == 1
                layer.Name = name;
            end

            % Set layer description
            layer.Description = 'Example regression layer with truncated l1 loss';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Returns the MAE loss between the predictions Y and the
            % training targets T

            % Calculate MAE
            K = size(Y,3);
            meanAbsoluteError = sum(min(abs(Y-T),0.3),3)/K;
    
            % Take mean over mini-batch
            N = size(Y,4);
            loss = sum(meanAbsoluteError)/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the MAE loss with respect to the predictions Y
            K = size(Y,3);
            N = size(Y,4);
            dLdY = (abs(Y-T)<0.3).*sign(Y-T)/(N*K);
        end
    end
end