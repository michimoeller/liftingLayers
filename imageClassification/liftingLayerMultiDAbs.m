classdef liftingLayerMultiDAbs < nnet.layer.Layer

    properties
        % (Optional) Layer properties
        labels

        % Layer properties go here
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters
        
        % Layer learnable parameters go here
    end
    
    methods
        function layer = liftingLayerMultiDAbs(minLift,maxLift,liftingDim,name)
            % (Optional) Create a myLayer
            % This function must have the same name as the layer
            if nargin == 4
                layer.Name = name;
            else
                layer.Name = 'liftingLayer';
            end
            % Layer constructor function goes here
            layer.labels=(linspace(minLift,maxLift,liftingDim));%gpuArray(single
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result
            %
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            
            % Layer forward function for prediction goes here
            
            Z = 0*repmat(X,[1, 1, length(layer.labels), 1]);
            
            chan = size(X,3);
            for i=1:length(layer.labels)-1
                L = layer.labels(i);
                U = layer.labels(i+1);
                temp = (X-L)./(U-L);  
                if length(layer.labels)==2
                    logi = true(size(temp));
                else
                    if i==1
                        logi = (temp<=1);
                    elseif i==length(layer.labels)-1
                         logi = (temp>0);
                    else
                         logi = (temp>0)&(temp<=1);
                    end
                end
                Z(:,:,1+chan*i:chan*(i+1),:) = Z(:,:,1+chan*i:chan*(i+1),:)+(logi).*temp*abs(U);
                temp = 1-temp;
                Z(:,:,1+chan*(i-1):chan*i,:) = Z(:,:,1+chan*(i-1):chan*i,:)+(logi).*temp*abs(L);
            end
        end


        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Output of layer forward function            
            %         dLdZ              - Gradient propagated from the deeper layer
            %         memory            - Memory value which can be used in
            %                             backward propagation
            % Output:
            %         dLdX              - Derivative of the loss with respect to the
            %                             input data
            %         dLdW1, ..., dLdWn - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            % Layer backward function goes here
            chan = size(X,3);
            dLdX = 0*X;
            for i=1:length(layer.labels)-1
                L = layer.labels(i);
                U = layer.labels(i+1);
                temp = (X-L)./(U-L);  
                if length(layer.labels)==2
                    logi = true(size(temp));
                else
                    if i==1
                        logi = (temp<=1);
                    elseif i==length(layer.labels)-1
                         logi = (temp>0);
                    else
                         logi = (temp>0)&(temp<=1);
                    end
                end
                dLdX = dLdX+(logi).* (dLdZ(:,:,1+chan*i:chan*(i+1),:)*abs(U)-dLdZ(:,:,1+chan*(i-1):chan*i,:)*abs(L))./(U-L);
            end
        end
    end
end