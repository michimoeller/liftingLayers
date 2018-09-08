classdef absLiftingLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties

        % Layer properties go here
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters
        
        % Layer learnable parameters go here
    end
    
    methods
        function layer = absLiftingLayer(name)
            % (Optional) Create a myLayer
            % This function must have the same name as the layer
            if nargin == 1
                layer.Name = name;
            else
                layer.Name = 'absLiftingLayer';
            end
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
            Z = cat(3,max(X,0), max(-X,0));
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
            nr = size(dLdZ,3);
            dLdX = (X>0).*dLdZ(:,:,1:nr/2,:) - (X<0).*dLdZ(:,:,1+nr/2:end,:);
        end
    end
end