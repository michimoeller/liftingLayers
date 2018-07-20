function layers = relatedArchitectureBig(ny,nx,nz,outNr)
if~exist('outNr','var')
    outNr = 10;
end
layers = [imageInputLayer([ny,nx,nz]);
          convolution2dLayer(5,32, 'padding', 'same');
          batchNormalizationLayer;
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          convolution2dLayer(5,128, 'padding', 'same');
          batchNormalizationLayer;
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(2048);
          dropoutLayer(0.4);
          batchNormalizationLayer;
          reluLayer();
          fullyConnectedLayer(outNr);
          softmaxLayer();
          classificationLayer()];