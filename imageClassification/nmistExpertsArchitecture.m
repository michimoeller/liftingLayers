function layers = nmistExpertsArchitecture(ny,nx,nz,outNr)
if~exist('outNr','var')
    outNr = 10;
end
layers = [imageInputLayer([ny,nx,nz]);
          convolution2dLayer(5,32, 'padding', 'same');
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          convolution2dLayer(5,64, 'padding', 'same');
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(1024);
          dropoutLayer(0.4);
          reluLayer();
          fullyConnectedLayer(outNr);
          softmaxLayer();
          classificationLayer()];