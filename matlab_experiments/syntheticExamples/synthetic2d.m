%% generate training data
%rng(42); %<- This seed generates the paper results, other seeds are often 
          %   similar or worse for the standard architectures
n=1000;
x1(1,1,1,1:n) = 2*pi*rand(n,1)';
x2(1,1,1,1:n) = 2*pi*rand(n,1)';
fun = @(x1,x2)cos(x2.*sin(x1));
y = fun(squeeze(x1),squeeze(x2));

dim = 20;
X1 = lift(x1,0,2*pi,dim);
X2 = lift(x2,0,2*pi,dim);

[x1_2d,x2_2d]=meshgrid(0:0.1:2*pi, 0:0.1:2*pi);
Z = fun(x1_2d,x2_2d);
figure, surf(x1_2d,x2_2d,Z);
title('Original function')
set(gca, 'fontsize', 16);
axis tight
view(170,40);
%% network architectures
layerslayersStandard = [ ...
    imageInputLayer([2 1 1])
    fullyConnectedLayer(40)
    reluLayer
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

layersLifting = [ ...
    imageInputLayer([2 dim 1])
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];
%% training + visualization after different number of epochs
trainingEpochs = [25,100,2000];
for i=1:length(trainingEpochs)
    nr = trainingEpochs(i);
    options = trainingOptions('sgdm','InitialLearnRate',0.1, ...
        'MaxEpochs',nr, 'ExecutionEnvironment','cpu');

    netStandard = trainNetwork(cat(1,x1,x2),y,layerslayersStandard,options);
    netLifting = trainNetwork(cat(1,X1,X2),y,layersLifting,options);

    % visualize learned function 
    XNet = reshape([x1_2d(:),x2_2d(:)]', [2,1,1,numel(Z)]);
    Z2 =  predict(netStandard,XNet);
    Z2 = reshape(Z2,size(Z));
    errorStandard(i) = sqrt(sum((Z2(:)-Z(:)).^2)/numel(Z(:)));
    
    figure, surf(x1_2d,x2_2d,Z2), hold on, plot3(squeeze(x1),squeeze(x2),y, 'rx');
    title(['Std-Net, ', num2str(nr),' epochs, RMSE ', num2str(errorStandard(i),3)])
    set(gca, 'fontsize', 16);
    axis tight
    view(170,40);
    drawnow;
    
    X1n = lift(XNet(1,:,:,:),0,2*pi,dim);
    X2n = lift(XNet(2,:,:,:),0,2*pi,dim);
    Z2 =  predict(netLifting,cat(1,X1n,X2n));
    Z2 = reshape(Z2,size(Z));
    errorOurs(i) = sqrt(sum((Z2(:)-Z(:)).^2)/numel(Z(:)));
    
    figure, surf(x1_2d,x2_2d,Z2), hold on, plot3(squeeze(x1),squeeze(x2),y, 'rx');
    title(['Lift-Net, ', num2str(nr),' epochs, RMSE ', num2str(errorOurs(i),3)])
    set(gca, 'fontsize', 16);
    axis tight
    view(170,40);
    drawnow;
end