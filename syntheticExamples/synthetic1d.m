%% generate training data and lift
%rng(42); %<- This seed generates the paper results, other seeds are often 
          %   similar or worse for the standard architectures
n=50;
x(1,1,1,1:n) = 2*pi*rand(n,1)';
y = sin(squeeze(x));

dim = 9;
X = liftUnscaled(x,0,2*pi,dim);
%% network architectures 
layersStandard = [ ...
    imageInputLayer([1 1 1])
    fullyConnectedLayer(9)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

layersLifting = [ ...
    imageInputLayer([1 dim 1], 'Normalization', 'none')
    fullyConnectedLayer(1,'BiasLearnRateFactor',0)
    regressionLayer];

%% training for various number of epochs
trainingEpochs = [25,75,200,4000];
for i=1:length(trainingEpochs)
    nr = trainingEpochs(i);
    options = trainingOptions('sgdm','InitialLearnRate',0.1, ...
    'MaxEpochs',nr,'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.9, ...
            'LearnRateDropPeriod', 200,...
            'ExecutionEnvironment','cpu');

    netStandard = trainNetwork(x,y,layersStandard,options);
    netLifting = trainNetwork(X,y,layersLifting,options);

    % visualize learned function 
    xLin(1,1,1,1:200) = linspace(0,2*pi,200);
    yPred = predict(netStandard,xLin);
    figure, plot(squeeze(xLin),yPred, 'linewidth', 4), 
    hold on, plot(squeeze(x),y,'x', 'linewidth', 2, 'markersize', 12);
    set(gca, 'fontsize', 16);
    grid on;
    title(['Std-Net approximation, ', num2str(nr),' epochs'])
    axis tight
    
    XLin = liftUnscaled(xLin,0,2*pi,dim);
    yPred = predict(netLifting,XLin);
    figure, plot(squeeze(xLin),yPred, 'linewidth', 4),
    hold on, plot(squeeze(x),y,'x', 'linewidth', 2, 'markersize', 12);
    set(gca, 'fontsize', 16);
    grid on;
    title(['Lift-Net approximation, ', num2str(nr),' epochs'])
    axis tight
end