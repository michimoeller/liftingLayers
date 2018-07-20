%% clean up and set a seed
clear all;
close all;
rng(15)   %<- This seed generates the paper results, different seeds often 
          %   differ, but the general tendency remains the same. Larger
          %   values of sample points n make the lifting (and ell^1) 
          %   results more consistent
%% define loss, generate training data
loss = @(x,y)min(abs(x-y), 0.3); %truncated linear
n=150; %number of training points
x(1:n) = 2*pi*rand(n,1)'; %training points x
y = sin(x); %training points y
per = randperm(n);
fractionOfOutlier = 0.4;
dest = round(n*fractionOfOutlier); %destroy 30% of the values completely...
y(per(1:dest)) = 0.3*randn(dest,1); % .. and set them to random values

%% lift data X, define output lifting dimension 
dim = 21;
dimOut = 51;
[X,labelX] = (liftUnscaled(permute(x, [1 4 3 2]),0,2*pi,dim));
X = squeeze(X);

%% compute loss matrix
v = zeros(dimOut,n);
labelY = linspace(-1.2,1.2,dimOut);
for i=1:n
    v(:,i) = loss(labelY,y(i));
end
lossMat = v*X';

%% Illustrate solution to the minimization problem
figure, imagesc(lossMat == repmat(min(lossMat,[],1), [size(lossMat,1),1]))
title('Columnwise minimum, i.e., optimal $\theta$', 'interpreter', 'latex')
set(gca, 'fontsize', 16);
[~,ind]=min(lossMat,[],1);
lossMat = lossMat./repmat(sum(lossMat,1), [size(lossMat,1),1]);
figure, imagesc(lossMat), title('Normalized cost matrix', 'interpreter', 'latex')
set(gca, 'fontsize', 16)
yPlot = labelY(ind);
figure, plot(labelX, yPlot, 'linewidth', 4)
hold on, plot(x, y,'x', 'linewidth', 1, 'markersize', 13)
grid on
title(['output lifting fit, ', num2str(fractionOfOutlier*100),'$\%$ outliers'], 'interpreter', 'latex')
set(gca, 'fontsize', 16)
axis([0 2*pi -1 1])


%% plain ell^1 minimization using the primal-dual hybrid gradient method
u=zeros(1,dim);
p=0;
maxiter = 5000;
normX = normest(X);
tau = 1/normX;
sigma = 1/normX;
for i=1:maxiter
    uold = u;
    p = max(min(p + sigma*((2*u-uold)*X-y),1),-1);
    u = u - tau*p*X';
end
xLin(1,1,1,1:200) = linspace(0,2*pi,200);
XLin = liftUnscaled(xLin,0,2*pi,dim);
figure, plot(squeeze(xLin),u*squeeze(XLin), 'linewidth', 4)
hold on, plot(squeeze(x), squeeze(y),'x', 'linewidth', 1, 'markersize', 13)
grid on
title(['$\ell^1$-fit lifted to scalar, ', num2str(fractionOfOutlier*100),'$\%$  outliers'], 'interpreter', 'latex')
set(gca, 'fontsize', 16)
axis([0 2*pi -1 1])

%% nonconvex optimization
X = liftUnscaled(permute(x, [1 4 3 2]),0,2*pi,dim);
layers = [ ...
    imageInputLayer([size(X,1), size(X,2),size(X,3)]);
    fullyConnectedLayer(1);
    regressionTruncatedL1Layer
    ];
layers(2).Weights = randn(1,dim);%sin(linspace(0,2*pi, dim));
options = trainingOptions('sgdm','InitialLearnRate',1, ...
        'MaxEpochs',3000,'MiniBatchSize',size(X,4), 'Momentum',0.9,...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',0.995,...
        'LearnRateDropPeriod',2,...
        'ExecutionEnvironment','cpu');

net = trainNetwork(X, permute(y, [1 4 3 2]), layers,options);

xplot = linspace(0,2*pi,1000);
Xplot = liftUnscaled(permute(xplot, [1 4 3 2]),0,2*pi,dim);
yplot = predict(net,Xplot);
figure, plot(xplot, yplot, 'linewidth', 4)
hold on, plot(squeeze(x), squeeze(y),'x', 'linewidth', 1, 'markersize', 13)
grid on
title(['Nonconvex optimization lifted to scalar, ', num2str(fractionOfOutlier*100),'$\%$  outliers'], 'interpreter', 'latex')
set(gca, 'fontsize', 16)
axis tight
