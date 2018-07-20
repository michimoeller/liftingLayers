%% generate training data
rng(42);
n=1000;
offs = 0.01;
x1(1,1,1,1:n) = 2*pi*rand(n,1)';
x2(1,1,1,1:n) = 2*pi*rand(n,1)';
fun = @(x1,x2)cos(x2.*sin(x1));
y = fun(squeeze(x1),squeeze(x2));

%% visualize ground truth
[X1,X2]=meshgrid(0:0.1:2*pi, 0:0.1:2*pi);
Z = fun(X1,X2);
figure, surf(X1,X2,Z);
title('Ground truth')
set(gca, 'fontsize', 16);
view(170,40);
drawnow;
axis tight;

%% compute lifting and optimal parameters
dim = 11;
[X,~] = liftImgVectorValued(cat(3,x1,x2),0-offs,2*pi+offs,dim);% for training
[XvecLift,points] = liftImgVectorValued(cat(3,X1,X2),0-offs,2*pi+offs,dim);% for testing
test = permute(reshape(XvecLift, [size(XvecLift,1)*size(XvecLift,2),1,size(XvecLift,3)]), [4 2 3 1]);

% 'training', i.e., computing weight matrix A
Xtemp = squeeze(X);
alph = 0*1e-1;%<- weight decay parameter if desired
A = (y'*Xtemp')*pinv(Xtemp*Xtemp' + alph*eye(size(Xtemp,1)));

% testing,i.e., preparing the visualization of the results
Z2 = reshape(A*squeeze(test), size(Z));

%% visualize result
figure, surf(X1,X2,Z2), hold on, plot3(squeeze(x1),squeeze(x2),y, 'rx');
title(['Vector valued lifting, ',num2str(dim),'^2 labels, RMSE ', num2str(sqrt(sum((Z2(:)-Z(:)).^2)/numel(Z(:))),3)])
set(gca, 'fontsize', 16);
view(170,40);
drawnow;
hold on, trisurf(delaunay(points(:,1),points(:,2)),points(:,1),points(:,2),points(:,1)*0-2.5);
axis tight;