%% validation loss

temp = mnistTypeInfo.ValidationLoss;
x = 1:length(temp);
logi = isnan(temp);
x(logi)=[];
temp(logi)=[];
figure, semilogy(x,temp, 'linewidth',4)
temp = smallInfo.ValidationLoss;
temp(isnan(temp))=[];
hold on, semilogy(x,temp, 'linewidth',4)
temp = bigInfo.ValidationLoss;
temp(isnan(temp))=[];
hold on, semilogy(x,temp, 'linewidth',4)
temp = ourInfo.ValidationLoss;
temp(isnan(temp))=[];
hold on, semilogy(x,temp, 'linewidth',4)
leg = legend('ME-model', 'ME-model+BN', 'Large ME-model+BN', 'Proposed')
xlabel('Iteration')
ylabel('Test Loss')
set(gca, 'fontsize', 16);

%% TrainingLoss
temp = imfilter(mnistTypeInfo.TrainingLoss, ones(1,151),'replicate');
x = 1:length(temp);
logi = isnan(temp);
x(logi)=[];
temp(logi)=[];
figure, semilogy(x,temp, 'linewidth',4)
temp = imfilter(smallInfo.TrainingLoss, ones(1,151),'replicate');;
temp(isnan(temp))=[];
hold on, semilogy(x,temp, 'linewidth',4)
temp = imfilter(bigInfo.TrainingLoss, ones(1,151),'replicate');;
temp(isnan(temp))=[];
hold on, semilogy(x,temp, 'linewidth',4)
temp = imfilter(ourInfo.TrainingLoss, ones(1,151),'replicate');
temp(isnan(temp))=[];
hold on, semilogy(x,temp, 'linewidth',4)
leg = legend('ME-model', 'ME-model+BN', 'Large ME-model+BN', 'Proposed')
xlabel('Iteration')
ylabel('Minibatch Training Loss')
set(gca, 'fontsize', 16);

%% validation accuracy
temp = 100-mnistTypeInfo.ValidationAccuracy;
x = 1:length(temp);
logi = isnan(temp);
x(logi)=[];
temp(logi)=[];
figure, semilogy(x,temp, 'linewidth',4)
temp = 100-smallInfo.ValidationAccuracy;
temp(isnan(temp))=[];
hold on, semilogy(x,temp, 'linewidth',4)
temp = 100-bigInfo.ValidationAccuracy;
temp(isnan(temp))=[];
hold on, semilogy(x,temp, 'linewidth',4)
temp = 100-ourInfo.ValidationAccuracy;
temp(isnan(temp))=[];
hold on, semilogy(x,temp, 'linewidth',4)
leg = legend('ME-model', 'ME-model+BN', 'Large ME-model+BN', 'Proposed')
xlabel('Iteration')
ylabel('Test Error in %')
set(gca, 'fontsize', 16);