function totalParams = totalNumberOfParams(L,channels)

params(1) = 5^2*channels(1)*3+channels(1);
params(2) = 5^2*(32*L)*channels(2)+channels(2);
params(3) = channels(3)*(8^2*(channels(2)*L)+1);
params(4) = channels(4)*(channels(3)*L+1);
totalParams = sum(params);

