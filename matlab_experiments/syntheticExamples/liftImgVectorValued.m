function [reshaped_X,points] = liftImgVectorValued(x,minVal,maxVal,labelNumber)

d = size(x,3);
if d>3
    error('Only supports vector-valued lifting up to d=3');
end
dimLiftedSpace = labelNumber^d;
label = (linspace(minVal,maxVal,labelNumber));
if d==1
    points = ndgrid(label);
    DT = delaunayTriangulation(points(:));
elseif d==2
    [pointsX,pointsY] = ndgrid(label);
    DT = delaunayTriangulation([pointsX(:),pointsY(:)]);
elseif d==3
    [pointsX,pointsY,pointsZ] = ndgrid(label);
    DT = delaunayTriangulation([pointsX(:),pointsY(:),pointsZ(:)]);
end

reshaped_x = reshape(permute(x, [1 2 4 3]),[size(x,1)*size(x,2)*size(x,4), size(x,3)]);
reshaped_X = zeros(size(x,1)*size(x,2)*size(x,4),dimLiftedSpace, 'single');
for i=1:length(DT.ConnectivityList)
    
    B = cartesianToBarycentric(DT,i*ones(size(reshaped_x,1),1),reshaped_x);
    logi = (max(B,[],2)<=1)&(min(B,[],2)>=0);
    reshaped_X(logi,DT.ConnectivityList(i,:))=B(logi,:);
end
reshaped_X = permute(reshape(reshaped_X, [size(x,1),size(x,2),size(x,4),dimLiftedSpace]), [1 2 4 3]);
points = DT.Points;