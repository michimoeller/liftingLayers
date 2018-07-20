function X = lift(x,minVal,maxVal,dim)

X = zeros(1,dim,1,size(x,4));
label = linspace(minVal,maxVal,dim);
for i=1:size(x,4)
    for j=1:dim-1
        if (x(1,1,1,i)>label(j))&&(x(1,1,1,i)<=label(j+1))
            X(1,j,1,i)=(label(j+1)-x(1,1,1,i))/(label(j+1)-label(j));
            X(1,j+1,1,i)=1-X(1,j,1,i);
            X(1,j,1,i)=X(1,j,1,i)*label(j);
            X(1,j+1,1,i)=X(1,j+1,1,i)*label(j+1);
        end
    end
end

