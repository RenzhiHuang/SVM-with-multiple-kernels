function [a,b] = RBF_SVM(traindata,labeldata,C,r)
m = size(labeldata,1);
for i=1:m
    for j =1:m
        H(i,j) = labeldata(i)*labeldata(j)*...
            exp(-r*((traindata(i,:)-traindata(j,:))*...
            (traindata(i,:)-traindata(j,:))'));
    end
end
f = -ones(m,1);
a = quadprog(H,f,[],[],labeldata',0,zeros(m,1),C*ones(m,1));
SV = find(a>1e-5);
SV1 = find(1e-5<a<C);
% SV = find(a>10^-7*(5+log10(C)));
% SV1 = find(10^-7*(5+log10(C))<a<C);
b = 0;
for i=1:size(SV1,1)
    temp = 0;
    for j=1:size(SV,1)
        temp = temp + a(SV(j))*labeldata(SV(j))*...
            exp(-r*((traindata(SV1(i),:)-traindata(SV(j),:))*...
            (traindata(SV1(i),:)-traindata(SV(j),:))'));
    end
    b = b + labeldata(SV1(i))-temp;
end
b = b / size(SV1,1);    
end