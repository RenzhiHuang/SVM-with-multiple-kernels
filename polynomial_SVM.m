% This function trains a polynomial classifier and returns the alpha and bias.
function [a,b] = polynomial_SVM(traindata,labeldata,C,q)
m = size(labeldata,1);
H = labeldata.* ((traindata*traindata'+ones(m,m)).^q).*labeldata';
f = -ones(m,1);
option = optimoptions('quadprog',...
    'MaxIterations',10^5);
a = quadprog(H,f,[],[],labeldata',0,zeros(m,1),C*ones(m,1),[],option);
SV = find(a>10^-7*(5+log10(C)));
SV1 = find(10^-7*(5+log10(C))<a<0.99999*C);
b = 0;
for i=1:size(SV1,1)
    temp = 0;
    for j=1:size(SV,1)
        temp = temp + a(SV(j))*labeldata(SV(j))*...
            (traindata(SV1(i),:)*(traindata(SV(j),:))'+1)^q;
    end
    b = b + labeldata(SV1(i))-temp;
end
b = b / size(SV1,1);    
end