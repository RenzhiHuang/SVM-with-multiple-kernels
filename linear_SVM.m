% This function trains a linear classifier and returns the weights and
% bias.
function [w,b] = linear_SVM(traindata,labeldata,C)
H = labeldata.*(traindata*traindata').*labeldata';
m = size(labeldata,1);
f = -ones(m,1);
a = quadprog(H,f,[],[],labeldata',0,zeros(m,1),C*ones(m,1));
SV = find(a>0.000001*C);
w = a' * (labeldata.*traindata);
temp = labeldata-traindata * w';
b=1 / size(SV,1) * sum(temp(SV));
end