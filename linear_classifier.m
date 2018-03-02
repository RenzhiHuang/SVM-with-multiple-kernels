% This function gives the predicted label and error with a linear
% classifier.
function [error,y_hat] = linear_classifier(data,label,w,b)
data = [data ones(size(data,1),1)];
y_hat = data *[w b]';
y_hat(y_hat >0) = 1;
y_hat(y_hat ==0) = 0;
y_hat(y_hat<0) = -1;
acc = label.* y_hat;
correct = acc > 0;
error = 1-(1/size(label,1)) * sum(correct);