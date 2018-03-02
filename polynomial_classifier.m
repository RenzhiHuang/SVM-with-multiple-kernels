% This function gives the predicted label and error with a polynomial
% classifier.
function [error,y_hat] = ...
    polynomial_classifier(traindata,trainlabel,testdata,testlabel,a,b,q)
h = sum(a.*trainlabel.*...
    (traindata*testdata'+ones(size(traindata,1),size(testdata,1))).^q)+b;
h(h>=0)=1;
h(h ==0) = 0;
h(h<0)=-1;
y_hat = h;
acc = y_hat'.*testlabel;
correct = acc > 0;
error = 1-(1/size(testlabel,1)) * sum(correct);
end