% This function gives the predicted label and error with a RBF
% classifier.
function [error,y_hat] = ...
    RBF_classifier(traindata,trainlabel,testdata,testlabel,a,b,r)
m_test = size(testdata,1);
m_train = size(traindata,1);
for i=1:m_test
    h(i)=0;
    for j=1:m_train
    h(i)=h(i)+a(j)*trainlabel(j)*...
            exp(-r*((testdata(i,:)-traindata(j,:))*...
            (testdata(i,:)-traindata(j,:))'));
    end
    h(i)=h(i)+b;
end
h(h>0)=1;
h(h==0)=0;
h(h<0)=-1;
y_hat = h;
acc = y_hat'.*testlabel;
correct = acc > 0;
error = 1-(1/size(testlabel,1)) * sum(correct);
end