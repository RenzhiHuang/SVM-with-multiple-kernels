% This code process data with linear kernel 
d = 2
%get the train data and train label
train_data = train(:,1:d);
train_label = train(:,d+1);
test_data = test(:,1:d);
test_label = test(:,d+1);

%model parameter
C = [10^-4,10^-3,10^-2,10^-1,1,10,10^2];
for i =1:7
    %% train error and test error
    [w,b] = linear_SVM(train_data,train_label,C(i));
    train_error(i) = linear_classifier(train_data,train_label,w,b);
    test_error(i) = linear_classifier(test_data,test_label,w,b);
    %% cross validation
    sum_train_error = 0;
    sum_test_error = 0;
    for fold=1:5
        cvtrain_data = cv_sub_train{fold}(:,1:d);
        cvtrain_label = cv_sub_train{fold}(:,d+1);
        cvtest_data = cv_sub_test{fold}(:,1:d);
        cvtest_label = cv_sub_test{fold}(:,d+1);
        [w,b] = linear_SVM(cvtrain_data,cvtrain_label,C(i));
        %error analysis
        sum_train_error = sum_train_error+linear_classifier(cvtrain_data,cvtrain_label,w,b);
        sum_test_error = sum_test_error+linear_classifier(cvtest_data,cvtest_label,w,b);
    end
    cv_train_error(i) = sum_train_error/5;
    cv_test_error(i) = sum_test_error/5;
end


