% This code process data with polynomial kernel 
d = 57
%get the train data and train label
train_data = train(:,1:d);
train_label = train(:,d+1);
test_data = test(:,1:d);
test_label = test(:,d+1);

%model parameter
C = [10^-4,10^-3,10^-2,10^-1,1,10,10^2];
q = [1,2,3,4,5];
for i =1:5
    for j = 1:7
        %% train error and test error
        [a,b] = polynomial_SVM(train_data,train_label,C(j),q(i));
        train_error(i,j) = polynomial_classifier...
            (train_data,train_label,train_data,train_label,a,b,q(i));
        test_error(i,j) = polynomial_classifier...
            (train_data,train_label,test_data,test_label,a,b,q(i));
        %% cross validation
        sum_train_error = 0;
        sum_test_error = 0;
        for fold=1:5
            cvtrain_data = cv_sub_train{fold}(:,1:d);
            cvtrain_label = cv_sub_train{fold}(:,d+1);
            cvtest_data = cv_sub_test{fold}(:,1:d);
            cvtest_label = cv_sub_test{fold}(:,d+1);
            [a,b] = polynomial_SVM(cvtrain_data,cvtrain_label,C(j),q(i));
            %error analysis
            sum_train_error = sum_train_error+...
                polynomial_classifier...
            (cvtrain_data,cvtrain_label,cvtrain_data,cvtrain_label,a,b,q(i));
            sum_test_error = sum_test_error+...
                polynomial_classifier...
            (cvtrain_data,cvtrain_label,cvtest_data,cvtest_label,a,b,q(i));
        end
        cv_train_error(i,j) = sum_train_error/5;
        cv_test_error(i,j) = sum_test_error/5;
    end
end
%% get the lowest cross validation error for each q
for i=1:5
    bestcv(i)=min(cv_test_error(i,:));
    position = find(cv_test_error(i,:)==bestcv(i));
    besttrain(i)=train_error(i,position(1));
    besttest(i)=test_error(i,position(1));
end

