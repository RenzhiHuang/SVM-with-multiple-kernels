% Decision boundary
% This code plots the decision boundary for a classifier with respect
% Directly taken from http://www.peteryu.ca/tutorials/matlab/visualize_decision_boundaries

% range of decision boundary (can be changed according to the need)
xrange = [0 25];
yrange = [0 12];

% step size for how finely you want to visualize the decision boundary (can be changed according to the need)
inc = 0.01;

% generate grid coordinates. This will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

% size of the (x, y) image, which will also be the size of the
% decision boundary image that is used as the plot background.
image_size = size(x);

xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.

xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
numxypairs = length(xy); % number of (x,y) pairs


% Important
%----------------------------------
% change this part depending on the classifier
% get the decision for each point (each point of 'xy')
% use your classifier here to classify each point
% store the labels (-1 or 1) in idx
% eg- logistic regression
% xy = [xy ones(size(xy,1),1)];
% scores = -(xy * [w b]');
% expscores = exp(scores);
% test accuracy
% probability = ones(size(xy,1),1)./(ones(size(xy,1),1)+expscores);
% idx = probability>0.5;

%-----------------------------------
[er,idx] = linear_classifier(xy,ones(size(xy,1),1),w,b);
% [er,idx]=...
%     polynomial_classifier(train_data,train_label,xy,ones(size(xy,1),1),a,b,q(2));
% [er,idx]=...
%     RBF_classifier(train_data,train_label,xy,ones(size(xy,1),1),a,b,r(2));
% reshape the idx (which contains the class label) into an image
decisionmap = reshape(idx, image_size);

figure;

%show the image
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');

% colormap for the classes:
cmap =  [0.0 0.8 1; 1 0.8 0.8];
colormap(cmap);

% plot the class test data.
temp1 = test_data((test_label==-1),:);
temp2 = test_data((test_label==1),:);
plot(temp1(:,1),temp1(:,2),'bx','linewidth',0.3);
plot(temp2(:,1),temp2(:,2), 'ro','linewidth',0.3);

% include legend
legend('Class +1','Class -1')
%title('Neural Network Decision Boundary');

% label the axes.
xlabel('x1');
ylabel('x2');

% store the figure
saveas(gcf,'File/Figs/decision.png')
