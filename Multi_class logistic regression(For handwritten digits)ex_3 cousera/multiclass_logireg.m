clear all
clc
load('ex3data1.mat');
lambda=(0:0.1:1)';
m1 = size(X, 1);
num_labels = 10;
rand_indices = randperm(m1);
train=floor(m1*0.70);
cross_validation=floor(m1*0.15);
test=m1-train-cross_validation;
X1 = X(rand_indices(1:train), :);
y1=y(rand_indices(1:train));
y3=y(rand_indices(train+1:train+cross_validation));
y4=y(rand_indices(train+cross_validation+1:m1));
m = size(X1, 1);
%lambda = 0.1;
n = size(X1, 2);
all_theta = zeros(num_labels, n + 1);
X1 = [ones(m, 1) X1];
size_lambda=size(lambda,1);
cross_validation_error=zeros(size_lambda,1);
size(cross_validation_error)
for counter=1:size_lambda
for i=1:num_labels
    y2= y1==i;
    temp_theta=zeros(n+1,1);
    options = optimset('GradObj', 'on', 'MaxIter', 400);


[temp_theta, J, exit_flag] = ...
	fmincg(@(t)(lrCostFunction(t, X1, y2, lambda(counter))),temp_theta, options);
    all_theta(i,:)=temp_theta';
end

X2= X(rand_indices(train+1:train+cross_validation), :);
m4=size(X2,1);
X2 = [ones(m4, 1) X2];
p2=zeros(size(X2, 1), 1);
for i=1:m4
   [~, p2(i)]=max(X2(i,:)*all_theta');
end
cross_validation_error(counter)=mean(double(p2 == y3)) * 100;
end
size(cross_validation_error);
plot(lambda,cross_validation_error);
[M,I]=max(cross_validation_error);
%Finding the theta values for the given lambda
for i=1:num_labels
    y2= y1==i;
    temp_theta=zeros(n+1,1);
    options = optimset('GradObj', 'on', 'MaxIter', 400);


[temp_theta, J, exit_flag] = ...
	fmincg(@(t)(lrCostFunction(t, X1, y2, lambda(I))),temp_theta, options);
    all_theta(i,:)=temp_theta';
end

X4= X(rand_indices(train+cross_validation+1:m1), :);
m4=size(X4,1);
X4 = [ones(m4, 1) X4];
p2=zeros(size(X4, 1), 1);
for i=1:m4
   [~, p2(i)]=max(X4(i,:)*all_theta');
end
fprintf('\n min_cost lambda value: %f\n', lambda(I));
fprintf('\n Test set accuracy: %f\n', mean(double(p2 == y4)) * 100);

