function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

mul=X*theta;
sigmul=sigmoid(mul);
for i=1:m
    J=J+((-y(i) * log(sigmul(i)) )-((1-y(i))*log(1-sigmul(i))));
end
J=J/m;
reg_parameter=0;
for i=2:theta(1)
    reg_parameter= reg_parameter+(theta(i)^2);
end
reg_parameter=(reg_parameter)*(lambda/(2*m));
J=J+reg_parameter;

Suz=size(theta);
for i=1:Suz(1)
    for j=1:m
      grad(i)=grad(i)+((sigmul(j)-y(j))*X(j,i));  
    end
end

grad=grad/m;

for i=2:Suz(1)
    grad(i)=grad(i)+((lambda/m)*theta(i));
end

% =============================================================

end
