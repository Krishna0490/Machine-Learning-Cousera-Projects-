function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m, 1) X];
k=size(Theta2,1);
middle_layer0=zeros(1,size(Theta1, 1));
middle_layer1=zeros(1,size(Theta1, 1)+1);
final_layer=zeros(1,size(Theta2,1));
m_layer0=zeros(1,size(Theta1, 1));
m_layer1=zeros(1,size(Theta1, 1)+1);

for i=1:m
m_layer0=X(i,:)*Theta1';
m_layer1=[1 m_layer0];
    middle_layer0=(sigmoid(X(i,:)*Theta1'));
    middle_layer1=[1 middle_layer0];
    final_layer=(sigmoid(middle_layer1*Theta2'));
    final_layer1=1-final_layer;
    sigmul1=(log(final_layer))';
sigmul2=(log(final_layer1))';
temp_y=zeros(k,1);
 temp_y(y(i))=1;
  J=J-sum(((sigmul1.*temp_y)+(sigmul2.*(1-temp_y))));
    delta_3=final_layer'-temp_y;
    %size(delta_3)
    delta_2=(Theta2'*delta_3).*sigmoidGradient((m_layer1'));
    %size(delta_2)
    delta_2=delta_2(2:end);
    %size(delta_2)
    %size(X(i,:))
    Theta1_grad=Theta1_grad+(delta_2*X(i,:));
    Theta2_grad=Theta2_grad+(delta_3*middle_layer1);
          
end
J=J/m;
Reg_theta1=Theta1(:,2:size(Theta1,2));
Reg_theta1=(Reg_theta1.^2);
Reg_theta2=Theta2(:,2:size(Theta2,2));
Reg_theta2=(Reg_theta2.^2);

J=J+((sum(sum(Reg_theta1))+sum(sum(Reg_theta2)))*(lambda/(2*m)));
Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;

Theta1_grad(:,2:size(Theta1_grad,2))= Theta1_grad(:,2:size(Theta1_grad,2))+(Theta1(:,2:size(Theta1,2))*(lambda/m));
Theta2_grad(:,2:size(Theta2_grad,2))=Theta2_grad(:,2:size(Theta2_grad,2))+(Theta2(:,2:size(Theta2,2))*(lambda/m));












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
