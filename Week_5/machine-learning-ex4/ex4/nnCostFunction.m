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

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Calculate htheta
A1 = [ones(m, 1) X];
A2 = sigmoid(A1*Theta1');
m2 = size(A2, 1);
A2 = [ones(m2, 1) A2];
A3 = sigmoid(A2*Theta2');

% Transform y to num_labels-dimension vector
size_y = size(y, 1);
yk = zeros(size_y,num_labels);

for i = 1:size_y
    yk(i,y(i))=1;
end

residual = -yk.*log(A3)-(1-yk).*log(1-A3);
J=1/m*sum(residual(:));

Theta1_squared = Theta1.^2;
Theta2_squared = Theta2.^2;

Theta1_squared_sum = sum(Theta1_squared);
Theta2_squared_sum = sum(Theta2_squared);

regularization = lambda/(2*m)*(sum(Theta1_squared_sum(2:end))+sum(Theta2_squared_sum(2:end)));

J = J+regularization;

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

for t = 1:m
% substep 1
    a1 = A1(t,:);
    z2 = a1*Theta1';
    a2 = sigmoid(z2);
    a2 = [1,a2];
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
% substep 2   
    d3 = a3 - yk(t,:);
% substep 3    
    d2 = (Theta2'*d3').*sigmoidGradient([1,z2])';
    d2 = d2(2:end);
    %size(d3'*a2)
% substep 4   
    D2 = D2 + d3'*a2;
    D1 = D1 + d2*a1;
end

Theta1_grad = 1/m*D1;
Theta2_grad = 1/m*D2;



%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
