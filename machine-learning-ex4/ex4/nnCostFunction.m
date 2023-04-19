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

% printf('size of Theta1 = %d %d\n', size(Theta1)) % 25 x 401
% printf('size of Theta2 = %d %d\n', size(Theta2)) % 10 x 26
% printf('size of X = %d %d\n', size(X)) % 5000 x 400
% printf('size of y = %d %d\n', size(y)) % 5000 x 1

%
% Part 1
%

% Input layer
a1 = [ones(m, 1), X];

% Hidden layer
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];

% Output layer
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3; 
% printf('size of h = %d %d\n', size(h)) % 5000 x 10

% Cost
total_cost = 0;
for i = 1 : m
    % v is the i-th example's y binary vector
    v = zeros(1, num_labels);
    v(y(i)) = 1;

    for k = 1 : num_labels
        cost = -1 * v(k) * log(h(i, k)) - (1 - v(k)) * log(1 - h(i, k));
        total_cost = total_cost + cost;
    end
end
J = sum(total_cost) / m;

% add the cost for the regularization terms (bias column excluded)
J = J + lambda * (sum((Theta1(:, 2:end) .^ 2)(:)) + sum((Theta2(:, 2:end) .^ 2)(:))) / (2 * m);

%
% Part 2 & 3
%

total_delta1 = zeros(size(Theta1)); % 𝛥1, the same size of Theta1
total_delta2 = zeros(size(Theta2)); % 𝛥2, the same size of Theta2

for t = 1 : m
    a1 = [1, X(t, :)]; % add bias unit
    % printf('size of a1 = %d %d\n', size(a1))
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [1, a2];
    % printf('size of a2 = %d %d\n', size(a2))
    z3 = a2 * Theta2';
    a3 = sigmoid(z3); % a 1x10 row vector
    % printf('size of a3 = %d %d\n', size(a3))

    % r is the binary vector of actual result
    r = zeros(1, num_labels); % a 1x10 row vector
    r(y(t)) = 1;

    % 𝛅3
    delta3 = a3 - r; % a 1x10 row vector
    % printf('size of delta3 = %d %d\n', size(delta3))

    % 𝛅2
    delta2 = (delta3 * Theta2)(2:end) .* sigmoidGradient(z2); % a 1 x 26 row vector
    % printf('size of delta2 = %d %d\n', size(delta2))

    total_delta2 = total_delta2 + delta3' * a2;
    total_delta1 = total_delta1 + delta2' * a1;
end


Theta1_grad = (total_delta1 + lambda * Theta1) / m;
Theta1_grad(:,1) = total_delta1(:,1) / m;

Theta2_grad = (total_delta2 + lambda * Theta2) / m;
Theta2_grad(:,1) = total_delta2(:,1) / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
