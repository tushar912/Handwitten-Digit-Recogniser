input_layer_size= 400;
num_labels =10;
hidden_layer_size = 25; 
load('ex3data1.mat');
m= size(X,1);

display1(X)
load('ex4weights.mat');
#loading trained parameters
param = [Theta1(:);Theta2(:)];
% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = Cost(param, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);


fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = Cost(param, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);


initial_Theta1 = randinit(input_layer_size, hidden_layer_size);
initial_Theta2 = randinit(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%  Check gradients by running checkNNGradients
checkNNGradients;



%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = Cost(param, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) Cost(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

