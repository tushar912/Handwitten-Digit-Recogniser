function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


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


X= [ones(m,1),X];
a1=X;
z2 =a1*Theta1';
a2=sigmoid(z2);
a2=[ones(size(a2,1),1),a2];
z3=a2*Theta2';
a3=sigmoid(z3);
h_x=a3;
y_v=(1:num_labels)==y;
J=(-1/m)*sum(sum((y_v.*log(h_x))+((1-y_v).*log(1-h_x))));

for t =1:m
  a1=X(t,:)';
  z2=Theta1*a1;
  a2=[1;sigmoid(z2)];
  z3=Theta2*a2;
  a3=sigmoid(z3);
  y_v=(1:num_labels)'==y(t);
  delta3= a3-y_v;
  delta2=(Theta2'*delta3).*[1;sigrad(z2)];
  delta2=delta2(2:end);
  Theta1_grad = Theta1_grad + (delta2 * a1'); 
   Theta2_grad = Theta2_grad + (delta3 * a2');
  
endfor
Theta1_grad = (1/m) * Theta1_grad; 
   Theta2_grad = (1/m) * Theta2_grad;

reg_term = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); %scalar
  
  %Costfunction With regularization
  J = J + reg_term; %scalar
  
  %Calculating gradients for the regularization
  Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25 x 401
  Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10 x 26
  
  %Adding regularization term to earlier calculated Theta_grad
  Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
  Theta2_grad = Theta2_grad + Theta2_grad_reg_term;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
