function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

z=X*theta;
Theta=theta.^2;
J=(sum(((z)-y).^2)+lambda*sum(Theta(2:end)))*0.5/m;
A=X';
grad=(1/m)*(A*(z-y))+(lambda/m)*theta;
grad(1,1)=(A(1,:)*(z-y))*(1/m);












% =========================================================================

grad = grad(:);

end
