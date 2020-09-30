function [ Y, L, U ] = runMultiLayer( X, W, V )
% RUNMULTILAYER Calculates output and labels of the net
%
%    Inputs:
%              X - Data samples to be classified (matrix)
%              W - Weights of the hidden neurons (matrix)
%              V - Weights of the output neurons (matrix)
%
%    Output:
%              Y - Output for each sample and class (matrix)
%              L - The resulting label of each sample (vector) 
%              U - Activation of hidden neurons (vector)

% Add your own code here
S = zeros(size(W,2),1); % Calculate the weighted sum of input signals (hidden neuron)
U = zeros(size(W,2)); % Calculate the activation of the hidden neurons (use hyperbolic tangent)
Y = zeros(size(V,2),1); % Calculate the weighted sum of the hidden neurons



S = X*W;
U = tanh(S);


Uadj = [U ones(size(U,1),1)];


Y = Uadj*V; 





% Calculate labels
[~, L] = max(Y,[],2);

end

