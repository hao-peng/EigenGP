% Predict using EigenGP
% x is the inputs of training data
% y is the responses of training data
% xs is the inputs of test data
%
% Author: Hao Peng
% Last edit: April 21, 2015

function [mu, s2] = predEigenGP(model, x, y, xs)
if strcmp(model.method, 'EigenGP')
  [mu, s2] = pred_upd_W(model, x, y, xs);
elseif strcmp(model.method, 'EigenGP*')
  [mu, s2] = pred_upd_unscaledkerBW(model, x, y, xs);
elseif strcmp(model.method, 'EigenGP+')
  [mu, s2] = pred_delta_W(model, x, y, xs);
else
  error('method is not supported\n');
end

end