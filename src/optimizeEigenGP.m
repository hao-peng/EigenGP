% Optimize a model by bfgs
% model is the eigenGP model
% x is the inputs of training data
% y is the responses of training data
% nIter is the max number of iteration for optimizer
%
% Author: Hao Peng
% Last edit: April 21, 2015

function model = optimizeEigenGP(model, x, y, nIter)
D = size(x,2);
M = model.M;
if strcmp(model.method, 'EigenGP')
  param = [model.lik;model.cov;reshape(model.B, M*D,1)];
  [new_param] = minimize(param, @(param) lik_upd_kerB(param, x, y, M), nIter);
  model.lik = new_param(1);
  model.cov = new_param(2:D+2);
  model.B = reshape(new_param(D+3:D+2+M*D),M,D);
  inv_ell = exp(-model.cov(1:D)); % inv_ell=1./ell;
  B_ell = scale_cols(model.B, inv_ell);
  jitter = 1e-6;
  sf2 = exp(2*model.cov(D+1));
  Kbb = sf2*exp(-0.5*sq_dist(B_ell'))+ jitter*eye(M);
  Kbb = (Kbb+Kbb')/2;
  Lambdaq = eig(Kbb);
  Lambdaq_sort = sort(abs(Lambdaq),'descend');
  model.lnw = log(Lambdaq_sort);
    
  param = [model.lnw];
  [new_param] = minimize(param, @(param) lik_upd_W(param, x, y, M, model), nIter);
  model.lnw = new_param;
elseif strcmp(model.method, 'EigenGP*')
  param = [model.lik;model.cov;reshape(model.B, M*D,1);model.lnw];
  %[new_param] = minimize(param, @(param) lik_upd_kerBW(param, x, y, M), nIter);
  [new_param] = minimize(param, @(param) lik_upd_unscaledkerBW(param, x, y, M), nIter);
  model.lik = new_param(1);
  model.cov = new_param(2:D+2);
  model.B = reshape(new_param(D+3:D+2+M*D),M,D);
  model.lnw = new_param(D+3+M*D:D+2+M+M*D);
elseif strcmp(model.method, 'EigenGP+')
  param = [model.lik;model.cov;reshape(model.B, M*D,1)];
  [new_param] = minimize(param, @(param) lik_delta_kerB(param, x, y, M), nIter);
  model.lik = new_param(1);
  model.cov = new_param(2:D+2);
  model.B = reshape(new_param(D+3:D+2+M*D),M,D);
  inv_ell = exp(-model.cov(1:D)); % inv_ell=1./ell;
  B_ell = scale_cols(model.B, inv_ell);
  jitter = 1e-6;
  sf2 = exp(2*model.cov(D+1));
  Kbb = sf2*exp(-0.5*sq_dist(B_ell'))+ jitter*eye(M);
  Kbb = (Kbb+Kbb')/2;
  Lambdaq = eig(Kbb);
  Lambdaq_sort = sort(abs(Lambdaq),'descend');
  model.lnw = log(Lambdaq_sort);

  param = [model.lnw];
  [new_param] = minimize(param, @(param) lik_delta_W(param, x, y, M, model), nIter);
  model.lnw = new_param;
else
  error('method is not supported\n');
end

end