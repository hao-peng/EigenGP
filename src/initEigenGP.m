% Initialize a model
% method is the model to be used, the current options are:
%    "EigenGP": EigenGP with sequential update
%    "EigenGP*": EigenGP with joint update
%    "EigenGP+": EigenGP with sequential update and diagonal delta term
% x is the inputs of training data
% y is the responses of training data
% M is the number of inducing inputs (basis points) used
% hyp is the hyperparameter for the ARD squared exponential kernel:
%    k(a,b) = sf^2 * exp(-0.5*(a-b)'*diag(1./ell.^2)*(a-b))
%    hyp.lik is is log(sn) - the log derivation of observation noise
%    hyp.cov(1:D) is log(ell) - the log scale length for each dimension.
%    hyp.cov(D+1) is log(sf) - the log signal ampilifier 
%
% Author: Hao Peng
% Last edit: April 21, 2015

function model = initEigenGP(method, x, y, M, hyp, B)
nTries = 10;
jitter = 1e-6;
model.method = method;

[N, D] = size(x);

if nargin < 4
  model.M = ceil(N/10);
else
  model.M = M;
end

if nargin < 5
  model.lik = log(std(y)/2);
  model.cov = [log((max(x)-min(x))/2); log(var(y,1))];
else
  model.lik = hyp.lik;
  model.cov = hyp.cov;
  if size(model.cov,2) ~= 1
    model.cov = model.cov';
  end
end

if nargin < 6
  last_nlZ = -Inf;
  for i = 1:nTries 
    %use kmeans with both x and y
    [~, B] = fkmeans(x', model.M);
    inv_ell = exp(-model.cov(1:D)); % inv_ell=1./ell;
    B_ell = scale_cols(B, inv_ell);
    sf2 = exp(2*model.cov(D+1));
    Kbb = sf2*exp(-0.5*sq_dist(B_ell'))+ jitter*eye(M);
    Kbb = (Kbb+Kbb')/2;
    Lambdaq = eig(Kbb);
    Lambdaq_sort = sort(abs(Lambdaq),'descend');

    if strcmp(model.method, 'EigenGP')
      lnw = log(Lambdaq_sort);
      param = [model.lik;model.cov;reshape(B, M*D,1)];
      nlZ = lik_upd_kerB(param, x, y, M);
    elseif strcmp(model.method, 'EigenGP*')
%       lnw = log(Lambdaq_sort);
%       param = [model.lik;model.cov;reshape(B, M*D,1);lnw];
%       nlZ = lik_upd_kerBW(param, x, y, M);
      lnw = -log(Lambdaq_sort);
      param = [model.lik;model.cov;reshape(B, M*D,1);lnw];
      nlZ = lik_upd_unscaledkerBW(param, x, y, M);
    elseif strcmp(model.method, 'EigenGP+')
      lnw = log(Lambdaq_sort);
      param = [model.lik;model.cov;reshape(B, M*D,1)];
      nlZ = lik_delta_kerB(param, x, y, M);
    else
      error('method is not supported\n');
    end
    if nlZ > last_nlZ
      model.lnw = lnw;
      model.B = B;
    end
  end
else
  model.B = B;
end
end