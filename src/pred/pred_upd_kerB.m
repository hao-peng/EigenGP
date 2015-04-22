% k(a,b) = sf^2 * exp(-0.5*(a-b)'*diag(1./ell.^2)*(a-b))
% tilde_Kxx = Kxb*inv(Kbb)*Kbx
% B: inducing variables/ basis points
% 
% Author: Hao Peng
% Last edit: April 21, 2015
function [mu,s2] = pred_upd_kerB(model, x, y, xs)
  M = model.M;
  [N,D] = size(x);
  sn2 = exp(2*model.lik);
  inv_ell = exp(-model.cov(1:D));
  sf2 = exp(2*model.cov(D+1));
  B = model.B;
  
  fixed_jitter = 1e-6; % jitter

  B_inv_ell = scale_cols(B, inv_ell);
  x_inv_ell = scale_cols(x, inv_ell);
  xs_inv_ell = scale_cols(xs, inv_ell);
  Kbb = sf2*exp(-0.5*sq_dist(B_inv_ell')) + fixed_jitter*eye(M);
  Kbb = (Kbb+Kbb')/2;
  Kxb = sf2*exp(-0.5*sq_dist(x_inv_ell', B_inv_ell'));
  Ksb = sf2*exp(-0.5*sq_dist(xs_inv_ell', B_inv_ell'));

  Q = Kbb+Kxb'*Kxb/sn2;
  chol_Q = jitChol(Q);

  beta = solve_tril(chol_Q', Kxb')/sn2; %(chol(Q)'\Kbx)/sn2
  beta_y = beta*y;

  mu = solve_tril(chol_Q', Ksb')'*beta_y;
  alpha = solve_tril(chol_Q', Ksb');
  s2 = sum(alpha.*alpha)'+sn2;
end