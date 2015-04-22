% k(a,b) = sf^2 * exp(-0.5*(a-b)'*diag(1./ell.^2)*(a-b))
% tilde_Kxx = Kxb*inv(Kbb)*Kbx + I.*(Kxx-Kxb*inv(Kbb)*Kbx)
% Phi(x) = Kxb*U*inv(Lambda)
% 
% Author: Hao Peng
% Last edit: April 21, 2015
function [mu,s2] = pred_delta_kerB(model, x, y, xs)
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

  chol_Kbb = jitChol(Kbb);
  invchol_Kbb_Kbx = solve_tril(chol_Kbb', Kxb');
  A = sn2+sf2*ones(N,1)-sum(invchol_Kbb_Kbx.*invchol_Kbb_Kbx)'; % diagonal term in CN
  invA = 1./A;

  Q = Kbb+Kxb'*scale_rows(Kxb, invA);
  chol_Q = jitChol(Q);

  beta = scale_cols(solve_tril(chol_Q', Kxb'), invA); %(chol(Q)'\Kbx)*diag(1/A)
  beta_y = beta*y;

  mu = solve_tril(chol_Q', Ksb')'*beta_y;
  invchol_Q_Kbs =  solve_tril(chol_Q', Ksb');
  invchol_Kbb_Kbs = solve_tril(chol_Kbb', Ksb');
  s2 = sn2+sf2-sum(invchol_Kbb_Kbs.*invchol_Kbb_Kbs)'+sum(invchol_Q_Kbs.*invchol_Q_Kbs)';
end