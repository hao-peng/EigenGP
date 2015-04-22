% k(a,b) = sf^2 * exp(-0.5*(a-b)'*diag(1./ell.^2)*(a-b))
% tilde_Kxx = Phi(x)*diag(w)*Phi(x)'+I.*(Kxx-Phi(x)*diag(w)*Phi(x)')
% Phi(x) = Kxb*U*inv(Lambda)
% 
% Author: Hao Peng
% Last edit: April 21, 2015
function [mu,s2] = pred_delta_W(model, x, y, xs)
  M = model.M;
  [N,D] = size(x);
  w = exp(model.lnw);
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

  [Uq, Lambdaq] = eig(Kbb);
  diag_lambdaq = real(diag(Lambdaq));
  [Lambdaq_sort,sort_ind] = sort(abs(diag_lambdaq),'descend');
  U = real(Uq(:,sort_ind(1:M)));
  U = scale_cols(U, 1./Lambdaq_sort);

  Phin = Kxb*U;

  A = sn2+sf2*ones(N,1)-sum(Phin.*scale_cols(Phin,w),2); % diagonal term in CN
  invA = 1./A;

  Q = diag(1./w)+Phin'*scale_rows(Phin, invA);
  chol_Q = jitChol(Q);
  %chol_Q = cholproj(Q);

  beta = scale_cols(solve_tril(chol_Q', Phin'), invA); %(chol(Q)'\Phin)*diag(1/A)
  beta_y = beta*y;

  Ksb_U = Ksb*U;
  mu = solve_tril(chol_Q', Ksb_U')'*beta_y;
  alpha = solve_tril(chol_Q', Ksb_U');
  s2 = sn2+sf2-sum(Ksb_U.*scale_cols(Ksb_U,w),2)+sum(alpha.*alpha)';
end