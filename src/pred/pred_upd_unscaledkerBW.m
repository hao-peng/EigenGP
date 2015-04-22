function [mu,s2] = pred_upd_unscaledkerBW(model, x, y, xs)
  M = model.M;
  [N,D] = size(x);
  sn2 = exp(2*model.lik);
  inv_ell = exp(-model.cov(1:D));
  sf2 = exp(2*model.cov(D+1));
  B = reshape(model.B, M, D);
  w = exp(model.lnw);

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
  U = real(Uq(:,sort_ind(1:M))); %U = Uq_sort

  Phin = Kxb*U;

  Q = diag(1./w)+Phin'*Phin/sn2;
  chol_Q = jitChol(Q);

  beta = solve_tril(chol_Q', Phin')/sn2; %(chol(Q)'\Phin)/sn2
  beta_y = beta*y;

  Ksb_U = Ksb*U;
  mu = solve_tril(chol_Q', Ksb_U')'*beta_y;
  alpha = solve_tril(chol_Q', Ksb_U');
  s2 = sum(alpha.*alpha)'+sn2;
end