% param = [ln(w)]
% k(a,b) = sf^2 * exp(-0.5*(a-b)'*diag(1./ell.^2)*(a-b))
% tilde_Kxx = Phi(x)*diag(w)*Phi(x)'+I.*max(0,(Kxx-Phi(x)*diag(w)*Phi(x)'))
% Phi(x) = Kxb*U*inv(Lambda), where U*Lambda*U = Kbb
% hyp.B: inducing variables/ basis points
% hyp.lik: log(sn)
% hyp.cov: [log(ell);log(sf)]
% 
% Author: Hao Peng
% Last edit: April 21, 2015
function [varargout] = lik_bnd_delta_W(param, x, y, M, hyp)
  [N,D] = size(x);
  w = exp(param);
  sn2 = exp(2*hyp.lik);
  inv_ell = exp(-hyp.cov(1:D));
  sf2 = exp(2*hyp.cov(D+1));
  B = hyp.B;

  fixed_jitter = 1e-6; % jitter

  B_inv_ell = scale_cols(B, inv_ell);
  x_inv_ell = scale_cols(x, inv_ell);
  Kbb = sf2*exp(-0.5*sq_dist(B_inv_ell')) + fixed_jitter*eye(M);
  Kbb = (Kbb+Kbb')/2;
  Kxb = sf2*exp(-0.5*sq_dist(x_inv_ell', B_inv_ell'));

  [Uq, Lambdaq] = eig(Kbb);
  diag_lambdaq = real(diag(Lambdaq));
  [Lambdaq_sort,sort_ind] = sort(abs(diag_lambdaq),'descend');
  U = real(Uq(:,sort_ind(1:M)));
  U = scale_cols(U, 1./Lambdaq_sort);

  Phin = Kxb*U;
  
  delta = sf2*ones(N,1)-sum(Phin.*scale_cols(Phin,w),2);
  diff0ind = delta >= 0;
  A = sn2+max(0,sf2*ones(N,1)-sum(Phin.*scale_cols(Phin,w),2)); % diagonal term in CN
  invA = 1./A;

  Q = diag(1./w)+Phin'*scale_rows(Phin, invA);
  chol_Q = jitChol(Q);
  %chol_Q = cholproj(Q);

  beta = scale_cols(solve_tril(chol_Q', Phin'), invA); %(chol(Q)'\Kbx)*diag(1/A)
  beta_y = beta*y;

  % negative log marginalized likelihood
  nlZ = sum(log(diag(chol_Q)))+0.5*sum(log(w))+0.5*sum(log(A))... % logdet(CN)/2
    +0.5*y'*(invA.*y)-0.5*(beta_y'*beta_y)... % y'inv(CN)y/2
    +0.5*N*log(2*pi);
  
  nlZ = real(nlZ);

  varargout = {nlZ};

  if nargout > 1
    dnlZ = zeros(size(param));
    % gradient w.r.t. w
    diag_invCN = invA-sum(beta.*beta)';
    invCN_y = invA.*y-beta'*beta_y;
    diag_P = invCN_y.*invCN_y; % P = diag(inv(CN)*y*y'*inv(CN))
    inv_chol_Q = inv_triu(chol_Q);
    tr_invCN_dCN = (1./w-1./w.^2.*sum(inv_chol_Q.*inv_chol_Q,2))...
      -sum(Phin.*scale_rows(Phin,diag_invCN.*diff0ind))';
    phin_invCN_y = scale_rows(solve_triu(chol_Q, solve_tril(chol_Q', scale_rows(Phin,invA)'*y)),1./w);
    yt_invCN_dCN_invCN_y = sum(phin_invCN_y.*phin_invCN_y,2)...
      -sum(Phin.*scale_rows(Phin,diag_P.*diff0ind))';
    dnlZ(1:M) = 0.5*w.*(tr_invCN_dCN-yt_invCN_dCN_invCN_y);
    
    varargout = {nlZ, dnlZ};
  end
end