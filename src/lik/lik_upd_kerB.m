% param = [log(sn); log(ell); log(sf); B]
% k(a,b) = sf^2 * exp(-0.5*(a-b)'*diag(1./ell.^2)*(a-b))
% tilde_Kxx = Kxb*inv(Kbb)*Kbx
% B: inducing variables/ basis points
% 
% Author: Hao Peng
% Last edit: April 21, 2015
function [varargout] = lik_upd_kerB(param, x, y, M)
[N,D] = size(x);
sn2 = exp(2*param(1));
inv_ell = exp(-param(2:D+1));

sf2 = exp(2*param(D+2));
B = reshape(param((D+3):(D+2+M*D)), M, D);

fixed_jitter = 1e-6; % jitter

B_inv_ell = scale_cols(B, inv_ell);
x_inv_ell = scale_cols(x, inv_ell);
Kbb = sf2*exp(-0.5*sq_dist(B_inv_ell')) + fixed_jitter*eye(M);
Kbb = (Kbb+Kbb')/2;
Kxb = sf2*exp(-0.5*sq_dist(x_inv_ell', B_inv_ell'));

Q = Kbb+Kxb'*Kxb/sn2;
chol_Q = jitChol(Q);
chol_Kbb = jitChol(Kbb);

beta = solve_tril(chol_Q', Kxb')/sn2; %(chol(Q)'\Kbx)/sn2
beta_y = beta*y;

% negative log marginalized likelihood
nlZ = sum(log(diag(chol_Q)))-sum(log(diag(chol_Kbb)))+0.5*log(sn2)*N... % logdet(CN)/2
  +0.5*(y'*y)/sn2-0.5*(beta_y'*beta_y)... % y'inv(CN)y/2
  +0.5*N*log(2*pi);

varargout = {nlZ};

if nargout > 1
  dnlZ = zeros(size(param));
  % gradient w.r.t. log(sn)
  invCN_y = y/sn2-beta'*beta_y;
  trace_invCN = N/sn2-sum(sum(beta.*beta));
  dnlZ(1) = sn2*(trace_invCN - sum(sum(invCN_y.*invCN_y)));
  % gradient w.r.t. log(ell)
  eta = inv_ell.^2;
  
  invKbb_Kbx_invCN = solve_triu(chol_Q,beta); %inv(Kbb)*Kbx*inv(CN) = inv(Q)*Kbx/sn2
  inv_chol_Q = inv_triu(chol_Q);
  inv_chol_Kbb = inv_triu(chol_Kbb);
  invKbb_Kbx_invCN_Kxb_invKbb = inv_chol_Kbb*inv_chol_Kbb'-inv_chol_Q*inv_chol_Q'; %inv(Kbb)*Kbx*inv(CN)*Kxb*inv(Kbb) = inv(Kbb)-inv(Q)
  invKbb_Kbx_invCN_y = invKbb_Kbx_invCN*y;
 
  R = invKbb_Kbx_invCN.*Kxb';
  S = invKbb_Kbx_invCN_Kxb_invKbb.*Kbb';
  
  tR = (invKbb_Kbx_invCN_y*invCN_y').*Kxb';
  tS = (invKbb_Kbx_invCN_y*invKbb_Kbx_invCN_y').*Kbb';
  
  B2 = B.*B;
  x2 = x.*x; % this can be saved
    
  tr_invCN_dCN = 2*sum(B.*(R*x))-sum(R*x2)-sum(R'*B2)-sum(B.*(S*B))+sum(S*B2);
  yt_invCN_dCN_invCN_y = 2*sum(B.*(tR*x))-sum(tR*x2)-sum(tR'*B2)-sum(B.*(tS*B))+sum(tS*B2);
  dnlZ(2:D+1) = -eta.*(tr_invCN_dCN-yt_invCN_dCN_invCN_y)';
  % gradient w.r.t. log(sf)
  dnlZ(D+2) = (N-sn2*trace_invCN)-sum(invCN_y.*y)+sn2*sum(invCN_y.*invCN_y);
  % gradient w.r.t. B
  x_eta = scale_cols(x, eta);
  B_eta = scale_cols(B, eta);
  tr_invCN_dCN = 2*(R*x_eta-repmat(sum(R,2),1,D).*B_eta)...
    -(2*S*B_eta-2*repmat(sum(S,2),1,D).*B_eta);
  yt_invCN_dCN_invCN_y = 2*(tR*x_eta-repmat(sum(tR,2),1,D).*B_eta)...
    -(2*tS*B_eta-2*repmat(sum(tS,2),1,D).*B_eta);
  dnlZ(D+3:D+2+M*D) = reshape(0.5*(tr_invCN_dCN-yt_invCN_dCN_invCN_y),M*D,1);
  varargout = {nlZ, dnlZ};
end
end