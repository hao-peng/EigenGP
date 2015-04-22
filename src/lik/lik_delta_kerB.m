% param = [log(sn); log(ell); log(sf); B]
% k(a,b) = sf^2*exp(-0.5*(a-b)'*diag(1./ell.^2)*(a-b))
% tilde_Kxx = Kxb*inv(Kbb)*Kbx + I.*(Kxx-Kxb*inv(Kbb)*Kbx)
% B: inducing variables/ basis points
% 
% Author: Hao Peng
% Last edit: April 21, 2015
function [varargout] = lik_delta_kerB(param, x, y, M)
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

chol_Kbb = jitChol(Kbb);
invchol_Kbb_Kbx = solve_tril(chol_Kbb', Kxb');
A = sn2+sf2*ones(N,1)-sum(invchol_Kbb_Kbx.*invchol_Kbb_Kbx)'; % diagonal term in CN
invA = 1./A;

Q = Kbb+Kxb'*scale_rows(Kxb, invA);
chol_Q = jitChol(Q);

beta = scale_cols(solve_tril(chol_Q', Kxb'), invA); %(chol(Q)'\Kbx)*diag(1/A)
beta_y = beta*y;

% negative log marginalized likelihood
nlZ = sum(log(diag(chol_Q)))-sum(log(diag(chol_Kbb)))+0.5*sum(log(A))... % logdet(CN)/2
  +0.5*y'*(invA.*y)-0.5*(beta_y'*beta_y)... % y'inv(CN)y/2
  +0.5*N*log(2*pi);

varargout = {nlZ};

if nargout > 1
  dnlZ = zeros(size(param));
  % gradient w.r.t. log(sn)
  diag_invCN = invA-sum(beta.*beta)';
  trace_invCN = sum(diag_invCN);
  invCN_y = invA.*y-beta'*beta_y;
  dnlZ(1) = sn2*(trace_invCN-sum(sum(invCN_y.*invCN_y)));
  % gradient w.r.t. log(ell)
  eta = inv_ell.^2;
  
  invKbb_Kbx_invCN = solve_triu(chol_Q,beta); %inv(Kbb)*Kbx*inv(CN) = inv(Q)*Kbx*diag(1./A)
  inv_chol_Q = inv_triu(chol_Q);
  inv_chol_Kbb = inv_triu(chol_Kbb);
  invKbb_Kbx_invCN_Kxb_invKbb = inv_chol_Kbb*inv_chol_Kbb'-inv_chol_Q*inv_chol_Q'; %inv(Kbb)*Kbx*inv(CN)*Kxb*inv(Kbb) = inv(Kbb)-inv(Q)
  invKbb_Kbx_invCN_y = invKbb_Kbx_invCN*y;
  
  R = invKbb_Kbx_invCN.*Kxb';
  S = invKbb_Kbx_invCN_Kxb_invKbb.*Kbb';
  tR = (invKbb_Kbx_invCN_y*invCN_y').*Kxb';
  tS = (invKbb_Kbx_invCN_y*invKbb_Kbx_invCN_y').*Kbb';
  
  % replacing inv(CN) by I.*inv(CN) in R and S
  % replacing inv(CN)*y*y'*inv(CN) by I.*(inv(CN)*y*y'*inv(CN)) in tR and tS
  invKbb_Kbx = solve_triu(chol_Kbb,solve_tril(chol_Kbb', Kxb'));
  invKbb_Kbx_diag_invCN = scale_cols(invKbb_Kbx, diag_invCN);
  invKbb_Kbx_diag_invCN_Kxb_invKbb = invKbb_Kbx*invKbb_Kbx_diag_invCN';
  diag_P = invCN_y.*invCN_y; % P = diag(inv(CN)*y*y'*inv(CN))
  invKbb_Kbx_diag_P = scale_cols(invKbb_Kbx, diag_P);
  invKbb_Kbx_diag_P_Kxb_invKbb = invKbb_Kbx*invKbb_Kbx_diag_P';
  
  R_ = invKbb_Kbx_diag_invCN.*Kxb';
  S_ = invKbb_Kbx_diag_invCN_Kxb_invKbb.*Kbb';
  tR_ = invKbb_Kbx_diag_P.*Kxb';
  tS_ = invKbb_Kbx_diag_P_Kxb_invKbb.*Kbb';
  
  B2 = B.*B;
  x2 = x.*x; % this can be saved
  
  tr_invCN_dCN = (2*sum(B.*(R*x))-sum(R*x2)-sum(R'*B2)-sum(B.*(S*B))+sum(S*B2))...
    -(2*sum(B.*(R_*x))-sum(R_*x2)-sum(R_'*B2)-sum(B.*(S_*B))+sum(S_*B2));
  yt_invCN_dCN_invCN_y = 2*sum(B.*(tR*x))-sum(tR*x2)-sum(tR'*B2)-sum(B.*(tS*B))+sum(tS*B2)...
    -(2*sum(B.*(tR_*x))-sum(tR_*x2)-sum(tR_'*B2)-sum(B.*(tS_*B))+sum(tS_*B2));
  dnlZ(2:D+1) = -eta.*(tr_invCN_dCN-yt_invCN_dCN_invCN_y)';
  % gradient w.r.t. log(sf)
  dnlZ(D+2) = (N-sn2*trace_invCN)-sum(invCN_y.*y)+sn2*sum(invCN_y.*invCN_y);
  % gradient w.r.t. B
  x_eta = scale_cols(x, eta);
  B_eta = scale_cols(B, eta);
  tr_invCN_dCN = 2*(R*x_eta-repmat(sum(R,2),1,D).*B_eta)-(2*S*B_eta-2*repmat(sum(S,2),1,D).*B_eta)...
    -(2*(R_*x_eta-repmat(sum(R_,2),1,D).*B_eta)-(2*S_*B_eta-2*repmat(sum(S_,2),1,D).*B_eta));
  yt_invCN_dCN_invCN_y = 2*(tR*x_eta-repmat(sum(tR,2),1,D).*B_eta)-(2*tS*B_eta-2*repmat(sum(tS,2),1,D).*B_eta)...
    -(2*(tR_*x_eta-repmat(sum(tR_,2),1,D).*B_eta)-(2*tS_*B_eta-2*repmat(sum(tS_,2),1,D).*B_eta));
  dnlZ(D+3:D+2+M*D) = reshape(0.5*(tr_invCN_dCN-yt_invCN_dCN_invCN_y),M*D,1);
  
  varargout = {nlZ, dnlZ};
end
end