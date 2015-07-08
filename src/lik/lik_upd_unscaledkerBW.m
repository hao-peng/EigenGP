% param = [ln(sn);ln(ell);log(sf);B;ln(w)]
% k(a,b) = sf^2 * exp(-0.5*(a-b)'*diag(1./ell.^2)*(a-b))
% tilde_Kxx = Phi(x)*diag(w)*Phi(x)'
% Phi(x) = Kxb*U, 
% where U = Uq and Uq*Lambda*Uq'= Kbb
% hyp.B: inducing variables/ basis points
% hyp.lik: log(sn)
% hyp.cov: [log(ell);log(sf)]
function [varargout] = lik_upd_unscaledkerBW(param, x, y, M)
[N,D] = size(x);
sn2 = exp(2*param(1));
inv_ell = exp(-param(2:D+1));
sf2 = exp(2*param(D+2));
B = reshape(param((D+3):(D+2+M*D)), M, D);
w = exp(param((D+3+M*D):(M+D+2+M*D)));

fixed_jitter = 1e-6; % jitter

B_inv_ell = scale_cols(B, inv_ell);
x_inv_ell = scale_cols(x, inv_ell);
Kbb = sf2*exp(-0.5*sq_dist(B_inv_ell')) + fixed_jitter*eye(M);
Kbb = (Kbb+Kbb')/2;
Kxb = sf2*exp(-0.5*sq_dist(x_inv_ell', B_inv_ell'));

% compute eigenvectors and eigenvalues of Kbb
[Uq, Lambdaq] = eig(Kbb);
diag_lambdaq = real(diag(Lambdaq));
[Lambdaq_sort,sort_ind] = sort(abs(diag_lambdaq),'descend');
U = real(Uq(:,sort_ind(1:M))); %U = Uq_sort

%O(NM2)
Phin = Kxb*U; 

Q = diag(1./w)+Phin'*Phin/sn2;
chol_Q = jitChol(Q);

%O(NM2)
beta = solve_tril(chol_Q', Phin')/sn2; %(chol(Q)'\Kbx)/sn2
beta_y = beta*y;

%negative log marginalized likelihood
nlZ = sum(log(diag(chol_Q)))+0.5*sum(log(w))+0.5*log(sn2)*N... % logdet(CN)/2
  +0.5*(y'*y)/sn2-0.5*(beta_y'*beta_y)... % y'inv(CN)y/2
  +0.5*N*log(2*pi);

varargout = {nlZ};

if nargout > 1
  dnlZ = zeros(size(param));
  % gradient w.r.t. log(sn)
  diag_invCN = 1/sn2-sum(beta.*beta)';
  trace_invCN = sum(diag_invCN);
  invCN_y = y/sn2-beta'*beta_y;
  dnlZ(1) = sn2*(trace_invCN-sum(sum(invCN_y.*invCN_y)));
  
  % gradient w.r.t. log(ell)
  eta = inv_ell.^2;
  
  U_diagW_Phin_invCN = solve_tril(chol_Q', U')'*beta; %inv(Kbb)*Uq*diag(W)*Phin'*inv(CN)
  R = U_diagW_Phin_invCN.*Kxb';
  U_diagW_Phin_invCN_y_y_invCN = U_diagW_Phin_invCN*y*invCN_y';
  tR = U_diagW_Phin_invCN_y_y_invCN.*Kxb';
  
  diagW_Phin_invCN_Kxb = solve_triu(chol_Q, beta*Kxb); 
  diagW_Phin_invCN_y_y_invCN_Kxb = solve_triu(chol_Q, beta_y)*(Kxb'*invCN_y)';
  T = zeros(M, M);
  tT = zeros(M, M);
  for j = 1 : M
    V = 1./(Lambdaq_sort - Lambdaq_sort(j)); % numerical problematic
    V((Lambdaq_sort - Lambdaq_sort(j))==0) = 0;
    T(j,:) = -((diagW_Phin_invCN_Kxb(j,:)*U).*V')*U';
    tT(j,:) = -((diagW_Phin_invCN_y_y_invCN_Kxb(j,:)*U).*V')*U';
  end
  T = (U*T).*Kbb;
  tT = (U*tT).*Kbb;
  T_T = T+T';
  tT_tT = tT+tT';
  
  B2 = B.*B;
  x2 = x.*x;
  
  tr_invCN_dCN = 2*sum(B.*(R*x))-sum(R*x2)-sum(R'*B2)...
    +sum(B.*(T_T*B))-sum(T_T*B2);
    
  yt_invCN_dCN_invCN_y = 2*sum(B.*(tR*x))-sum(tR*x2)-sum(tR'*B2)...
    +sum(B.*(tT_tT*B))-sum(tT_tT*B2);

  dnlZ(2:D+1) =  -eta.*(tr_invCN_dCN-yt_invCN_dCN_invCN_y)';
  
  % gradient w.r.t. log(sf)
  dnlZ(D+2) = 2*((N-sn2*trace_invCN)-sum(invCN_y.*y)+sn2*sum(invCN_y.*invCN_y));
  
  % gradient w.r.t. B
  x_eta = scale_cols(x, eta);
  B_eta = scale_cols(B, eta);
  
  tr_invCN_dCN = 2*R*x_eta-2*repmat(sum(R,2),1,D).*B_eta...
    +2*T_T*B_eta-2*repmat(sum(T_T,2),1,D).*B_eta;
  
  yt_invCN_dCN_invCN_y = 2*tR*x_eta-2*repmat(sum(tR,2),1,D).*B_eta...
    +2*tT_tT*B_eta-2*repmat(sum(tT_tT,2),1,D).*B_eta;
   
  dnlZ((D+3):(D+2+M*D)) = reshape(0.5*(tr_invCN_dCN-yt_invCN_dCN_invCN_y),M*D,1);
   
  % gradient w.r.t. w
  inv_chol_Q = inv_triu(chol_Q);
  tr_invCN_dCN = 1./w-1./w.^2.*sum(inv_chol_Q.*inv_chol_Q,2);
  
  phin_invCN_y = scale_rows(solve_triu(chol_Q, solve_tril(chol_Q', Phin'*y)),1./w)/sn2;
  yt_invCN_dCN_invCN_y = sum(phin_invCN_y.*phin_invCN_y,2);
  dnlZ((D+3+M*D):(M+D+2+M*D)) = 0.5*w.*(tr_invCN_dCN-yt_invCN_dCN_invCN_y);
  
  varargout = {nlZ, dnlZ};
end
end