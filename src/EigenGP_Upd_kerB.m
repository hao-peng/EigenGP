% no decomposition for compute derivatives. Update B and kernel paramters
function [varargout] = EigenGP_Upd_kerB(param, x, y, M, xs)
[N,D] = size(x);

% parameters to optimize
lik = param(1); %  parameter in likelhiood function. here it is
cov = param(2:(D+2)); % par in covariance function
B = reshape(param((D+3):(D+2+M*D)), M, D);
% kernel parameters for ARD kernel
% ell = exp(cov(1:D));    % ell is the standard deviation of Gaussian kernel
% eta = ell.^-2;          % eta is the precision 
% inv_ell=1./ell;
inv_ell = exp(-cov(1:D));    % ell is the standard deviation of Gaussian kernel
eta = inv_ell.^2;          % eta is the precision 
sf2 = exp(2*cov(D+1));
% diagonal added to kernel
epsi =  1e-10;
% Varaince of observation noise
sn2 = exp(2*lik);
% modified ARD kernel matrix on basis points
% Kernel Matrix of basis points

if 1
  B_ell = scale_cols(B, inv_ell);
  Kbb = sf2*exp(-0.5*sq_dist(B_ell'))+ epsi*eye(M);
  % Cross Kernel Matrix between the training points and basis points
  Kxb = sf2*exp(-0.5*sq_dist(scale_cols(x,inv_ell)', B_ell'));
else
  Kbb = sf2*exp(-sq_dist(diag(1./ell)*B')/2)+ epsi*eye(M);
  Kxb = sf2*exp(-sq_dist(diag(1./ell)*x',diag(1./ell)*B')/2);
end

Q = Kbb*sn2 + Kxb'*Kxb;
% For reuse:
if 0
  Chol_Q = linfactor(Q);
  iKbb_Kbx_iCN = linfactor(Chol_Q, Kxb');
else
  Chol_Q  = chol(Q, 'lower');
  lower.LT = true ;   upper.LT = true ;    upper.TRANSA = true ;
  iKbb_Kbx_iCN = linsolve (Chol_Q, linsolve (Chol_Q , Kxb', lower), upper);
end

%tmp_iKbb_Kxb_iCN = Q\Kxb';
if 0
Chol_Q = chol(Q);
opts.UT = true; % opts.TRANSA = false;
iKbb_Kxb_iCN = Chol_Q'*linsolve(Chol_Q, Kxb',opts); % M by N
end

iKbb_Kbx_iCN_Kxb = iKbb_Kbx_iCN*Kxb;
% invKbb_Kbx = Kbb\Kxb';  M by N
if 0
  Chol_Kbb = linfactor(Kbb);
  iKbb_Kbx_iCN_Kxb_iKbb = linfactor(Chol_Kbb, iKbb_Kbx_iCN_Kxb'); % same as iKbb*Kxb*inv(CN)*Kxb*iKbb
else
  Chol_Kbb  = chol(Kbb, 'lower');
  iKbb_Kbx_iCN_Kxb_iKbb = linsolve (Chol_Kbb, linsolve (Chol_Kbb , iKbb_Kbx_iCN_Kxb', lower), upper);        
  iKbb_Kbx_iCN_Kxb_iKbb = (iKbb_Kbx_iCN_Kxb_iKbb +  iKbb_Kbx_iCN_Kxb_iKbb')/2;
  if 0 % ~isposdef(iKbb_Kbx_iCN_Kxb_iKbb)
    warning('iKbb_Kbx_iCN_Kxb_iKbb is not isposdef');
  end
end

% Sigma_1 = invKbb*Kxb'*invCN.*Kxb'
Sigma_1 = iKbb_Kbx_iCN.*Kxb';
% tilde_R_1 = (invKbb*Kxb'*invCN*Kxb*invKbb).*Kbb, where invKbb*Kxb*invCN = invQ*Kbx*invA
tilde_R_1 = iKbb_Kbx_iCN_Kxb_iKbb.*Kbb;

B_eta = scale_cols(B, eta); 

iKbb_Kbx_iCN_y = iKbb_Kbx_iCN*y;
invCN_y = (y - Kxb*iKbb_Kbx_iCN_y)/sn2;

%invCN_y = invA.*y-Kbx_invA'*(iKbb_Kxb_iCN*y);

invCNdiag = (1 - sum(Kxb.*iKbb_Kbx_iCN', 2)) / sn2;

% Sigma_3 = invKbb*Kxb'*invCN*y*y'*invCN.*Kxb'
Sigma_3 = (iKbb_Kbx_iCN_y*invCN_y').*Kxb';

% tilde_R_3 = (invKbb*Kxb'*invCN*y*y'*invCN*Kxb*invKbb).*Kbb
tilde_R_3 = (iKbb_Kbx_iCN_y*iKbb_Kbx_iCN_y').*Kbb;
% x2 = x.*x
x2 = x.*x;
% B2 = B.*B
B2 = B.*B;

%% -log marg lik
%nlZ = ( sum(log(diag(Chol_Q.L))) - sum(log(diag(Chol_Kbb.L)))) + log(sn2)*N/2 + y'*invCN_y/2 + log(2*pi)*N/2;  
%nlZ = ( sum(log(diag(Chol_Q))) - sum(log(diag(Chol_Kbb)))) + log(sn2)*N/2 + y'*invCN_y/2 + log(2*pi)*N/2;  
nlZ = ( sum(log(diag(Chol_Q))) - sum(log(diag(Chol_Kbb)))) + log(sn2)*(N-M)/2 + y'*invCN_y/2 + log(2*pi)*N/2;  
% nlZ2  = logdet(Kxb*(Kbb\Kxb')+sn2*eye(N))/2  + y'*invCN_y/2 + log(2*pi)*N/2; nlZ-nlZ2 % Sanity check for evidence. 
%nlZ = (logdet(Q)  -logdet(Kbb)+sum(log(A)))/2 + y'*invCN_y/2 + N*log(2*pi)/2;  
%%nlZ = (logdet(Q)-logdet(Kbb)+sum(log(A)))/2;
assert(isreal(nlZ));
% nlZ = real(nlZ);

%% Check number of outputs
if nargout == 1
    varargout = {nlZ};
elseif nargout == 2
    nlZd = zeros(size(param));
    
    %% gradient over observation noise i.e. log(sn)
    nlZd(1) = sn2*(sum(invCNdiag)-invCN_y'*invCN_y);
    
    %% gradient over kernel widths, i.e. log(ell)
    %% for tr(invCN * d(tilde_Kbb))
    % eq10_1 = tr(invCN*dKxb*invKbb*Kbx)
    eq10_1 = 2*sum(B'.*(x'*Sigma_1'), 2)-B2'*sum(Sigma_1,2)-x2'*sum(Sigma_1,1)';
    % eq_11_1 = tr(invCN*Kxb*dinvKbb*Kbx)
    eq11_1 = -2*sum(B.*(tilde_R_1*B),1)'+2*B2'*sum(tilde_R_1, 1)';
    
    tr_invCN_dCN = 2*eq10_1+eq11_1;
    
    % for tr(invCN*y*y'*invCN*d(tilde_Kbb))
    % eq10_3 = tr(invCN*dKxb*invKbb*Kbx)
    eq10_3 = 2*sum(B'.*(x'*Sigma_3'), 2)-B2'*sum(Sigma_3,2)-x2'*sum(Sigma_3,1)';
    % eq_11_3 = tr(invCN*Kxb*dinvKbb*Kbx)
    eq11_3 = -2*sum(B.*(tilde_R_3*B),1)'+2*B2'*sum(tilde_R_3, 1)';
    
    yt_invCN_dCN_invCN_y = 2*eq10_3+eq11_3;
    nlZd(2:1+D) = -eta.*(tr_invCN_dCN-yt_invCN_dCN_invCN_y)/2;       
    
    %% gradient over kernel amplifier, i.e. log(sf)
    nlZd(D+2) = sum(1-sn2*invCNdiag) - sum(invCN_y.*y)+sn2*sum(invCN_y.*invCN_y);
            
    %% gradient over basis
    %% for tr(invCN * d(tilde_Kbb))
    % eq3_1 = tr(invCN*dKxb*invKbb*Kxb')
    eq3_1 = scale_cols(Sigma_1*x,eta)-repmat(sum(Sigma_1,2),1,D).*B_eta;
    % eq5_1 = tr(invCN*Kxb*dinvKbb*Kxb')
    eq5_1 = -2*scale_cols(tilde_R_1*B,eta)+2*repmat(sum(tilde_R_1,2),1,D).*B_eta;
    tr_invCN_dCN = 2*eq3_1+eq5_1;
    
    %% for tr(invCN*y*y'*invCN*d(tilde_Kbb))
    % eq3_3 = tr(invCN*y*y'*invCN*dKxb*invKbb*Kxb')
    eq3_3 = scale_cols(Sigma_3*x,eta)-repmat(sum(Sigma_3,2),1,D).*B_eta;
    % eq5_3 = tr(invCN*y*y'*invCN*Kxb*dinvKbb*Kxb')
    eq5_3 = -2*scale_cols(tilde_R_3*B,eta)+2*repmat(sum(tilde_R_3,2),1,D).*B_eta;      
    yt_invCN_dCN_invCN_y = 2*eq3_3+eq5_3;
    
    %%
    nlZd((D+3):(D+2+M*D)) = reshape(tr_invCN_dCN-yt_invCN_dCN_invCN_y, M*D, 1)/2;
    nlZd = real(nlZd);
    varargout = {nlZ nlZd};
else
    %beta = invKbb*Kbx*invCN*Kxb*invKbb
    %beta = scale_cols(invKbb_Kbx,invA)*invKbb_Kbx'-(invKbb_Kbx*Kbx_invA')*(iKbb_Kxb_iCN*invKbb_Kbx');
    beta = iKbb_Kbx_iCN_Kxb_iKbb;
    % number of test data points
    Ns = size(xs,1);
    % number of data points per mini batch
    nperbatch = 1000;
    % number of already processed test data points
    nact = 0;
    % Allocate memory for predictive mean
    mu = zeros(Ns,1);
    % Allocate memory for predictive variance
    S2 = mu;
    
    % process minibatches of test cases to save memory
    while nact<Ns
        % Data points to process
        id = (nact+1):min(nact+nperbatch,Ns);
        % Cross Kernel Matrix between the testing points and basis points
        % Ksb = sf2*exp(-0.5*sq_dist(scale_cols(xs(id,:),inv_ell)',
        % B_ell'));
        Ksb = covSEard(cov, xs(id,:), B);
        % Eigen functions on testing points
        %Phis = Ksq*U;
        % cross-covariances. k in (6.67)
        % diagonal covariance. c - sn2 in (6.67)
        %Kss = covSEard(cov, xs(id,:), 'diag')+ epsi;
        % Predictive mean
        mu(id) = Ksb * iKbb_Kbx_iCN_y;
        % Predictive variance
                
        % S2temp = sum(Ksb/Kbb.*Ksb, 2)+epsi+sn2-sum(Ksb*beta.*Ksb, 2);
        iKbb_Kbs = linsolve (Chol_Kbb, linsolve (Chol_Kbb , Ksb', lower), upper);        
        S2(id) = sum(iKbb_Kbs'.*Ksb, 2) - sum(Ksb*beta.*Ksb, 2)  + sn2;
        % if any(S2<0),   S2(S2<0)';    end
        
        nact = id(end);
    end
    varargout = {nlZ mu S2};
end
end