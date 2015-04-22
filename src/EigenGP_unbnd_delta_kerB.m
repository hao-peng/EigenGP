% no decomposition for compute derivatives. Update B and kernel paramters
function [varargout] = EigenGP_unbnd_delta_kerB(param, x, y, M, xs)
[N,D] = size(x);

% parameters to optimize
lik = param(1);
cov = param(2:(D+2));
B = reshape(param((D+3):(D+2+M*D)), M, D);
% kernel parameters for ARD kernel
ell = exp(cov(1:D));  
eta = ell.^-2;
sf2 = exp(2*cov(D+1));
% diagonal added to kernel
epsi = 1e-10;
% Varaince of observation noise
sn2 = exp(2*lik);
% modified ARD kernel matrix on basis points
% Kernel Matrix of basis points
Kbb = sf2*exp(-sq_dist(diag(1./ell)*B')/2)+ epsi*eye(M);
% Cross Kernel Matrix between the training points and basis points
Kxb = sf2*exp(-sq_dist(diag(1./ell)*x',diag(1./ell)*B')/2);
% Diagonals of the Kernel Matrix of the training points
Kxxdiag = sf2*ones(N,1)+epsi;

invKbb_Kbx = Kbb\Kxb';
% A = sn2*I + diag(diag(Kbb) - diag(Kxb*invKbb*Kxb'))
A = sn2 + Kxxdiag - sum(Kxb.*invKbb_Kbx', 2);
invA = 1./A;
% Q = Kbb + Kxb*diag(invA)*Kxb'
Q = Kbb + Kxb'*scale_rows(Kxb, invA);
% invCN = invA - invA*Kxb*invQ*Kxb'*invA
% For reuse:
Kbx_invA = scale_cols(Kxb', invA);
invQ_Kbx_invA = Q\Kbx_invA;
B_eta = scale_cols(B, eta);
invCN_y = invA.*y-Kbx_invA'*(invQ_Kbx_invA*y);
invKbb_Kbx_invCN_y = invKbb_Kbx*invCN_y;
invCNdiag = invA - sum(Kbx_invA'.*invQ_Kbx_invA', 2);
% Pdiag = diag(invCN*y*y'*invCN)
Pdiag = invCN_y.*invCN_y;
% Sigma_1 = invKbb*Kxb'*invCN.*Kxb'
Sigma_1 = invQ_Kbx_invA.*Kxb';
% Sigma_2 = invKbb.*I*Kxb'*invCN.*Kxb'
Sigma_2 = scale_cols(invKbb_Kbx, invCNdiag).*Kxb';
% Sigma_3 = invKbb*Kxb'*invCN*y*y'*invCN.*Kxb'
Sigma_3 = invKbb_Kbx_invCN_y*invCN_y'.*Kxb';
% Sigma_4 = Pdiag*Kxb'*invCN.*Kxb'
Sigma_4 = scale_cols(invKbb_Kbx, Pdiag).*Kxb';
% tilde_R_1 = (invKbb*Kxb'*invCN*Kxb*invKbb).*Kbb, where invKbb*Kxb*invCN = invQ*Kbx*invA
tilde_R_1 = (invQ_Kbx_invA*invKbb_Kbx').*Kbb;
% tilde_R_2 = (invKbb*Kxb'*(invCN.*I)*Kxb*invKbb).*Kbb
tilde_R_2 = (scale_cols(invKbb_Kbx,invCNdiag)*invKbb_Kbx').*Kbb;
% tilde_R_3 = (invKbb*Kxb'*invCN*y*y'*invCN*Kxb*invKbb).*Kbb
tilde_R_3 = (invKbb_Kbx_invCN_y*invKbb_Kbx_invCN_y').*Kbb;
% tilde_R_4 = (invKbb*Kxb'*(Pdiag)*Kxb*invKbb).*Kbb
tilde_R_4 = (scale_cols(invKbb_Kbx,Pdiag)*invKbb_Kbx').*Kbb;
% x2 = x.*x
x2 = x.*x;
% B2 = B.*B
B2 = B.*B;

%% -log marg lik
nlZ = (logdet(Q)-logdet(Kbb)+sum(log(A)))/2 + y'*invCN_y/2 + N*log(2*pi)/2;  
%nlZ = (logdet(Q)-logdet(Kbb)+sum(log(A)))/2;
nlZ = real(nlZ);


%% Check number of outputs
if nargout == 1
    varargout = {nlZ};
elseif nargout == 2
    nlZd = param;
    
    %% gradient over observation noise i.e. log(sn)
    nlZd(1) = sn2*(sum(invCNdiag)-invCN_y'*invCN_y);
    
    %% gradient over kernel widths, i.e. log(ell)
    %% for tr(invCN * d(tilde_Kbb))
    % eq10_1 = tr(invCN*dKxb*invKbb*Kbx)
    eq10_1 = 2*sum(B'.*(x'*Sigma_1'), 2)-B2'*sum(Sigma_1,2)-x2'*sum(Sigma_1,1)';
    % eq_11_1 = tr(invCN*Kxb*dinvKbb*Kbx)
    eq11_1 = -2*sum(B.*(tilde_R_1*B),1)'+2*B2'*sum(tilde_R_1, 1)';
    
    %% for tr(invCN * d(I.*tilde_Kbb)) by replacing invCN with invCNdiag
    % eq10_2 = tr((I.*invCN)*dKxb*invKbb*Kbx)
    eq10_2 = 2*sum(B'.*(x'*Sigma_2'), 2)-B2'*sum(Sigma_2,2)-x2'*sum(Sigma_2,1)';
    % eq11_2 = tr((I.*invCN)*Kxb*dinvKbb*Kbx)
    eq11_2 = -2*sum(B.*(tilde_R_2*B),1)'+2*B2'*sum(tilde_R_2, 1)';
    
    %%
    tr_invCN_dCN = (2*eq10_1+eq11_1)-(2*eq10_2+eq11_2);
    
    %% for tr(invCN*y*y'*invCN*d(tilde_Kbb))
    % eq10_3 = tr(invCN*dKxb*invKbb*Kbx)
    eq10_3 = 2*sum(B'.*(x'*Sigma_3'), 2)-B2'*sum(Sigma_3,2)-x2'*sum(Sigma_3,1)';
    % eq_11_3 = tr(invCN*Kxb*dinvKbb*Kbx)
    eq11_3 = -2*sum(B.*(tilde_R_3*B),1)'+2*B2'*sum(tilde_R_3, 1)';
    
    %% for tr(invCN*y*y'*invCN* d(I.*tilde_Kbb))
    % eq10_4 = tr((I.*invCN)*dKxb*invKbb*Kbx)
    eq10_4 = 2*sum(B'.*(x'*Sigma_4'), 2)-B2'*sum(Sigma_4,2)-x2'*sum(Sigma_4,1)';
    % eq11_4 = dtr((I.*invCN)*Kxb*dinvKbb*Kbx)
    eq11_4 = -2*sum(B.*(tilde_R_4*B),1)'+2*B2'*sum(tilde_R_4, 1)';
    
    %%
    yt_invCN_dCN_invCN_y = (2*eq10_3+eq11_3)-(2*eq10_4+eq11_4);
    %% 
    nlZd(2:1+D) = -eta.*(tr_invCN_dCN-yt_invCN_dCN_invCN_y)/2;
    
    
    
    %% gradient over kernel amplifier, i.e. log(sf)
    % eq10_1 = tr(invCN*dKxb*invKbb*Kbx)
    nlZd(D+2) = sum(1-sn2*invCNdiag) - sum(invCN_y.*y)+sn2*sum(invCN_y.*invCN_y);
    
    
    %% gradient over basis
    %% for tr(invCN * d(tilde_Kbb))
    % eq3_1 = tr(invCN*dKxb*invKbb*Kxb')
    eq3_1 = scale_cols(Sigma_1*x,eta)-repmat(sum(Sigma_1,2),1,D).*B_eta;
    % eq5_1 = tr(invCN*Kxb*dinvKbb*Kxb')
    eq5_1 = -2*scale_cols(tilde_R_1*B,eta)+2*repmat(sum(tilde_R_1,2),1,D).*B_eta;
    
    
    %% for tr(invCN * d(I.*tilde_Kbb)) by replacing invCN with invCNdiag
    % eq3_2 = tr(invCN.*I*dKxb*invKbb*Kxb')
    eq3_2 = scale_cols(Sigma_2*x,eta)-repmat(sum(Sigma_2,2),1,D).*B_eta;
    % eq5_2 = tr(invCN.*I*Kxb*dinvKbb*Kxb')
    eq5_2 = -2*scale_cols(tilde_R_2*B,eta)+2*repmat(sum(tilde_R_2,2),1,D).*B_eta;
    
    %%
    tr_invCN_dCN = (2*eq3_1+eq5_1)-(2*eq3_2+eq5_2);
    
    
    %% for tr(invCN*y*y'*invCN*d(tilde_Kbb))
    % eq3_3 = tr(invCN*y*y'*invCN*dKxb*invKbb*Kxb')
    eq3_3 = scale_cols(Sigma_3*x,eta)-repmat(sum(Sigma_3,2),1,D).*B_eta;
    % eq5_3 = tr(invCN*y*y'*invCN*Kxb*dinvKbb*Kxb')
    eq5_3 = -2*scale_cols(tilde_R_3*B,eta)+2*repmat(sum(tilde_R_3,2),1,D).*B_eta;
    
    %% for tr(invCN*y*y'*invCN* d(I.*tilde_Kbb))
    % eq3_4 = tr(Pdiag*dKxb*invKbb*Kxb')
    eq3_4 = scale_cols(Sigma_4*x,eta)-repmat(sum(Sigma_4,2),1,D).*B_eta;
    % eq5_4 = tr(Pdiag*Kxb*dinvKbb*Kxb')
    eq5_4 = -2*scale_cols(tilde_R_4*B,eta)+2*repmat(sum(tilde_R_4,2),1,D).*B_eta;
    
    
    %%
    yt_invCN_dCN_invCN_y = (2*eq3_3+eq5_3)-(2*eq3_4+eq5_4);
    
    %%
    nlZd((D+3):(D+2+M*D)) = reshape(tr_invCN_dCN-yt_invCN_dCN_invCN_y, M*D, 1)/2;
    nlZd = real(nlZd);
    varargout = {nlZ nlZd};
else
    %beta = invKbb*Kbx*invCN*Kxb*invKbb
    beta = scale_cols(invKbb_Kbx,invA)*invKbb_Kbx'-(invKbb_Kbx*Kbx_invA')*(invQ_Kbx_invA*invKbb_Kbx');
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
        Ksb = covSEard(cov, xs(id,:), B);
        % Eigen functions on testing points
        %Phis = Ksq*U;
        % cross-covariances. k in (6.67)
        % diagonal covariance. c - sn2 in (6.67)
        Kss = covSEard(cov, xs(id,:), 'diag');
        % Predictive mean
        mu(id) = Ksb * invKbb_Kbx_invCN_y;
        % Predictive variance
        S2(id) = Kss + sn2 - sum(Ksb*beta.*Ksb, 2);
        
        nact = id(end);
    end
    varargout = {nlZ mu S2};
end
end