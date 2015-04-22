% Update W
function [varargout] = EigenGP_Upd_W(param, x, y, opt, xs)
[N,D] = size(x);
B = opt.B;
M = size(B, 1);
lik = opt.lik;
cov = opt.cov;
lnW = param(1:M);
W = exp(lnW);

% kernel parameters for ARD kernel
ell = exp(cov(1:D));  
sf2 = exp(2*cov(D+1));
% diagonal added to kernel
epsi = 1e-10;
% Varaince of observation noise
sn2 = exp(2*lik);
% modified ARD kernel matrix on basis points
% Kernel Matrix of basis points
scaledB = scale_rows(B',1./ell);
scaledx = scale_rows(x',1./ell);
% modified ARD kernel matrix on basis points
% Kernel Matrix of basis points
Kbb = sf2*exp(-sq_dist(scaledB)/2)+ epsi*eye(M);
% Cross Kernel Matrix between the training points and basis points
Kxb = sf2*exp(-sq_dist(scaledx,scaledB)/2);
% Eigen decomposition
[Uq, Lambdaq] = eig(Kbb);


diag_lambdaq = real(diag(Lambdaq));

[Lambdaq_sort,sort_ind] = sort(abs(diag_lambdaq),'descend');
U = real(Uq(:,sort_ind(1:M)));
U = scale_cols(U, 1./Lambdaq_sort);

W_ind = W>1e-10;
W = W(W_ind);
U = U(:,W_ind);

% Eigen functions evaluated at the training points
% O(N*M*M)
Phin = Kxb*U;
% the diagnoal part \sigma^2*I
% O(N*M)
% A = sn2*I
A = sn2*ones(N,1);
invA = 1./A;
% Precomputation of invA * Phin
% O(N*M)
invA_Phin = scale_rows(Phin,invA);
% Precomputation of low rank part
%Q = diag(1./W(indW)) + Phin'*invA_Phin;
% O(M*N*M)
Q = diag(1./W) + Phin'*invA_Phin;
% O(N*M)
y_invA_Phin = y'*invA_Phin;
% O(M^3+M*M*N);
invQ_Phin_invA = Q\invA_Phin';
% O(M*N)
invQ_Phin_invAy = invQ_Phin_invA*y;
% -log marg lik
% O(N+M^3)
nlZ = y'.*invA'*y/2 - y_invA_Phin*invQ_Phin_invAy/2 + ...
    (logdet(Q)+sum(log(W))+sum(log(A)))/2 + N*log(2*pi)/2;  
nlZ = real(nlZ);

if nargout == 1
    varargout = {nlZ};
elseif nargout == 2
    % allocate memorey for derivative
    nlZd = zeros(size(param));
    Phin2 = Phin.*Phin;
    %Phin2_modi(diff0_ind) = 0;
    % eqA = tr(invCN*PhindW*Phin')
    eqA = sum(scale_rows(Phin2, invA)-Phin.*(invA_Phin*(invQ_Phin_invA*Phin)),1);
    invCN_y = invA.*y-invA_Phin*(invQ_Phin_invA*y);
    Phin_invCN_y = Phin'*invCN_y;
    % eqC = tr(t'*invCN*dCN*invCN*t)
    eqC = (Phin_invCN_y.*Phin_invCN_y)';
    nlZd(W_ind) = (eqA-eqC)'.*W/2;
    nlZd = real(nlZd);
    varargout = {nlZ nlZd};
else
    % Precomputation for mean
    %alpha = U*(diag(W) * Phin'* (diag(invA)*y- diag(invA)*Phin*(invQ_Phin_invA*y)));
    alpha = U * invQ_Phin_invAy;
    
    % k*CNinv*k' = k*U*W*Phin'*CNinv*Phin*W*U'*k
    % beta = U*W*Phin'*CNinv*Phin*W*U
    beta = scale_cols(U,W)*U' - U*(Q\U');
    %beta = U*diag(W)*Phin'*((diag(A)+Phin*diag(W)*Phin')\Phin)*diag(W)*U';
    %beta = 0;
    gamma = scale_cols(U, W)*U';
    
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
    
    %PAP = Phin' * invA_Phin;
    %scale_cols(PAP,W)*U';
    % process minibatches of test cases to save memory
    while nact<Ns
        % Data points to process
        id = (nact+1):min(nact+nperbatch,Ns);
        % Cross Kernel Matrix between the testing points and basis points
        %Ksb = covSEard(cov, xs(id,:), B);
        Ksb = sf2*exp(-sq_dist(scale_rows(xs(id,:)', 1./ell), scaledB)/2);
        % Eigen functions on testing points
        %Phis = Ksb*U;
        % cross-covariances. k in (6.67)
        % diagonal covariance. c - sn2 in (6.67)
        %Kss = covSEard(cov, xs(id,:), 'diag') + epsi;
        % Predictive mean
        mu(id) = Ksb * alpha;
        
        % Predictive variance
        S2(id) = sum(Ksb*gamma.*Ksb, 2)+epsi+sn2-sum(Ksb*beta.*Ksb, 2);
        
        nact = id(end);
    end
    varargout = {nlZ mu S2};
end
end