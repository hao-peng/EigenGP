% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
function [varargout] = EigenGP_Upd_kerBW(param, x, y, M, xs)
[N,D] = size(x);
% parameters to optimize
sn2 = exp(2*param(1)); % Varaince of observation noise
inv_ell = exp(-param(2:(D+1))); % inv_ell=1./ell;
eta = inv_ell.^2;          % eta is the precision
sf2 = exp(2*param(D+2));
W = exp(param(D+3:M+D+2)); % weights for basis functions Kbb = U*inv(W)*U'
B = reshape(param(M+D+3:M+D+2+M*D), M, D); % basis points
epsi =  1e-10; % diagonal added to kernel

% Compute part of the gram matrix
x_ell = scale_cols(x, inv_ell);
B_ell = scale_cols(B, inv_ell);
Kbb = sf2*exp(-0.5*sq_dist(B_ell'))+ epsi*eye(M);
Kxb = sf2*exp(-0.5*sq_dist(x_ell', B_ell'));

% Eigen decomposition
[Uq, Lambdaq] = eig(Kbb);
[Lambda, sort_ind] = sort(abs(diag(Lambdaq)),'descend');
U = real(Uq(:,sort_ind(1:M)));

W_ind = W>-Inf; % no prunning
L = sum(W_ind);
W = W(W_ind);
Lambda = Lambda(W_ind);
U = U(:,W_ind);

% Eigen functions evaluated at the training points
% O(N*M*M)
Phin = Kxb*U;
% A = sn2*I
% Q is used in the matrix inverse/determinant lemma
Q = diag(1./W) + Phin'*Phin/sn2;
% Q = CholQ * CholQ'
CholQ  = chol(Q, 'lower');
%invA_Phin_invCholQ = inv(A)*Phin*inv(CholQ')
invA_Phin_invCholQ = Phin/CholQ'/sn2;
% y_invA_Phin_invCholQ = y'*inv(A)*Phin*inv(CholQ)
y_invA_Phin_invCholQ = y'*invA_Phin_invCholQ;
%invCN_Phin = inv(CN)*Phin
invCN_Phin = Phin/sn2-invA_Phin_invCholQ*(invA_Phin_invCholQ'*Phin);
%U_W_Phin_invCN = U*diag(W)*Phin'*inv(CN)
U_W_Phin_invCN = scale_cols(U,W)*invCN_Phin';


% negative log likelihood 
nlZ = (y'*y/sn2 - y_invA_Phin_invCholQ*y_invA_Phin_invCholQ' + ...
     sum(log(diag(CholQ)))*2+sum(log(W))+log(sn2)*N + N*log(2*pi))/2;  
%nlZ = (sum(log(diag(CholQ)))*2+sum(log(W))+log(sn2)*N )/2;

%% Check number of outputs
if nargout == 1
    varargout = {nlZ};
elseif nargout == 2
    %invCNdiag = diag(inv(CN))
    invCNdiag = 1/sn2 - sum(invA_Phin_invCholQ.*invA_Phin_invCholQ,2);
    % y_invCN_Phin = y'*inv(CN)*Phin
    y_invCN_Phin = y'*invCN_Phin;
    %invCN_y = inv(CN)*y
    invCN_y = y/sn2 - invA_Phin_invCholQ*(invA_Phin_invCholQ'*y);
    %U_W_Phin_invCN_y_y_invCN = U*diag(W)*Phin'*invCN *y*y'*invCN
    U_W_Phin_invCN_y_y_invCN = U_W_Phin_invCN*y*invCN_y';

    
    %UVU is [U'*V1*U ... U'*VM*U] in S
    U_V_U = zeros(M, M*L);
    UU = kr(U,U);
    for j = 1 : L
        V = 1./(Lambda - Lambda(j));
        V(j) = 0;
         U_V_U(1:M, (j-1)*M+1:j*M) = reshape(UU*V, M, M);
    end
%     U_V_U2 = zeros(M, M*L);
%     for j = 1:L
%         V = 1./(Lambda - Lambda(j));
%         V(j) = 0;
%         U_V_U2(1:M, (j-1)*M+1:j*M) = scale_cols(U, V)*U';
%     end
    
    % Sigma_1 = U*W*Phin'*invCN.*Kxb'
    Sigma_1 = U_W_Phin_invCN .*Kxb';
    % Sigma_2 = U*inv(W)*Phin'*invCN*y*y'*invCN.*Kxb'
    Sigma_2 = U_W_Phin_invCN_y_y_invCN.*Kxb';
    
    %Kbx_invCN_Kxb_U_W = Kbx*inv(CN)*Kxb*U*inv(W)
    Kbx_invCN_Kxb_U_W = scale_cols(Kxb'*invCN_Phin, W);
    tildeS1 = scale_cols(U_V_U, reshape(Kbx_invCN_Kxb_U_W, M*L, 1))*kron(U',ones(M,1)).*Kbb;
    tildeS_tildeS1 = tildeS1+tildeS1';
    
    %Kbx_invCN_y_y_invCN_Kxb_U_W = Kbx*inv(CN)*y*y'*inv(CN)*Kxb*U*diag(W)
    Kbx_invCN_y_y_invCN_Kxb_U_W = scale_cols(Kxb'*invCN_y*(y'*invCN_Phin), W);
    tildeS2 = scale_cols(U_V_U, reshape(Kbx_invCN_y_y_invCN_Kxb_U_W, M*L, 1))*kron(U',ones(M,1)).*Kbb;
    tildeS_tildeS2 = tildeS2+tildeS2';
    
    B_eta = scale_cols(B, eta);
    x_eta = scale_cols(x, eta);
    B2 = B.*B;
    x2 = x.*x;
    
    % create memory for derivatives
    dnlZ = zeros(size(param));
    
    % gradient over observation noise i.e. log(sn)
    dnlZ(1) = sn2*(sum(invCNdiag)-invCN_y'*invCN_y);
    
    % gradient over kernel widths, i.e. log(ell)
    % eq10 = tr(invCN*dKxb*invKbb*Kbx)/d(eta)
    eq10 = 2*sum(B'.*(x'*Sigma_1'), 2)-B2'*sum(Sigma_1,2)-x2'*sum(Sigma_1,1)';
    % eq_11 = tr(invCN*Kxb*dinvKbb*Kbx)/d(eta)
    eq11 = -2*sum(B.*(tildeS_tildeS1*B),1)'+2*B2'*sum(tildeS_tildeS1, 1)';
    tr_invCN_dCN = 2*eq10+eq11;
    % eq10 = tr(invCN*y*y'*invCN*dKxb*invKbb*Kbx)/d(eta)
    eq10 = 2*sum(B'.*(x'*Sigma_2'), 2)-B2'*sum(Sigma_2,2)-x2'*sum(Sigma_2,1)';
    % eq_11 = tr(invCN*y*y'(invCN*Kxb*dinvKbb*Kbx)/d(eta)
    eq11 = -2*sum(B.*(tildeS_tildeS2*B),1)'+2*B2'*sum(tildeS_tildeS2, 1)';
    yt_invCN_dCN_invCN_y = 2*eq10+eq11;
    dnlZ(2:D+1) = -eta.*(tr_invCN_dCN-yt_invCN_dCN_invCN_y)/2;
    %dnlZ(2:D+1) = -eta.*(yt_invCN_dCN_invCN_y)/2;

    
    % gradient over kernel amplifier, i.e. log(sf)
    dnlZ(D+2) = 2*sum(sum(U_W_Phin_invCN'.*Kxb, 1))-...
        2*sum(sum(U_W_Phin_invCN_y_y_invCN'.*Kxb,1));    
    
    % gradient over ln(W)
    % eqA = tr(invCN*Phin*d(invW)*Phin')/d(invW)
    eqA = sum(Phin.*invCN_Phin, 1)';
    % eqC = tr(t'*inv(CN)*d(CN)*inv(CN)*t)/d(invW)
    eqC = (y_invCN_Phin.*y_invCN_Phin)';
    % d(nlZ)/d(ln(W)) = -d(nlZ)/d(W)./W
    dnlZdW = zeros(M,1);
    dnlZdW(W_ind) = (eqA-eqC).*W/2;
    dnlZ((D+3):(M+D+2)) = dnlZdW;
    
    
    % gradient over basis B
    % eq4 = tr(invCN*d(Kxb)*invKbb*Kxb')/dB
    eq4 = Sigma_1*x_eta-repmat(sum(Sigma_1,2),1,D).*B_eta;
    % eq18 = tr(invCN*Kxb*d(inv(Kbb))*Kxb')/dB
    eq18 = -2*(tildeS_tildeS1)*B_eta+2*repmat(sum(tildeS_tildeS1,2),1,D).*B_eta;
    %tmp = tildeS1 + tildeS1'-diag(diag(tildeS1));
    %eq18 = -4*tmp*B_eta+4*repmat(sum(tmp,2),1,D).*B_eta;
    tr_invCN_dCN = 2*eq4+eq18;
    %tr_invCN_dCN = eq18;
    % eq4 = tr(invCN*y*y'*invCN*d(Kxb)*invKbb*Kxb')/dB
    eq4 = Sigma_2*x_eta-repmat(sum(Sigma_2,2),1,D).*B_eta;
    % eq18 = tr(invCN*y*y'*invCN*Kxb*d(inv(Kbb))*Kxb')/dB
    eq18 = -2*(tildeS_tildeS2)*B_eta+2*repmat(sum(tildeS_tildeS2,2),1,D).*B_eta;
    %tmp = tildeS2 + tildeS2'-diag(diag(tildeS2));
    %eq18 = -4*tmp*B_eta+4*repmat(sum(tmp,2),1,D).*B_eta;
    yt_invCN_dCN_invCN_y = 2*eq4+eq18;
    dnlZ(M+D+3:M+D+2+M*D) = reshape(tr_invCN_dCN-yt_invCN_dCN_invCN_y, M*D, 1)/2;
    %dnlZ(M+D+3:M+D+2+M*D) = reshape(tr_invCN_dCN, M*D, 1)/2;
    
    varargout = {nlZ dnlZ};
else
    alpha = U_W_Phin_invCN*y;
    beta = U_W_Phin_invCN*Phin*scale_cols(U, W)';
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
    
    % process minibatches of test cases to save memory
    while nact<Ns
        % Data points to process
        id = (nact+1):min(nact+nperbatch,Ns);
        % Cross Kernel Matrix between the testing points and basis points
        % Ksb = sf2*exp(-0.5*sq_dist(scale_cols(xs(id,:),inv_ell)', B_ell'));
        xs_ell = scale_cols(xs(id,:), inv_ell);
        Ksb = sf2*exp(-0.5*sq_dist(xs_ell', B_ell'));
        % Eigen functions on testing points
        % cross-covariances. k in (6.67)
        % diagonal covariance. c - sn2 in (6.67)
        % Predictive mean
        mu(id) = Ksb * alpha;
        % Predictive variance
                
        % S2temp = sum(Ksb/Kbb.*Ksb, 2)+epsi+sn2-sum(Ksb*beta.*Ksb, 2);
        S2(id) = sum(Ksb*gamma.*Ksb, 2)+epsi+sn2-sum(Ksb*beta.*Ksb, 2);
        % if any(S2<0),   S2(S2<0)';    end
        
        nact = id(end);
    end
    varargout = {nlZ mu S2};
end
end