% UI for EigenGP which only update B and kernel width
% [NMSE, mu, S2, NMLP, post] when ys is not []
% [mu, S2, post] when ys is []
%

function [varargout]  = EigenGP_Upd_kerB_UI(x, y, xs, ys, M, opt)
[N,D] = size(x);

optDefault = struct('cov',log([ones(D, 1)/4; 1]), 'lik', log(0.1),'L',M,...
    'optimization', 'none','nIter',1000, 'numKmeans', 10, 'fkmeans', true);
% optDefault = struct('lik', log(0.1),'L',M,...

% Overwrite default values if they are specified by the input
opt = mergestruct(opt,optDefault);

% Regression assumes zero-mean functions, substract mean
meanp=mean(y);
%y=y-meanp;


if size(opt.cov, 2) > 1
    opt.cov = opt.cov';
end


if 0 %~isfield(opt, 'B')
    best_sumd = Inf;
    for i = 1:1
        %use kmeans with both x and y
        [IDX, By] = fkmeans([x 0*y]', M);
        sumd = 0;
        for j = 1:N
           sumd = sum(([x(j,:), 0*y(j)]-By(IDX(j),:)).^2);
        end
        if sumd < best_sumd
            opt.B = By(:,1:end-1);
        end
    end
end


if ~isfield(opt, 'B')
    best_nlZ = Inf;
    for i = 1:10
        %use kmeans with both x and y
        [IDX, By] = fkmeans([x 0*y]', M);
        B = By(:,1:end-1);
        
        param = [opt.lik; opt.cov; reshape(B,M*D,1)];
        [nlZ] = EigenGP_Upd_kerB(param, x, y, M, xs);
        if nlZ < best_nlZ
           best_nlZ = nlZ;
           opt.B = B;
        end
    end
    %opt.B = x(randsample(N,M),:);
end


if 0 %~isfield(opt, 'B')
    %use kmeans with both x and y
    [IDX, By] = fkmeans([x 0.1*y]', M);
    opt.B = By(:,1:end-1);
end

param = [opt.lik; opt.cov; reshape(opt.B,M*D,1)];
[new_param] = minimize(param, @(param) EigenGP_Upd_kerB(param, x, y, M), opt.nIter);
opt.lik = new_param(1);
opt.cov = new_param(2:D+2);
opt.B = reshape(new_param(D+3:D+2+M*D),M,D);

%save('demo.mat');

[nlZ mu S2] = EigenGP_Upd_kerB(new_param, x, y, M, xs);
post.nlZ = nlZ;
post.opt = opt;
%mu = mu + meanp;
if isempty(ys)
    varargout = {mu, S2, post};
else
    % Test Mean Negative Log Probability
    % S2(S2<0) = exp(2*opt.lik);
    if any(S2<0), warning('some predictive variance is smalelr than 0'); end
    %S2 = max(S2, exp(2*opt.lik));
    NMLP = -0.5*mean(-(mu-ys).^2./S2-log(2*pi)-log(S2));
    % Test Normalized Mean Square Error
    NMSE = mean((mu-ys).^2)/mean((meanp-ys).^2);
    varargout = {NMSE, mu, S2, NMLP, post};
end
end
