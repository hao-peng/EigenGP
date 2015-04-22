% UI for EigenGP which only update B and kernel width
% [NMSE, mu, S2, NMLP, post] when ys is not []
% [mu, S2, post] when ys is []
%

function [varargout]  = EigenGP_unbnd_delta_kerB_UI(x, y, xs, ys, M, opt)
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

if ~isfield(opt, 'B')
    [IDX, B] = fkmeans(x', M);
    opt.B = B;
end


param = [opt.lik; opt.cov; reshape(opt.B,M*D,1)];
[new_param] = minimize(param, @(param) EigenGP_unbnd_delta_kerB(param, x, y, M), opt.nIter);
opt.lik = new_param(1);
opt.cov = new_param(2:D+2);
opt.B = reshape(new_param(D+3:D+2+M*D),M,D);

%save('sinc_updateB.mat');

[nlZ mu S2] = EigenGP_unbnd_delta_kerB(new_param, x, y, M, xs);
post.nlZ = nlZ;
post.opt = opt;
%mu = mu + meanp;
if isempty(ys)
    varargout = {mu, S2, post};
else
    % Test Mean Negative Log Probability
    NMLP = -0.5*mean(-(mu-ys).^2./S2-log(2*pi)-log(S2));
    % Test Normalized Mean Square Error
    NMSE = mean((mu-ys).^2)/mean((meanp-ys).^2);
    varargout = {NMSE, mu, S2, NMLP, post};
end
end
