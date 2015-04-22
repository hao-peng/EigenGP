% opt.cov  D characteristic length scales + 1 singal variance
% [NMSE, mu, S2, NMLP, post] when ys is not []
% [mu, S2, post] when ys is []
%

function [varargout]  = EigenGP_Upd_W_UI(x, y, xs, ys, M, opt)
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
    % use kmeans with both x and y
    [IDX, By] = fkmeans([x y]', M);
    opt.B = By(:,1:end-1);
end


%% initial W if it is now assigned
ell = exp(opt.cov(1:D));  
sf2 = exp(2*opt.cov(D+1));
epsi = 1e-10;
Kbb = sf2*exp(-sq_dist(diag(1./ell)*opt.B')/2)+ epsi*eye(M);
Lambdaq = eig(Kbb);
Lambdaq_sort = sort(abs(Lambdaq),'descend'); 
if ~isfield(opt, 'lnW')
    opt.lnW = log(Lambdaq_sort);
end

%save('sinc_updateW.mat');
%% optimizer over W
if 1
param = [opt.lnW];
%options.maxIter = opt.nIter;
%[new_param] = minConf_TMP(@(param) EigenGP_Upd_W(param, x, y, opt),param,-inf(M,1),log(2*Lambdaq_sort),options);
[new_param] = minimize(param, @(param) EigenGP_Upd_W(param, x, y, opt), opt.nIter);
% if opt.nIter < 0
%     options = optimset('MaxFunEvals', -opt.nIter, 'Display', 'iter');
% else
%     options = optimset('MaxIter', opt.nIter, 'Display', 'iter');
% end
% [new_param] = fminunc(@(param) EigenGP_Upd_W(param, x, y, opt), param, options);
else
if opt.nIter < 0
    options = optimset('MaxFunEvals', -opt.nIter, 'Display', 'iter');
else
    options = optimset('MaxIter', opt.nIter, 'Display', 'iter');
end
[new_param] = fmincon(@(param) EigenGP_Upd_W(param, x, y, opt), opt.lnW,...
    [], [], [], [], 1e-20*ones(M,1), log(Lambdaq_sort), [], options);
end
opt.lnW = new_param(1:M);

%new_param = opt.lnW;

[nlZ mu S2] = EigenGP_Upd_W(new_param, x, y, opt, xs);
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
