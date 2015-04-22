% a demo file that test sequential update and joint update of EigenGP on a
% synthetic dataset
% 
% Author: Hao Peng
% Last edit: April 21, 2015
addpath('src');
addpath('src/utils');

% number of basis
M = 7;

% load data
load('data/syn.mat');
[N,D] = size(x);

% use the range from -1 to 7
range = 48:231; %[-1 7]

% reset rng
rng(1);
 
% initialize hyperparameters
opt.cov(1:D) = -log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
opt.cov(D+1) = log(var(y,1)); % log size 
opt.lik = log(var(y,1)/4); % log noise
opt.nIter = 30;


%% only update kernel parameters and B
[nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_UI(x, y, xtest, zeros(size(xtest,1),1), M, opt);

figure(1)
set(gcf,'defaultlinelinewidth',1.5);
hold on
plot(x,y,'.m', 'MarkerSize', 12)% data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
plot(post.opt.B,-2.75*ones(size(post.opt.B)),'k+','markersize',20)
hold off
axis([-3 9 -3 2]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [7.5 6]);
set(gcf, 'PaperPositionMode', 'auto')
title('EigenGP kerB');

%% update W based, but fix kernel parameters and B learned above
[nmse, mu, s2, nmlp, post] = EigenGP_Upd_W_UI(x, y, xtest, zeros(size(xtest,1),1), M, post.opt);

figure(2)
set(gcf,'defaultlinelinewidth',1.5);
hold on
plot(x,y,'.m', 'MarkerSize', 12)% data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
plot(post.opt.B,-2.75*ones(size(post.opt.B)),'k+','markersize',20)
hold off
axis([-1 7.5 -3 2]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [7.5 6]);
set(gcf, 'PaperPositionMode', 'auto')
title('EigenGP seq');


%% update kernerl paramters, B and W jointly
[nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerBW_UI(x, y, xtest, zeros(size(xtest,1),1), M, opt);

figure(3)
set(gcf,'defaultlinelinewidth',1.5);
hold on
plot(x,y,'.m', 'MarkerSize', 12)% data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
plot(post.opt.B,-2.75*ones(size(post.opt.B)),'k+','markersize',20)
hold off
axis([-1 7.5 -3 2]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [7.5 6]);
set(gcf, 'PaperPositionMode', 'auto')
title('EigenGP joint');

