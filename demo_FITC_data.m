clear; close; clc;
addpath('src');
addpath('src/lik');
addpath('src/utils');
addpath('src/pred');

% number of basis
M = 7;

% number of interations
nIter = 400;

% load data
load('data/syn.mat');
[N,D] = size(x);

% use the range from -1 to 7
range = 48:231; %[-1 7]

% reset rng
rng(0);

%% sequential update
model = initEigenGP('EigenGP', x, y, M);
model = optimizeEigenGP(model, x, y, nIter);
[mu, s2] = predEigenGP(model, x, y, xtest);

figure(1)
set(gcf,'defaultlinelinewidth',1.5);
plot(x,y,'.m', 'MarkerSize', 12)% data points in magenta
hold on
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
plot(model.B,-2.75*ones(size(model.B)),'k+','markersize',20)
hold off
axis([-3 9 -3 2]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [7.5 6]);
set(gcf, 'PaperPositionMode', 'auto')
title('EigenGP');

%% joint update
model = initEigenGP('EigenGP*', x, y, M);
model = optimizeEigenGP(model, x, y, nIter);
[mu, s2] = predEigenGP(model, x, y, xtest);

figure(2)
set(gcf,'defaultlinelinewidth',1.5);
plot(x,y,'.m', 'MarkerSize', 12)% data points in magenta
hold on
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
plot(model.B,-2.75*ones(size(model.B)),'k+','markersize',20)
hold off
axis([-3 9 -3 2]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [7.5 6]);
set(gcf, 'PaperPositionMode', 'auto')
title('EigenGP*');

%% sequential update with diagonal delta term
model = initEigenGP('EigenGP+', x, y, M);
model = optimizeEigenGP(model, x, y, nIter);
[mu, s2] = predEigenGP(model, x, y, xtest);

figure(3)
set(gcf,'defaultlinelinewidth',1.5);
plot(x,y,'.m', 'MarkerSize', 12)% data points in magenta
hold on
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
plot(model.B,-2.75*ones(size(model.B)),'k+','markersize',20)
hold off
axis([-3 9 -3 2]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [7.5 6]);
set(gcf, 'PaperPositionMode', 'auto')
title('EigenGP+');