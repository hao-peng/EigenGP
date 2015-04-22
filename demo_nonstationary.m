clear; close; clc;
addpath('src');
addpath('src/lik');
addpath('src/utils');
addpath('src/pred');

% reset rng
rng(0);

% parameters
N = 800;
Ns = 1000;
D = 1;
M = 20;
nIter = 200;

% generate data
x = linspace(0,3,N)';
y = x.*sin(x.^3) + randn(N, 1)*0.5;
xtest = linspace(0, 3, Ns)';
ytest = xtest.*sin(xtest.^3);

%% sequential update
model = initEigenGP('EigenGP', x, y, M);
model = optimizeEigenGP(model, x, y, nIter);
[mu, s2] = predEigenGP(model, x, y, xtest);

figure(1)
set(gcf,'defaultlinelinewidth',1.5);
plot(xtest, ytest, '-', 'Color', [0 .5 0]);
hold on
plot(x,y,'.m', 'MarkerSize', 15) % data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
plot(model.B,-2.75*ones(size(model.B)),'k+','markersize',20)
hold off
axis([-0 3 -4 5])
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
title('EigenGP');

%% joint update
model = initEigenGP('EigenGP*', x, y, M);
model = optimizeEigenGP(model, x, y, nIter);
[mu, s2] = predEigenGP(model, x, y, xtest);

figure(2)
set(gcf,'defaultlinelinewidth',1.5);
plot(xtest, ytest, '-', 'Color', [0 .5 0]);
hold on
plot(x,y,'.m', 'MarkerSize', 15) % data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
plot(model.B,-2.75*ones(size(model.B)),'k+','markersize',20)
hold off
axis([-0 3 -4 5])
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
title('EigenGP*');

%% sequential update with diagonal delta term
model = initEigenGP('EigenGP+', x, y, M);
model = optimizeEigenGP(model, x, y, nIter);
[mu, s2] = predEigenGP(model, x, y, xtest);

figure(3)
set(gcf,'defaultlinelinewidth',1.5);
plot(xtest, ytest, '-', 'Color', [0 .5 0]);
hold on
plot(x,y,'.m', 'MarkerSize', 15) % data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
plot(model.B,-2.75*ones(size(model.B)),'k+','markersize',20)
hold off
axis([-0 3 -4 5])
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
title('EigenGP+');