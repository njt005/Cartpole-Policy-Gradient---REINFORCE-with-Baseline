%% Guassian Processes FA

l = 1;
noise_var = 0.0001;

% obtained samples
X = [-5,-3,-1,1,3,5];
y = sin(X);

% query samples
xt = -5:0.1:5;

kernel = @(x,xp) exp(-(x-xp).^2/(2*l^2));
K = zeros(length(X));
for i=1:length(X)
    for j=1:length(X)
        K(i,j) = kernel(X(i),X(j));
    end
end
f_est = @(xt) kernel(xt,X)*inv(K+noise_var*eye(size(K)))*y';
var_est = @(xt) 1-kernel(xt,X)*inv(K+noise_var*eye(size(K)))*kernel(xt,X)';

ft = zeros(size(xt));
sigmat = zeros(size(xt));
for i=1:length(xt)
    ft(i) = f_est(xt(i));
    sigmat(i) = sqrt(var_est(xt(i)));
end

figure;
hold all
plot(X,y,'ro')
plot(xt,ft,'b')
plot(xt,sin(xt),'k')
fp = fill([xt,fliplr(xt)],[ft+2*sigmat,fliplr(ft-2*sigmat)],[0,0,1],'linestyle', 'none');
set(fp,'facealpha',.2)
legend('samples','GP mean','sin(x)')
ylabel('y')
xlabel('x')
title('GP function approximation')