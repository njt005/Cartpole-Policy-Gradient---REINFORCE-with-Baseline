clear all
close all
clc

MSE = Inf;
thr = 0.0005;

samps = -5:0.1:5;
samps = [samps,fliplr(samps)];
MSE_all = [];

% initialize the GMM with one Gaussian
f1 = [0];
fz = [0,0];
fzz = [1,0,0,1];
alpha = [1];
mu = [0,0];
sigma = [1,0,0,1]; % transform this using reshape(S,2,2) before each use

GMM = table(f1,alpha,fz,mu,fzz,sigma,...
            'VariableNames',...
            {'f1','alpha','fz','mu','fzz','sigma'});
a = 0.9;
b = 2;
thr_e = 0.005;
thr_dens = 0.1;
lambda = @(t) 1-(1-a)/(a*t+b);

while MSE>thr
    % training sweep
    for t=1:length(samps)
        %%% generate new sample
        x = samps(t);
        y = sin(x);
        %%% update the model
        % calculate weights for each Gaussian
        w = zeros(size(GMM,1),1);
        for i=1:length(w)
            w(i) = GMM.alpha(i)*mvnpdf([x,y],GMM.mu(i,:),reshape(GMM.sigma(i,:),2,2));
        end
        w = w/sum(w); % this is the w(t,:) from the paper
        density = 0;
        for i=1:size(GMM,1)
            density = density + GMM.alpha(i)*mvnpdf([x,y],GMM.mu(i,:),reshape(GMM.sigma(i,:),2,2));
        end
        nx = sum(GMM.f1)*10*2*1e-4*density;
        % update the GMM parameters
        for i=1:size(GMM,1)
            GMM.f1(i) = lambda(nx)^w(i)*GMM.f1(i) + (1-lambda(nx)^w(i))/(1-lambda(nx));
            GMM.fz(i,:) = lambda(nx)^w(i)*GMM.fz(i,:) + (1-lambda(nx)^w(i))/(1-lambda(nx))*[x,y];
            GMM.fzz(i,:) = lambda(nx)^w(i)*GMM.fzz(i,:) + (1-lambda(nx)^w(i))/(1-lambda(nx))*reshape([x,y]'*[x,y],1,4);
        end
        for i=1:size(GMM,1)
            GMM.alpha(i) = GMM.f1(i)/sum(GMM.f1);
            GMM.mu(i,:) = GMM.fz(i,:)/GMM.f1(i);
            GMM.sigma(i,:) = GMM.fzz(i,:)/GMM.f1(i) - reshape(GMM.mu(i,:)'*GMM.mu(i,:),1,4);
        end
        % calculate mean of y given x based on the current GMM
        mus = zeros(size(GMM,1),1);
        betas = zeros(size(GMM,1),1);
        for i=1:size(GMM,1)
            mus(i) = GMM.mu(i,2) + GMM.sigma(i,2)/GMM.sigma(i,1)*(x-GMM.mu(i,1));
            betas(i) = GMM.alpha(i)*mvnpdf(x,GMM.mu(i,1),GMM.sigma(i,1));
        end
        betas = betas/sum(betas);
        sample_mu = betas'*mus;
        % calculate the error
        e = (y-sample_mu)^2;
        % check for generating new Gaussians
        if e>thr_e
            density = 0;
            for i=1:size(GMM,1)
                density = density + GMM.alpha(i)*mvnpdf([x,y],GMM.mu(i,:),reshape(GMM.sigma(i,:),2,2));
            end
            if density<thr_dens
                % add another Gaussian
                w_new = 0.95;
                f1 = 1;
                alpha = f1/(sum(GMM.f1)+f1);
                mu = [x,y];
                fz = mu*f1;
                % calculate C^2 from paper
                temps = 0;
                for i=1:size(GMM,1)
                    temps = temps + GMM.f1(i)*mvnpdf([x,y],GMM.mu(i,:),reshape(GMM.sigma(i,:),2,2));
                end
                C2 = 1/(2*pi)*(w_new/(1-w_new)*temps*10*2)^(-1);
                sigma = [C2*10^2,0,0,C2*2^2];
                fzz = (sigma+reshape(mu'*mu,1,4))*f1;
                new_G = {f1,alpha,fz,mu,fzz,sigma};
                GMM = [GMM;new_G];
            end
        end
%         figure(100);clf;
%         x1 = -5:0.2:5;
%         x2 = -5:0.2:5;
%         [X1,X2] = meshgrid(x1,x2);
%         X = [X1(:) X2(:)];
%         y = GMM.alpha(1,:)*mvnpdf(X,GMM.mu(1,:),reshape(GMM.sigma(1,:),2,2));
%         for i=2:size(GMM,1)
%             y = y + GMM.alpha(i,:)*mvnpdf(X,GMM.mu(i,:),reshape(GMM.sigma(i,:),2,2));
%         end
%         y = reshape(y,length(x2),length(x1));
%         ycap = zeros(size(x1));
%         for j=1:length(x1)
%             mus_plot = zeros(size(GMM,1),1);
%             betas_plot = zeros(size(GMM,1),1);
%             for i=1:size(GMM,1)
%                 mus_plot(i) = GMM.mu(i,2) + GMM.sigma(i,2)/GMM.sigma(i,1)*(x1(j)-GMM.mu(i,1));
%                 betas_plot(i) = GMM.alpha(i)*mvnpdf(x1(j),GMM.mu(i,1),GMM.sigma(i,1));
%             end
%             betas_plot = betas_plot/sum(betas_plot);
%             ycap(j) = betas_plot'*mus_plot;
%         end
%         hold all
%         contour(x1,x2,y,logspace(-6,-2,5))
%         plot(x1,ycap,'r')
%         xlabel('x1')
%         ylabel('x2')
%         pause(0.05)
    end
    % evaluation sweep
    MSE = 0;
    for k=1:length(samps)/2
        % generate sample
        x = samps(k);
        y = sin(x);
        % obtain approximation from model
        mus = zeros(size(GMM,1),1);
        betas = zeros(size(GMM,1),1);
        for i=1:size(GMM,1)
            mus(i) = GMM.mu(i,2) + GMM.sigma(i,2)/GMM.sigma(i,1)*(x-GMM.mu(i,1));
            betas(i) = GMM.alpha(i)*mvnpdf(x,GMM.mu(i,1),GMM.sigma(i,1));
        end
        betas = betas/sum(betas);
        ycap = betas'*mus;
        % add to MSE
        MSE = MSE + (ycap-y)^2/(length(samps)/2);
    end
    MSE_all = [MSE_all,MSE];
end

figure;
plot(MSE_all)
title('MSE convergence')
xlabel('sweep')
ylabel('MSE')

figure;
x1 = -5:0.2:5;
x2 = -5:0.2:5;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
y = GMM.alpha(1,:)*mvnpdf(X,GMM.mu(1,:),reshape(GMM.sigma(1,:),2,2));
for i=2:size(GMM,1)
    y = y + GMM.alpha(i,:)*mvnpdf(X,GMM.mu(i,:),reshape(GMM.sigma(i,:),2,2));
end
y = reshape(y,length(x2),length(x1));
ycap = zeros(size(x1));
scap = zeros(size(x1));
for j=1:length(x1)
    mus_plot = zeros(size(GMM,1),1);
    sig2_plot = zeros(size(GMM,1),1);
    betas_plot = zeros(size(GMM,1),1);
    for i=1:size(GMM,1)
        mus_plot(i) = GMM.mu(i,2) + GMM.sigma(i,2)/GMM.sigma(i,1)*(x1(j)-GMM.mu(i,1));
        betas_plot(i) = GMM.alpha(i)*mvnpdf(x1(j),GMM.mu(i,1),GMM.sigma(i,1));
    end
    betas_plot = betas_plot/sum(betas_plot);
    ycap(j) = betas_plot'*mus_plot;
    for i=1:size(GMM,1)
        sig2_plot(i) = GMM.sigma(i,4)-GMM.sigma(i,3)/GMM.sigma(i,1)*GMM.sigma(i,2);
    end
    scap(j) = sqrt(betas_plot'*(sig2_plot+(mus_plot-ycap(j)).^2));
end
hold all
plot(x1,sin(x1),'b','LineWidth',2)
plot(x1,ycap,'r-.','LineWidth',2)
fa = fill([x1,fliplr(x1)],[ycap+scap,fliplr(ycap-scap)],[1,0,0],'linestyle','none');
set(fa,'facealpha',.5)
xlabel('x')
ylabel('y')
grid on
title('Comparison of approximation and original function')
legend('function','approximation')