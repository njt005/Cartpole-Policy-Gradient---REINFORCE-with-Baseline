function updatedGMM = update_model(GMM, g, new_samp)

    a = 0.999;  % 0.1, 0.01, 0.001...
    b = 0.005;    % 5-10
    thr_e = 200000;
    thr_dens = 0.00001;
    thr_f1 = 1;
    lambda = @(t) 1-(1-a)/(a*t+b);
    
    % extract alphas, mus and sigmas from the table for the speed-up
    GMM_f1 = GMM.f1;
    GMM_fz = GMM.fz;
    GMM_fzz = GMM.fzz;
    GMM_alphas = GMM.alpha;
    GMM_mus = GMM.mu;
    GMM_sigmas = GMM.sigma;

    % calculate weights for each Gaussian
    w = zeros(size(GMM_alphas,1),1);
    for i=1:length(w)
        try
            w(i) = GMM_alphas(i)*mvnpdf(new_samp,GMM_mus(i,:),reshape(GMM_sigmas(i,:),4,4));
        catch
            disp('Error')
        end
    end
    density = sum(w);
    if density>1e-18
        w = w/density; % this is the w(t,:) from the paper
    else
        disp('Density is small!')
    end
%     nz = sum(GMM_f1)*10*2*pi*4*pi*1e-4*density;

    % update the GMM parameters
    for i=1:size(GMM_alphas,1)
        GMM_f1(i) = 0.995^w(i)*GMM_f1(i) + w(i);%(1-lambda(nz)^w(i))/(1-lambda(nz));
        GMM_fz(i,:) = 0.995^w(i)*GMM_fz(i,:) + w(i)*new_samp;%(1-lambda(nz)^w(i))/(1-lambda(nz))*new_samp;
        GMM_fzz(i,:) = 0.995^w(i)*GMM_fzz(i,:) + w(i)*reshape(new_samp'*new_samp,1,16);%(1-lambda(nz)^w(i))/(1-lambda(nz))*reshape(new_samp'*new_samp,1,16);
    end
    for i=1:size(GMM_alphas,1)
        GMM_alphas(i) = GMM_f1(i)/sum(GMM_f1);
        GMM_mus(i,:) = GMM_fz(i,:)/GMM_f1(i);
        GMM_sigmas(i,:) = GMM_fzz(i,:)/GMM_f1(i) - reshape(GMM_mus(i,:)'*GMM_mus(i,:),1,16);
        % maybe we need to do regularization here
        if min(diag(reshape(GMM_sigmas(i,:),4,4)))<1e-9
            GMM_sigmas(i,:) = reshape(reshape(GMM_sigmas(i,:),4,4)+1e-6*eye(4,4),1,16);
            GMM_fzz(i,:) = (GMM_sigmas(i,:)+reshape(GMM_mus(i,:)'*GMM_mus(i,:),1,16))*GMM_f1(i);
        end
    end
%     % calculate mean of y given x based on the current GMM
%     mus = zeros(size(GMM_alphas,1),1);
%     betas = zeros(size(GMM_alphas,1),1);
%     for i=1:size(GMM_alphas,1)
%         covmtx = reshape(GMM_sigmas(i,:),4,4);
%         sigma_xx = covmtx(1:3,1:3);
%         sigma_yx = covmtx(4,1:3);
%         mus(i) = GMM_mus(i,4) + sigma_yx*inv(sigma_xx)*(new_samp(1:3)-GMM_mus(i,1:3))';
%         try
%             betas(i) = GMM_alphas(i)*mvnpdf(new_samp(1:3),GMM_mus(i,1:3),sigma_xx);
%         catch
%             disp('Error')
%         end
%     end
%     if sum(betas)>1e-9
%         betas = betas/sum(betas);
%     end
%     Qcap = betas'*mus;
%     % calculate the error
%     e = abs(new_samp(4)-Qcap);
    
    GMM = table(GMM_f1,GMM_alphas,GMM_fz,GMM_mus,GMM_fzz,GMM_sigmas,...
                'VariableNames',...
                {'f1','alpha','fz','mu','fzz','sigma'});
%     % check for generating new Gaussians
%     if e>thr_e
%         density = 0;
%         for i=1:size(GMM_alphas,1)
%             try
%                 density = density + GMM_alphas(i)*mvnpdf(new_samp,GMM_mus(i,:),reshape(GMM_sigmas(i,:),4,4));
%             catch
%                 disp('Error')
%             end
%         end
%         if density<thr_dens && min(GMM_f1)>thr_f1
%             % add another Gaussian
%             w_new = 0.95;
%             f1 = 1;
%             alpha = f1/(sum(GMM_f1)+f1);
%             mu = new_samp;
%             fz = mu*f1;
%             % calculate C^2 from paper
%             temps = 0;
%             for i=1:size(GMM_alphas,1)
%                 try
%                     temps = temps + GMM_f1(i)*mvnpdf(new_samp,GMM_mus(i,:),reshape(GMM_sigmas(i,:),4,4));
%                 catch
%                     disp('Error')
%                 end
%             end
%             C2 = 1/(2*pi)*(w_new/(1-w_new)*temps*10*2*pi*4*pi)^(-1);
%             sigma = reshape(C2*diag([(pi)^2,(2*pi)^2,5^2,(pi/(1-g))^2]),1,16);
%             fzz = (sigma+reshape(mu'*mu,1,16))*f1;
%             new_G_left = {f1,alpha,fz,mu,fzz,sigma};
%             %fz(1:3) = -fz(1:3);
%             %mu(1:3) = -mu(1:3);
%             %fzz = (sigma+reshape(mu'*mu,1,16))*f1;
%             %new_G_right = {f1,alpha,fz,mu,fzz,sigma};
%             GMM = [GMM;new_G_left];%;new_G_right];
%         end
%     end
    updatedGMM = GMM;
end

