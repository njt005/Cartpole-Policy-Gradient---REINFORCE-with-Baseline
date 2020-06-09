function [Qmean_max,samp_a,mean_a,Qsamps,Qmeans] = sample_model(GMM, state)

    % extract alphas, mus and sigmas from the table for the speed-up
    GMM_f1 = GMM.f1;
    GMM_fz = GMM.fz;
    GMM_fzz = GMM.fzz;
    GMM_alphas = GMM.alpha;
    GMM_mus = GMM.mu;
    GMM_sigmas = GMM.sigma;

    Qsamps = [];
    Qmeans = [];
    Qsamp_max = -Inf;
    samp_a = -5;
    Qmean_max = -Inf;
    mean_a = -5;
    for a=linspace(-5,5,10)
%         a = ak-10/49+rand*20/49;
%         if a<-5
%             a = -5;
%         elseif a>5
%             a = 5;
%         end
        new_samp = [state',a];
        % calculate mean of y given x based on the current GMM
        mus = zeros(size(GMM_alphas,1),1);
        sig2 = zeros(size(GMM_alphas,1),1);
        betas = zeros(size(GMM_alphas,1),1);
        for i=1:size(GMM_alphas,1)
            covmtx = reshape(GMM_sigmas(i,:),4,4);
            sigma_xx = covmtx(1:3,1:3);
            sigma_yx = covmtx(4,1:3);
            mus(i) = GMM_mus(i,4) + sigma_yx*inv(sigma_xx)*(new_samp-GMM_mus(i,1:3))';
            try
                betas(i) = GMM_alphas(i)*mvnpdf(new_samp,GMM_mus(i,1:3),sigma_xx);
            catch
                disp('Error')
            end
            sig2(i) = GMM_sigmas(i,16)-sigma_yx*inv(sigma_xx)*sigma_yx';
        end
        if sum(betas)>1e-9
            betas = betas/sum(betas);
        end
        muQ = betas'*mus;
        Qmeans = [Qmeans,muQ];
        if muQ>Qmean_max
            Qmean_max = muQ;
            mean_a = a;
        end
        
        sigmaQ = sqrt(betas'*(sig2+(mus-muQ).^2));
        % generate sample of Q
        Qsamp = muQ + randn*sigmaQ;
        Qsamps = [Qsamps,Qsamp];
        if Qsamp>Qsamp_max
            Qsamp_max = Qsamp;
            samp_a = a;
        end
    end
end

