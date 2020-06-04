% clear all
% close all
% clc

g = 0.95;

% initialize the GMM with 27 Gaussians
f1 = 0.1*ones(27,1);
alpha = 1/27*ones(27,1);
[mu_s1,mu_s2,mu_a] = meshgrid(linspace(-pi,pi,3),...
                              linspace(-2*pi,2*pi,3),...
                              linspace(-5,5,3));
mu = [mu_s1(:),...
      mu_s2(:),...
      mu_a(:),...
      zeros(27,1)];  % [s1, s2, a, Q_est]
fz = mu.*f1;
sigma = repmat(reshape(diag([(0.5*pi)^2,(0.5*2*pi)^2,(0.5*5)^2,(0.5*pi/(1-g))^2]),1,16),27,1); % transform this using reshape(S,4,4) before each use
fzz = zeros(size(sigma));
for k=1:27
    fzz(k,:) = (sigma(k,:)+reshape(mu(k,:)'*mu(k,:),1,16))*f1(k);
end

GMM = table(f1,alpha,fz,mu,fzz,sigma,...
            'VariableNames',...
            {'f1','alpha','fz','mu','fzz','sigma'});

action_array = [];
Qsamps = [];
Qmeans = [];
sum_r_total = [];
num_g_total = [];
all_samples = zeros(4,1);

for episode=1:100
    GMM_old = GMM;
    
    % training iterations
    s = [-pi+0.0001*randn;0]; % always start from around the bottom
    if s(1)<-pi
        s(1) = s(1)+2*pi;
    end
    for iter=1:500
        % action selection
        [~,a,~,~,~] = sample_model(GMM,s);
        % execute a and obtain r
        [snext, rnext] = simulator(s, a, 0.1, 0.001);
        % Qmax = max(Q(snext,:));
        [Qmax,~,~,~,~] = sample_model(GMM,snext);
        q = rnext + g*Qmax;
        all_samples = [all_samples,[s;a;q]];
        % update_model
        GMM = update_model(GMM,g,[s',a,q]);
        s = snext;
    end
    % test iterations
    s = [-pi;0]; % always start from the bottom
    sum_r = 0;
    action_array_old = action_array;
    action_array = [];
    Qsamps_old = Qsamps;
    Qmeans_old = Qmeans;
    Qsamps = [];
    Qmeans = [];
    for iter=1:500
        % action selection
        [~,~,a,Qsamp,Qmean] = sample_model(GMM,s);
        Qsamps = [Qsamps,Qsamp'];
        Qmeans = [Qmeans,Qmean'];
        action_array = [action_array,a];
        % execute a and obtain r
        [snext, rnext] = simulator(s, a, 0.1, 0.001);
        sum_r = sum_r + rnext;
        s = snext;
    end
    if episode>2 && sum_r_total(end)>sum_r+500
        figure; hold all; plot(action_array_old); plot(action_array)
%         figure; hold all; plot(Qmeans_old'); plot(Qmeans')
        disp('Big drop!')
    end
    sum_r_total = [sum_r_total,sum_r];
    num_g_total = [num_g_total,size(GMM,1)];
%     figure(1111);clf;
%         scatter(all_samples(1,end-500:end),all_samples(2,end-500:end),2,all_samples(4,end-500:end))
%         for k=1:size(GMM,1)
%             covmtx = reshape(GMM.sigma(k,:),4,4);
%             plot_gaussian_ellipsoid(GMM.mu(k,1:2), covmtx(1:2,1:2), 2);
%         end
%         colorbar
%         colormap('jet')
%         caxis([-pi/(1-g) 0])
%         xlim([-pi pi])
%         xlabel('theta')
%         ylim([-2*pi 2*pi])
%         ylabel('theta\_dot')
%     figure(1112);clf;
%         scatter(all_samples(1,end-500:end),all_samples(3,end-500:end),2,all_samples(4,end-500:end))
%         for k=1:size(GMM,1)
%             covmtx = reshape(GMM.sigma(k,:),4,4);
%             plot_gaussian_ellipsoid(GMM.mu(k,[1,3]), covmtx([1,3],[1,3]), 2);
%         end
%         colorbar
%         colormap('jet')
%         caxis([-pi/(1-g) 0])
%         xlim([-pi pi])
%         xlabel('theta')
%         ylim([-5 5])
%         ylabel('a')
%     figure(113);clf;
%         scatter(all_samples(1,end-500:end),all_samples(2,end-500:end),2,all_samples(4,end-500:end))
%         for k=1:size(GMM_old,1)
%             covmtx = reshape(GMM_old.sigma(k,:),4,4);
%             plot_gaussian_ellipsoid(GMM_old.mu(k,1:2), covmtx(1:2,1:2), 2);
%         end
%         colorbar
%         colormap('jet')
%         caxis([-pi/(1-g) 0])
%         xlim([-pi pi])
%         xlabel('theta')
%         ylim([-2*pi 2*pi])
%         ylabel('theta\_dot')
%     figure(114);clf;
%         scatter(all_samples(1,end-500:end),all_samples(3,end-500:end),2,all_samples(4,end-500:end))
%         for k=1:size(GMM_old,1)
%             covmtx = reshape(GMM_old.sigma(k,:),4,4);
%             plot_gaussian_ellipsoid(GMM_old.mu(k,[1,3]), covmtx([1,3],[1,3]), 2);
%         end
%         colorbar
%         colormap('jet')
%         caxis([-pi/(1-g) 0])
%         xlim([-pi pi])
%         xlabel('theta')
%         ylim([-5 5])
%         ylabel('a')
%     figure(1115);
%         subplot(1,2,1); plot(sum_r_total)
%         subplot(1,2,2); plot(num_g_total)
%     figure; clf; hold all; plot(Qmeans_old'); plot(Qmeans')
end

% %% visualizing the final Q
% % test iterations
% s = [-pi;0]; % always start from the bottom
% for iter=1:500
%     % action selection
%     [~,~,a] = sample_model(GMM,s);
%     % execute a and obtain r
%     [snext, rnext] = simulator(s, a, 0.1, 0.001);
%     s = snext;
%     visualization(s);
%     pause(0.01)
% end

%% plotting
% figure; plot(sum_r_total)
% figure;
% hold all
% scatter(all_samples(1,2:end),all_samples(2,2:end),2,all_samples(4,2:end))
% colorbar
% colormap('jet')
% for k=1:size(Qrep,1)
%     rectangle('Position',[Qrep.s1min(k),Qrep.s2min(k),Qrep.s1max(k)-Qrep.s1min(k),Qrep.s2max(k)-Qrep.s2min(k)])
%     %text(Qrep.s1min(k),(Qrep.s2min(k)+Qrep.s2max(k))/2,num2str(round(Qrep.mean(k),2)),'FontSize',6)
%     xlim([-pi pi])
%     xlabel('theta')
%     ylim([-2*pi 2*pi])
%     ylabel('theta\_dot')
% end
% figure;
% hold all
% scatter(all_samples(1,2:end),all_samples(3,2:end),2,all_samples(4,2:end))
% colorbar
% colormap('jet')
% for k=1:size(Qrep,1)
%     rectangle('Position',[Qrep.s1min(k),Qrep.amin(k),Qrep.s1max(k)-Qrep.s1min(k),Qrep.amax(k)-Qrep.amin(k)])
%     %text(Qrep.s1min(k),(Qrep.amin(k)+Qrep.amax(k))/2,num2str(round(Qrep.mean(k),2)),'FontSize',6)
%     xlim([-pi pi])
%     xlabel('theta')
%     ylim([-5 5])
%     ylabel('a')
% end
% figure;
% hold all
% scatter(all_samples(2,2:end),all_samples(3,2:end),2,all_samples(4,2:end))
% colorbar
% colormap('jet')
% for k=1:size(Qrep,1)
%     rectangle('Position',[Qrep.s2min(k),Qrep.amin(k),Qrep.s2max(k)-Qrep.s2min(k),Qrep.amax(k)-Qrep.amin(k)])
%     %text(Qrep.s2min(k),(Qrep.amin(k)+Qrep.amax(k))/2,num2str(round(Qrep.mean(k),2)),'FontSize',6)
%     xlim([-2*pi 2*pi])
%     xlabel('theta\_dot')
%     ylim([-5 5])
%     ylabel('a')
% end
