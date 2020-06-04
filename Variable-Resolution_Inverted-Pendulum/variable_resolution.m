clear all
close all
clc

% initial partitioning is in 4 equal-size regions
index = [1;2;3;4];
s1min = [-pi;-pi;0;0];
s1max = [0;0;pi;pi];
s2min = [-2*pi;-2*pi;-2*pi;-2*pi];
s2max = [2*pi;2*pi;2*pi;2*pi];
amin = [0;-5;0;-5];
amax = [5;0;5;0];
nsamp = [0;0;0;0];
mean = [0.001*randn;0.001*randn;0.001*randn;0.001*randn];
var = [10000;10000;10000;10000];
Qrep = table(index,s1min,s1max,s2min,s2max,amin,amax,nsamp,mean,var,...
             'VariableNames',...
             {'index','s1min','s1max','s2min','s2max','amin','amax','nsamp','mean','var'});
         
g = 0.995;
%alpha = 0.3;
alpha = 0.05;
beta = 4;
sum_r_total = [];
all_samples = zeros(4,1);

for episode=1:300
    % training iterations
    s = [-pi+0.0001*randn;0]; % always start from around the bottom
    if s(1)<-pi
        s(1) = s(1)+2*pi;
    end
    for iter=1:500
        % action selection
        Qsel = Qrep(Qrep.s1min<=s(1) & Qrep.s1max>=s(1)...
                    & Qrep.s2min<=s(2) & Qrep.s2max>=s(2),:);  % select corresponding regions
        Qsamp = normrnd(Qsel.mean,sqrt(Qsel.var));  % generate normal random q
        [~,ind] = max(Qsamp);                       % select region which produced maxq
        orig_ind = Qsel.index(ind);                 % orig_ind used for FA step
        a = rand*(Qsel.amax(ind)-Qsel.amin(ind))+Qsel.amin(ind);
        % execute a and obtain r
        [snext, rnext] = simulator(s, a, 0.1, 0.001);
        % Qmax = max(Q(snext,:));
        Qsel = Qrep(Qrep.s1min<=snext(1) & Qrep.s1max>=snext(1)...
                    & Qrep.s2min<=snext(2) & Qrep.s2max>=snext(2),:);  % select corresponding regions
%         Qsamp = normrnd(Qsel.mean,sqrt(Qsel.var));                     % generate normal random q
        Qmax = max(Qsel.mean);                                                            
        q = rnext + g*Qmax;
        all_samples = [all_samples,[s;a;q]];
        % function approximation
        Qrep.nsamp(orig_ind) = Qrep.nsamp(orig_ind) + 1;
        lr = 1/(alpha*Qrep.nsamp(orig_ind)+beta);
        if lr<0.05
            lr = 0.05;
        end
        Qrep.mean(orig_ind) = Qrep.mean(orig_ind) + lr*(q-Qrep.mean(orig_ind));
        Qrep.var(orig_ind) = Qrep.var(orig_ind) + lr*((q-Qrep.mean(orig_ind))^2-Qrep.var(orig_ind));
        % check for splitting
        if Qrep.var(orig_ind)>1 && Qrep.nsamp(orig_ind)>100
            asize = (Qrep.amax(orig_ind)-Qrep.amin(orig_ind))/5;
            s1size = (Qrep.s1max(orig_ind)-Qrep.s1min(orig_ind))/pi;
            s2size = (Qrep.s2max(orig_ind)-Qrep.s2min(orig_ind))/(2*pi);
            
            if s2size>asize && s2size>=s1size
                % split along state2 axis
                divpt = (Qrep.s2max(orig_ind)+Qrep.s2min(orig_ind))/2;
                r1 = {orig_ind,Qrep.s1min(orig_ind),Qrep.s1max(orig_ind),...
                               Qrep.s2min(orig_ind),divpt,...
                               Qrep.amin(orig_ind),Qrep.amax(orig_ind),...
                               0,Qrep.mean(orig_ind),Qrep.var(orig_ind)};
                r2 = {Qrep.index(end)+1,Qrep.s1min(orig_ind),Qrep.s1max(orig_ind),...
                                        divpt,Qrep.s2max(orig_ind),...
                                        Qrep.amin(orig_ind),Qrep.amax(orig_ind),...
                                        0,Qrep.mean(orig_ind),Qrep.var(orig_ind)};
                Qrep(orig_ind,:) = r1;
                Qrep = [Qrep;r2];
            elseif s1size>asize && s1size>=s2size
                % split along state1 axis
                divpt = (Qrep.s1max(orig_ind)+Qrep.s1min(orig_ind))/2;
                r1 = {orig_ind,Qrep.s1min(orig_ind),divpt,...
                               Qrep.s2min(orig_ind),Qrep.s2max(orig_ind),...
                               Qrep.amin(orig_ind),Qrep.amax(orig_ind),...
                               0,Qrep.mean(orig_ind),Qrep.var(orig_ind)};
                r2 = {Qrep.index(end)+1,divpt,Qrep.s1max(orig_ind),...
                                        Qrep.s2min(orig_ind),Qrep.s2max(orig_ind),...
                                        Qrep.amin(orig_ind),Qrep.amax(orig_ind),...
                                        0,Qrep.mean(orig_ind),Qrep.var(orig_ind)};
                Qrep(orig_ind,:) = r1;
                Qrep = [Qrep;r2];
            else
                % split along action axis
                divpt = (Qrep.amax(orig_ind)+Qrep.amin(orig_ind))/2;
                r1 = {orig_ind,Qrep.s1min(orig_ind),Qrep.s1max(orig_ind),...
                               Qrep.s2min(orig_ind),Qrep.s2max(orig_ind),...
                               Qrep.amin(orig_ind),divpt,...
                               0,Qrep.mean(orig_ind),Qrep.var(orig_ind)};
                r2 = {Qrep.index(end)+1,Qrep.s1min(orig_ind),Qrep.s1max(orig_ind),...
                                        Qrep.s2min(orig_ind),Qrep.s2max(orig_ind),...
                                        divpt,Qrep.amax(orig_ind),...
                                        0,Qrep.mean(orig_ind),Qrep.var(orig_ind)};
                Qrep(orig_ind,:) = r1;
                Qrep = [Qrep;r2];
            end
            Qrep = sortrows(Qrep,'index');
        end
        s = snext;
    end
    % test iterations
    s = [-pi;0]; % always start from the bottom
    sum_r = 0;
    for iter=1:500
        % action selection
        Qsel = Qrep(Qrep.s1min<=s(1) & Qrep.s1max>=s(1)...
                    & Qrep.s2min<=s(2) & Qrep.s2max>=s(2),:);  % select corresponding regions
        [~,ind] = max(Qsel.mean);
        orig_ind = Qsel.index(ind);                            % orig_ind used for FA step
        a = rand*(Qsel.amax(ind)-Qsel.amax(ind))+Qsel.amin(ind);
        % execute a and obtain r
        [snext, rnext] = simulator(s, a, 0.1, 0.001);
        sum_r = sum_r + rnext;
        s = snext;
    end
    if sum_r>max(sum_r_total)
        Qrep_max = Qrep;
    end
    sum_r_total = [sum_r_total,sum_r];
    close all
end

%% visualizing the final Qrep
% test iterations
s = [-pi;0]; % always start from the bottom
for iter=1:500
    % action selection
    Qsel = Qrep(Qrep.s1min<=s(1) & Qrep.s1max>=s(1)...
                    & Qrep.s2min<=s(2) & Qrep.s2max>=s(2),:);  % select corresponding regions
    [~,ind] = max(Qsel.mean);
    orig_ind = Qsel.index(ind);                            % orig_ind used for FA step
    a = rand*(Qsel.amax(ind)-Qsel.amax(ind))+Qsel.amin(ind);
    % execute a and obtain r
    [snext, rnext] = simulator(s, a, 0.1, 0.001);
    
    visualization(s);
    pause(0.01)

    s = snext;
end

%% plotting
figure; plot(sum_r_total)
figure;
hold all
scatter(all_samples(1,2:end),all_samples(2,2:end),2,all_samples(4,2:end))
colorbar
colormap('jet')
for k=1:size(Qrep,1)
    rectangle('Position',[Qrep.s1min(k),Qrep.s2min(k),Qrep.s1max(k)-Qrep.s1min(k),Qrep.s2max(k)-Qrep.s2min(k)])
    %text(Qrep.s1min(k),(Qrep.s2min(k)+Qrep.s2max(k))/2,num2str(round(Qrep.mean(k),2)),'FontSize',6)
    xlim([-pi pi])
    xlabel('theta')
    ylim([-2*pi 2*pi])
    ylabel('theta\_dot')
end
figure;
hold all
scatter(all_samples(1,2:end),all_samples(3,2:end),2,all_samples(4,2:end))
colorbar
colormap('jet')
for k=1:size(Qrep,1)
    rectangle('Position',[Qrep.s1min(k),Qrep.amin(k),Qrep.s1max(k)-Qrep.s1min(k),Qrep.amax(k)-Qrep.amin(k)])
    %text(Qrep.s1min(k),(Qrep.amin(k)+Qrep.amax(k))/2,num2str(round(Qrep.mean(k),2)),'FontSize',6)
    xlim([-pi pi])
    xlabel('theta')
    ylim([-5 5])
    ylabel('a')
end
figure;
hold all
scatter(all_samples(2,2:end),all_samples(3,2:end),2,all_samples(4,2:end))
colorbar
colormap('jet')
for k=1:size(Qrep,1)
    rectangle('Position',[Qrep.s2min(k),Qrep.amin(k),Qrep.s2max(k)-Qrep.s2min(k),Qrep.amax(k)-Qrep.amin(k)])
    %text(Qrep.s2min(k),(Qrep.amin(k)+Qrep.amax(k))/2,num2str(round(Qrep.mean(k),2)),'FontSize',6)
    xlim([-2*pi 2*pi])
    xlabel('theta\_dot')
    ylim([-5 5])
    ylabel('a')
end

% figure;
% hold all
% scatter3(all_samples(1,2:end),all_samples(2,2:end),all_samples(3,2:end),2,all_samples(4,2:end))
% colorbar
% colormap('jet')
% xlim([-pi pi])
% xlabel('theta')
% ylim([-2*pi 2*pi])
% ylabel('theta\_dot')
% zlim([-5 5])
% zlabel('a')
