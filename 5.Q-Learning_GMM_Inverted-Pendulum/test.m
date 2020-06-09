clear all
close all
clc

tot_sum_r = zeros(10,100);

for global_ind=0:9
%     test1run;
%     save(['run',num2str(global_ind)],'GMM','sum_r_total')
    load(['run',num2str(global_ind)])
    tot_sum_r(global_ind+1,:) = sum_r_total;
    if global_ind==9
        %% visualizing the final Q
        % test iterations
        s = [-pi;0]; % always start from the bottom
        for iter=1:500
            % action selection
            [~,~,a] = sample_model(GMM,s);
            % execute a and obtain r
            [snext, rnext] = simulator(s, a, 0.1, 0.001);
            s = snext;
            visualization(s);
            pause(0.01)
        end
    end
end

median_sum_r = median(tot_sum_r,1);
mean_sum_r = mean(tot_sum_r,1);
std_sum_r = std(tot_sum_r,0,1);

figure;
hold all
plot(mean_sum_r,'b')
plot(median_sum_r,'r')
fp = fill([1:100,100:-1:1],[mean_sum_r+std_sum_r,fliplr(mean_sum_r-std_sum_r)],[0,0,1],'linestyle', 'none');
set(fp,'facealpha',.5)
title('Sum of reward, Q learning with GMM')
legend('Mean sum','Median sum')
xlabel('Episodes')