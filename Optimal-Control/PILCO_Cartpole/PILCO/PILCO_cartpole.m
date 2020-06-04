clear all
close all
clc

%% Accumulated Real Cost

load('cartPole_15_H40.mat','realCost');
cumulative_cost = [];
for i=1:length(realCost)
    cumulative_cost = [cumulative_cost,sum(realCost{i})];
end
figure;
plot(0:15,cumulative_cost)
ylabel('\Sigma cost')
xlabel('iteration')
title('Cumulative cost of PILCO cartpole')