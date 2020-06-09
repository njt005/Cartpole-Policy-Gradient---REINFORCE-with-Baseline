clear all
close all
clc

load('Qrep.mat');
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
close all