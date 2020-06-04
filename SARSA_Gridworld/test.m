%% Q learning
clear all
close all
clc
g = 0.9; % discount factor
eta = 0.02; % learning rate: high in the beginning, we can decrease it with k
epsilon = 0.6;
iters = 100000;
Q = zeros(12,4);
s = randi(12);
if s==5
    s=6;
end
td = zeros(iters,1);
for i=1:iters
    % epsilon greedy policy
    if rand>epsilon
        [~, a] = max(Q(s,:));
    else
        a = randi(4);
    end
    % execute a and obtain r
    [snext, r] = simulator(s, a);
    Qmax = max(Q(snext,:));
    delta = r + g*Qmax - Q(s,a);
    Q(s,a) = Q(s,a) + eta*delta;
    td(i) = abs(eta*delta);
    s = snext;
end
Q = reshape(Q,3,4,4);
[Q_act, pi] = max(Q,[],3);
disp('~~~Q-Learning~~~')
disp(' ')
disp('Optimal action-value function Q* (given by actions) is: ')
disp(Q)
disp('Optimal action-value function Q* is: ')
disp(Q_act)
disp('Optimal policy derived from Q* is: ')
disp(pi)
%% Sarsa
clear all

g = 1; % discount factor
eta = 0.001; % learning rate: high in the beginning, we can decrease it with k
eta_end = 0.0000001;
epsilon = 0.4;
iters = 100000;
eta_vect = linspace(eta, eta_end, iters);

Q = zeros(12,4);
while 1
    s=randi(12);
    if s~=5
        break
    end
end

[~, a] = max(Q(s,:));
for i=1:iters
    eta = eta_vect(i);
    if i==iters/10
        epsilon = epsilon/2;
    elseif i==iters/2
        epsilon = epsilon/2;
    elseif i==iters/1.5
        epsilon = epsilon/2;
    end
    % execute a and obtain r
    [snext, r] = simulator(s, a);
    
    % epsilon greedy policy
    if rand>epsilon
        [~, anext] = max(Q(snext,:));
    else
        anext = randi(4);
    end
    q = r + g*Q(snext,anext);
    Q(s,a) = Q(s,a) + eta*(q-Q(s,a));
    s = snext;
    a = anext;
    if i>iters/2
        if s==10
            while 1
                s=randi(12);
                if s~=5
                    break
                end
            end
        end
    end
        
end
Q = reshape(Q,3,4,4);
[Q_act, pi] = max(Q,[],3);
disp('~~~SARSA~~~')
disp(' ')
disp('Optimal action-value function Q* (given by actions) is: ')
disp(Q)
disp('Optimal action-value function Q* is: ')
disp(Q_act)
disp('Optimal policy derived from Q* is: ')
disp(pi)