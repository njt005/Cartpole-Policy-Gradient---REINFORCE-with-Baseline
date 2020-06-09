clear all
close all
clc

% our gridworld looks like this
% 1  4  7  10
% 2  5  8  11
% 3  6  9  12
% actions: up 1, down 2, left 3, right 4

% probs for state trainsitions for prefered direction
P = zeros(12,12,4);
% action up
P(1,1,1) = 0.9;
P(1,4,1) = 0.1;
P(2,1,1) = 0.8;
P(2,2,1) = 0.2;
P(3,2,1) = 0.8;
P(3,3,1) = 0.1;
P(3,6,1) = 0.1;
P(4,4,1) = 0.8;
P(4,1,1) = 0.1;
P(4,7,1) = 0.1;
P(6,6,1) = 0.8;
P(6,3,1) = 0.1;
P(6,9,1) = 0.1;
P(7,7,1) = 0.8;
P(7,4,1) = 0.1;
P(7,10,1) = 0.1;
P(8,8,1) = 0.1;
P(8,7,1) = 0.8;
P(8,11,1) = 0.1;
P(9,8,1) = 0.8;
P(9,6,1) = 0.1;
P(9,12,1) = 0.1;
P(10,10,1) = 0.9;
P(10,7,1) = 0.1;
P(11,10,1) = 0.8;
P(11,11,1) = 0.1;
P(11,8,1) = 0.1;
P(12,11,1) = 0.8;
P(12,12,1) = 0.1;
P(12,9,1) = 0.1;
% action down
P(1,1,2) = 0.1;
P(1,2,2) = 0.8;
P(1,4,2) = 0.1;
P(2,2,2) = 0.2;
P(2,3,2) = 0.8;
P(3,3,2) = 0.9;
P(3,6,2) = 0.1;
P(4,4,2) = 0.8;
P(4,1,2) = 0.1;
P(4,7,2) = 0.1;
P(6,6,2) = 0.8;
P(6,3,2) = 0.1;
P(6,9,2) = 0.1;
P(7,8,2) = 0.8;
P(7,4,2) = 0.1;
P(7,10,2) = 0.1;
P(8,8,2) = 0.1;
P(8,9,2) = 0.8;
P(8,11,2) = 0.1;
P(9,9,2) = 0.8;
P(9,6,2) = 0.1;
P(9,12,2) = 0.1;
P(10,10,2) = 0.1;
P(10,7,2) = 0.1;
P(10,11,2) = 0.8;
P(11,11,2) = 0.1;
P(11,8,2) = 0.1;
P(11,12,2) = 0.8;
P(12,12,2) = 0.9;
P(12,9,2) = 0.1;
% action left
P(1,1,3) = 0.9;
P(1,2,3) = 0.1;
P(2,1,3) = 0.1;
P(2,2,3) = 0.8;
P(2,3,3) = 0.1;
P(3,3,3) = 0.9;
P(3,2,3) = 0.1;
P(4,4,3) = 0.2;
P(4,1,3) = 0.8;
P(6,6,3) = 0.2;
P(6,3,3) = 0.8;
P(7,7,3) = 0.1;
P(7,4,3) = 0.8;
P(7,8,3) = 0.1;
P(8,8,3) = 0.8;
P(8,7,3) = 0.1;
P(8,9,3) = 0.1;
P(9,9,3) = 0.1;
P(9,6,3) = 0.8;
P(9,8,3) = 0.1;
P(10,10,3) = 0.1;
P(10,7,3) = 0.8;
P(10,11,3) = 0.1;
P(11,10,3) = 0.1;
P(11,8,3) = 0.8;
P(11,12,3) = 0.1;
P(12,12,3) = 0.1;
P(12,9,3) = 0.8;
P(12,9,3) = 0.1;
% action right
P(1,1,4) = 0.1;
P(1,2,4) = 0.1;
P(1,4,4) = 0.8;
P(2,1,4) = 0.1;
P(2,2,4) = 0.8;
P(2,3,4) = 0.1;
P(3,3,4) = 0.1;
P(3,2,4) = 0.1;
P(3,6,4) = 0.8;
P(4,4,4) = 0.2;
P(4,7,4) = 0.8;
P(6,6,4) = 0.2;
P(6,9,4) = 0.8;
P(7,7,4) = 0.1;
P(7,10,4) = 0.8;
P(7,8,4) = 0.1;
P(8,11,4) = 0.8;
P(8,7,4) = 0.1;
P(8,9,4) = 0.1;
P(9,9,4) = 0.1;
P(9,12,4) = 0.8;
P(9,8,4) = 0.1;
P(10,10,4) = 0.9;
P(10,11,4) = 0.1;
P(11,10,4) = 0.1;
P(11,11,4) = 0.8;
P(11,12,4) = 0.1;
P(12,12,4) = 0.9;
P(12,11,4) = 0.1;

r = [ 0 0 0 1; 0 0 0 -100; 0 0 0 0];
threshold = 0.01;
pi = randi(4,3,4);
pi(5) = 0;
g = 0.9;

% optimal pi is this:
% pi = [4, 4, 4, 1;
%       1, 0, 3, 3;
%       1, 3, 3, 2]

%% policy iteration
pi_old = zeros(3,4);
while ~isequal(pi_old, pi)
    % policy evaluation
    update=1;
    V = zeros(3,4);
    while update>=threshold
        V_old = V;
        for k=[1:4,6:12]
            sum = 0;
            for i=1:12
                sum = sum + P(k,i,pi(k))*V_old(i);
            end
            V(k) = r(k) + g*sum;
        end
        update = max(abs(V(:) - V_old(:)));
    end
    % policy improvement
    pi_old = pi;
    for k=[1:4,6:12]
        Vp = zeros(4,1);
        for a=1:4
            sum = 0;
            for i=1:12
                sum = sum + P(k,i,a)*V(i);
            end
            Vp(a) = r(k) + g*sum;
        end
        [~, pi(k)] = max(Vp(:));
    end
end
disp('~~~Policy iteration~~~')
disp(' ')
disp('Optimal value function V* is: ')
disp(V)
disp('Optimal policy derived from V* is: ')
disp(pi)

%% action-value iteration
update=1;
Q = zeros(12,4);
while update>=threshold
    Q_old = Q;
    for k=1:12
        for a=1:4
            sum = 0;
            for i=1:12
                sum = sum + P(k,i,a)*max(Q_old(i,:));
            end
            Q(k,a) = r(k) + g*sum;
        end
    end
    update = max(abs(Q(:) - Q_old(:)));
end
Q = reshape(Q,3,4,4);
[Q_act, pi] = max(Q,[],3);
pi(5)=0;
disp('~~~Q iteration~~~')
disp(' ')
disp('Optimal action-value function Q* (given by actions) is: ')
disp(Q)
disp('Optimal action-value function Q* is: ')
disp(Q_act)
disp('Optimal policy derived from Q* is: ')
disp(pi)