clear all
close all
clc

% set up the optimization problem
DYNCST  = @(x,u,i) car_dyn_cst(x,u);
T       = 500;              % horizon
x0      = [pi;0];           % initial state
u0      = .1*randn(1,T);    % initial controls
Op.lims  = [-5 5];          % torque limits

% === run the optimization!
% [x,u]= iLQG(DYNCST, x0, u0, Op);
% 
% save('states_actions.mat','x','u')

load('states_actions.mat')

% visualization
for s=x
    visualization(s);
    pause(0.01)
end

function [f,c,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = car_dyn_cst(x,u)
    % combine car dynamics and cost
    % use helper function finite_difference() to compute derivatives
    u(isnan(u)) = 0;
    
    if nargout == 2
        f = simulator(x,u);
        c = cost(x,u);
    else
        % state and control indices
        ix = 1:2;
        iu = 3;

        % dynamics first derivatives
        xu_dyn = @(xu) simulator(xu(ix,:),xu(iu,:));
        J      = finite_difference(xu_dyn, [x; u]);
        fx     = J(:,ix,:);
        fu     = J(:,iu,:);

        [fxx,fxu,fuu] = deal([]);   

        % cost first derivatives
        xu_cost = @(xu) cost(xu(ix,:),xu(iu,:));
        J       = squeeze(finite_difference(xu_cost, [x; u]));
        cx      = J(ix,:);
        cu      = J(iu,:);

        % cost second derivatives
        xu_Jcst = @(xu) squeeze(finite_difference(xu_cost, xu));
        JJ      = finite_difference(xu_Jcst, [x; u]);
        JJ      = 0.5*(JJ + permute(JJ,[2 1 3])); %symmetrize
        cxx     = JJ(ix,ix,:);
        cxu     = JJ(ix,iu,:);
        cuu     = JJ(iu,iu,:);

        [f,c] = deal([]);
    end
end

function J = finite_difference(fun, x, h)
    % simple finite-difference derivatives
    % assumes the function fun() is vectorized

    if nargin < 3
        h = 2^-17;
    end

    [n, K]  = size(x);
    H       = [zeros(n,1) h*eye(n)];
    H       = permute(H, [1 3 2]);
    X       = pp(x, H);
    X       = reshape(X, n, K*(n+1));
    Y       = fun(X);
    m       = numel(Y)/(K*(n+1));
    Y       = reshape(Y, m, K, n+1);
    J       = pp(Y(:,:,2:end), -Y(:,:,1)) / h;
    J       = permute(J, [1 3 2]);
end

% utility functions: singleton-expanded addition and multiplication
function c = pp(a,b)
    c = bsxfun(@plus,a,b);
end

function c = tt(a,b)
    c = bsxfun(@times,a,b);
end