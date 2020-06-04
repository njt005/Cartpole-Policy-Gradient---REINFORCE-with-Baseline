function snext = simulator(s, a, dur_a, dt)

    if nargin < 3
        dur_a = 0.01;
        dt = 0.01;
    end
    % v is the derivative of theta
    % s = [theta;v]
    
    % consts
    m = 1;
    l = 1;
    g = 9.8;
    mu = 0.01;
    % a in range(-5,5)
    
    dthetadt = @(theta,v) v;
    snext = [];
    for k=1:size(s,2)
        dvdt = @(theta,v) -mu/(m*l^2) * v + g/l * sin(theta) + 1/(m*l^2) * a(:,k);
        functions = {dthetadt,dvdt};
    
        [sn,~] = forward_euler(functions,dt,dur_a,s(:,k));
        snext = [snext,sn(:,end)];
    end
end