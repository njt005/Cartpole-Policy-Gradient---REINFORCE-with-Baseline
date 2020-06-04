function [snext, rnext] = simulator(s, a, dur_a, dt)
    % v is the derivative of theta
    % s = [theta;v]
    
    % consts
    m = 1;
    l = 1;
    g = 9.8;
    mu = 0.01;
    % a in range(-5,5)
    
    dthetadt = @(theta,v) v;
    dvdt = @(theta,v) -mu/(m*l^2) * v + g/l * sin(theta) + 1/(m*l^2) * a;
    functions = {dthetadt,dvdt};
    
    [snext,~] = forward_euler(functions,dt,dur_a,s);
    snext = snext(:,end);
    rnext = -abs(snext(1));
end

