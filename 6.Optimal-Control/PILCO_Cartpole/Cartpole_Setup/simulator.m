function [snext, rnext] = simulator(s, a, dur_a, dt)
    % w is the derivative of theta, v is the derivative of x
    % s = [theta;w;x;v]
    
    % consts
    mp = 0.15;
    mc = 1;
    l = 0.75;
    g = 9.81;
    % a in range(-50,50); it's the force now
    
    dthetadt = @(theta,w,x,v) w;
    F = a;
    if (s(3)==-1.5 && a<0) || (s(3)==1.5 && a>0)
        F = 0;
    end
    dwdt = @(theta,w,x,v) (g*sin(theta)+cos(theta)*(-F-mp*l*w*sin(theta)))/...
                          l*(4/3-mp*cos(theta)^2/(mc+mp));
    dxdt = @(theta,w,x,v) v;
    dvdt = @(theta,w,x,v) (F+mp*l*(w*sin(theta)-dwdt(theta,w,x,v)*cos(theta)))/...
                          (mc+mp);
    functions = {dthetadt,dwdt,dxdt,dvdt};
    
    [snext,~] = forward_euler(functions,dt,dur_a,s);
    snext = snext(:,end);
    rnext = -abs(snext(1));
end

