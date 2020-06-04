function [x,t] = forward_euler(dxdt,tstep,tend,xbeg)

    t = 0:tstep:tend;
    x = zeros(2,length(t));
    x(:,1) = xbeg;
    dxdt1 = dxdt{1};
    dxdt2 = dxdt{2};
    
    for k=2:length(t)    
        x(1,k) = x(1,k-1)+tstep*dxdt1(x(1,k-1),x(2,k-1));
        if x(1,k)>pi
            x(1,k) = x(1,k)-2*pi;
        elseif x(1,k)<=-pi
            x(1,k) = x(1,k)+2*pi;
        end
        x(2,k) = x(2,k-1)+tstep*dxdt2(x(1,k-1),x(2,k-1));
        % add a second term from Taylor expansion
        if x(2,k)>2*pi
            x(2,k)=2*pi;
        elseif x(2,k)<=-2*pi
            x(2,k)=-2*pi;
        end
    end
end