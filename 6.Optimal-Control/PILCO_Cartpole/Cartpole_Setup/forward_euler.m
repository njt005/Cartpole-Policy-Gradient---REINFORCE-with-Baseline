function [x,t] = forward_euler(dxdt,tstep,tend,xbeg)

    t = 0:tstep:tend;
    x = zeros(4,length(t));
    x(:,1) = xbeg;
    dxdt1 = dxdt{1};
    dxdt2 = dxdt{2};
    dxdt3 = dxdt{3};
    dxdt4 = dxdt{4};
    
    for k=2:length(t)    
        x(1,k) = x(1,k-1)+tstep*dxdt1(x(1,k-1),x(2,k-1),x(3,k-1),x(4,k-1));
        if x(1,k)>pi
            x(1,k) = x(1,k)-2*pi;
        elseif x(1,k)<=-pi
            x(1,k) = x(1,k)+2*pi;
        end
        x(2,k) = x(2,k-1)+tstep*dxdt2(x(1,k-1),x(2,k-1),x(3,k-1),x(4,k-1));
        % add a second term from Taylor expansion
        if x(2,k)>2*pi
            x(2,k)=2*pi;
        elseif x(2,k)<=-2*pi
            x(2,k)=-2*pi;
        end
        x(3,k) = x(3,k-1)+tstep*dxdt3(x(1,k-1),x(2,k-1),x(3,k-1),x(4,k-1));
        if x(3,k)>=1.5
            x(3,k) = 1.5;
        elseif x(3,k)<=-1.5
            x(3,k) = -1.5;
        end
        x(4,k) = x(4,k-1)+tstep*dxdt4(x(1,k-1),x(2,k-1),x(3,k-1),x(4,k-1));
        if x(4,k)>=10
            x(4,k) = 10;
        elseif x(4,k)<=-10
            x(4,k) = -10;
        end
    end
end