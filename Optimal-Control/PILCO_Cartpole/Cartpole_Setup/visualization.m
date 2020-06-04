function visualization(s)
    xc = s(3);
    yc = 0;
    x = sin(s(1))+xc;
    y = cos(s(1))+yc;
    figure(100)
    plot([xc,x], [yc,y],'Marker','o','MarkerSize',6,'LineWidth',2,'Color','b')
    rectangle('Position',[xc-0.2,yc-0.2,0.4,0.2],'LineWidth',2,'EdgeColor','b')
    xlim([-2.5,2.5])
    ylim([-2,2])
end

