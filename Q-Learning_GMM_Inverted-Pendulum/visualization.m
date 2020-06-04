function visualization(s)
    x = sin(s(1));
    y = cos(s(1));
    figure(100)
    plot([0,x], [0,y],'Marker','o','MarkerSize',6,'LineWidth', 2)
    axis equal
    xlim([-2,2])
    ylim([-2,2])
end

