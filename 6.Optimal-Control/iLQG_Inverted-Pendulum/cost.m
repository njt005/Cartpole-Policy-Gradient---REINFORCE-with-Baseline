function c = cost(s, a)
    c = [];
    for k = 1:size(s,2)
        c = [c,abs(s(1,k))];
    end
end

