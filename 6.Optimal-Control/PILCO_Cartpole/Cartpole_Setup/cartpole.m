clear all
close all
clc

s = [pi;0;0;0];

for k=1:600
    a = rand*100-50;
    [snext, rnext] = simulator(s, a, 1/60, 1/60);
    visualization(s);
    pause(1/60)
    s = snext;
end