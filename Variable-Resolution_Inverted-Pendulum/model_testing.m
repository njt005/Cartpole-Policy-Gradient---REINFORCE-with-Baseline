clear all
close all
clc

s = [pi;0];

for k=1:200
    a = rand*10-5;
    [snext, rnext] = simulator(s, a, 0.1, 0.001);
    visualization(s);
    pause(0.05)
    s = snext;
end