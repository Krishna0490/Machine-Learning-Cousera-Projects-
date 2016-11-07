clear all
clc
A=imread('test.JPG');
I = rgb2gray(A);
output_size=[20 20];
B=imresize(I,output_size);
B=B/255;
B(:);
pred = predictOneVsAll(all_theta, B);