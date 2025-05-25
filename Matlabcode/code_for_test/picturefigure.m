clear;
close all;
clc;
I=imread('picture.jpg');
[H,W,G]=size(I);
H1=H/S;H2=H*2/3;W1=W/10;W2=W*85/100;
S=(W2-W1)*(H2-H1);
I=imcrop(I,[W1,H1,W2,H2]);
subplot(2,3,1),imshow(I),title('orginalpicture');
I1=rgb2gray(I);
I2=im2bw(I1,0.6);
subplot(2,3,2),imshow(I2),title('thegrayone')
I1=~I2;


