I=imread('seven.JPG');
I1=rgb2gray(I);
%I1=(I1/255);
%size(I1)
I2=imresize(I1,[20 20]);
%imshow(I2)
I2=double(I2);
I2=(I2-122)/255;
I2=(I2(:))';

%size(I2)
%pred = predict(Theta1, Theta2, I2)
sel = randperm(size(X, 1));
sel = sel(1:10);
size(X(sel,:))
imshow(X(sel,:));
%displayData(X(sel, :));