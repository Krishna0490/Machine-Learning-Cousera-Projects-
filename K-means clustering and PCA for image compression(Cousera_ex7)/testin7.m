i = [10 10; 6 2; 8 5];
size(i);
a=[2 2];
b=a-i;
c=b.^2
d=sum(c,2)
[e,f]=min(d);
a2=[2 ;3;4];
i./a2;
A = double(imread('bird_small.png'));
X = reshape(A, img_size(1) * img_size(2), 3);
size(X)