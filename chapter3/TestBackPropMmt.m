D = [0 1 1 0];
X =[0 0 1;0 1 1;1 0 1;1 1 1];
alpha = 0.9;
N = 4;   %有4组训练数据%
enpch = 10000;%训练10000次%
W1 = 2*rand(4,3)-1;
W2 = 2*rand(1,4)-1;
for i = 1:enpch
    [W1,W2] = BackPropMmt(W1,W2,D,X);
end
for i =1:N
   x= X(i,:)'; 
   v1 = W1*x;
   y1  = Sigmoid(v1);
   v2 = W2*y1;
   y2 = Sigmoid(v2);   %y2为输出% 
   disp(y2);
end