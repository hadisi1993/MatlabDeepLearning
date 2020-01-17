function [W1,W2] = BackPropMmt(W1,W2,D,X)  
%W1,W2为权重矩阵，d为正确的输出，x为初始输入数据%
%维数:x:3*1 W1:4*3,W2:1*4,y:1*1%
alpha = 0.9;
beta = 0.9;
m1 = zeros(1,4);
m2 = zeros(4,3);
N = 4; %输入数据的组数%
for i = 1:N
   d = D(i);
   x= X(i,:)'; 
   v1 = W1*x;
   y1  = Sigmoid(v1);
   v2 = W2*y1;
   y2 = Sigmoid(v2);   %y2为输出%
   e = d - y2;
   delta = e.*y2.*(1-y2);
   dW2 = alpha*delta*y1';
   m1 = dW2+beta*m1;
   W2 = W2+m1;      %更新第二个权重矩阵的权重%
   e2 = W2'*delta;
   delta2 = e2.*y1.*(1-y1);
   dW1 = alpha*delta2*x';
   m2=dW1+beta*m2;
   W1 = W1+m2;      %更新第一个权重矩阵的权重%
end 
   
   
    

