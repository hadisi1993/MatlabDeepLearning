%使用SGD方法的单层神经网络%
function  W = DeltaSGD(W,x,d)  
alpha = 0.9;  %取学习率为0.9%
v = W*x;  
y = Sigmoid(v);
e = d-y;%误差%
delta = alpha*y.*(1-y).*e;
for i = 1:length(x)
    for j = 1:length(x)
       W(i,j) = W(i,j)+alpha*delta(i)*x(j); 
    end
end