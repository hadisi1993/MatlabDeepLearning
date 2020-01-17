%ʹ��SGD�����ĵ���������%
function  W = DeltaSGD(W,x,d)  
alpha = 0.9;  %ȡѧϰ��Ϊ0.9%
v = W*x;  
y = Sigmoid(v);
e = d-y;%���%
delta = alpha*y.*(1-y).*e;
for i = 1:length(x)
    for j = 1:length(x)
       W(i,j) = W(i,j)+alpha*delta(i)*x(j); 
    end
end