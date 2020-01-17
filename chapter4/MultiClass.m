function [W1,W2] = MultiClass(W1,W2,D,X)
%训练过程和第三章大致相同%
alpha = 0.9;
N = 5;
for i = 1:N
    x = reshape(X(:,:,i),25,1);   %将输入化为25*1的矩阵%
    d = D(i,:)';    %正确的输出%
    v1 = W1*x;
    y1 = Sigmoid(v1);
    v = W2*y1;
    y = Softmax(v);
    e = d - y ;
    delta = e;
    e1 = W2'*delta;
    delta1 = y1.*(1-y1).*e1;
    dW2 = alpha*delta*y1';
    W2 = W2+dW2;
    dW1 = alpha*delta1*x';
    W1 =W1+dW1;
end 