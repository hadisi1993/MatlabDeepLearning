function [W1,W2,W3] = MinstCNN(W1,W2,W3,D,X)
%这里X是一个28*28*N的矩阵,D是一个10*10的向量，W1是一个9*9*20的矩阵，W2是一个100*2000的矩阵，W3是一个10*100的矩阵%
alpha = 0.01;
beta = 0.9;
N = length(D);
%使用batch的方法%
bsize = 100;
blist =1:bsize:(N-bsize+1);
%使用动量来更新权重%
%动量更新公式%
% m = alpha*dW+beta*m%
% W = W + m%
momentum1 = zeros(size(W1));
momentum2 = zeros(size(W2));
momentum3 = zeros(size(W3));
for batch = 1:length(blist)
   dW1 = zeros(size(W1));
   dW2 = zeros(size(W2));
   dW3 = zeros(size(W3));
   begin = blist(batch);
   for k = begin:(begin+bsize-1)
       %原始图像矩阵%
       img = X(:,:,k);
       %进行卷积操作，输出的是一个20*20*20的矩阵%
       y1 = Conv(W1,img);
       y2 = Relu(y1);
       %池化%
       y3 = Pooling(y2);
       %池化结束以后进入BP神经网络进行分类%
       y4 = reshape(y3,[],1);
       v1 = W2*y4;
       y5 = Relu(v1);
       v = W3*y5;
       y = Softmax(v);
       %独热编码，读进来的数据是0~9之间的一个数，下面的两行函数就是为构造一个正确的向量%
       d = zeros(10,1);
       d(sub2ind(size(d),D(k),1))=1;
       %下面就是BP神经网络误差的反向传播过程%
       e = d -y;
       delta = e;
       e2 = W3'*delta;
       delta2 = (v1>0).*e2;
       e3 = W2'*delta2;
       %误差原封不动的传递，要改变的只有误差矩阵的结构，从原来的2000*1变为10*10*20%
       e4 = reshape(e3,size(y3));
       %下面这一段代码的作用是将误差从输出层传递到卷积层当中，具体实现细节暂时不深究

       e5 = zeros(size(y2));
       W4 = ones(size(y2))/(2*2);
       for c = 1:20
          e5(:,:,c) = kron(e4(:,:,c),ones([2,2])).*W4(:,:,c); 
       end

       delta3 = (y2>0).*e5;
       delta4_x = zeros(size(W1));
       for c=1:20
          delta4_x(:,:,c) = conv2(img(:,:),rot90(delta3(:,:,c),2),'valid'); 
       end
       %更新所有权重%
       dW1 = dW1+delta4_x;
       dW2 = dW2+delta2*y4';
       dW3 = dW3+delta*y5';
   end
   
   dW1 = dW1/bsize;
   dW2 = dW2/bsize;
   dW3 = dW3/bsize;
   
   momentum1 = alpha*dW1+beta*momentum1;
   W1 = W1+momentum1;
   momentum2 =alpha*dW2 + beta*momentum2 ;
   W2 = W2+momentum2;
   momentum3 = alpha*dW3 + beta*momentum3;
   W3 = W3+momentum3;
end

