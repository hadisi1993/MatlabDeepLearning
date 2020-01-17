function [W1,W2,W3] = MinstCNN(W1,W2,W3,D,X)
%����X��һ��28*28*N�ľ���,D��һ��10*10��������W1��һ��9*9*20�ľ���W2��һ��100*2000�ľ���W3��һ��10*100�ľ���%
alpha = 0.01;
beta = 0.9;
N = length(D);
%ʹ��batch�ķ���%
bsize = 100;
blist =1:bsize:(N-bsize+1);
%ʹ�ö���������Ȩ��%
%�������¹�ʽ%
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
       %ԭʼͼ�����%
       img = X(:,:,k);
       %���о���������������һ��20*20*20�ľ���%
       y1 = Conv(W1,img);
       y2 = Relu(y1);
       %�ػ�%
       y3 = Pooling(y2);
       %�ػ������Ժ����BP��������з���%
       y4 = reshape(y3,[],1);
       v1 = W2*y4;
       y5 = Relu(v1);
       v = W3*y5;
       y = Softmax(v);
       %���ȱ��룬��������������0~9֮���һ��������������к�������Ϊ����һ����ȷ������%
       d = zeros(10,1);
       d(sub2ind(size(d),D(k),1))=1;
       %�������BP���������ķ��򴫲�����%
       e = d -y;
       delta = e;
       e2 = W3'*delta;
       delta2 = (v1>0).*e2;
       e3 = W2'*delta2;
       %���ԭ�ⲻ���Ĵ��ݣ�Ҫ�ı��ֻ��������Ľṹ����ԭ����2000*1��Ϊ10*10*20%
       e4 = reshape(e3,size(y3));
       %������һ�δ���������ǽ���������㴫�ݵ�����㵱�У�����ʵ��ϸ����ʱ���

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
       %��������Ȩ��%
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

