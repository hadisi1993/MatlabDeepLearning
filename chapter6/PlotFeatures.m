clear all 
load('MnistCNN.mat')
%ȡ�ڶ���ͼ��%
k = 2;
image = X(:,:,k);
y1 = Conv(W1,image);
y2 = Relu(y1);
y3 = Pooling(y2);
y4 = reshape(y3,[],1);
v5 = W2*y4;
y5 = Relu(v5);
v = W3*y5;
y = Softmax(v);

%��ʼ������ͼ��%
figure;
display_network(image(:));
title('Input image');

%��ѵ���ľ���˾����%
convFilters = zeros(9*9,20);
for i = 1:20
    filter = W1(:,:,i);
    convFilters(:,i) = filter(:);
end
figure;
display_network(convFilters);
title('Convolution Filters');

%ͼ�񾭹�����㴦���Ľ��%

fList = zeros(20*20,20);
for i = 1:20
   feature = y1(:,:,i);
   fList(:,i) = feature(:);
end
figure;
display_network(fList);
title('Features [Convolution]');

%���������Relu�����ͼ��%
fList = zeros(20*20,20);
for i = 1:20
   feature = y2(:,:,i);
   fList(:,i) = feature(:);
end
figure;
display_network(fList);
title('Features [Convolution+Relu]');

%���������Relu�ͳػ������ͼ��%
fList = zeros(10*10,20);
for i = 1:20
   feature = y3(:,:,i);
   fList(:,i) = feature(:);
end
figure;
display_network(fList);
title('Features [Convolution+Relu+mean pooling]');

    