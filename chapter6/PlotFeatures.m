clear all 
load('MnistCNN.mat')
%取第二幅图像%
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

%初始的输入图像%
figure;
display_network(image(:));
title('Input image');

%被训练的卷积核卷积核%
convFilters = zeros(9*9,20);
for i = 1:20
    filter = W1(:,:,i);
    convFilters(:,i) = filter(:);
end
figure;
display_network(convFilters);
title('Convolution Filters');

%图像经过卷积层处理后的结果%

fList = zeros(20*20,20);
for i = 1:20
   feature = y1(:,:,i);
   fList(:,i) = feature(:);
end
figure;
display_network(fList);
title('Features [Convolution]');

%经过卷积和Relu处理的图像%
fList = zeros(20*20,20);
for i = 1:20
   feature = y2(:,:,i);
   fList(:,i) = feature(:);
end
figure;
display_network(fList);
title('Features [Convolution+Relu]');

%经过卷积、Relu和池化处理的图像%
fList = zeros(10*10,20);
for i = 1:20
   feature = y3(:,:,i);
   fList(:,i) = feature(:);
end
figure;
display_network(fList);
title('Features [Convolution+Relu+mean pooling]');

    