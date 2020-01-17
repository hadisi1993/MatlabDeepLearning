%图片路径%
imagepath = 'C:\Users\wzq\Desktop\wzq\研究生\机器学习和深度学习\数据集\Matlab deep learning\t10k-images.idx3-ubyte';
%图片%
Images = loadMNISTImages(imagepath);
%将图像的大小限定为28*28%
Images = reshape(Images,28,28,[]);
labelpath = 'C:\Users\wzq\Desktop\wzq\研究生\机器学习和深度学习\数据集\Matlab deep learning\t10k-labels.idx1-ubyte';
%标签%
Labels = loadMNISTLabels(labelpath);
%将标签集中所有0转化为10%
Labels(Labels == 0) =10;
%取前8000张图片作为训练集%
X=Images(:,:,1:8000);
%取8000张标签,标签就是数字,代表第几个数%
D = Labels(1:8000);
%训练过程%

%为卷积核赋值：9*9*20%
W1 = 1e-2*randn([9 9 20]);
%为隐藏层和输入层之间的权重赋值,输入是10*10*20的矩阵，转化为2000*1的向量%
W2 = (2*rand(100,2000)-1)*sqrt(6)/sqrt(360+2000);
%为隐藏层和输出层之间的权重赋值%
W3 = (2*rand(10,100)-1)*sqrt(6)/sqrt(10+100);
%训练3次%
for epoch =1:3
    [W1,W2,W3] = MinstCNN(W1,W2,W3,D,X);
end
%保存当前所有的变量%
save('MnistCNN.mat');
%测试过程%
X = Images(:,:,8001:10000);
D = Labels(8001:10000);
N = length(D);

%使用变量acc来判断精确度%
acc =0;
for k =1:N
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
    %找到概率最大的数的下标%
    [~,i] = max(y);
    if i == D(k)
        acc = acc+1;
    end
end
acc = acc/N;
fprintf('Accuracy is %f\n',acc);
