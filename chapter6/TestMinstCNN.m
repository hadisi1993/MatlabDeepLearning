%ͼƬ·��%
imagepath = 'C:\Users\wzq\Desktop\wzq\�о���\����ѧϰ�����ѧϰ\���ݼ�\Matlab deep learning\t10k-images.idx3-ubyte';
%ͼƬ%
Images = loadMNISTImages(imagepath);
%��ͼ��Ĵ�С�޶�Ϊ28*28%
Images = reshape(Images,28,28,[]);
labelpath = 'C:\Users\wzq\Desktop\wzq\�о���\����ѧϰ�����ѧϰ\���ݼ�\Matlab deep learning\t10k-labels.idx1-ubyte';
%��ǩ%
Labels = loadMNISTLabels(labelpath);
%����ǩ��������0ת��Ϊ10%
Labels(Labels == 0) =10;
%ȡǰ8000��ͼƬ��Ϊѵ����%
X=Images(:,:,1:8000);
%ȡ8000�ű�ǩ,��ǩ��������,����ڼ�����%
D = Labels(1:8000);
%ѵ������%

%Ϊ����˸�ֵ��9*9*20%
W1 = 1e-2*randn([9 9 20]);
%Ϊ���ز�������֮���Ȩ�ظ�ֵ,������10*10*20�ľ���ת��Ϊ2000*1������%
W2 = (2*rand(100,2000)-1)*sqrt(6)/sqrt(360+2000);
%Ϊ���ز�������֮���Ȩ�ظ�ֵ%
W3 = (2*rand(10,100)-1)*sqrt(6)/sqrt(10+100);
%ѵ��3��%
for epoch =1:3
    [W1,W2,W3] = MinstCNN(W1,W2,W3,D,X);
end
%���浱ǰ���еı���%
save('MnistCNN.mat');
%���Թ���%
X = Images(:,:,8001:10000);
D = Labels(8001:10000);
N = length(D);

%ʹ�ñ���acc���жϾ�ȷ��%
acc =0;
for k =1:N
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
    %�ҵ��������������±�%
    [~,i] = max(y);
    if i == D(k)
        acc = acc+1;
    end
end
acc = acc/N;
fprintf('Accuracy is %f\n',acc);
