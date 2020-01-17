function y = Conv(W,X)
%卷积函数，输入是一个28*28的图像矩阵和9*9*20的卷积核，输出是一个20*20*20的矩阵%
[wrows,wcols,numfilter] = size(W);
[xrows,xcols] = size(X);
%计算卷积后矩阵的维度，结果为20*20%
yrows = xrows-wrows+1;
ycols = xcols-wcols+1;
y = zeros(yrows,ycols,numfilter);
for k =1:numfilter
    %进行卷积操作%
    %过滤器%
    filter = W(:,:,k);
    %做两次旋转%
    filter = rot90(filter,2);
    %计算卷积核%
    y(:,:,k) = conv2(X,filter,'valid');
end

