function y=Pooling(X)
%池化函数，输入是一个20*20*20的矩阵，输出是一个10*10*20的特征图%
%使用卷积的思想来完成池化%
filter = ones(2)/(2*2);
%规模%
[xrows,xcols,numfilters] = size(X);
y = zeros(xrows/2,xcols/2,numfilters);
for k = 1:numfilters
   image = conv2(X(:,:,k),filter,'valid'); 
   y(:,:,k) = image(1:2:end,1:2:end);
end

