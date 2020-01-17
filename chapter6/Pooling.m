function y=Pooling(X)
%�ػ�������������һ��20*20*20�ľ��������һ��10*10*20������ͼ%
%ʹ�þ����˼������ɳػ�%
filter = ones(2)/(2*2);
%��ģ%
[xrows,xcols,numfilters] = size(X);
y = zeros(xrows/2,xcols/2,numfilters);
for k = 1:numfilters
   image = conv2(X(:,:,k),filter,'valid'); 
   y(:,:,k) = image(1:2:end,1:2:end);
end

