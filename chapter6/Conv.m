function y = Conv(W,X)
%���������������һ��28*28��ͼ������9*9*20�ľ���ˣ������һ��20*20*20�ľ���%
[wrows,wcols,numfilter] = size(W);
[xrows,xcols] = size(X);
%������������ά�ȣ����Ϊ20*20%
yrows = xrows-wrows+1;
ycols = xcols-wcols+1;
y = zeros(yrows,ycols,numfilter);
for k =1:numfilter
    %���о������%
    %������%
    filter = W(:,:,k);
    %��������ת%
    filter = rot90(filter,2);
    %��������%
    y(:,:,k) = conv2(X,filter,'valid');
end

