function r= Dropout(x,radio)
%�ú��������������ֽڵ��ֵ%
[n,m] =size(x); 
r = zeros(n,m);
d = 1/(1-radio);  %��0��ȡ��ֵ����֤��������������%
num = round(n*m*(1-radio));
idx = randperm(n*m,num);
r(idx) = d;


