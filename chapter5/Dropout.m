function r= Dropout(x,radio)
%该函数用来舍弃部分节点的值%
[n,m] =size(x); 
r = zeros(n,m);
d = 1/(1-radio);  %非0数取该值，保证舍弃后总量不变%
num = round(n*m*(1-radio));
idx = randperm(n*m,num);
r(idx) = d;


