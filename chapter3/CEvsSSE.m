clear all;
X = [0 0 1;0 1 1;1 0 1;1 1 1];
D = [0 0 1 1];
N =4;
%ŒÛ≤Ó%
E1 = zeros(1000,1);    
E2 = zeros(1000,1); 
%»®÷ÿæÿ’Û%
W11 = 2*rand(4,3)-1;   
W12 = 2*rand(1,4)-1;
W21 = W11;
W22 = W12;

for epoch =1:1000
    [W11,W12] = BackPropXOR(W11,W12,D,X);
    [W21,W22] = BackPropCE(W21,W22,D,X);
    eps1 = 0;
    eps2 = 0;
    for k=1:N
        x = X(k,:)';
        d = D(k);
        v11 = W11*x;
        v12 = W21*x;
        y11 = Sigmoid(v11);
        y12 = Sigmoid(v12);
        v21 = W12*y11;
        v22 = W22*y12;
        y21 = Sigmoid(v21);
        y22 = Sigmoid(v22);
        eps1 = eps1 + (d-y21)^2;
        eps2 = eps2 + (d-y22)^2;
    end
        E1(epoch) = eps1/N;
        E2(epoch) = eps2/N;
end
    
plot(E1,'r');
hold on 
plot(E2,'b:')
xlabel('Epoch');
ylabel('Average of Tranin error')
legend('Sum of squared error','Cross Entroy');