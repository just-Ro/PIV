function [K, R, T]=qrcamera(P)
% [K, R, T]=qrcamera(P)
[q r]=qr(inv(P(:,1:3)));
K=inv(r);
D = diag(sign(diag(K)));
K=K*D;
%K(3,3)=1
T=inv(K)*P(:,4);
Ks=K(3,3);
K=K/Ks;
R=D*q';
end
