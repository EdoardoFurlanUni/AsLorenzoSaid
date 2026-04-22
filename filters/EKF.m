%  EKF
function [Xekf,V]=EKF(x0,y,V0,B,D,f,h,sym_x,T)

%% EKF
n=size(B,1);
p=size(D,1);
Q=B*B';
R=D*D';
Xekf=zeros(n,T+1);
Xekf(:,1)=x0;
Xn=zeros(n,T);
V(:,:,1)=V0;
V=zeros(n,n,T+1);
A=zeros(n,n,T); 
C=zeros(p,n,T);
G=zeros(n,p,T);
th=zeros(1,T);
for i=1:T
    %C_t
    h_l=jacobian(h,sym_x);
    C(:,:,i)=subs(h_l,sym_x,Xekf(:,i));
    %L_t
    L=V(:,:,i)*C(:,:,i)'*inv(C(:,:,i)*V(:,:,i)*C(:,:,i)'+R); 
    %h(\hat x_t,u_t)
    hn=subs(h,sym_x,Xekf(:,i));
    hn = eval(hn);
    %\hat x_t|t
    Xn(:,i)=Xekf(:,i)+L*(y(:,i)-hn);
    %A_t
    f_l=jacobian(f,sym_x);
    A(:,:,i)=subs(f_l,sym_x,Xn(:,i));
    %G_t
    G(:,:,i)=A(:,:,i)*L;
    %\hat x_t+1
    Xekf(:,i+1)=subs(f,sym_x,Xn(:,i));     
    %V_t+1
    V(:,:,i+1)=A(:,:,i)*V(:,:,i)*A(:,:,i)'-A(:,:,i)*V(:,:,i)*C(:,:,i)'*inv(C(:,:,i)*V(:,:,i)*C(:,:,i)'+R)*C(:,:,i)*V(:,:,i)*A(:,:,i)'+Q; 
end