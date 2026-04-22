%  EKF: modified -> added sym_u in function f & matlabFunctions instead of subs !
function [Xekf,V]=EKF_fast(x0,y,u,V0,B,D,f_num,h_num,f_jac_num, h_jac_num,T)

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
    C(:,:,i)=h_jac_num(Xekf(:,i));
    %L_t
    L=V(:,:,i)*C(:,:,i)'*inv(C(:,:,i)*V(:,:,i)*C(:,:,i)'+R); 
    %h(\hat x_t,u_t)
    hn=h_num(Xekf(:,i));
    %\hat x_t|t
    Xn(:,i)=Xekf(:,i)+L*(y(:,i)-hn);
    %A_t
    A(:,:,i)=f_jac_num(Xn(:,i), u(:,i));
    %G_t
    G(:,:,i)=A(:,:,i)*L;
    %\hat x_t+1
    Xekf(:,i+1)=f_num(Xn(:,i), u(:,i));     
    %V_t+1
    V(:,:,i+1)=A(:,:,i)*V(:,:,i)*A(:,:,i)'-A(:,:,i)*V(:,:,i)*C(:,:,i)'*inv(C(:,:,i)*V(:,:,i)*C(:,:,i)'+R)*C(:,:,i)*V(:,:,i)*A(:,:,i)'+Q; 
end
