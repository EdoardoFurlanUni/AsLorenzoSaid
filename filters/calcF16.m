function [F] = calcF16(Delta_t,Delta_theta_m,x_t,Delta_v_m,q0,q1,q2,q3)
% Delta_theta_m,Delta_v_m: Input
% x_t: State filter $\hat x_{t|t}$ at time t
% q0,q1,q2,q3: Estimate quaternion

F=zeros(16,16);
%% F^{q_{k+1}}_{q_k} in Eq. 78
F(1,1) = 1;
F(1,2) = -(Delta_theta_m(1)-x_t(11))/2;
F(1,3) = -(Delta_theta_m(2)-x_t(12))/2;
F(1,4) = -(Delta_theta_m(3)-x_t(13))/2;
F(2,1) = -F(1,2);F(2,2) = 1;F(2,3) = -F(1,4);F(2,4)=F(1,3);
F(3,1) = -F(2,4);F(3,2)=F(1,4);F(3,3)=1;F(3,4)=-F(1,2);
F(4,1) = -F(1,4);F(4,2) = -F(1,3);F(4,3) =F(1,2);F(4,4)=1;

%% F^{q_{k+1}}_{Delta_theta_b^k} in Eq. 79
F(1,11)= q1/2;  F(1,12)=q2/2;  F(1,13)=q3/2;
F(2,11)= -q0/2; F(2,12)=q3/2;  F(2,13)=-q2/2;
F(3,11)= -q3/2; F(3,12)=-q0/2; F(3,13)=q1/2;
F(4,11)= q2/2;  F(4,12)=-q1/2; F(4,13)=-q0/2;

%% F^{v_{k+1}}_{q_k} in Eq. 81
zh1=[q0 -q3 q2;q1 q2 q3;-q2 q1 q0;-q3 -q0 q1;];
zh2=[q3 q0 -q1;q2 -q1 -q0;q1 q2 q3;q0 -q3 q2;];
zh3=[-q2 q1 q0;q3 q0 -q1;-q0 q3 -q2;q1 q2 q3;];

F(5,1:4)=2*(zh1*(Delta_v_m(1:3)-x_t(14:16)))';
F(6,1:4)=2*(zh2*(Delta_v_m(1:3)-x_t(14:16)))';
F(7,1:4)=2*(zh3*(Delta_v_m(1:3)-x_t(14:16)))';

%% F
F(5:7,5:7)=eye(3);F(8:10,8:10)=eye(3);F(11:13,11:13)=eye(3);F(14:16,14:16)=eye(3);
F(5:7,14:16)=-[q0^2+q1^2-q2^2-q3^2 2*q1*q2-2*q0*q3 2*q0*q2+2*q1*q3;...
    2*q0*q3+2*q1*q2 q0^2-q1^2+q2^2-q3^2 2*q2*q3-2*q0*q1;...
    2*q1*q3-2*q0*q2 2*q0*q1+2*q2*q3 q0^2-q1^2-q2^2+q3^2];
F(8:10,5:7)=[Delta_t 0 0;0 Delta_t 0;0 0 Delta_t];
end