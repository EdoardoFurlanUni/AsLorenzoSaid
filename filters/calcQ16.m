function [Q] = calcQ16(wb,ab,Delta_theta_n,Delta_v_n,q0,q1,q2,q3)
% calcQ16 constructs the 16-dimensional process noise covariance matrix Q.
%
% Inputs:
%   wb:
%       Standard deviation associated with the gyroscope bias error state
%       component: \Delta_{w,k+1}=\Delta_{w,k}+wb*v, v WGN with variance 1, see [1].
%
%   ab:
%       Standard deviation associated with the accelerometer bias error state
%       component: \Delta_{a,k+1}=\Delta_{a,k}+ab*v, v WGN with variance 1, see [1]. in [1].
%
%   Delta_theta_n:
%       Standard deviation of the angular increment noise, corresponding to
%       \Delta_{\sigma_w,k} in [1].
%
%   Delta_v_n:
%       Standard deviation of the velocity increment noise, corresponding to
%       \Delta_{\sigma_v,k} in [1].
%
%       In [1], both \Delta_{\sigma_w,k} and \Delta_{\sigma_v,k} are treated
%       as scalar quantities. In this implementation, they are extended to
%       three-dimensional vectors (the 3d dimension could have a different value), 
%       which provides more tuning degrees of
%       freedom for the noise covariance design.
%
%   q0, q1, q2, q3:
%       Components of the unit quaternion used to construct the noise
%       propagation matrix G.
%
% Output:
%   Q:
%       The 16-by-16 process noise covariance matrix, i.e. Qe+Gk*Qe,k*Gk' in [1] .
%
% Reference:
%   [1] Yi, Shenglun, et al. "Data-driven robust UAV position estimation in GPS signal-challenged environment." 2025 American Control Conference (ACC). IEEE, 2025.
%  
% Suggested values 
%
% Delta_theta_n = [2.6e-5; 2.6e-5; 2.6e-5;]; 
% 
% Delta_v_n = [1.66e-3; 1.66e-3; 1.66e-3;]; 
%ab = [1.66e-4; 1.66e-4; 1.66e-4;];
%wb = [2.6e-6; 2.6e-6; 2.6e-6;];  

%% Noise propagation matrix G
G=zeros(16,6);
G(1,1)=q1/2;G(1,2)=q2/2;G(1,3)=q3/2;G(2,1)=-q0/2;G(2,2)=q3/2;G(2,3)=-q2/2;
G(3,1)=-q3/2;G(3,2)=-q0/2;G(3,3)=q1/2;G(4,1)=q2/2;G(4,2)=-q1/2;G(4,3)=-q0/2;
G(5:7,4:6)=-[q0^2+q1^2-q2^2-q3^2 2*q1*q2-2*q0*q3 2*q0*q2+2*q1*q3;2*q0*q3+2*q1*q2 q0^2-q1^2+q2^2-q3^2 2*q2*q3-2*q0*q1;2*q1*q3-2*q0*q2 2*q0*q1+2*q2*q3 q0^2-q1^2-q2^2+q3^2];

%% Process noise covariance induced by angular and velocity increment noises
Q=G(:,:)*diag([Delta_theta_n(1)^2 Delta_theta_n(2)^2 Delta_theta_n(3)^2 Delta_v_n(1)^2 Delta_v_n(2)^2 Delta_v_n(3)^2])*G(:,:)';

%% Additional process noise variances for bias-related states
processNoiseVariance = [zeros(1,10), (wb').^2, (ab').^2];

% A small diagonal regularization term can be added to avoid numerical
% singularity. It is currently disabled by multiplying it by zero.
kapa = 0*[zeros(1,7), 10^-10, 10^-10, 10^-10, zeros(1,6)];

% Add the bias-related process noise variances and the optional
% regularization term to the diagonal entries of Q.
for i = 1:16
    Q(i,i) = Q(i,i) + processNoiseVariance(i) + kapa(i);
end
end