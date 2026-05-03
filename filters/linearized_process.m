function [A,Q] = linearized_process(xn,Delta_theta_m,Delta_v_m,Delta_theta_n,Delta_v_n,wb,ab,Delta_t)
% xn: State filter $\hat x_{t|t}$ at time t
% Delta_theta_m,Delta_v_m: Input
% Delta_theta_n,Delta_v_n: Parameters w.r.t. process noise
% Delta_t: Sampling time

q0=xn(1);q1=xn(2);q2=xn(3);q3=xn(4); % Estimate quaternion 

%% A and Q
[A] = calcF16(Delta_t,Delta_theta_m(:),xn,Delta_v_m(:),q0,q1,q2,q3);
[Q] = calcQ16(wb,ab,Delta_theta_n,Delta_v_n,q0,q1,q2,q3);

end

