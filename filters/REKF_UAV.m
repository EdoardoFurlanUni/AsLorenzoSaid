function [x_new, P_new, theta] = REKF_UAV(x, y, P, R, dtheta, dv, Delta_theta_n, Delta_v_n, wb, ab, dt, mode, c)
%% REKF for 16D UAV state  [ref: EFFI_EKF/EKF.m, filters/REKF.m, Algorithm steps 1-9]
%
% Inputs:
%   x    : 16x1  x_hat_t  (state prediction at time t)
%   y    : measurement vector (6x1 GPS+baro or 3x1 flow+baro)
%   P    : 16x16 P_t  (state covariance at time t)
%   R    : measurement noise covariance DD^T
%   dtheta, dv    : IMU angular/velocity increments (1x3 each)
%   Delta_theta_n : 3x1 angular increment noise std dev
%   Delta_v_n     : 3x1 velocity increment noise std dev
%   wb            : 3x1 gyro bias noise std dev
%   ab            : 3x1 accel bias noise std dev
%   dt   : sampling interval (s)
%   mode : 'gps' or 'flow'
%   c    : robustness parameter (scalar, > 0)
%    
%
% Outputs:
%   x_new  : 16x1  x_hat_{t+1}   (predicted state)
%   P_new  : 16x16 P_{t+1}       (predicted covariance, robustified)
%   theta  : scalar theta_t found at this step

n = 16;

%% Step 1: C_t
if strcmp(mode, 'gps')
    C = [zeros(3,4), eye(3), zeros(3,9);
         zeros(3,4), zeros(3,3), eye(3), zeros(3,6)];
elseif strcmp(mode, 'flow')
    C = calcC_h(x);
end

%% Step 2: L_t
S = C * P * C' + R;
L = P * C' / S;

%% Step 3: x_hat_{t|t}
if strcmp(mode, 'gps')
    xn = x + L * (y - C*x);
elseif strcmp(mode, 'flow')
    xn = x + L * (y - func_h(x));
end
xn(1:4) = xn(1:4) / norm(xn(1:4));

%% Step 4: A_t, Q_t  (Q_t time-varying: depends on quaternion of x_hat_{t|t})
[A, Q] = linearized_process(xn, dtheta, dv, Delta_theta_n, Delta_v_n, wb, ab, dt);

%% Step 5: x_hat_{t+1}
x_new = func_f(xn, dtheta, dv, dt);

%% Steps 6-7: P_{t+1} 
P_n   = (eye(n) - L*C) * P;   
P_new = A * P_n * A' + Q;

%% Steps 8-9: find theta_t s.t. gamma(P_{t+1}, theta_t) = c
%  gamma(P, theta) = trace(inv(I - theta*P) - I) + log(det(I - theta*P))
%  Binary search on theta in (0, 1/max_eig(P_new)).
%
%  Special case c = 0: gamma(P, 0) = 0 already satisfies the equation,
%  so theta = 0 and P_new is left unchanged. Avoiding the bisection
%  here also prevents numerical drift due to two matrix inversions.
if c <= 0
    theta = 0;
    return
end

P_new = (P_new + P_new')/2 + 1e-10 * eye(n);

e  = eig(P_new);
r  = max(abs(e));
t1 = 0;
t2 = (1 - 1e-5) / r;

value    = 1;
max_iter = 100;
iter     = 0;
while abs(value) >= 1e-9 && iter < max_iter
    iter  = iter + 1;
    t     = 0.5 * (t1 + t2);
    M     = eye(n) - t * P_new;
    invM  = M \ eye(n);                          
    % eig(I - t*P) = 1 - t*lambda_i
    value = trace(invM - eye(n)) + sum(log(1 - t * e)) - c;
    if value > 0
        t2 = t;
    else
        t1 = t;
    end
end

theta = t;
P_new = (P_new \ eye(n) - theta * eye(n)) \ eye(n);      % inv(inv(P)-theta*I) 
P_new = (P_new + P_new')/2 + 1e-10 * eye(n);

end
