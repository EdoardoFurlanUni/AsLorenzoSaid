%% Task 3: EKF and UKF Testing on the Optical Flow Model
clear; clc; close all;

project_dir = fileparts(mfilename('fullpath'));
filters_dir = fullfile(project_dir, '..', 'filters');
addpath(project_dir);   % optical_flow_model.m
addpath(filters_dir);   % func_f.m, func_h.m

% 1. Load data
data_path = fullfile(project_dir, '..', 'Data', 'mat', 'data_sync.mat');
load(data_path);

% 2. Setup Symbolic Variables
syms q0 q1 q2 q3 vn ve vd pn pe pd wbx wby wbz abx aby abz real % real to force the symbolic variables to be real
syms dthx dthy dthz dvx dvy dvz dt real % dt to allow the possibility of changing the sampling frequency
sym_x = [q0; q1; q2; q3; vn; ve; vd; pn; pe; pd; wbx; wby; wbz; abx; aby; abz];
sym_u = [dthx; dthy; dthz; dvx; dvy; dvz];

% functions for state and measurement model
f_sym = func_f(sym_x, sym_u, dt);
h_sym = func_h(sym_x);

% 3. Initialization
start_idx = 1000; % Start evaluation earlier (sample 1000 is 10s)
T = 1000; % Simulate for 50 seconds (fits within the 6935 total length)
dt_val = 1/Delta;

% Initial State: [q; v; p; wb; ab]
x0 = [q_sync(start_idx,:)'; gps_gt(start_idx,1:3)'; gps_gt(start_idx,4:6)'; zeros(3,1); zeros(3,1)];
P0 = diag([1e-4*ones(4,1); 0.1*ones(3,1); 0.1*ones(3,1); 1e-6*ones(3,1); 1e-4*ones(3,1)]);

% Noise Covariances (Tuning for smoother tracking)
% - Aggressively decreased velocity process noise (from 0.05 to 0.01) to strongly trust the smoothed IMU.
B_mat = diag([1e-3*ones(4,1); 0.01*ones(3,1); 0.01*ones(3,1); 1e-7*ones(3,1); 1e-5*ones(3,1)]);
D_mat = diag([0.15; 0.15; 1.0]); % [flow vbx | flow vby | baro h]

% Measurement Vector y: [v_body_x; v_body_y; p_d]
%   flow_v(:,1:2)  <- estimator_optical_flow (body frame)
%   baro_h(:,1)    <- barometer altitude
y_meas = [flow_v(start_idx:start_idx+T-1, 1:2), baro_h(start_idx:start_idx+T-1, 1)]';

% 4. Optimization: Convert Symbolic to Numeric Functions
fprintf('Optimizing symbolic functions...\n');
f_num = matlabFunction(f_sym, 'Vars', {sym_x, sym_u, dt});
h_num = matlabFunction(h_sym, 'Vars', {sym_x});

f_jac_sym = jacobian(f_sym, sym_x);
f_jac_num = matlabFunction(f_jac_sym, 'Vars', {sym_x, sym_u, dt});

h_jac_sym = jacobian(h_sym, sym_x);
h_jac_num = matlabFunction(h_jac_sym, 'Vars', {sym_x});

% 5. Filters Execution
fprintf('Running EKF...\n');
Xekf = zeros(16, T+1); Xekf(:,1) = x0;
Vekf = zeros(16, 16, T+1); Vekf(:,:,1) = P0;
Q_base = B_mat*B_mat'; R_base = D_mat*D_mat'; % Base process and measurement noise covariances

tic;
for i = 1:T
    i
    u_i = [dtheta(start_idx+i-1,:)'; dv(start_idx+i-1,:)'];
    
    % --- Matrici Covarianza Tempo Varianti (EKF) ---
    % 1. Process Noise Q: Aumenta sulle velocità se le accelerazioni IMU sono alte (manovre brusche/vibrazioni)
    accel_mag = norm(u_i(4:6)) / dt_val; % approssimazione accelerazione m/s^2
    Q_k = Q_base;
    Q_k(5:7, 5:7) = Q_base(5:7, 5:7) * (1 + 0.1 * accel_mag);
    
    % 2. Measurement Noise R: Il rumore della velocità misurata aumenta con la quota (v = flow * h)
    % Usiamo la quota stimata corrente (h > 0.1 per evitare singolarità a terra)
    current_h = max(0.1, abs(Xekf(10, i))); 
    R_k = R_base;
    R_k(1:2, 1:2) = R_base(1:2, 1:2) * (current_h^2);

    % EKF Predict
    X_pred = f_num(Xekf(:,i), u_i, dt_val); %prediction using state equation
    A = f_jac_num(Xekf(:,i), u_i, dt_val);  %Jacobian
    V_pred = A * Vekf(:,:,i) * A' + Q_k;    %covariance prediction con Q_k
    % EKF Update
    C = h_jac_num(X_pred); %Jacobian
    K = V_pred * C' / (C * V_pred * C' + R_k); %Kalman gain con R_k
    h_val = h_num(X_pred); %measurement prediction
    Xekf(:,i+1) = X_pred + K * (y_meas(:,i) - h_val); %state update
    % Normalize quaternion to prevent norm drift corrupting R(q)
    Xekf(1:4,i+1) = Xekf(1:4,i+1) / norm(Xekf(1:4,i+1));
    Vekf(:,:,i+1) = (eye(16) - K * C) * V_pred;
end
t_ekf = toc;
fprintf('EKF finished in %.4f seconds.\n', t_ekf);

fprintf('Running UKF (Numerical)...\n');
Xukf = zeros(16, T+1); Xukf(:,1) = x0;
Vukf = zeros(16, 16, T+1); Vukf(:,:,1) = P0;
n = 16; m = 3; % state dim = 16, measurement dim = 3
alpha = 0.1; kapa = 3-n; beta = 2;
lambda = alpha^2 * (n + kapa) - n;
Wm = [lambda/(n+lambda), ones(1, 2*n)/(2*(n+lambda))];
Wc = Wm; Wc(1) = Wc(1) + (1 - alpha^2 + beta);

tic;
for i = 1:T
    u_i = [dtheta(start_idx+i-1,:)'; dv(start_idx+i-1,:)'];
    
    % --- Matrici Covarianza Tempo Varianti (UKF) ---
    accel_mag = norm(u_i(4:6)) / dt_val; 
    Q_k = Q_base;
    Q_k(5:7, 5:7) = Q_base(5:7, 5:7) * (1 + 0.1 * accel_mag);
    
    current_h = max(0.1, abs(Xukf(10, i))); 
    R_k = R_base;
    R_k(1:2, 1:2) = R_base(1:2, 1:2) * (current_h^2);
    
    % 1. Sigma Points (Prediction)
    P_sqrt = chol((n + lambda) * Vukf(:,:,i))';
    sigma_x = [Xukf(:,i), Xukf(:,i) + P_sqrt, Xukf(:,i) - P_sqrt];
    
    % 2. Propagate Sigma Points (Process)
    sigma_x_pred = zeros(n, 2*n+1);
    for j = 1:2*n+1
        sigma_x_pred(:,j) = f_num(sigma_x(:,j), u_i, dt_val);
    end
    
    % 3. Predicted Mean and Covariance
    x_pred = sum(Wm .* sigma_x_pred, 2);
    P_pred = Q_k;
    for j = 1:2*n+1
        diff_x_pred = sigma_x_pred(:,j) - x_pred;
        P_pred = P_pred + Wc(j) * (diff_x_pred * diffMulti(diff_x_pred)); % Custom mult for safety
    end
    P_pred = (P_pred + P_pred')/2; % Enforce symmetry

    % 4. Sigma Points (Update)
    P_sqrt_pred = chol((n + lambda) * P_pred)';
    sigma_x_upd = [x_pred, x_pred + P_sqrt_pred, x_pred - P_sqrt_pred];
    
    % 5. Propagate Sigma Points (Measurement)
    sigma_y_pred = zeros(m, 2*n+1);
    for j = 1:2*n+1
        sigma_y_pred(:,j) = h_num(sigma_x_upd(:,j));
    end
    
    % 6. Predicted Measurement and Cross-Covariance
    y_pred = sum(Wm .* sigma_y_pred, 2);
    Py = R_k; Pxy = zeros(n, m);
    for j = 1:2*n+1
        diff_x = sigma_x_upd(:,j) - x_pred;
        diff_y = sigma_y_pred(:,j) - y_pred;
        Py = Py + Wc(j) * (diff_y * diff_y');
        Pxy = Pxy + Wc(j) * (diff_x * diff_y');
    end
    
    % 7. Final Update
    K = Pxy / Py;
    Xukf(:,i+1) = x_pred + K * (y_meas(:,i) - y_pred);
    % Normalize quaternion to prevent norm drift corrupting R(q)
    Xukf(1:4,i+1) = Xukf(1:4,i+1) / norm(Xukf(1:4,i+1));
    Vukf(:,:,i+1) = P_pred - K * Py * K';
    Vukf(:,:,i+1) = (Vukf(:,:,i+1) + Vukf(:,:,i+1)')/2; % Symmetry

    if mod(i, 100) == 0, fprintf('Step %d/%d\n', i, T); end
end
t_ukf = toc;
fprintf('UKF finished in %.4f seconds.\n', t_ukf);

function out = diffMulti(d)
    out = d';
end

% 6. Visualization
% The filter estimates the full state without GPS.
% We compare against gps_gt only as an EXTERNAL REFERENCE (not used by the filter).
figure('Name', 'EKF vs UKF Performance', 'NumberTitle', 'off', 'Color', 'w');

time_axis = t_sync(start_idx:start_idx+T-1) - t_sync(start_idx);

% --- North Velocity: filter estimate vs GPS reference ---
subplot(2, 2, 1);
plot(time_axis, gps_gt(start_idx:start_idx+T-1, 1), 'k--', 'DisplayName', 'GPS ref (not used)'); hold on;
plot(time_axis, Xekf(5, 2:T+1), 'r', 'LineWidth', 1.5, 'DisplayName', 'EKF estimate');
plot(time_axis, Xukf(5, 2:T+1), 'b--', 'LineWidth', 1.5, 'DisplayName', 'UKF estimate');
yaxis_label = 'v_N (m/s)'; ylabel(yaxis_label); title('North Velocity');
grid on; legend('Location', 'best');

% --- East Velocity: filter estimate vs GPS reference ---
subplot(2, 2, 2);
plot(time_axis, gps_gt(start_idx:start_idx+T-1, 2), 'k--', 'DisplayName', 'GPS ref (not used)'); hold on;
plot(time_axis, Xekf(6, 2:T+1), 'r', 'LineWidth', 1.5, 'DisplayName', 'EKF estimate');
plot(time_axis, Xukf(6, 2:T+1), 'b--', 'LineWidth', 1.5, 'DisplayName', 'UKF estimate');
yaxis_label = 'v_E (m/s)'; ylabel(yaxis_label); title('East Velocity');
grid on; legend('Location', 'best');

% --- Altitude: filter estimate vs distance sensor measurement ---
subplot(2, 2, 3); % TODO: baro_h o dist_h ???, sec me baro_h
plot(time_axis, movmean(baro_h(start_idx:start_idx+T-1),10), 'k.', 'MarkerSize', 3, 'DisplayName', 'Dist sensor (noisy)'); hold on;
plot(time_axis, Xekf(10, 2:T+1), 'r', 'LineWidth', 1.5, 'DisplayName', 'EKF estimate');
plot(time_axis, Xukf(10, 2:T+1), 'b--', 'LineWidth', 1.5, 'DisplayName', 'UKF estimate');
yaxis_label = 'p_d (m)'; ylabel(yaxis_label); title('Altitude (Position Down)');
grid on; legend('Location', 'best');

sgtitle('Task 3: EKF vs UKF — GPS-Free Navigation with Optical Flow');

saveas(gcf, 'Task3.png')
saveas(gcf, 'Task3', 'epsc')