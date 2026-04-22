%% Task 3: run EKF and UKF using our optical flow measurment model
clear; clc; close all;

%% 1. Load synchronized data
% project_dir = fileparts(mfilename('fullpath'));
project_dir = pwd;
filters_dir = fullfile(project_dir, '..', 'filters');
addpath(project_dir);   % optical_flow_model.m
addpath(filters_dir);   % func_f.m, func_h.m

% 1. Load data
data_path = fullfile(project_dir, '..', 'Data', 'mat', 'data_sync.mat');
load(data_path);

%% 2. Setup Symbolic Variables
syms q0 q1 q2 q3 vn ve vd pn pe pd wbx wby wbz abx aby abz real % real to force the symbolic variables to be real
syms dthx dthy dthz dvx dvy dvz dt real % dt to allow the possibility of changing the sampling frequency
sym_x = [q0; q1; q2; q3; vn; ve; vd; pn; pe; pd; wbx; wby; wbz; abx; aby; abz];
sym_u = [dthx; dthy; dthz; dvx; dvy; dvz];

% 3. Initialization
start_idx = 1000; % Start evaluation earlier (sample 1000 is 10s)
T = 500; % Simulate for 50 seconds (fits within the 6935 total length)
len = T;
dt_val = 1/Delta;

% functions for state and measurement model
f_sym = func_f(sym_x, sym_u, dt);
h_sym = func_h(sym_x);

f_sym = subs(f_sym, dt, dt_val);

% matlabFunctions to be faster 
f_num = matlabFunction(f_sym, 'Vars', {sym_x, sym_u});
h_num = matlabFunction(h_sym, 'Vars', {sym_x});

% 1. Jacobian of f
f_jac_sym = jacobian(f_sym, sym_x);
f_jac_num = matlabFunction(f_jac_sym, 'Vars', {sym_x, sym_u});

% 2. Jacobianof h
h_jac_sym = jacobian(h_sym, sym_x);
h_jac_num = matlabFunction(h_jac_sym, 'Vars', {sym_x});

% Initial State: [q; v; p; wb; ab]
x0 = [q_sync(start_idx,:)'; gps_gt(start_idx,1:3)'; gps_gt(start_idx,4:5)'; baro_h(start_idx, 1); zeros(3,1); zeros(3,1)];
P0 = diag([1e-4*ones(4,1); 0.1*ones(3,1); 0.1*ones(3,1); 1e-6*ones(3,1); 1e-4*ones(3,1)]);
V0 = P0;
u = [dtheta(start_idx:start_idx+T-1, :)'; dv(start_idx:start_idx+T-1, :)'];

% Noise Covariances (Tuning for smoother tracking)
% - Aggressively decreased velocity process noise (from 0.05 to 0.01) to strongly trust the smoothed IMU.
B_mat = diag([1e-2*ones(4,1); 0.1*ones(3,1); 0.01*ones(3,1); 1e-6*ones(3,1); 1e-4*ones(3,1)]);
D_mat = diag([1.2; 1.2; 0.8]); % [flow vbx | flow vby | baro h]

% Measurement Vector y: [v_body_x; v_body_y; p_d]
%   flow_v(:,1:2)  <- estimator_optical_flow (body frame)
%   baro_h(:,1)    <- barometer altitude
y_meas = [flow_v(start_idx:start_idx+T-1, 1:2), baro_h(start_idx:start_idx+T-1, 1)]';

% GPS denied
% y_meas = denied(y_meas', 5/Delta, 10/Delta, Delta)'; % 2s of no gps data after 5s of simulation

fprintf("Running EKF...\n")
tic
[Xekf, V] = EKF_fast(x0, y_meas, u, V0, B_mat, D_mat, f_num, h_num, f_jac_num, h_jac_num, T);
t_ekf = toc;
fprintf('EKF finished in %.4f seconds.\n', t_ekf);

fprintf("Running UKF...\n")
tic
[Xukf, P_pred] = UKF_fast(x0, y_meas, u, P0, B_mat, D_mat, f_num, h_num, len);
t_ukf = toc;
fprintf('UKF finished in %.4f seconds.\n', t_ukf);

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
subplot(2, 2, 3);
plot(time_axis, movmean(baro_h(start_idx:start_idx+T-1),10), 'k.', 'MarkerSize', 3, 'DisplayName', 'Barometer (noisy)'); hold on;
plot(time_axis, Xekf(10, 2:T+1), 'r', 'LineWidth', 1.5, 'DisplayName', 'EKF estimate');
plot(time_axis, Xukf(10, 2:T+1), 'b--', 'LineWidth', 1.5, 'DisplayName', 'UKF estimate');
yaxis_label = 'p_d (m)'; ylabel(yaxis_label); title('Altitude (Position Down)');
grid on; legend('Location', 'best');

sgtitle('Task 3: EKF vs UKF — GPS-Free Navigation with Optical Flow');

saveas(gcf, 'Task3_base.png')
saveas(gcf, 'Task3_base', 'epsc')

% figure; 
% plot(y_meas(1,:)); 
% title('Verifica GPS Denied: dovresti vedere un tratto piatto');