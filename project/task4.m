%% Task 4: Robust EKF and UKF Testing (REKF / RUKF)
clear; clc; close all;

project_dir = fileparts(mfilename('fullpath'));
filters_dir = fullfile(project_dir, '..', 'filters');
utils_dir   = fullfile(project_dir, '..', 'utils'); % if any
addpath(project_dir, filters_dir, utils_dir);

% 1. Load the pre-processed synchronized data
load('../Data/mat/data_sync.mat');

T = 6000; % Simulation steps (4 seconds at 100Hz)
dt_val = 1/Delta;

% 2. Define Symbolic Vectors and Equations
syms q0 q1 q2 q3 vn ve vd pn pe pd wbx wby wbz abx aby abz real
syms d_thx d_thy d_thz dvx dvy dvz dt real

sym_x = [q0; q1; q2; q3; vn; ve; vd; pn; pe; pd; wbx; wby; wbz; abx; aby; abz];
sym_u = [d_thx; d_thy; d_thz; dvx; dvy; dvz];

% Define Model functions
f_sym = func_f(sym_x, sym_u, dt);
h_sym = func_h(sym_x);

% Optimization: Convert Symbolic to Numeric Functions (CRITICAL for speed)
fprintf('Optimizing symbolic functions (please wait a few seconds)...\n');
f_num = matlabFunction(f_sym, 'Vars', {sym_x, sym_u, dt});
h_num = matlabFunction(h_sym, 'Vars', {sym_x});

f_jac_sym = jacobian(f_sym, sym_x);
f_jac_num = matlabFunction(f_jac_sym, 'Vars', {sym_x, sym_u, dt});

h_jac_sym = jacobian(h_sym, sym_x);
h_jac_num = matlabFunction(h_jac_sym, 'Vars', {sym_x});

% 3. Initialize Variables
x0 = [q_sync(1,:)';         % Initial Attitude
      gps_mea(1, 1:3)';     % Initial Velocity (NED)
      gps_mea(1, 4:6)';     % Initial Position (NED)
      zeros(3,1);           % Gyro bias
      zeros(3,1)];          % Accel bias

P0 = diag([1e-4*ones(4,1); 0.1*ones(3,1); 0.1*ones(3,1); 1e-6*ones(3,1); 1e-4*ones(3,1)]);

% Noise Covariances
B_mat = diag([1e-3*ones(4,1); 0.2*ones(3,1); 0.01*ones(3,1); 1e-7*ones(3,1); 1e-5*ones(3,1)]);
D_mat = diag([0.05; 0.05; 5.0; 0.3]); % [flow vbx | flow vby | baro vd | dist pd]

% Compute vertical velocity vd from barometer
baro_h_smooth = movmean(baro_h(:,1), 20);          
baro_vd       = -[0; diff(baro_h_smooth)] * Delta; 

% Measurement Vector y: [v_body_x; v_body_y; v_ned_z; p_d]
y_meas = [flow_v(1:T, 1:2), baro_vd(1:T), dist_h(1:T)]';

% Input Matrix U for filters
U = [dtheta(1:T, :)'; dv(1:T, :)'];

Q = B_mat*B_mat'; 
R = D_mat*D_mat';

% Robustness parameter c (can be tuned, 1 is a standard choice)
c_robust = 1.0;

% -------------------------------------------------------------
% 4. Filters Execution (Robust EKF and Robust UKF)
% -------------------------------------------------------------
fprintf('Running REKF_fast...\n');
tic;
[Xrekf, Vekf, th_rekf] = REKF_fast(x0, y_meas, P0, Q, R, f_num, f_jac_num, h_num, h_jac_num, U, dt_val, c_robust);
t_rekf = toc;
fprintf('REKF finished in %.4f seconds.\n', t_rekf);


fprintf('Running RUKF_fast...\n');
tic;
[Xrukf, Vukf, th_rukf] = RUKF_fast(x0, y_meas, P0, Q, R, f_num, h_num, U, dt_val, c_robust);
t_rukf = toc;
fprintf('RUKF finished in %.4f seconds.\n', t_rukf);

% -------------------------------------------------------------
% 5. Visualization
% -------------------------------------------------------------
% Extract GPS Ground Truth for plotting
gps_vn = gps_mea(1:T+1, 1);
gps_ve = gps_mea(1:T+1, 2);
dist_truth = dist_h(1:T+1);
time_axis = t_sync(1:T+1) - t_sync(1);

figure('Name', 'Robust Filters: Velocity Tracking', 'NumberTitle', 'off');

subplot(3, 1, 1);
plot(time_axis, gps_vn, 'k-', 'LineWidth', 1.5, 'DisplayName', 'GPS (Ground Truth)'); hold on;
plot(time_axis, Xrekf(5, :), 'b--', 'LineWidth', 1.2, 'DisplayName', 'REKF');
plot(time_axis, Xrukf(5, :), 'r-.', 'LineWidth', 1.2, 'DisplayName', 'RUKF');
xlabel('Time (s)'); ylabel('v_N (m/s)'); title('North Velocity v_N');
legend('Location', 'best'); grid on;

subplot(3, 1, 2);
plot(time_axis, gps_ve, 'k-', 'LineWidth', 1.5, 'DisplayName', 'GPS (Ground Truth)'); hold on;
plot(time_axis, Xrekf(6, :), 'b--', 'LineWidth', 1.2, 'DisplayName', 'REKF');
plot(time_axis, Xrukf(6, :), 'r-.', 'LineWidth', 1.2, 'DisplayName', 'RUKF');
plot(time_axis, flow_v(1:T+1, 3), 'g:', 'LineWidth', 1.5, 'DisplayName', 'PX4 EKF (Optical Flow Only)');
xlabel('Time (s)'); ylabel('v_E (m/s)'); title('East Velocity v_E');
legend('Location', 'best'); grid on;

subplot(3, 1, 3);
plot(time_axis, baro_vd(1:T+1), 'k-', 'LineWidth', 1, 'DisplayName', 'Barometer v_D (Derived)'); hold on;
plot(time_axis, Xrekf(7, :), 'b--', 'LineWidth', 1.5, 'DisplayName', 'REKF');
plot(time_axis, Xrukf(7, :), 'r-.', 'LineWidth', 1.5, 'DisplayName', 'RUKF');
xlabel('Time (s)'); ylabel('v_D (m/s)'); title('Down Velocity v_D');
legend('Location', 'best'); grid on;

figure('Name', 'Robust Filters: Altitude Tracking', 'NumberTitle', 'off');
plot(time_axis, dist_truth, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Distance Sensor (Raw)'); hold on;
plot(time_axis, Xrekf(10, :), 'b--', 'LineWidth', 1.2, 'DisplayName', 'REKF');
plot(time_axis, Xrukf(10, :), 'r-.', 'LineWidth', 1.2, 'DisplayName', 'RUKF');
xlabel('Time (s)'); ylabel('p_d (m)'); title('Altitude Tracking (Down Position)');
legend('Location', 'best'); grid on;

figure('Name', 'Robust Parameters Over Time', 'NumberTitle', 'off');
plot(time_axis(1:end-1), th_rekf, 'b-', 'DisplayName', 'REKF \theta'); hold on;
plot(time_axis(1:end-1), th_rukf, 'r-', 'DisplayName', 'RUKF \theta');
xlabel('Time (s)'); ylabel('\theta'); title('Covariance Inflation Factor over Time');
legend; grid on;
