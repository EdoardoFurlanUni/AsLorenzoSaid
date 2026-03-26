%% Task 2: Accuracy Verification of the Optical Flow Model
clear; clc; close all;

% 1. Load synchronized data
% Path to data_sync.mat (adjust if necessary)
data_path = '../Data/mat/data_sync.mat';

if ~exist(data_path, 'file')
    error('Please run Data/mat/DATA_PROCESS.m first to generate data_sync.mat');
end

fprintf('Loading data from %s...\n', data_path);
load(data_path);

% 2. Initialization
N = length(t_sync);
y_pred = zeros(N, 4);
y_real = zeros(N, 4);

% Prepare real data for comparison
% y_real = [v_body_x; v_body_y; v_ned_z; pos_ned_z]
y_real(:, 1:2) = flow_v(:, 1:2); % Body frame velocities from flow sensor
y_real(:, 3)   = gps_gt(:, 3);   % Down velocity from GPS
y_real(:, 4)   = dist_h;         % Height from distance sensor (as p_d)

fprintf('Running verification loop for %d samples...\n', N);

% 3. Verification Loop
for k = 1:N
    % Construct state vector x
    % Format: [q0,q1,q2,q3, vn,ve,vd, pn,pe,pd, wb_x,wb_y,wb_z, ab_x,ab_y,ab_z]
    
    q = q_sync(k, :)';       % Quaternions from attitude log
    v_ned = gps_gt(k, 1:3)'; % Velocity NED
    p_ned = gps_gt(k, 4:6)'; % Position NED
    
    % Biases are assumed 0 for verification of the kinematic model
    wb = zeros(3, 1);
    ab = zeros(3, 1);
    
    x = [q; v_ned; p_ned; wb; ab];
    
    % Model prediction
    y_pred(k, :) = optical_flow_model(x)';
end

% 4. Error Calculation (RMSE)
rmse = sqrt(mean((y_real - y_pred).^2));
fprintf('\nRoot Mean Square Error (RMSE):\n');
fprintf('  v_body_x: %.4f m/s\n', rmse(1));
fprintf('  v_body_y: %.4f m/s\n', rmse(2));
fprintf('  v_ned_z:  %.4f m/s\n', rmse(3));
fprintf('  p_d:      %.4f m\n', rmse(4));

% 5. Visualization
figure('Name', 'Optical Flow Model Verification', 'NumberTitle', 'off');

% Velocity Body X
subplot(2, 2, 1);
plot(t_sync, y_real(:, 1), 'b', 'DisplayName', 'Real (Flow Sensor)'); hold on;
plot(t_sync, y_pred(:, 1), 'r--', 'DisplayName', 'Model Prediction');
xlabel('Time (s)'); ylabel('v_x (m/s)');
title('Body Velocity X');
legend('Location', 'best');

% Velocity Body Y
subplot(2, 2, 2);
plot(t_sync, y_real(:, 2), 'b', 'DisplayName', 'Real (Flow Sensor)'); hold on;
plot(t_sync, y_pred(:, 2), 'r--', 'DisplayName', 'Model Prediction');
xlabel('Time (s)'); ylabel('v_y (m/s)');
title('Body Velocity Y');
legend('Location', 'best');

% Velocity NED Z (Down)
subplot(2, 2, 3);
plot(t_sync, y_real(:, 3), 'b', 'DisplayName', 'Real (GPS)'); hold on;
plot(t_sync, y_pred(:, 3), 'r--', 'DisplayName', 'Model Prediction');
xlabel('Time (s)'); ylabel('v_z (m/s)');
title('NED Velocity Down');
legend('Location', 'best');

% Position NED Z (Down)
subplot(2, 2, 4);
plot(t_sync, y_real(:, 4), 'b', 'DisplayName', 'Real (Dist Sensor)'); hold on;
plot(t_sync, y_pred(:, 4), 'r--', 'DisplayName', 'Model Prediction');
xlabel('Time (s)'); ylabel('p_d (m)');
title('Position Down (Height)');
legend('Location', 'best');

sgtitle('Comparison: Real Data vs Model Predictions');
