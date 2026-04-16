%% Task 2: Accuracy Verification of the Optical Flow Model
clear; clc; close all;

% 1. Load synchronized data
% Build absolute path from the script's own directory (works regardless of MATLAB CWD)
project_dir = fileparts(mfilename('fullpath'));
addpath(project_dir);  % Ensure optical_flow_model.m is on the path
data_path = fullfile(project_dir, '..', 'Data', 'mat', 'data_sync.mat');

if ~exist(data_path, 'file')
    error('Please run Data/mat/DATA_PROCESS.m first to generate data_sync.mat');
end

fprintf('Loading data from %s...\n', data_path);
load(data_path);

% 2. Initialization
N = length(t_sync);
y_pred = zeros(N, 2);  % Only v_body_x and v_body_y
y_real = zeros(N, 2);

% Real measurements: body frame velocities from optical flow sensor
% Source: estimator_optical_flow (columns 1-2 are body frame vx, vy)
% These are the only outputs of h(x) that can be validated without GPS,
% since the optical flow sensor does not measure vertical velocity or position.
y_real(:, 1) = flow_v(:, 1); % v_body_x from optical flow sensor
y_real(:, 2) = flow_v(:, 2); % v_body_y from optical flow sensor

fprintf('Running verification loop for %d samples...\n', N);

% 3. Verification Loop
for k = 1:N
    % Construct state vector x
    % Format: [q0,q1,q2,q3, vn,ve,vd, pn,pe,pd, wb_x,wb_y,wb_z, ab_x,ab_y,ab_z]

    % Quaternions from attitude log (no GPS)
    q = q_sync(k, :)';

    % NED horizontal velocities from estimator_optical_flow (no GPS).
    % flow_v(:,3:4) = vn, ve estimated by PX4 EKF using optical flow data.
    % vd (down velocity) is not observable from optical flow -> set to 0.
    v_ned = [flow_v(k, 3); flow_v(k, 4); 0];

    % Position: not used by h(x) for v_body_x/y, set to zero.
    p_ned = zeros(3, 1);

    % Biases assumed 0 for kinematic verification
    wb = zeros(3, 1);
    ab = zeros(3, 1);

    x = [q; v_ned; p_ned; wb; ab];

    % Model prediction - take only v_body_x and v_body_y (outputs 1 and 2)
    % v_ned_z and p_d are excluded: they are not measurable by optical flow
    % and will instead be estimated by the EKF/UKF in Task 3.
    y_full = optical_flow_model(x);
    y_pred(k, :) = y_full(1:2)';
end

% 4. Error Calculation (RMSE)
rmse = sqrt(mean((y_real - y_pred).^2));
fprintf('\nRoot Mean Square Error (RMSE):\n');
fprintf('  v_body_x: %.4f m/s\n', rmse(1));
fprintf('  v_body_y: %.4f m/s\n', rmse(2));

% 5. Visualization
figure('Name', 'Optical Flow Model Verification', 'NumberTitle', 'off');

% Velocity Body X
subplot(2, 1, 1);
plot(t_sync, y_real(:, 1), 'b', 'DisplayName', 'Real (estimator\_optical\_flow)'); hold on;
plot(t_sync, y_pred(:, 1), 'r--', 'DisplayName', 'Model Prediction h(x)');
xlabel('Time (s)'); ylabel('v_x (m/s)');
title('Body Velocity X');
legend('Location', 'best');
grid on;

% Velocity Body Y
subplot(2, 1, 2);
plot(t_sync, y_real(:, 2), 'b', 'DisplayName', 'Real (estimator\_optical\_flow)'); hold on;
plot(t_sync, y_pred(:, 2), 'r--', 'DisplayName', 'Model Prediction h(x)');
xlabel('Time (s)'); ylabel('v_y (m/s)');
title('Body Velocity Y');
legend('Location', 'best');
grid on;

sgtitle('Task 2: Optical Flow Model Verification (GPS-Free)');

saveas(gcf, 'Task2.png')
saveas(gcf, 'Task2', 'epsc')