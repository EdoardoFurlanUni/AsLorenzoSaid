%% Task 2: Accuracy Verification of the Optical Flow Model
% =========================================================================
%   GPS NED velocity (ground truth) is fed through the observer model h(x)
%   (which rotates v_NED → body frame via the quaternion attitude) to
%   produce the PREDICTED body-frame velocities.
%
%   Level 1 | Raw (sensor_optical_flow)
%              vel = (pixel_flow / timespan_s) * distance_m
%              No gyro compensation → noisiest.
%
%   Level 2 | Gyro-compensated (vehicle_optical_flow)
%              Same formula; PX4 driver already removed the rotational
%              contribution of the IMU gyro from pixel_flow.
%
%   Level 3 | EKF output (estimator_optical_flow_vel)
%              PX4 internal EKF fused + smoothed body velocity. Cleanest.
%
% =========================================================================

clear; clc; close all;

%% 1. Load synchronized data
% project_dir = fileparts(mfilename('fullpath'));
project_dir = pwd;
addpath(project_dir);
data_path = fullfile(project_dir, '..', 'Data', 'mat', 'data_sync.mat');

if ~exist(data_path, 'file')
    error('Please run Data/mat/DATA_PROCESS.m first to generate data_sync.mat');
end
fprintf('Loading data from %s...\n', data_path);
load(data_path);
% Variables loaded: t_sync, gps_gt, q_sync, flow_v, raw_flow_v, veh_flow_v, ...

%% 2. Model prediction: h(GPS_velocity, attitude) → predicted body velocity
N      = length(t_sync);
y_pred = zeros(N, 2);

fprintf('Running model prediction over %d samples...\n', N);
for k = 1 : N
    q     = q_sync(k, :)';          % quaternion from vehicle_attitude log
    v_ned = gps_gt(k, 1:3)';        % GPS NED velocity [vn; ve; vd] @ 100 Hz

    % State vector for optical_flow_model (h(x)):
    %   [q0,q1,q2,q3 | vn,ve,vd | pn,pe,pd | wb | ab]
    x = [q; v_ned; zeros(9,1)];

    y_full      = optical_flow_model(x);   % returns [v_body_x; v_body_y; pd]
    y_pred(k,:) = y_full(1:2)';
end

%% 3. Measurement vectors (all at 100 Hz from data_sync.mat)
y_raw = raw_flow_v(:, 1:2);   % Lv1 – raw pixel flow velocity [vx, vy]
y_veh = veh_flow_v(:, 1:2);   % Lv2 – gyro-compensated pixel flow velocity
y_ekf = flow_v(:, 1:2);       % Lv3 – PX4 EKF body-velocity estimate

fprintf('\n=== Signal amplitude check (max|v| over full recording) ===\n');
fprintf('  GPS prediction (h(x)|GPS)   vx=%.4f  vy=%.4f m/s\n', ...
    max(abs(y_pred(:,1))), max(abs(y_pred(:,2))));
fprintf('  Level 1  raw pixel flow     vx=%.4f  vy=%.4f m/s\n', ...
    max(abs(y_raw(:,1))), max(abs(y_raw(:,2))));
fprintf('  Level 2  gyro-compensated   vx=%.4f  vy=%.4f m/s\n', ...
    max(abs(y_veh(:,1))), max(abs(y_veh(:,2))));
fprintf('  Level 3  PX4 EKF output     vx=%.4f  vy=%.4f m/s\n', ...
    max(abs(y_ekf(:,1))), max(abs(y_ekf(:,2))));

%% 4. RMSE (each level vs prediction)
rmse_raw = sqrt(mean((y_raw - y_pred).^2));
rmse_veh = sqrt(mean((y_veh - y_pred).^2));
rmse_ekf = sqrt(mean((y_ekf - y_pred).^2));

fprintf('\n=== RMSE: h(x)|GPS  vs  Optical Flow measurements ===\n');
fprintf('  Level 1  raw pixel flow      : vx=%.4f m/s  vy=%.4f m/s\n', rmse_raw(1), rmse_raw(2));
fprintf('  Level 2  gyro-compensated    : vx=%.4f m/s  vy=%.4f m/s\n', rmse_veh(1), rmse_veh(2));
fprintf('  Level 3  PX4 EKF output      : vx=%.4f m/s  vy=%.4f m/s\n', rmse_ekf(1), rmse_ekf(2));

%% 5. Helper: plot one level vs prediction
% (defined at the bottom of this script as a nested function)

% Colour scheme
c_pred = [0.05 0.05 0.05];   % near-black  – model prediction (GPS)
c_meas = [0.15 0.50 0.85];   % steel-blue  – actual measurement

% -------------------------------------------------------------------------
%% Figure 1 – Level 1: Raw pixel flow (no gyro compensation)
fig1 = figure('Name', 'Task2 – Level 1: Raw pixel flow', ...
              'NumberTitle', 'off', 'Position', [60 500 1100 500]);

subplot(2, 1, 1);
plot_comparison(t_sync, y_pred(:,1), y_raw(:,1), c_pred, c_meas, ...
    'v_{body,x}  (m/s)', ...
    sprintf('Level 1 – RAW pixel flow  |  RMSE vx = %.4f m/s', rmse_raw(1)));

subplot(2, 1, 2);
plot_comparison(t_sync, y_pred(:,2), y_raw(:,2), c_pred, c_meas, ...
    'v_{body,y}  (m/s)', ...
    sprintf('Level 1 – RAW pixel flow  |  RMSE vy = %.4f m/s', rmse_raw(2)));

sgtitle({'Task 2 – Optical Flow Verification', ...
         'Level 1: sensor\_optical\_flow  (raw, no gyro compensation)'}, ...
        'FontWeight', 'bold');

% -------------------------------------------------------------------------
%% Figure 2 – Level 2: Gyro-compensated (vehicle_optical_flow)
fig2 = figure('Name', 'Task2 – Level 2: Gyro-compensated', ...
              'NumberTitle', 'off', 'Position', [80 300 1100 500]);

subplot(2, 1, 1);
plot_comparison(t_sync, y_pred(:,1), y_veh(:,1), c_pred, [0.85 0.45 0.05], ...
    'v_{body,x}  (m/s)', ...
    sprintf('Level 2 – Gyro-compensated  |  RMSE vx = %.4f m/s', rmse_veh(1)));

subplot(2, 1, 2);
plot_comparison(t_sync, y_pred(:,2), y_veh(:,2), c_pred, [0.85 0.45 0.05], ...
    'v_{body,y}  (m/s)', ...
    sprintf('Level 2 – Gyro-compensated  |  RMSE vy = %.4f m/s', rmse_veh(2)));

sgtitle({'Task 2 – Optical Flow Verification', ...
         'Level 2: vehicle\_optical\_flow  (gyro rotation removed by PX4 driver)'}, ...
        'FontWeight', 'bold');

% -------------------------------------------------------------------------
%% Figure 3 – Level 3: PX4 EKF output (estimator_optical_flow_vel)
fig3 = figure('Name', 'Task2 – Level 3: PX4 EKF output', ...
              'NumberTitle', 'off', 'Position', [100 100 1100 500]);

subplot(2, 1, 1);
plot_comparison(t_sync, y_pred(:,1), y_ekf(:,1), c_pred, [0.15 0.70 0.35], ...
    'v_{body,x}  (m/s)', ...
    sprintf('Level 3 – PX4 EKF output  |  RMSE vx = %.4f m/s', rmse_ekf(1)));

subplot(2, 1, 2);
plot_comparison(t_sync, y_pred(:,2), y_ekf(:,2), c_pred, [0.15 0.70 0.35], ...
    'v_{body,y}  (m/s)', ...
    sprintf('Level 3 – PX4 EKF output  |  RMSE vy = %.4f m/s', rmse_ekf(2)));

sgtitle({'Task 2 – Optical Flow Verification', ...
         'Level 3: estimator\_optical\_flow\_vel  (PX4 EKF, fused & smoothed)'}, ...
        'FontWeight', 'bold');

% -------------------------------------------------------------------------
%% Save all figures
saveas(fig1, fullfile(project_dir, 'Task2_Lv1_raw.png'));
saveas(fig2, fullfile(project_dir, 'Task2_Lv2_vehicle.png'));
saveas(fig3, fullfile(project_dir, 'Task2_Lv3_ekf.png'));
fprintf('\nFigures saved: Task2_Lv1_raw.png / Task2_Lv2_vehicle.png / Task2_Lv3_ekf.png\n');

% =========================================================================
%% Local helper function
function plot_comparison(t, pred, meas, c_pred, c_meas, ylbl, ttl)
% Plots model prediction vs one measurement signal on the current axes.
    hold on; grid on;
    plot(t, meas, '-',  'Color', [c_meas 0.6], 'LineWidth', 0.8, ...
        'DisplayName', 'Measurement');
    plot(t, pred, '-',  'Color', c_pred, 'LineWidth', 1.8, ...
        'DisplayName', 'h(x) | GPS (prediction)');
    xlabel('Time (s)');
    ylabel(ylbl);
    title(ttl, 'FontSize', 9);
    legend('Location', 'best', 'FontSize', 8);
    xlim([t(1), t(end)]);
end