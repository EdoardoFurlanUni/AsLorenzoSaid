%% GRID SEARCH: Fine-tuning filter parameters
% =========================================================================
%   Phase 1: Tune R_gps scaling using EKF only (fastest filter)
%   Phase 2: Tune c_rekf and c_rukf with the best R_gps from Phase 1
%   Alpha tuning: done separately on UKF by modifying UKF_UAV.m
%
%   Run this script from the project/ folder.
%   Change data_num below to test different datasets.
% =========================================================================
clear; clc; close all;

data_num = '49';  % <-- CHANGE THIS FOR EACH DATASET

%% Setup paths and load data
project_dir = fileparts(mfilename('fullpath'));
filters_dir = fullfile(project_dir, '..', 'filters');
data_dir    = fullfile(project_dir, '..', 'Data', 'mat');
addpath(filters_dir);
addpath(data_dir);

data_path = fullfile(data_dir, sprintf('data_sync_%s.mat', data_num));
if ~exist(data_path, 'file')
    error('Run DATA_PROCESS.m first for dataset %s', data_num);
end
fprintf('Loading dataset %s...\n', data_num);
load(data_path);

N  = length(t_sync);
dt = 1 / Delta;

%% GPS-denied interval (same as task34)
T_deny = 100; I_deny = 100;
gps_denied = denied(gps_mea, T_deny, I_deny, Delta);

deny_start = T_deny * Delta + 1;
deny_end   = min((T_deny + I_deny) * Delta, N);
mode_vec   = repmat({'gps'}, N, 1);
mode_vec(deny_start:deny_end) = {'flow'};

%% Initialization (same as task34)
x0 = [q_sync(1,:)'; gps_gt(1,1:3)'; gps_gt(1,4:5)'; -dist_h(1);
      2.7556e-6*ones(3,1); 6.7600e-11*ones(3,1)];
P0 = 1e-4 * eye(16);

%% Noise parameters (fixed)
Delta_theta_n = [2.6e-5; 2.6e-5; 2.6e-5];
Delta_v_n     = [1.66e-3; 1.66e-3; 1.66e-3];
wb            = [2.6e-6; 2.6e-6; 2.6e-6];
ab            = [1.66e-4; 1.66e-4; 1.66e-4];

R_flow = diag([0.5^2; 0.4^2; 0.4^2]);

%% Gating thresholds (dynamic, same as task34)
mean_x = mean(abs(veh_flow_v(:,1)), 'omitnan');
std_x  = std(veh_flow_v(:,1), 'omitnan');
mean_y = mean(abs(veh_flow_v(:,2)), 'omitnan');
std_y  = std(veh_flow_v(:,2), 'omitnan');
VEL_THRESHOLD_x = mean_x + 3 * std_x;
VEL_THRESHOLD_y = mean_y + 3 * std_y;

%% Ground truth
idx = 1:N-1;
gt_vel = gps_gt(idx, 1:3);
gt_pos = [gps_gt(idx, 4:5), -dist_h(idx)];
calc_rmse = @(est, gt) sqrt(mean(sum((est - gt).^2, 2)));

%% ========================================================================
%% PHASE 1: R_gps scaling grid search (EKF only)
%% ========================================================================
fprintf('\n=== PHASE 1: R_gps scaling (EKF) ===\n');

% Base R_gps structure: diag([vel_h^2, vel_h^2, vel_d^2, pos_h^2, pos_h^2, pos_d^2])
% We test different scaling factors for velocity and position components
vel_scales = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0];    % multiply the velocity std devs
pos_scales = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0];    % multiply the position std devs

% Base values from task34
base_vel_h = 0.05; base_vel_d = 0.1;   % m/s
base_pos_h = 0.3;  base_pos_d = 0.4;   % m

n_vel = length(vel_scales);
n_pos = length(pos_scales);
rmse_p_grid = zeros(n_vel, n_pos);
rmse_v_grid = zeros(n_vel, n_pos);

total_runs = n_vel * n_pos;
run_count = 0;

for iv = 1:n_vel
    for ip = 1:n_pos
        run_count = run_count + 1;
        
        sv = vel_scales(iv);
        sp = pos_scales(ip);
        
        R_gps_test = diag([(sv*base_vel_h)^2; (sv*base_vel_h)^2; (sv*base_vel_d)^2;
                           (sp*base_pos_h)^2; (sp*base_pos_h)^2; (sp*base_pos_d)^2]);
        
        % Run EKF
        X_ekf = zeros(16, N); X_ekf(:,1) = x0; P_ekf = P0;
        
        for k = 1:N-1
            md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
            if strcmp(md, 'gps')
                y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps_test;
            else
                y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
                if abs(y_k(1)) > VEL_THRESHOLD_x, R_k(1,1) = R_k(1,1) * 1e6; end
                if abs(y_k(2)) > VEL_THRESHOLD_y, R_k(2,2) = R_k(2,2) * 1e6; end
            end
            [X_ekf(:,k+1), P_ekf] = EKF_UAV(X_ekf(:,k), y_k, P_ekf, R_k, dth, dvk, ...
                Delta_theta_n, Delta_v_n, wb, ab, dt, md);
        end
        
        rmse_p_grid(iv, ip) = calc_rmse(X_ekf(8:10, idx)', gt_pos);
        rmse_v_grid(iv, ip) = calc_rmse(X_ekf(5:7, idx)', gt_vel);
        
        fprintf('[%2d/%2d] vel_scale=%.2f, pos_scale=%.2f => Pos RMSE=%.4f m, Vel RMSE=%.4f m/s\n', ...
            run_count, total_runs, sv, sp, rmse_p_grid(iv,ip), rmse_v_grid(iv,ip));
    end
end

% Find best R_gps
[min_rmse_p, min_idx] = min(rmse_p_grid(:));
[best_iv, best_ip] = ind2sub(size(rmse_p_grid), min_idx);
best_vel_scale = vel_scales(best_iv);
best_pos_scale = pos_scales(best_ip);

fprintf('\n>>> BEST R_gps: vel_scale=%.2f, pos_scale=%.2f => Pos RMSE=%.4f m\n\n', ...
    best_vel_scale, best_pos_scale, min_rmse_p);

% Build best R_gps for Phase 2
R_gps_best = diag([(best_vel_scale*base_vel_h)^2; (best_vel_scale*base_vel_h)^2; (best_vel_scale*base_vel_d)^2;
                    (best_pos_scale*base_pos_h)^2; (best_pos_scale*base_pos_h)^2; (best_pos_scale*base_pos_d)^2]);

%% ========================================================================
%% PHASE 2: c grid search for REKF and RUKF (with best R_gps)
%% ========================================================================
fprintf('=== PHASE 2: c tuning for REKF and RUKF ===\n');

c_grid = [0, 1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04];

n_c = length(c_grid);
rmse_p_rekf = zeros(n_c, 1);
rmse_v_rekf = zeros(n_c, 1);
rmse_p_rukf = zeros(n_c, 1);
rmse_v_rukf = zeros(n_c, 1);

% --- REKF ---
fprintf('\n--- REKF ---\n');
for ic = 1:n_c
    c_test = c_grid(ic);
    
    X_rekf = zeros(16, N); X_rekf(:,1) = x0; P_rekf = P0;
    
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps_best;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
            if abs(y_k(1)) > VEL_THRESHOLD_x, R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THRESHOLD_y, R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X_rekf(:,k+1), P_rekf, ~] = REKF_UAV(X_rekf(:,k), y_k, P_rekf, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md, c_test);
    end
    
    rmse_p_rekf(ic) = calc_rmse(X_rekf(8:10, idx)', gt_pos);
    rmse_v_rekf(ic) = calc_rmse(X_rekf(5:7, idx)', gt_vel);
    
    fprintf('  c=%.1e => Pos RMSE=%.4f m, Vel RMSE=%.4f m/s\n', c_test, rmse_p_rekf(ic), rmse_v_rekf(ic));
end

[best_rmse_rekf, best_ic_rekf] = min(rmse_p_rekf);
fprintf('>>> BEST REKF: c=%.1e => Pos RMSE=%.4f m\n\n', c_grid(best_ic_rekf), best_rmse_rekf);

% --- RUKF ---
fprintf('--- RUKF ---\n');
for ic = 1:n_c
    c_test = c_grid(ic);
    
    X_rukf = zeros(16, N); X_rukf(:,1) = x0; P_rukf = P0;
    
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps_best;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
            if abs(y_k(1)) > VEL_THRESHOLD_x, R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THRESHOLD_y, R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X_rukf(:,k+1), P_rukf, ~] = RUKF_UAV(X_rukf(:,k), y_k, P_rukf, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md, c_test);
    end
    
    rmse_p_rukf(ic) = calc_rmse(X_rukf(8:10, idx)', gt_pos);
    rmse_v_rukf(ic) = calc_rmse(X_rukf(5:7, idx)', gt_vel);
    
    fprintf('  c=%.1e => Pos RMSE=%.4f m, Vel RMSE=%.4f m/s\n', c_test, rmse_p_rukf(ic), rmse_v_rukf(ic));
end

[best_rmse_rukf, best_ic_rukf] = min(rmse_p_rukf);
fprintf('>>> BEST RUKF: c=%.1e => Pos RMSE=%.4f m\n\n', c_grid(best_ic_rukf), best_rmse_rukf);

%% ========================================================================
%% SUMMARY
%% ========================================================================
fprintf('\n==========================================================\n');
fprintf('          TUNING RESULTS - Dataset %s\n', data_num);
fprintf('==========================================================\n');
fprintf('Best R_gps:  vel_scale=%.2f, pos_scale=%.2f\n', best_vel_scale, best_pos_scale);
fprintf('             R_gps = diag([%.4f, %.4f, %.4f, %.4f, %.4f, %.4f])\n', diag(R_gps_best)');
fprintf('Best c_rekf: %.1e  (Pos RMSE = %.4f m)\n', c_grid(best_ic_rekf), best_rmse_rekf);
fprintf('Best c_rukf: %.1e  (Pos RMSE = %.4f m)\n', c_grid(best_ic_rukf), best_rmse_rukf);
fprintf('==========================================================\n');

%% Plots
figure('Name', sprintf('R_gps Grid - Dataset %s', data_num), ...
       'NumberTitle', 'off', 'Units', 'normalized', 'OuterPosition', [0.1 0.3 0.5 0.5]);
imagesc(pos_scales, vel_scales, rmse_p_grid);
colorbar; colormap('jet');
xlabel('Position Scale'); ylabel('Velocity Scale');
title(sprintf('3D Position RMSE (m) - EKF - Dataset %s', data_num));
set(gca, 'XTick', pos_scales, 'YTick', vel_scales);
% Add text annotations
for iv = 1:n_vel
    for ip = 1:n_pos
        text(pos_scales(ip), vel_scales(iv), sprintf('%.3f', rmse_p_grid(iv,ip)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'w');
    end
end

figure('Name', sprintf('c Tuning - Dataset %s', data_num), ...
       'NumberTitle', 'off', 'Units', 'normalized', 'OuterPosition', [0.55 0.3 0.45 0.5]);
c_labels = arrayfun(@(x) sprintf('%.0e', x), c_grid, 'UniformOutput', false);
c_labels{1} = '0';

subplot(1,2,1);
bar(rmse_p_rekf);
set(gca, 'XTickLabel', c_labels, 'XTickLabelRotation', 45);
xlabel('c_{rekf}'); ylabel('3D Pos RMSE (m)');
title('REKF'); grid on;

subplot(1,2,2);
bar(rmse_p_rukf);
set(gca, 'XTickLabel', c_labels, 'XTickLabelRotation', 45);
xlabel('c_{rukf}'); ylabel('3D Pos RMSE (m)');
title('RUKF'); grid on;

sgtitle(sprintf('c Tuning - Dataset %s', data_num));

fprintf('\nDone! Now change data_num and re-run for the other datasets.\n');
