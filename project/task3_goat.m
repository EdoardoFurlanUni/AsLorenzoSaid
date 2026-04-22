%% Task 3 - HYBRID: GPS available → GPS denied (optical flow) → GPS recovered
%
% MEASUREMENT SWITCH LOGIC:
%   GPS phase    : y_k = C * x_k + noise        (linear, 6D: vNED + pNED)
%   GPS-denied   : y_k = h_flow(x_k) + noise    (non-linear, 3D: v_body_x/y + p_d)
%
% The EKF and UKF receive the correct h / h_jac / R at every step.
%
% denied() from the professor is NOT used here because it just freezes the
% GPS value — it does not switch the measurement model.  Our approach is
% the correct one for a real GPS-denied scenario.

clear; clc; close all;

%% 1. Load data
project_dir = pwd;
filters_dir  = fullfile(project_dir, '..', 'filters');
addpath(project_dir);
addpath(filters_dir);
load(fullfile(project_dir, '..', 'Data', 'mat', 'data_sync.mat'));

%% 2. Symbolic setup
syms q0 q1 q2 q3 vn ve vd pn pe pd wbx wby wbz abx aby abz real
syms dthx dthy dthz dvx dvy dvz dt real
sym_x = [q0;q1;q2;q3; vn;ve;vd; pn;pe;pd; wbx;wby;wbz; abx;aby;abz];
sym_u = [dthx;dthy;dthz; dvx;dvy;dvz];

%% 3. Simulation window
start_idx = 1000;
T         = 900;          % 9 s total
dt_val    = 1 / Delta;

% GPS denial window  (seconds relative to start)
t_deny_start = 3;         % GPS lost  after 3 s
t_deny_dur   = 4;         % GPS denied for 4 s  → recovered at 7 s
k_deny_start = round(t_deny_start / dt_val) + 1;
k_deny_end   = round((t_deny_start + t_deny_dur) / dt_val);

fprintf('GPS denied from k=%d to k=%d  (%.1f s – %.1f s)\n', ...
    k_deny_start, k_deny_end, t_deny_start, t_deny_start+t_deny_dur);

%% 4. Build process model (same for both phases)
f_sym     = subs(func_f(sym_x, sym_u, dt), dt, dt_val);
f_num     = matlabFunction(f_sym, 'Vars', {sym_x, sym_u});
f_jac_num = matlabFunction(jacobian(f_sym, sym_x), 'Vars', {sym_x, sym_u});

%% 5. Build OPTICAL FLOW measurement model  h_flow  (GPS-denied phase)
h_flow_sym     = func_h(sym_x);          % 3x1: [v_bx; v_by; p_d]
h_flow_num     = matlabFunction(h_flow_sym, 'Vars', {sym_x});
h_flow_jac_num = matlabFunction(jacobian(h_flow_sym, sym_x), 'Vars', {sym_x});

%% 6. Build GPS measurement model  h_gps  (GPS-available phase)
%    y_gps = [vN; vE; vD; pN; pE; pD]  →  linear, C matrix
C_gps  = [zeros(3,4), eye(3), zeros(3,3), zeros(3,3), zeros(3,3);   % velocity NED
           zeros(3,4), zeros(3,3), eye(3), zeros(3,3), zeros(3,3)];  % position NED
h_gps_sym     = C_gps * sym_x;           % 6x1
h_gps_num     = matlabFunction(h_gps_sym, 'Vars', {sym_x});
h_gps_jac_num = matlabFunction(jacobian(h_gps_sym, sym_x), 'Vars', {sym_x});
% (Jacobian is just C_gps, but we keep the function handle for generality)

%% 7. Noise covariances
B_mat = diag([1e-2*ones(4,1); 0.1*ones(3,1); 0.01*ones(3,1); 1e-6*ones(3,1); 1e-4*ones(3,1)]);

% GPS measurement noise (6D)
D_gps = diag([0.1; 0.1; 0.1;   % velocity NED  (m/s)
               0.5; 0.5; 0.3]); % position NED  (m)

% Optical flow measurement noise (3D)
D_flow = diag([1.2; 1.2; 0.8]);

R_gps  = D_gps  * D_gps';
R_flow = D_flow * D_flow';

%% 8. Measurement data
%   GPS phase   : [vN vE vD pN pE] from gps_gt  +  pD from baro
%   Optical flow: [v_bx v_by] from flow_v  +  pD from baro

idx = start_idx : start_idx+T-1;

y_gps_full  = [gps_gt(idx, 1:5), baro_h(idx,1)]';   % 6 x T
y_flow_full = [flow_v(idx, 1:2),  baro_h(idx,1)]';  % 3 x T

%% 9. Initial state
x0 = [q_sync(start_idx,:)'; ...
      gps_gt(start_idx, 1:3)'; ...
      gps_gt(start_idx, 4:5)', baro_h(start_idx,1); ...
      zeros(3,1); zeros(3,1)];

P0 = diag([1e-4*ones(4,1); 0.1*ones(3,1); 0.1*ones(3,1); 1e-6*ones(3,1); 1e-4*ones(3,1)]);
V0 = P0;

u = [dtheta(idx,:)'; dv(idx,:)'];

%% 10. EKF with switching measurement model
fprintf('Running EKF (hybrid)...\n'); tic;
[Xekf, ~] = EKF_hybrid(x0, y_gps_full, y_flow_full, u, V0, B_mat, ...
                         R_gps, R_flow, ...
                         f_num, h_gps_num, h_flow_num, ...
                         f_jac_num, h_gps_jac_num, h_flow_jac_num, ...
                         T, k_deny_start, k_deny_end);
fprintf('EKF done in %.3f s\n', toc);

%% 11. UKF with switching measurement model
fprintf('Running UKF (hybrid)...\n'); tic;
[Xukf, ~] = UKF_hybrid(x0, y_gps_full, y_flow_full, u, P0, B_mat, ...
                         R_gps, R_flow, ...
                         f_num, h_gps_num, h_flow_num, T, ...
                         k_deny_start, k_deny_end);
fprintf('UKF done in %.3f s\n', toc);

%% 12. Plot
time_axis = t_sync(idx) - t_sync(start_idx);

deny_patch = @(ax) patch(ax, ...
    [t_deny_start, t_deny_start+t_deny_dur, t_deny_start+t_deny_dur, t_deny_start], ...
    [ax.YLim(1), ax.YLim(1), ax.YLim(2), ax.YLim(2)], ...
    [1 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName','GPS denied');

figure('Name','Task3 Hybrid','NumberTitle','off','Color','w','Position',[50 50 1200 700]);

% North velocity
ax1 = subplot(2,2,1);
plot(time_axis, gps_gt(idx,1), 'k--', 'DisplayName','GPS ref'); hold on;
plot(time_axis, Xekf(5,2:T+1), 'r',   'LineWidth',1.5, 'DisplayName','EKF');
plot(time_axis, Xukf(5,2:T+1), 'b--', 'LineWidth',1.5, 'DisplayName','UKF');
ylim_val = ylim; patch([t_deny_start t_deny_start+t_deny_dur t_deny_start+t_deny_dur t_deny_start], ...
    [ylim_val(1) ylim_val(1) ylim_val(2) ylim_val(2)],[1 0.8 0.8],'FaceAlpha',0.3,'EdgeColor','none','DisplayName','GPS denied');
ylabel('v_N (m/s)'); xlabel('Time (s)'); title('North Velocity'); grid on; legend('Location','best');

% East velocity
ax2 = subplot(2,2,2);
plot(time_axis, gps_gt(idx,2), 'k--', 'DisplayName','GPS ref'); hold on;
plot(time_axis, Xekf(6,2:T+1), 'r',   'LineWidth',1.5, 'DisplayName','EKF');
plot(time_axis, Xukf(6,2:T+1), 'b--', 'LineWidth',1.5, 'DisplayName','UKF');
ylim_val = ylim; patch([t_deny_start t_deny_start+t_deny_dur t_deny_start+t_deny_dur t_deny_start], ...
    [ylim_val(1) ylim_val(1) ylim_val(2) ylim_val(2)],[1 0.8 0.8],'FaceAlpha',0.3,'EdgeColor','none','DisplayName','GPS denied');
ylabel('v_E (m/s)'); xlabel('Time (s)'); title('East Velocity'); grid on; legend('Location','best');

% Altitude
ax3 = subplot(2,2,3);
plot(time_axis, movmean(baro_h(idx),10), 'k.', 'MarkerSize',3, 'DisplayName','Barometer'); hold on;
plot(time_axis, Xekf(10,2:T+1), 'r',   'LineWidth',1.5, 'DisplayName','EKF');
plot(time_axis, Xukf(10,2:T+1), 'b--', 'LineWidth',1.5, 'DisplayName','UKF');
ylim_val = ylim; patch([t_deny_start t_deny_start+t_deny_dur t_deny_start+t_deny_dur t_deny_start], ...
    [ylim_val(1) ylim_val(1) ylim_val(2) ylim_val(2)],[1 0.8 0.8],'FaceAlpha',0.3,'EdgeColor','none','DisplayName','GPS denied');
ylabel('p_d (m)'); xlabel('Time (s)'); title('Altitude (Position Down)'); grid on; legend('Location','best');

% Denial indicator
ax4 = subplot(2,2,4);
gps_flag = ones(1,T);
gps_flag(k_deny_start:k_deny_end) = 0;
area(time_axis, gps_flag, 'FaceColor',[0.2 0.7 0.3],'FaceAlpha',0.5,'EdgeColor','none'); hold on;
area(time_axis, 1-gps_flag,'FaceColor',[0.9 0.2 0.2],'FaceAlpha',0.5,'EdgeColor','none');
ylim([0 1.2]); yticks([0 1]); yticklabels({'Optical flow','GPS'});
xlabel('Time (s)'); title('Active Measurement Source'); grid on;

sgtitle('Task 3 – Hybrid Navigation: GPS → Optical Flow → GPS');
saveas(gcf, 'Task3_hybrid.png');
fprintf('Saved: Task3_hybrid.png\n');


%% =========================================================================
%%  LOCAL FUNCTIONS
%% =========================================================================

function [Xekf, V] = EKF_hybrid(x0, y_gps, y_flow, u, V0, B, ...
                                  R_gps, R_flow, ...
                                  f_num, h_gps_num, h_flow_num, ...
                                  f_jac_num, h_gps_jac_num, h_flow_jac_num, ...
                                  T, k_deny_start, k_deny_end)
% EKF with run-time switch between GPS and optical-flow measurement models.

    n    = size(B,1);
    Q    = B * B';
    Xekf = zeros(n, T+1);
    Xekf(:,1) = x0;
    V    = zeros(n, n, T+1);
    V(:,:,1) = V0;
    Xn   = zeros(n, T);

    for i = 1:T
        % --- select measurement model for this step ---
        if i >= k_deny_start && i <= k_deny_end
            % GPS DENIED → optical flow (non-linear)
            y_i     = y_flow(:, i);
            h_fun   = h_flow_num;
            hj_fun  = h_flow_jac_num;
            R       = R_flow;
        else
            % GPS AVAILABLE → linear GPS model
            y_i     = y_gps(:, i);
            h_fun   = h_gps_num;
            hj_fun  = h_gps_jac_num;
            R       = R_gps;
        end

        C_i = hj_fun(Xekf(:,i));
        S_i = C_i * V(:,:,i) * C_i' + R;
        L_i = V(:,:,i) * C_i' / S_i;           % Kalman gain

        hn       = h_fun(Xekf(:,i));
        Xn(:,i)  = Xekf(:,i) + L_i * (y_i - hn);

        A_i          = f_jac_num(Xn(:,i), u(:,i));
        Xekf(:,i+1) = f_num(Xn(:,i), u(:,i));
        V(:,:,i+1)  = A_i*V(:,:,i)*A_i' - A_i*V(:,:,i)*C_i'/S_i*C_i*V(:,:,i)*A_i' + Q;
    end
end


function [x_pred, P_pred] = UKF_hybrid(x_0, y_gps, y_flow, u, P0, B, ...
                                         R_gps, R_flow, ...
                                         f_num, h_gps_num, h_flow_num, len, ...
                                         k_deny_start, k_deny_end)
% UKF with run-time switch between GPS and optical-flow measurement models.

    Q = B * B';
    n = size(P0,1);

    alpha  = 0.1;
    kapa   = 3 - n;
    lambda = alpha^2 * (kapa + n) - n;
    beta   = 2;

    Wm = [repmat(1/(2*(n+lambda)), 1, 2*n), lambda/(n+lambda)];
    Wc = Wm;
    Wc(end) = Wc(end) + 1 - alpha^2 + beta;

    x_pred        = zeros(n, len+1);
    x_pred(:,1)   = x_0;
    x_hat         = zeros(n, len);
    P_pred        = zeros(n, n, len+1);
    P_pred(:,:,1) = P0;

    for t = 1:len
        % --- select measurement model ---
        if t >= k_deny_start && t <= k_deny_end
            y_t   = y_flow(:, t);
            h_fun = h_flow_num;
            R     = R_flow;
        else
            y_t   = y_gps(:, t);
            h_fun = h_gps_num;
            R     = R_gps;
        end

        m = size(R, 1);

        %% Sigma points from x_pred(:,t)
        X = zeros(n, 2*n+1);
        sqrtP = chol(P_pred(:,:,t));
        for i = 1:n
            X(:,i)   = x_pred(:,t) + sqrt(n+lambda) * sqrtP(i,:)';
            X(:,i+n) = x_pred(:,t) - sqrt(n+lambda) * sqrtP(i,:)';
        end
        X(:,2*n+1) = x_pred(:,t);

        %% Predicted measurement mean
        Y      = zeros(m, 2*n+1);
        y_mean = zeros(m, 1);
        for i = 1:2*n+1
            Y(:,i)  = h_fun(X(:,i));
            y_mean  = y_mean + Wm(i) * Y(:,i);
        end

        %% Innovation covariance & Kalman gain
        P_y  = R;
        P_xy = zeros(n, m);
        for i = 1:2*n+1
            ey   = Y(:,i) - y_mean;
            P_y  = P_y  + Wc(i) * (ey * ey');
            P_xy = P_xy + Wc(i) * (X(:,i) - x_pred(:,t)) * ey';
        end
        L = P_xy / P_y;

        %% Update  (FIX: y_t is the full column vector)
        x_hat(:,t) = x_pred(:,t) + L * (y_t - y_mean);
        P_hat      = P_pred(:,:,t) - L * P_y * L';
        P_hat      = (P_hat + P_hat')/2 + 1e-10*eye(n);

        %% Sigma points from x_hat(:,t)
        X_hat = zeros(n, 2*n+1);
        sqrtP_hat = chol(P_hat);
        for i = 1:n
            X_hat(:,i)   = x_hat(:,t) + sqrt(n+lambda) * sqrtP_hat(i,:)';
            X_hat(:,i+n) = x_hat(:,t) - sqrt(n+lambda) * sqrtP_hat(i,:)';
        end
        X_hat(:,2*n+1) = x_hat(:,t);

        %% Propagate & compute prediction covariance
        X_pred       = zeros(n, 2*n+1);
        x_pred(:,t+1) = zeros(n,1);
        for i = 1:2*n+1
            X_pred(:,i)    = f_num(X_hat(:,i), u(:,t));
            x_pred(:,t+1)  = x_pred(:,t+1) + Wm(i) * X_pred(:,i);
        end

        P_next = Q;
        for i = 1:2*n+1
            ex     = X_pred(:,i) - x_pred(:,t+1);
            P_next = P_next + Wc(i) * (ex * ex');
        end
        P_pred(:,:,t+1) = (P_next + P_next')/2 + 1e-10*eye(n);
    end
end