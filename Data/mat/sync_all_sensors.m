function [t_sync, Delta, dtheta, dv, gps_gt, gps_mea, baro_h, flow_v, dist_h, q_sync] = ...
    sync_all_sensors(imu_tbl, gps_tbl, baro_tbl, flow_tbl, dist_tbl, att_tbl)
%SYNC_ALL_SENSORS Align multiple PX4 sensor data to the highest frequency sensor
%
% INPUTS:
%   imu_tbl   - IMU table, first column timestamp, 2-7: angular & velocity increments
%   gps_tbl   - GPS table, first column timestamp, next columns position and velocity
%   baro_tbl  - Barometer table, first column timestamp, height/pressure/temp
%   flow_tbl  - Optical flow table, first column timestamp, velocity data
%   dist_tbl  - Distance sensor table, first column timestamp, height
%   att_tbl   - Attitude table, first column timestamp, 2-5: quaternions
%
% OUTPUTS:
%   t_sync    - unified time vector
%   Delta     - average time step
%   dtheta    - angular increments [rad] (aligned)
%   dv        - velocity increments [m/s] (aligned)
%   gps_gt    - GPS ground truth velocities & positions [m/s & m] (linear interp)
%   gps_mea   - GPS measurements [m/s & m] (zero-order hold)
%   baro_h    - barometer height (m)
%   flow_v    - optical flow velocity in body 2-3, and ned 4-5 (m/s)
%   dist_h    - distance sensor height (m)
%   q_sync    - synchronized quaternions

%% 1. 提取时间
t_imu  = imu_tbl(:,1)* 1e-6;
t_gps  = gps_tbl(:,1)* 1e-6;
t_baro = baro_tbl(:,1)* 1e-6;
t_flow = flow_tbl(:,1)* 1e-6;
t_dist = dist_tbl(:,1)* 1e-6;
t_att  = att_tbl(:,1)* 1e-6;

%% Step 2: Determine highest frequency sensor
dt_list = [median(diff(t_imu)), median(diff(t_gps)), ...
           median(diff(t_baro)), median(diff(t_flow)), median(diff(t_dist)), median(diff(t_att))];
[dt_min, ~] = min(dt_list);

% Generate unified time axis within common overlapping interval
t_start = max([t_imu(1), t_gps(1), t_baro(1), t_flow(1), t_dist(1), t_att(1)]);
t_end   = min([t_imu(end), t_gps(end), t_baro(end), t_flow(end), t_dist(end), t_att(end)]);
t_sync  = (t_start : dt_min : t_end)';

%% 3. 对齐 IMU 数据
dtheta = interp1(t_imu, imu_tbl(:,2:4), t_sync, 'linear', 'extrap');  % angular increments
dv     = interp1(t_imu, imu_tbl(:,5:7), t_sync, 'linear', 'extrap');  % velocity increments

%% 4. 对齐 GPS 数据
% 假设 gps_tbl(:,5:7) 是位置 NED, (:,2:4) 是速度 NED
gps_pos = gps_tbl(:,5:7);
gps_vel = gps_tbl(:,2:4);

gps_gt  = [interp1(t_gps, gps_vel, t_sync, 'linear', 'extrap'), ...
           interp1(t_gps, gps_pos, t_sync, 'linear', 'extrap')];

gps_mea = [interp1(t_gps, gps_vel, t_sync, 'previous', 'extrap'), ...
           interp1(t_gps, gps_pos, t_sync, 'previous', 'extrap')];  % zero-order hold

%% 5. 对齐 Barometer
%% Step 5: Convert barometer pressure to height
% references: https://en.wikipedia.org/wiki/Hypsometric_equation, 
% https://geo.libretexts.org/Bookshelves/Meteorology_and_Climate_Science/Practical_Meteorology_%28Stull%29/01%3A_Atmospheric_Basics/1.10%3A_Hypsometric_Equation?
baro_p_sync = interp1(t_baro, baro_tbl(:,2:3), t_sync, 'previous');
baro_pressure = baro_p_sync(:,1);
baro_temperature = baro_p_sync(:,2); %Raw data values from the barometer sensor
N_baro = length(baro_temperature);

%define constants 
kelvin_const = 273.15;
tlr = 0.0065; % temperature lapse rate;
P_0 = 101325; % reference sea level pressure
baro_kelvin = baro_temperature + (kelvin_const * ones(N_baro,1));
% R = 287.05; % specific gas constant 
% g0 = 9.80665; %
exp_cst = 0.1903; % (g0.M)/R*.L

% use of Barometric Formula to calculate altitude
baro_alt = -(baro_kelvin)/tlr.*(1 - (baro_pressure/P_0).^(exp_cst));

% calculate velocity with discrete derivatives
baro_vel = zeros(N_baro,1);
for k = 2:N_baro
    baro_vel(k) = 1000*(baro_alt(k)-baro_alt(k-1))./(t_sync(k)-t_sync(k-1));
end
baro_h = [baro_alt, baro_vel]; % Size Number_samples x Number_meas
offset = baro_h(1,1);
baro_h(:,1) = baro_h(:,1) - offset;

%% 6. 对齐 Optical Flow
flow_v = interp1(t_flow, flow_tbl(:,2:5), t_sync, 'linear', 'extrap'); % x/y velocity in body and ned

%% 7. 对齐 Distance sensor
dist_h = interp1(t_dist, dist_tbl(:,2), t_sync, 'linear', 'extrap');   % 高度

%% 8. 对齐 Attitude (Quaternions)
q_sync = interp1(t_att, att_tbl(:,2:5), t_sync, 'linear', 'extrap');

Delta = round(1/dt_min); % frequence
end
