clc
clear
close all
% This code is to align all sensor data
%% load real data (Select the folder number to read)
folder_num = input('Please enter the data number to be read:','s');
% 46_2025-10-18-10-11-28
% 47_2025-10-18-10-28-26
% 48_2025-10-18-10-40-54
% 49_2025-10-18-10-53-38
% 50_2025-10-18-11-09-00

% Constructing a folder path
% base_path = fileparts(mfilename('fullpath'));
base_path = pwd;
folder_name = sprintf('log_%s', folder_num);
full_folder_path = fullfile(base_path, folder_name);

%%%%%%%%%%%%%%%%%%%%%%%%% IMU %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Concatenate the full file path
file_IMU = fullfile(full_folder_path, sprintf('log_%s_vehicle_imu_0.csv', folder_num));

% data
imu = readtable(file_IMU);
% 2:    Timestamp
% 5-7:  IMU angular increments [rad] in body
% 8-10: IMU velocity increments [m/s] in body
% 11-12: dt for angle and velocity [us]

% Convert increments to true rates (rad/s and m/s^2) using exact delta_t
dt_ang = table2array(imu(:,11)) * 1e-6; % microseconds -> seconds
dt_vel = table2array(imu(:,12)) * 1e-6;

rate_ang = table2array(imu(:,5:7)) ./ dt_ang; % rad -> rad/s
rate_vel = table2array(imu(:,8:10)) ./ dt_vel; % m/s -> m/s²

% Save as rates in imu_tbl
imu_tbl = [table2array(imu(:,2)), rate_ang, rate_vel];

%%%%%%%%%%%%%%%%%%%%%%% GPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Concatenate the full file path
file_gps = fullfile(full_folder_path, sprintf('log_%s_sensor_gps_0.csv', folder_num));

% data
gps = readtable(file_gps);

%% GPS original data -- NED position data
lat = table2array(gps(:,3)); % GPS original latitude
lon = table2array(gps(:,4)); % GPS original longitude
alt = table2array(gps(:,5)); % GPS original altitude

xyzNED_pos = GPS_NED(lat,lon,alt)'; % Position in NED

%% GPS original data -- NED velocity data
vel_n = table2array(gps(:,18)); % north velocity
vel_e = table2array(gps(:,19)); % east velocity
vel_d = table2array(gps(:,20)); % down velocity

xyzNED_vel = [vel_n, vel_e, vel_d]; % Velocity in NED

% Velocty-lon/lat/alt [m/s] and Position-lon/lat/alt [m] in NED
gps_tbl = [table2array(gps(:,1)), xyzNED_vel, xyzNED_pos]; 

%%%%%%%%%%%%%%%%%%%%%%%% Barometer %%%%%%%%%%%%%%%%%%%%%%%
% Concatenate the full file path
file_baro = fullfile(full_folder_path, sprintf('log_%s_sensor_baro_0.csv', folder_num));

% data
baro1 = readtable(file_baro);

% Pressure [Pa] and Temperature [C]
baro_tbl = table2array(baro1(:,[2,4,5])); 

%%%%%%%%%%%%%%%%%%%%%%%%% Attitude %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Concatenate the full file path
file_att = fullfile(full_folder_path, sprintf('log_%s_vehicle_attitude_0.csv', folder_num));

% data
att = readtable(file_att);
% 1: Timestamp
% 3-6: Quaternions q[0], q[1], q[2], q[3]
att_tbl = table2array(att(:,[1,3:6]));

%%%%%%%%%%%%%%%%%%%%%%%%% Optical flow and Distance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Level 3: Estimator optical flow
file_flow = fullfile(full_folder_path, sprintf('log_%s_estimator_optical_flow_vel_0.csv', folder_num));
optic = readtable(file_flow);
% vel-x/y [m/s] in body (cols 3-4) and vel-N/E [m/s] (cols 5-6)
flow_tbl = table2array(optic(:, [2, 3, 4, 5, 6]));

%% Level 2: Vehicle optical flow
file_veh_flow = fullfile(full_folder_path, sprintf('log_%s_vehicle_optical_flow_0.csv', folder_num));
veh_flow = readtable(file_veh_flow);
% Cols: timestamp(1), pixel_flow[0](4), pixel_flow[1](5), distance_m(9), integration_timespan_us(10)
veh_flow_tbl = table2array(veh_flow(:, [1, 4, 5, 9, 10]));

%% Level 1: Raw sensor optical flow (no gyro compensation)
file_raw_flow = fullfile(full_folder_path, sprintf('log_%s_sensor_optical_flow_0.csv', folder_num));
raw_flow = readtable(file_raw_flow);
% Here we use distance_sensor_0 (dist_tbl) for height, passed into sync_all_sensors.
% Extract: [timestamp(1), pixel_flow[0](4), pixel_flow[1](5), integration_timespan_us(10)]
raw_flow_tbl = table2array(raw_flow(:, [1, 4, 5, 10]));  % 4 cols (no distance)

%% Distance data
file_dist = fullfile(full_folder_path, sprintf('log_%s_distance_sensor_0.csv', folder_num));
dist = readtable(file_dist);
% Height [m] from the ground
dist_tbl = table2array(dist(:, [1, 5]));

%% Alignment
[t_sync, Delta, dtheta, dv, gps_gt, gps_mea, baro_h, flow_v, dist_h, q_sync, raw_flow_v, veh_flow_v] = ...
    sync_all_sensors(imu_tbl, gps_tbl, baro_tbl, flow_tbl, dist_tbl, att_tbl, raw_flow_tbl, veh_flow_tbl);

% t_sync     : unified time vector
% Delta      : data frequency (Hz)
% dtheta     : angular increments (rad)
% dv         : velocity increments (m/s)
% gps_gt     : GPS ground truth velocities & positions (linear interp)
% gps_mea    : GPS measurements (zero-order hold)
% baro_h     : height from barometer (m)
% flow_v     : Level 3 - estimator_optical_flow body/NE velocities (m/s)
% dist_h     : distance to ground (m)
% raw_flow_v : Level 1 - body velocities from raw sensor_optical_flow (m/s)
% veh_flow_v : Level 2 - body velocities from vehicle_optical_flow, gyro-comp (m/s)


%% GPS-denied setting 
Time_start    = 100; % Start time of denial
Time_duration = 100;  % Duration of the denial
gps_den = denied(gps_mea, Time_start,... % Start time of denial
               Time_duration,...% Duration of the denial
               Delta... % Data frequency
               );

save('data_sync.mat');

%% Figure
figure(1)
plot(gps_gt(:,4), 'b'); hold on,
plot(gps_mea(:,4), 'c'); hold on,
plot(gps_den(:,4), 'r');           
xlabel('North');
ylabel('South');
legend('Ground truth','Measurements','Denied data');

figure(2)
plot(gps_gt(:,6), 'b'); hold on,
plot(baro_h(:,1),'r')
ylabel('Altitude');
legend('Alt_{GPS}','Alt_{Baro}');

figure(3)
subplot(1,2,1)
plot(flow_v(:,3), 'c-'); hold on,
plot(gps_gt(:,1), 'b'); 
ylabel('x');
subplot(1,2,2)
plot(flow_v(:,4), 'c-'); hold on,
plot(gps_gt(:,2), 'b'); 
ylabel('y');
legend('v_{flow}','v_{gps}');

figure(4)
subplot(1,3,1)
plot(dtheta(:,1), 'c-'); 
ylabel('x');
subplot(1,3,2)
plot(dtheta(:,2), 'c-');  
ylabel('y');
subplot(1,3,3)
plot(dtheta(:,3), 'c-'); 
ylabel('z');
legend('\Delta_{\theta}');

figure(5)
subplot(1,3,1)
plot(dv(:,1), 'c-'); 
ylabel('x');
subplot(1,3,2)
plot(dv(:,2), 'c-'); 
ylabel('y');
subplot(1,3,3)
plot(dv(:,3), 'c-'); 
ylabel('z'); % z not y (?)
legend('\Delta_{v}');

%% IMU Scale Bug Verification
fprintf('\n--- IMU Scale Bug Check ---\n');

mean_acc_norm  = mean(sqrt(dv(:,1).^2     + dv(:,2).^2     + dv(:,3).^2))     * Delta;
mean_gyro_norm = mean(sqrt(dtheta(:,1).^2 + dtheta(:,2).^2 + dtheta(:,3).^2)) * Delta;

fprintf('Mean |acc|  = %.4f m/s^2  (expected ~9.81)\n', mean_acc_norm);
fprintf('Mean |gyro| = %.4f rad/s  (expected < 2.0)\n', mean_gyro_norm);

if abs(mean_acc_norm - 9.81) < 2.0
    fprintf('[OK] Accelerometer scale looks correct.\n');
else
    fprintf('[WARNING] Accelerometer scale looks wrong! Bug may still be present.\n');
end

if mean_gyro_norm < 2.0
    fprintf('[OK] Gyroscope scale looks correct.\n');
else
    fprintf('[WARNING] Gyroscope scale looks wrong! Bug may still be present.\n');
end