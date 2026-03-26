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
base_path = fileparts(mfilename('fullpath'));
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
imu_tbl = table2array(imu(:,[2,5:10]));   

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
%% Optical flow data
% Concatenate the full file path
file_flow = fullfile(full_folder_path, sprintf('log_%s_estimator_optical_flow_vel_0.csv', folder_num));

% data
optic=readtable(file_flow);

% vel-x/y [m/s] in body and vel-x/y [m/s] in ne
flow_tbl = table2array(optic(:,[2,3,4,5,6])); 

%% Distance data
% Concatenate the full file path
file_dist = fullfile(full_folder_path, sprintf('log_%s_distance_sensor_0.csv', folder_num));

% data
dist=readtable(file_dist);

% Height [m] from the ground
dist_tbl = table2array(dist(:,[1,5])); 

%% Alignment
[t_sync, Delta, dtheta, dv, gps_gt, gps_mea, baro_h, flow_v, dist_h, q_sync] = ...
    sync_all_sensors(imu_tbl, gps_tbl, baro_tbl, flow_tbl, dist_tbl, att_tbl);

% t_sync : unified time vector
% Delta  : Data frequence
% dtheta : angular increments (rad)
% dv     : velocity increments (m/s) 
% gps    : velocity (m/s) and positions (m)
% where gt = grouth truth w.r.t linear interp and mea = measurements w.r.t
% zero hold interp
% baro_h : height from barometer (m)
% flow_v : velocity from optical flow (m/s) in body and ne
% dist_h : distance to ground (m)


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
ylabel('y');
legend('\Delta_{v}');