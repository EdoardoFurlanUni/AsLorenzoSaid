%% Plot Optical Flow velocity with Quality overlay
% For each dataset, plots vx and vy from vehicle_optical_flow,
% highlighting low-quality samples (quality < 245) in red.
clc; clear; close all;

base_path = fullfile(fileparts(mfilename('fullpath')), '..', 'Data', 'mat');

folders = {
    'log_46_2025-10-18-10-11-28', '46';
    'log_47_2025-10-18-10-28-26', '47';
    'log_48_2025-10-18-10-40-54', '48';
    'log_49_2025-10-18-10-53-38', '49';
    'log_50_2025-10-18-11-09-00', '50';
};

for f = 1:size(folders, 1)
    folder = folders{f, 1};
    label  = folders{f, 2};
    
    csv_file = fullfile(base_path, folder, [folder '_vehicle_optical_flow_0.csv']);
    dist_file = fullfile(base_path, folder, [folder '_distance_sensor_0.csv']);
    
    if ~exist(csv_file, 'file'), continue; end
    
    % Read vehicle_optical_flow
    tbl = readtable(csv_file);
    t_us   = tbl.timestamp;
    pf_x   = tbl.pixel_flow_0_;   % pixel_flow[0]
    pf_y   = tbl.pixel_flow_1_;   % pixel_flow[1]
    dt_us  = tbl.integration_timespan_us;
    qual   = tbl.quality;
    dist_m = tbl.distance_m;
    
    % Convert to velocity (same formula as sync_all_sensors)
    dt_s = dt_us * 1e-6;
    vx = pf_x ./ dt_s .* dist_m;
    vy = pf_y ./ dt_s .* dist_m;
    
    % Time in seconds from start
    t_sec = (t_us - t_us(1)) * 1e-6;
    
    % Classify quality
    idx_good = (qual == 245);
    idx_mid  = (qual > 0 & qual < 245);   % quality ~120
    idx_bad  = (qual == 0);
    
    n_good = sum(idx_good);
    n_mid  = sum(idx_mid);
    n_bad  = sum(idx_bad);
    
    fprintf('Dataset %s: %d good (245), %d mid (%s), %d bad (0) / %d total\n', ...
        label, n_good, n_mid, ...
        mat2str(unique(qual(idx_mid))'), n_bad, length(qual));
    
    % Create figure
    figure('Name', sprintf('Dataset %s - Quality Overlay', label), ...
           'NumberTitle', 'off', 'Units', 'normalized', ...
           'OuterPosition', [0.05 0.05 0.9 0.9]);
    
    % --- VX ---
    subplot(2,1,1);
    hold on;
    % All points as small grey dots
    scatter(t_sec, vx, 4, [0.7 0.7 0.7], 'filled', 'DisplayName', sprintf('q=245 (%d)', n_good));
    % Low quality on top, same x/y, bigger colored markers
    if any(idx_mid)
        scatter(t_sec(idx_mid), vx(idx_mid), 60, [1 0.6 0], 'o', 'LineWidth', 1.5, ...
             'DisplayName', sprintf('q~120 (%d)', n_mid));
    end
    if any(idx_bad)
        scatter(t_sec(idx_bad), vx(idx_bad), 80, [1 0 0], 'x', 'LineWidth', 2, ...
             'DisplayName', sprintf('q=0 (%d)', n_bad));
    end
    ylabel('v_x body (m/s)');
    title(sprintf('Dataset %s — v_x', label));
    legend('Location', 'best'); grid on;
    hold off;
    
    % --- VY ---
    subplot(2,1,2);
    hold on;
    scatter(t_sec, vy, 4, [0.7 0.7 0.7], 'filled', 'DisplayName', sprintf('q=245 (%d)', n_good));
    if any(idx_mid)
        scatter(t_sec(idx_mid), vy(idx_mid), 60, [1 0.6 0], 'o', 'LineWidth', 1.5, ...
             'DisplayName', sprintf('q~120 (%d)', n_mid));
    end
    if any(idx_bad)
        scatter(t_sec(idx_bad), vy(idx_bad), 80, [1 0 0], 'x', 'LineWidth', 2, ...
             'DisplayName', sprintf('q=0 (%d)', n_bad));
    end
    ylabel('v_y body (m/s)');
    xlabel('Time (s)');
    title(sprintf('Dataset %s — v_y', label));
    legend('Location', 'best'); grid on;
    hold off;
end

fprintf('\nDone! Check the plots:\n');
fprintf('  Grey dots  = quality 245 (good)\n');
fprintf('  Orange circles = quality ~120 (medium)\n');
fprintf('  Red crosses = quality 0 (bad)\n');
fprintf('If red/orange points coincide with spikes, the quality flag is useful for gating.\n');
