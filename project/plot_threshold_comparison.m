%% Plot Optical Flow with Mean+3Sigma thresholds and Quality overlay
% Shows all data as scatter, draws threshold lines, highlights:
%   - Orange circles: quality ~120
%   - Red X: quality = 0
%   - Blue diamonds: exceed mean+3sigma threshold (but quality=245)
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
    if ~exist(csv_file, 'file'), continue; end
    
    % Read CSV
    tbl = readtable(csv_file);
    pf_x   = tbl.pixel_flow_0_;
    pf_y   = tbl.pixel_flow_1_;
    dt_us  = tbl.integration_timespan_us;
    qual   = tbl.quality;
    dist_m = tbl.distance_m;
    t_us   = tbl.timestamp;
    
    % Convert to velocity
    dt_s = dt_us * 1e-6;
    vx = pf_x ./ dt_s .* dist_m;
    vy = pf_y ./ dt_s .* dist_m;
    t_sec = (t_us - t_us(1)) * 1e-6;
    
    % Compute mean+3sigma thresholds (same formula as task34)
    thr_x = mean(abs(vx), 'omitnan') + 3 * std(vx, 'omitnan');
    thr_y = mean(abs(vy), 'omitnan') + 3 * std(vy, 'omitnan');
    
    % Classify
    idx_mid  = (qual > 0 & qual < 245);
    idx_bad  = (qual == 0);
    idx_outlier_x = (abs(vx) > thr_x) & (qual == 245);  % outliers NOT caught by quality
    idx_outlier_y = (abs(vy) > thr_y) & (qual == 245);
    
    % Both flagged by quality AND threshold
    idx_both_x = (abs(vx) > thr_x) & (qual < 245);
    idx_both_y = (abs(vy) > thr_y) & (qual < 245);
    
    fprintf('Dataset %s:\n', label);
    fprintf('  Threshold X: %.3f m/s,  Y: %.3f m/s\n', thr_x, thr_y);
    fprintf('  Quality<245: %d mid, %d bad\n', sum(idx_mid), sum(idx_bad));
    fprintf('  |vx|>thr (q=245): %d,  |vy|>thr (q=245): %d  <-- outliers MISSED by quality\n', ...
        sum(idx_outlier_x), sum(idx_outlier_y));
    fprintf('  |vx|>thr (q<245): %d,  |vy|>thr (q<245): %d  <-- caught by BOTH\n', ...
        sum(idx_both_x), sum(idx_both_y));
    fprintf('\n');
    
    % Create figure
    figure('Name', sprintf('Dataset %s — Thresholds', label), ...
           'NumberTitle', 'off', 'Units', 'normalized', ...
           'OuterPosition', [0.05 0.05 0.9 0.9]);
    
    % --- VX ---
    subplot(2,1,1); hold on;
    % All data
    scatter(t_sec, vx, 4, [0.7 0.7 0.7], 'filled', 'HandleVisibility', 'off');
    % Threshold lines
    yline( thr_x, '--r', 'LineWidth', 1.5, 'DisplayName', sprintf('+threshold = %.2f', thr_x));
    yline(-thr_x, '--r', 'LineWidth', 1.5, 'DisplayName', sprintf('-threshold = %.2f', -thr_x));
    % Quality markers
    if any(idx_mid)
        scatter(t_sec(idx_mid), vx(idx_mid), 60, [1 0.6 0], 'o', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('q~120 (%d)', sum(idx_mid)));
    end
    if any(idx_bad)
        scatter(t_sec(idx_bad), vx(idx_bad), 80, [1 0 0], 'x', 'LineWidth', 2, ...
            'DisplayName', sprintf('q=0 (%d)', sum(idx_bad)));
    end
    % Outliers with good quality (missed by quality flag!)
    if any(idx_outlier_x)
        scatter(t_sec(idx_outlier_x), vx(idx_outlier_x), 70, [0 0.4 1], 'd', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('|v|>thr but q=245 (%d)', sum(idx_outlier_x)));
    end
    ylabel('v_x body (m/s)');
    title(sprintf('Dataset %s — v_x', label));
    legend('Location', 'best'); grid on; hold off;
    
    % --- VY ---
    subplot(2,1,2); hold on;
    scatter(t_sec, vy, 4, [0.7 0.7 0.7], 'filled', 'HandleVisibility', 'off');
    yline( thr_y, '--r', 'LineWidth', 1.5, 'DisplayName', sprintf('+threshold = %.2f', thr_y));
    yline(-thr_y, '--r', 'LineWidth', 1.5, 'DisplayName', sprintf('-threshold = %.2f', -thr_y));
    if any(idx_mid)
        scatter(t_sec(idx_mid), vy(idx_mid), 60, [1 0.6 0], 'o', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('q~120 (%d)', sum(idx_mid)));
    end
    if any(idx_bad)
        scatter(t_sec(idx_bad), vy(idx_bad), 80, [1 0 0], 'x', 'LineWidth', 2, ...
            'DisplayName', sprintf('q=0 (%d)', sum(idx_bad)));
    end
    if any(idx_outlier_y)
        scatter(t_sec(idx_outlier_y), vy(idx_outlier_y), 70, [0 0.4 1], 'd', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('|v|>thr but q=245 (%d)', sum(idx_outlier_y)));
    end
    ylabel('v_y body (m/s)'); xlabel('Time (s)');
    title(sprintf('Dataset %s — v_y', label));
    legend('Location', 'best'); grid on; hold off;
end

fprintf('Legend:\n');
fprintf('  Grey dots      = all samples\n');
fprintf('  Red dashed     = mean+3sigma threshold\n');
fprintf('  Orange circles = quality ~120 (sensor says medium)\n');
fprintf('  Red X          = quality 0 (sensor says bad)\n');
fprintf('  Blue diamonds  = exceed threshold BUT quality=245 (outliers MISSED by quality flag)\n');
