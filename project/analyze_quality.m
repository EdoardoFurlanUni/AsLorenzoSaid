%% Analyze Optical Flow Quality Flag across all datasets
% Reads the 'quality' column (col 14) from vehicle_optical_flow CSVs
% and shows the distribution for each dataset.
clc; clear; close all;

base_path = fullfile(fileparts(mfilename('fullpath')), '..', 'Data', 'mat');

% All available log folders
folders = {
    'log_46_2025-10-18-10-11-28', '46';
    'log_47_2025-10-18-10-28-26', '47';
    'log_48_2025-10-18-10-40-54', '48';
    'log_49_2025-10-18-10-53-38', '49';
    'log_50_2025-10-18-11-09-00', '50';
};

figure('Name', 'Quality Analysis', 'NumberTitle', 'off', ...
       'Units', 'normalized', 'OuterPosition', [0.05 0.1 0.9 0.85]);

for f = 1:size(folders, 1)
    folder = folders{f, 1};
    label  = folders{f, 2};
    
    csv_file = fullfile(base_path, folder, [folder '_vehicle_optical_flow_0.csv']);
    
    if ~exist(csv_file, 'file')
        fprintf('Dataset %s: CSV not found, skipping.\n', label);
        continue;
    end
    
    % Read the CSV
    tbl = readtable(csv_file);
    quality = tbl.quality;
    
    % Print statistics
    fprintf('========== Dataset %s ==========\n', label);
    fprintf('  Total samples:    %d\n', length(quality));
    fprintf('  Min quality:      %d\n', min(quality));
    fprintf('  Max quality:      %d\n', max(quality));
    fprintf('  Mean quality:     %.1f\n', mean(quality));
    fprintf('  Median quality:   %.1f\n', median(quality));
    fprintf('  Std quality:      %.1f\n', std(quality));
    
    % Count unique values
    [unique_vals, ~, ic] = unique(quality);
    counts = accumarray(ic, 1);
    
    fprintf('  Unique values:    ');
    for j = 1:length(unique_vals)
        pct = 100 * counts(j) / length(quality);
        fprintf('%d (%.1f%%)  ', unique_vals(j), pct);
    end
    fprintf('\n');
    
    % Samples below common thresholds
    for thr = [50, 100, 150, 200]
        n_below = sum(quality < thr);
        fprintf('  quality < %3d:    %d samples (%.2f%%)\n', thr, n_below, 100*n_below/length(quality));
    end
    fprintf('\n');
    
    % Plot histogram
    subplot(2, 3, f);
    histogram(quality, 0:5:260, 'FaceColor', [0.3 0.5 0.9], 'EdgeColor', 'none');
    xlabel('Quality'); ylabel('Count');
    title(sprintf('Dataset %s (n=%d)', label, length(quality)));
    xlim([0 260]); grid on;
    
    % Plot quality over time
    if f <= 5
        subplot(2, 3, 6);
        hold on;
        t_sec = (0:length(quality)-1) / 40;  % ~40 Hz approx
        plot(t_sec, quality, '.', 'MarkerSize', 2, 'DisplayName', ['DS ' label]);
    end
end

subplot(2, 3, 6);
xlabel('Approx time (s)'); ylabel('Quality');
title('Quality over time (all datasets)');
legend('Location', 'best'); grid on;
ylim([0 260]);

fprintf('=== DONE ===\n');
fprintf('Look at the histograms and the console output.\n');
fprintf('If quality is almost always 245, the flag is not useful for gating.\n');
fprintf('If there are clusters at low values, those correspond to unreliable measurements.\n');
