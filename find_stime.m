function Stime = find_stime(x, fs, sti_freq)
%% find_stime: detect the locations of stimulus artifacts
% input:
    % x: signal with artifact
    % fs: sampling rate
    % sti_freq: stimulation frequency
% output:
    % Stime: time labels for all stimulation artifacts

    n = length(x);
    [b_hp2,a_hp2] = butter(3, 300/(fs/2), 'high');
    x = filtfilt(b_hp2, a_hp2, x);
    x = x - movmean(x, 1*fs);
    shift = round(0.5e-3*fs);
    xs = smooth(abs(x), 10);
    q = quantile(xs, 0.95);
    interval = floor(fs/sti_freq) - shift;
    [~, Stime] = findpeaks(xs, 'MinPeakDistance', interval, 'MinPeakHeight', q);
    Stime = Stime - shift;
    sd = diff(Stime);
    interval_neighbor = floor(fs/sti_freq) + shift;
    idx = find([sd; 0] > interval_neighbor & [0; sd] > interval_neighbor);
    Stime(idx) = [];
end