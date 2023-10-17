clear;clc;close all;
sti_freq = 130;
% load data
load('simulatedLFP.mat');
% apply notch filters to remove line noise
for jj = 1:10
    w0 = 60*jj/(fs/2);
    [b, a] = iirnotch(w0, w0/200);
    x_add = filtfilt(b, a, x_add);
    x_ori = filtfilt(b, a, x_ori);
end
% apply a high-pass filter to remove low-frequency noise
[b_fil, a_fil] = butter(2, 3/(fs/2), 'high');
x_add = filtfilt(b_fil, a_fil, x_add);
x_ori = filtfilt(b_fil, a_fil, x_ori);
% artifact detection
stime = find_stime(x_add, fs, sti_freq);
% apply SMARTA
[y, sa] = run_SMARTA(x_add, stime, fs, sti_freq);

% check the result
tt = (0:length(x_add)-1)/fs;
figure(1);
hold on;
set(gca, 'fontsize', 20);
plot(tt, x_ori, 'k', 'linewidth', 3);
plot(tt, x_add, 'b', 'linewidth', 1.5);
plot(tt, y, 'r', 'linewidth', 1.5);
xlim([11, 11.1]);
ylim([-50, 50]);
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
legend('Clean LFP', 'Raw data', 'LFP estimated by SMARTA');
set(gcf, 'position',  get(0, 'screensize'));