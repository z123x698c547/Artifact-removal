function [z, sa] = run_SMARTA(y, Stime, fs, sti_freq)
%% run_SMARTA: apply Shrinkage and Manifold-based Artifact Removal using Template Adaptation (SMARTA) for artifact removal
% input:
    % y: signal with artifact
    % Stime: locations of stimulus artifacts
    % fs: sampling rate
    % sti_freq: stimulation frequency
% output: 
    % z: recovered LFP
    % sa: estimated stimulus artifacts

    x = y;
    noise_est = 5*fs;
    [b_hp2,a_hp2] = butter(3, 300/(fs/2), 'high');
    xtmp = filtfilt(b_hp2, a_hp2, x);
    sigma = std(xtmp(1:round(noise_est)));
    
    st_point = -1*round(0.5e-3*fs);
    ed_point = round(1/sti_freq*fs);
    mid_point = round((ed_point-st_point)/2);
    if (Stime(end)+ed_point)>length(x)
        Stime(end) = [];
    end
    if (Stime(1)+st_point)<1
        Stime(1) = [];
    end
    
    N = length(Stime);
    X_all = zeros(N, ed_point-st_point+1);
    for idx = 1:N
        tt = Stime(idx)+st_point:Stime(idx)+ed_point;
        X_all(idx, :) = x(tt);
    end
    
    X_hp_all = filtfilt(b_hp2, a_hp2, X_all');
    X_hp_all = X_hp_all';
    
    nrange = 2000;
    nframe = floor(N/nrange);
    Xc_os_all = zeros(size(X_hp_all));
    for ii= 1:nframe-1
        idx = (ii-1)*nrange+1:ii*nrange;
        X_tmp = X_hp_all(idx, :);
        [X_tmp, eta, r_p] = optimal_shrinkage_color_fast(X_tmp, 'fro', 10, 15);
        Xc_os_all(idx, :) = real(X_tmp);
    end
    
    X_tmp = X_hp_all((nframe-1)*nrange+1:end, :);
    [X_tmp, eta, r_p] = optimal_shrinkage_color_fast(X_tmp, 'fro', 10, 15);
    Xc_os_all((nframe-1)*nrange+1:end, :) = real(X_tmp);

    nrange2 = 500;
    D = calDisAll(Xc_os_all, 1);
    Ncand = 10:10:100;
    Nmax = Ncand(end);
    sa = zeros(size(x));
    z = x;
    Xc_all = zeros(size(X_all));
    for ii = 1:N
        idx1 = knnSelf(ii, D, nrange2, 2);
        idx0 = find(idx1 == ii);
        X_tmp = X_hp_all(idx1, :);
        [X_tmp, eta, r_p] = optimal_shrinkage_color_fast(X_tmp, 'fro', 10, 15);

        idx2 = knnsearch(X_tmp, X_tmp(idx0, :), 'k', Nmax+1);
        idx2(idx2==idx0) = [];
        idx_comb = idx1(idx2);
        
        tt = Stime(ii)+st_point:Stime(ii)+ed_point;
        xtmp = x(tt);
        ar_all = zeros(length(Ncand), 1);
        for jj = 1:length(Ncand)
            tmp = xtmp - median(X_all(idx_comb(1:Ncand(jj)), :), 1);
            ar_all(jj) = ARquick(tmp, mid_point);
        end
        [~, idx] = min(ar_all);
        Nbest = Ncand(idx);
        
        tmp = median(X_all(idx_comb(1:Nbest), :), 1);
        if ii ~= 1
            gap = ed_point-(Stime(ii)-Stime(ii-1)+st_point);
            win = ones(1, size(X_all, 2));
            win(1: gap) = sin(pi*(0:gap-1)/2/gap).^2;
        else
            win = ones(1, size(X_all, 2));
        end

        if ii ~= N
            gap = ed_point-(Stime(ii+1)-Stime(ii)+st_point);
            win(end - gap + 1: end) = cos(pi*(1 : gap)/ 2 / gap).^2 ;
        end
        Xc_all(ii, :) = tmp.*win;
    end
    for ii = 1:N
        tmp = Xc_all(ii, :);
        tt = Stime(ii)+st_point:Stime(ii)+ed_point;
        sa(tt) = sa(tt) + tmp;
        z(tt) = z(tt) - tmp;
    end
end

function ar = ARquick(x, midpoint)
    Ei = x(midpoint+1:end);
    tEi = x(1:midpoint);
    tmp1 = 0.5*(median(abs(tEi-median(tEi)))/median(abs(Ei-median(Ei))) + median(abs(Ei-median(Ei)))/median(abs(tEi-median(tEi))));
    tmp2 = 0.5*(quantile(abs(tEi-median(tEi)), 0.95)/max(abs(Ei-median(Ei))) + max(abs(Ei-median(Ei)))/quantile(abs(tEi-median(tEi)), 0.95));
    ar= abs(log(tmp1*tmp2));
end

function D = calDisAll(X_all, type)
    [n, p] = size(X_all);
    D = zeros(n, n);
    if type == 2
        xmin = min(X_all(:));
        xmax = max(X_all(:));
        nbins = xmin:(xmax-xmin)/20:xmax;
        nbins = nbins(2:end);
    elseif type == 3
        coeff = pca(X_all);
        trans = coeff(:, 1:10);
        X_trans = X_all*trans;
    end
    for ii = 1:n
%         if mod(ii, 1000) == 0
%             fprintf('Distance calculation #%d\n', ii);
%         end
        x = X_all(ii, :);
        for jj = ii+1:n
            y = X_all(jj, :);
            if type == 1
                D(ii, jj) = norm(x-y);
            elseif type == 2
                % nbins = 20;
                D(ii, jj) = calOTD(x, y, nbins);
            elseif type == 3
                x = X_trans(ii, :);
                y = X_trans(jj, :);
                D(ii, jj) = norm(x-y);
            end
        end
    end
    D = D + D';
end

function idx = knnSelf(ii, D, K, type)
% type: (1) not include ii (2) include ii
    dis = D(ii, :);
    [~, idx0] = sort(dis);
    if type == 1
        idx = idx0(2:K+1);
    elseif type == 2
        idx = idx0(1:K+1);
    end
end