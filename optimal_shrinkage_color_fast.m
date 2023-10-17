function [Y_os_out,eta_out,r_p_out, k_sav] = optimal_shrinkage_color_fast(Y_ori,loss,k_l,k_h)
%Optimal singular value shrinkage over color noise
%=========input====================
% Y : Noisy data matrix;
% loss: = 'fro', 'op', ,'op2', 'nuc', 'rank'
% k_l : lower bound for search of k
% k_h : upper bound for search of k
%=========output===================
% Y_os_out: denoised matrix
% eta_out: shrinked singular values
% r_p_out: estimated rank
% Pei-Chun Su, 09/2022

[p,n] = size(Y_ori);
transpose = 0;
if p>n
    Y_ori = Y_ori';
    transpose = 1;
end
[p,n] = size(Y_ori);

p_ = 50;
RR = randn(n, p_) ;
for jj = 1: p_
    RR(:, jj) = RR(:, jj) ./ norm(RR(:, jj)) ;
end

YY = Y_ori*RR;
[u,l,v] = svd(YY);
O = u(:, 1:min(p_, size(u, 2)));
Y = O'*Y_ori;

[U,s,V] = svd(Y);
s = diag(s);

err = inf;
k_sav = 1;
for k = k_l:k_h
    u = eig(Y'*Y); u = sort(u,'descend');
    lab = eig(Y*Y'); lab = sort(lab,'descend');
    fZ = createPseudoNoise(s, k, 'i');
    r_p = max(find(lab>(fZ(1)^2)));
    ov = lab(1:r_p);
    eta = zeros(1,length(lab));
    eta_a = zeros(1,length(lab));

    for j = 1:r_p
        lab(1:p_) = fZ.^2;
        u(1:p_) = fZ.^2;
        m1 = (1/p_ *sum(1./(lab(1:end)-ov(j))));
        dm1 = (1/p_ *sum(1./(lab(1:end)-ov(j)).^2));
        %m2 = (1/n *sum(1./(u(1:end)-ov(j))));
        %dm2 = (1/n *sum(1./(u(1:end)-ov(j)).^2));
        m2 = -(1-p_/n)/ov(j) + m1*p_/n;
        dm2 = (1-p_/n)/(ov(j)^2) + dm1*p_/n;
        Tau = ov(j)*m1*m2; dTau = m1*m2 + ov(j)*dm1*m2 + ov(j)*m1*dm2;
        d = 1/sqrt(ov(j)*m1*m2);
        a1 = abs(m1/(d^2*dTau)); a2 = (m2/(d^2*dTau));

        if loss == "fro"
            eta(j) = d*sqrt(a1*a2);
        elseif loss == "op"
            eta(j) = d;%*sqrt(min(a1,a2)/max(a1,a2));
        elseif loss == "op2"
            eta(j) = d*sqrt(min(a1,a2)/max(a1,a2));
        elseif loss == "nuc"
            eta(j) = abs(d*(sqrt(a1*a2)- sqrt((1-a1)*(1-a2))));
        elseif loss == "rank"
            eta(j) = s(j);
        end
        %eta_a(j) = sqrt(a1*a2);
        %if sqrt(a1*a2)<w
        %    eta(j) = 0;
        %    eta_a(j) =0;
        %end

    end

    r_p = sum(eta>0);
    Y_os = O*U*diag(eta)*V(:,1:p_)';
%     Y_os = Y_os'*O';
%     Y_os = Y_os';
%     if transpose
%         Y_os = Y_os';
%     end
    sub_lambda1 = max(svd(Y_ori-Y_os));
    error = abs(sub_lambda1 - sqrt(lab(1)));
    if error<err
        err = error;
        k_sav = k;
        if transpose
            Y_os_out = Y_os';
        else
            Y_os_out = Y_os;
        end
        eta_out = eta;
        r_p_out = r_p;
%     else
%         break;
    end
end
end
