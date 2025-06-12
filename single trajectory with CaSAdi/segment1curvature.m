
% ====================================================
% 多段分段 + 按曲率加密 + 调整初始猜测
% 使用 CasADi（import casadi.*）做轨迹优化

function newversion_multisegment_curvature()
    import casadi.*

    %% === 0. 读取赛道数据 & 构造插值函数 ===
    data = readmatrix('Monza2.csv');
    x_data = data(:,1);
    y_data = data(:,2);
    wr_orig = data(:,3);
    wl_orig = data(:,4);

    % M：原始离散点数
    M = numel(x_data);

    % “原始索引” s_data = 1:M
    s_data = (1:M)';

    % 先算出离散导数，并做插值
    dX_data = gradient(x_data) ./ gradient(s_data);
    dY_data = gradient(y_data) ./ gradient(s_data);
    phi_data = atan2(dY_data, dX_data);

    % 把赛道中心线和导数都用 CasADi 插值封装
    
    X_ref   = interpolant('X_ref',   'linear', {s_data}, x_data');
    Y_ref   = interpolant('Y_ref',   'linear', {s_data}, y_data');
    dXdu_f  = interpolant('dXdu_ref','linear', {s_data}, dX_data');
    dYdu_f  = interpolant('dYdu_ref','linear', {s_data}, dY_data');
    phi_ref = interpolant('phi_ref', 'linear', {s_data}, phi_data');
    wr_ref  = interpolant('wr_ref',  'linear', {s_data}, wr_orig');
    wl_ref  = interpolant('wl_ref',  'linear', {s_data}, wl_orig');

    %% === 1. 分段参数 & 总段数设定 ===
    numSegments = 2;     % 把跑道分成段
    N_sub       = 240;    % 每段内部离散步数（自行调整）

    % 均匀把 [1, M] 分成 4 段，得到每段的起点 s_start、终点 s_end（整数索引）
    segBounds = round(linspace(1, M, numSegments+1));
    % 比如 segBounds = [1, 450, 900, 1350, M]
    

    % numSegments = 3;      % 假设分成 4 段
    segTime    = zeros(1, numSegments);      % 存每段用时
    segSstart  = zeros(1, numSegments);      % 存每段起始 s
    segSend    = zeros(1, numSegments);      % 存每段结束 s
    segXstart  = cell(1, numSegments);       % 存每段的起点状态（7×1）
    segXend    = cell(1, numSegments);       % 存每段的终点状态（7×1）
    segXk_all  = cell(1, numSegments);       % 存每段完整的状态轨迹
    segUk_all  = cell(1, numSegments);       % 存每段完整的控制序列


    % 全局拼接结果用的变量
    X_all = [];    % (7×总点数) 所有段拼在一起的状态
    U_all = [];    % (2×总步数) 所有段拼在一起的控制
    T_all = 0;     % 累计整个赛道走完所需的时间

    exitStatus  = cell(numSegments,1);
    % 上一段的末点状态，用作“下一段”的初始状态
    X_end_prev = [];  % 7×1 向量
    % -----------------------------------------------------------------
    % 循环调用下面的嵌套函数 solveSegment
    for k = 1:numSegments
        s_start = segBounds(k);
        s_end   = segBounds(k+1);

        if k == 1
            X_start_val = [];  % 第一段没有上一段末状态，留空
        else
            X_start_val = X_end_prev;  % 直接把上一段末状态当作本段初始
        end

        % 调用嵌套子函数：解第 k 段
 [Xk_opt, Uk_opt, Tk_opt, ret_stat] = solveSegment(k, X_start_val, s_start, s_end, N_sub);
 fprintf('=== 段 %d 求解结束：T_segment = %.3f s， return_status = %s ===\n', ...
             k, full(Tk_opt), ret_stat);

      exitStatus{k} = ret_stat;  % 保存这一段的 IPOPT 退出状态
         % —— 把本段关键数据“存”进去 —— 
       segTime(k)   = full(Tk_opt);                  % 本段用时
       segSstart(k) = s_start;                       % 本段起始 s
       segSend(k)   = s_end;                         % 本段结束 s
       segXstart{k} = Xk_opt(:,1);                   % 本段的起点状态（7×1）
       segXend{k}   = Xk_opt(:,end);                 % 本段的终点状态（7×1）
       segXk_all{k} = full(Xk_opt);                  % 本段整条解的状态
       segUk_all{k} = full(Uk_opt);                  % 本段整条解的控制



        if k == 1
            % 第一段，把所有 N_sub+1 个点一次性收进 X_all
            X_all = Xk_opt;          % (7 × (N_sub+1))
            U_all = Uk_opt;          % (2 × N_sub)
            T_all = Tk_opt;          % 累计时间
        else
            % 下一段与上一段 “末点” 会重复一个时间点，所以要去掉 Xk_opt(:,1)
            X_all = [ X_all,           Xk_opt(:,2:end) ];  %#ok<AGROW>
            U_all = [ U_all,           Uk_opt          ];  %#ok<AGROW>
            T_all = T_all + Tk_opt;
        end

        % 保存本段的“末状态” = Xk_opt(:,end)，给下一段用
        X_end_prev = Xk_opt(:, end);


        fprintf('== 段 %d 解算完成：T_segment = %.3f s， s 范围 [%d → %d]\n', ...
                 k, full(Tk_opt), s_start, s_end);
    end
fprintf('==== 多段累计总时间 T_all = %.3f s ====\n', full(T_all) );
% 【新增】—— 在主循环结束后，统一输出每段的 s_start → s_end 及 return_status
fprintf('\n================ 各段退出状态 汇总 ================\n');
fprintf(' 段号 |  s_start → s_end  |   return_status  \n');
fprintf('------+-------------------+--------------------\n');
for k = 1:numSegments
    fprintf('  %2d   |   %4d  →  %4d   |   %s\n', ...
        k, segBounds(k), segBounds(k+1), exitStatus{k});
end
fprintf('===================== 汇总结束 ====================\n\n');
    
fprintf('\n==================== 各段汇总====================\n');
fprintf(' 段号 |  s_start → s_end  |  本段用时 [s]  |  起点速度 [km/h]  |  终点速度 [km/h]\n');
fprintf(' ----+-------------------+----------------+-------------------+---------------------\n');
for k = 1:numSegments
    v_start_k = segXstart{k}(5)*3.6;   % 把 m/s 换算成 km/h
    v_end_k   = segXend{k}(5)*3.6;
    fprintf(' %2d   |    %4d → %4d    |    %6.3f     |      %6.2f      |      %6.2f\n', ...
            k, segSstart(k), segSend(k), segTime(k), v_start_k, v_end_k );
end
fprintf(' ==================== 汇总结束====================\n');


    %% === 3. 最终可视化：把拼接好的 X_all, U_all 以及赛道一起画出来 ===
    % 3.1. 先画一个细致的赛道中心线与走廊边界
    u_plot = linspace(1, M, 400);
    XbP    = full( X_ref(u_plot) );
    YbP    = full( Y_ref(u_plot) );
    dXuP   = full( dXdu_f(u_plot) );
    dYuP   = full( dYdu_f(u_plot) );
    LpP    = sqrt(dXuP.^2 + dYuP.^2)+1e-6;
    nX     = -dYuP ./ LpP;
    nY     =  dXuP ./ LpP;
    margin = 4.0;

    XcorrL = XbP + nX*margin;
    YcorrL = YbP + nY*margin;
    XcorrR = XbP - nX*margin;
    YcorrR = YbP - nY*margin;

    % 3.2. 画图：一张子图显示“中心线+走廊+优化轨迹”，另一张画速度曲线
    figure('Name','Multi‐Segment Trajectory & Speed','NumberTitle','off');
    subplot(2,1,1);
    hold on; grid on; axis equal;
    plot(XcorrL, YcorrL, 'k--');
    plot(XcorrR, YcorrR, 'k--');
    plot(XbP, YbP, 'k-', 'LineWidth',1.5);
    plot(X_all(1,:), X_all(2,:), 'b-', 'LineWidth',1.8);
    xlabel('X [m]'); ylabel('Y [m]');
    title('Multi‐Segment: Centerline, Corridor & Optimized Path');
    legend('Corridor Left','Corridor Right','Centerline','Optimized','Location','best');
  

    subplot(2,1,2);
    tgrid = linspace(0, full(T_all), size(X_all,2));
    plot(tgrid, X_all(5,:)*3.6, 'b-o', 'MarkerSize',3, 'LineWidth',1);
    grid on;
    xlabel('Time [s]'); ylabel('Speed [km/h]');
    title('Multi‐Segment Speed Profile');
    sgtitle('Multi‐Segment Trajectory Optimization');



    %% ===== 嵌套函数：解第 k 段的子问题 =====
    function [Xk_opt, Uk_opt, Tk_opt, ret_stat] = solveSegment(k, X_start_value, s_start, s_end, N_sub)
        % 说明：
        %   k：第几段（用来打印日志，没别的）
        %   X_start_value：上段末状态；如果是第一段，则为空 []
        %   s_start：本段开始的赛道整数索引
        %   s_end  ：本段结束的赛道整数索引
        %   N_sub  ：本段内部的离散步数
        %
        import casadi.*

        % —— 3.1. 本段自由终点时间变量 & 时间步长 dt_sub
        opti_sub = Opti();
        Tseg     = opti_sub.variable();  
        dt_sub   = Tseg / N_sub;
        opti_sub.subject_to( Tseg > 0 );

        % —— 3.2. 本段的决策变量：状态 Xk (7×(N_sub+1)), 控制 Uk (2×N_sub)
        Xk = opti_sub.variable(7, N_sub+1);
        Uk = opti_sub.variable(2, N_sub);

        % 把 Xk、Uk 拆行，方便写约束
        Xp_k   = Xk(1,:);   Yp_k  = Xk(2,:);   psi_k = Xk(3,:);
        beta_k = Xk(4,:);   v_k   = Xk(5,:);   r_k   = Xk(6,:);
        s_k    = Xk(7,:);

        delta_k = Uk(1,:);  Fdr_k = Uk(2,:);

        % —— 3.3. “初始/末端”约束
        if isempty(X_start_value)
            % 第一段：强制从 s = s_start 的赛道索引出发
            opti_sub.subject_to( s_k(1) == s_start );
            opti_sub.subject_to( Xp_k(1) == X_ref(s_start) );
            opti_sub.subject_to( Yp_k(1) == Y_ref(s_start) );
            opti_sub.subject_to( psi_k(1) == phi_ref(s_start) );
            opti_sub.subject_to( beta_k(1)== 0 );
            opti_sub.subject_to( v_k(1)   == 5 );
            opti_sub.subject_to( r_k(1)   == 0 );
        else
            % 非第一段：直接把上段末状态当作本段初始
            opti_sub.subject_to( Xk(:,1) == X_start_value );
            % （包含 s、X、Y、psi、beta、v、r 一并固定）
        end

        % 强制本段末端到达 s = s_end
        opti_sub.subject_to( s_k(N_sub+1) == s_end );
        opti_sub.subject_to( Xp_k(N_sub+1) == X_ref(s_end) );
        opti_sub.subject_to( Yp_k(N_sub+1) == Y_ref(s_end) );
        % （末端的 psi、beta、v、r 不加硬约束，自由调整）

        % —— 3.4. 保证 s 单调 & 范围限制
        for i = 1:N_sub
            opti_sub.subject_to( s_k(i+1) >= s_k(i) );
        end
        opti_sub.subject_to( s_k >= s_start );
        opti_sub.subject_to( s_k <= s_end );

        % 限制速度 v ≥ 0
        opti_sub.subject_to( v_k >= 0 );

        % —— 3.5. 控制/状态上下界
        delta_max = pi/4;  beta_max  = pi/4;  psi_max = pi;
        Fdr_max   = 7100;  Fdr_min   = -21000;
        opti_sub.subject_to( delta_k <=  delta_max );
        opti_sub.subject_to( delta_k >= -delta_max );
        opti_sub.subject_to( beta_k  <=  beta_max );
        opti_sub.subject_to( beta_k  >= -beta_max );
        opti_sub.subject_to( psi_k   <=  psi_max );
        opti_sub.subject_to( psi_k   >= -psi_max );
        opti_sub.subject_to( Fdr_k   <=  Fdr_max );
        opti_sub.subject_to( Fdr_k   >=  Fdr_min );

        % —— 3.6. 本段车辆动力学参数（同前面）
        m   = 1500;  Jz = 2500;
        lf  = 1.2;   lr = 1.3;
        wf  = 1.5;   wr_ = 1.5;
        g   = 9.81;
        Fz_f = (m*g)*(lr/(lf+lr))/2;
        Fz_r = (m*g)*(lf/(lf+lr))/2;
        C1 = 1.2; C2 = 2.0; C3 = 0.3;
        eps = 1e-6;

        % —— 3.7. 定义本段动态函数 f_dyn_sub(x,u) → xdot
        x_sub = SX.sym('x',7);
        u_sub = SX.sym('u',2);

        Xp    = x_sub(1);  Yp    = x_sub(2);
        Psi   = x_sub(3);  B     = x_sub(4);
        V     = x_sub(5);  R     = x_sub(6);
        s_par = x_sub(7);

        delta = u_sub(1);  Fdr   = u_sub(2);

        vx    = V*cos(B);  vy    = V*sin(B);
        vx_fl = vx - R*(wf/2);  vy_fl = vy + R*lf;
        vx_fr = vx + R*(wf/2);  vy_fr = vy + R*lf;
        vx_rl = vx - R*(wr_/2); vy_rl = vy - R*lr;
        vx_rr = vx + R*(wr_/2); vy_rr = vy - R*lr;

        alpha_fl = delta - atan2(vy_fl, vx_fl+eps);
        alpha_fr = delta - atan2(vy_fr, vx_fr+eps);
        alpha_rl =       - atan2(vy_rl, vx_rl+eps);
        alpha_rr =       - atan2(vy_rr, vx_rr+eps);

        Sfl = sqrt(max(2-2*cos(alpha_fl),0)+eps);
        Sfr = sqrt(max(2-2*cos(alpha_fr),0)+eps);
        Srl = sqrt(max(2-2*cos(alpha_rl),0)+eps);
        Srr = sqrt(max(2-2*cos(alpha_rr),0)+eps);

        mu_fl = C1*(1-exp(-C2*Sfl)) - C3*Sfl;
        mu_fr = C1*(1-exp(-C2*Sfr)) - C3*Sfr;
        mu_rl = C1*(1-exp(-C2*Srl)) - C3*Srl;
        mu_rr = C1*(1-exp(-C2*Srr)) - C3*Srr;

        Fyf_fl = (sin(alpha_fl)/(Sfl+eps))*mu_fl*Fz_f;
        Fyf_fr = (sin(alpha_fr)/(Sfr+eps))*mu_fr*Fz_f;
        Fyf_rl = (sin(alpha_rl)/(Srl+eps))*mu_rl*Fz_r;
        Fyf_rr = (sin(alpha_rr)/(Srr+eps))*mu_rr*Fz_r;

        Fxf_fl = Fdr;  Fxf_fr = Fdr;
        Fxf_rl = Fdr;  Fxf_rr = Fdr;

        % 前轮坐标系旋转到车辆坐标系
        Fxf_fl_c = Fxf_fl*cos(delta) - Fyf_fl*sin(delta);
        Fyf_fl_c = Fxf_fl*sin(delta) + Fyf_fl*cos(delta);
        Fxf_fr_c = Fxf_fr*cos(delta) - Fyf_fr*sin(delta);
        Fyf_fr_c = Fxf_fr*sin(delta) + Fyf_fr*cos(delta);
        Fxf_rl_c = Fxf_rl;  Fyf_rl_c = Fyf_rl;
        Fxf_rr_c = Fxf_rr;  Fyf_rr_c = Fyf_rr;

        Fxf_tot = Fxf_fl_c + Fxf_fr_c + Fxf_rl_c + Fxf_rr_c;
        Fyf_tot = Fyf_fl_c + Fyf_fr_c + Fyf_rl_c + Fyf_rr_c;

        dXpos = V*cos(Psi+B);
        dYpos = V*sin(Psi+B);
        dPsi  = R;
        dB    = -R + 1/(m*(V+eps)) * ( (Fyf_fl_c+Fyf_fr_c)*cos(delta-B) + ...
                                       (Fyf_rl_c+Fyf_rr_c)*cos(-B) - Fdr*sin(B) );
        dR    = 1/Jz * ( ...
                   Fyf_fl_c*(lf*cos(delta) - (wf/2)*sin(delta)) + ...
                   Fyf_fr_c*(lf*cos(delta) + (wf/2)*sin(delta)) + ...
                   Fyf_rl_c*(-lr) + Fyf_rr_c*(-lr) + ...
                   Fxf_fl_c*sin(delta)*lf + Fxf_fr_c*sin(delta)*lf );
        dV    = 1/m * ( (Fyf_fl_c+Fyf_fr_c)*sin(B-delta) + (Fyf_rl_c+Fyf_rr_c)*sin(B) + Fdr*cos(B) );

        % ds/dt = V * cosr / Lp，其中 cosr = (cos(Psi+B)*dXu + sin(Psi+B)*dYu)/Lp
        dXu  = dXdu_f(s_par);   dYu  = dYdu_f(s_par);
        Lp   = sqrt(dXu.^2 + dYu.^2) + eps;
        cosr = cos(Psi+B)*(dXu/Lp) + sin(Psi+B)*(dYu/Lp);
        ds   = V * cosr / Lp;

        xdot_sub = [dXpos; dYpos; dPsi; dB; dV; dR; ds];
        f_dyn_sub = Function('f_dyn_sub',{x_sub,u_sub},{xdot_sub});

        % —— 3.8. 添加“Euler 离散”动力学约束
        for i = 1:N_sub
            xki  = Xk(:,i);
            uki  = Uk(:,i);
            xki1 = xki + dt_sub * f_dyn_sub(xki, uki);
            opti_sub.subject_to( Xk(:,i+1) == xki1 );
        end

        % —— 3.9. 本段 Hard‐Corridor 约束
        margin = 4.0;
        for i = 1:(N_sub+1)
            xr_i  = X_ref(  s_k(i) );
            yr_i  = Y_ref(  s_k(i) );
            dXu_i = dXdu_f(s_k(i));
            dYu_i = dYdu_f(s_k(i));
            Lp_i  = sqrt(dXu_i^2 + dYu_i^2) + eps;
            sinp_i= dYu_i / Lp_i;
            cosp_i= dXu_i / Lp_i;
            ek_i  = -( Xp_k(i) - xr_i )*sinp_i + ( Yp_k(i) - yr_i )*cosp_i;
            opti_sub.subject_to( ek_i <=  margin );
            opti_sub.subject_to( ek_i >= -margin );
        end

        % —— 3.10. 为本段状态和控制做“每个格点单独”的初始猜测
        % 构造 X_guess (7×(N_sub+1))：让每个网格点 i 都落到跑道中心线上
        X_guess = zeros(7, N_sub+1);
        for i = 1:(N_sub+1)
            % 把 s_guess(i) 均匀分布在 [s_start, s_end]
            s_guess_i = s_start + (s_end - s_start)*(i-1)/N_sub;
            x_ref_i   = full( X_ref(   s_guess_i ) );
            y_ref_i   = full( Y_ref(   s_guess_i ) );
            psi_ref_i = full( phi_ref( s_guess_i ) );
            X_guess(:,i) = [ x_ref_i;    % X
                             y_ref_i;    % Y
                             psi_ref_i;  % psi
                             0;          % beta
                             5;          % v
                             0;          % r
                             s_guess_i]; % s
        end
        opti_sub.set_initial( Xk, X_guess );

        % 构造 Uk_guess (2×N_sub)：先猜 delta=0, Fdr=0
        Uk_guess = zeros(2, N_sub);
        opti_sub.set_initial( Uk, Uk_guess );

        % 给 Tseg 一个粗略猜测：假设平均速度 5 m/s
        T_guess = (s_end - s_start)/5;
        opti_sub.set_initial( Tseg, T_guess );

        % —— 3.11. 本段目标函数 & 求解 IPOPT
        w1 = 1.0;  w2 = 1.5;
        Jsub = w1*Tseg + w2*dt_sub*( sum(delta_k.^2) + sum(Fdr_k.^2) );
        opti_sub.minimize(Jsub);

        opts = struct('print_time',0, 'ipopt',struct('tol',1e-2, 'max_iter',1e5));
        opti_sub.solver('ipopt', opts);

        try
            sol_sub = opti_sub.solve();
            ret_stat = sol_sub.stats.return_status;
        catch
            sol_sub = opti_sub.debug;
            ret_stat = 'DEBUG_ONLY';
        end

        Xk_opt = sol_sub.value(Xk);
        Uk_opt = sol_sub.value(Uk);
        Tk_opt = sol_sub.value(Tseg);
    end
    % —— 嵌套函数结束 —— 
end
