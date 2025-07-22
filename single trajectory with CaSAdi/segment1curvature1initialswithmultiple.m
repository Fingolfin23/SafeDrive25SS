% ====================================================
% 多段分段 + 按曲率加密 + 调整初始猜测

function segment1curvature1initialswithmultiple()
import casadi.*

%% === 0. 读取赛道数据 & 构造插值函数 ===
data = readmatrix('Monza2.csv');
x_data = data(:,1);
y_data = data(:,2);
wr_data = data(:,3);
wl_data = data(:,4); 
x_data_orig     = x_data;
y_data_orig     = y_data;

% —— 原始弧长（累计欧氏距离）
s_raw  = [0; cumsum( hypot(diff(x_data), diff(y_data)) )];

% —— 原始曲率计算（用切线角变化率）
dx_ds = gradient(x_data, s_raw);
dy_ds = gradient(y_data, s_raw);
theta = atan2(dy_ds, dx_ds);
dtheta_ds = gradient(theta, s_raw);
kappa_raw = abs(dtheta_ds);   % 绝对曲率

% —— 密度权重（按曲率归一化）
a = 1.8;    % 曲率加密系数，可自行调整
kappa_norm = kappa_raw / max(kappa_raw + 1e-6);  % 防止除零
density_weight = 1 + a * kappa_norm;

% —— 每段“加权后”长度
ds_orig = diff(s_raw);   % 原始每段长度 (M-1)
ds_weighted = ds_orig .* density_weight(1:end-1);
s_dense = [0; cumsum(ds_weighted)];

% —— 按目标步长重新生成 s_uniform
% ds_target =1.0;    % 目标步长（可调节）
% s_uniform = (0:ds_target : s_dense(end))';
% if s_uniform(end) < s_dense(end)
%     s_uniform = [s_uniform; s_dense(end)];
% end

%% === 1. 分段参数 & 总段数设定 ===
numSegments = 6;     % 把跑道分成段
N_sub       = 90;    % 每段内部离散步数（自行调整）;

s_uniform = linspace(0, s_dense(end), N_sub*numSegments); 
% —— inverse，得到 s_uniform 对应的原始 s_raw 坐标
s_interp = interp1(s_dense, s_raw, s_uniform, 'pchip');

% —— 插值采样原始轨迹数据
x_data_new = interp1(s_raw, x_data, s_interp, 'pchip');
y_data_new = interp1(s_raw, y_data, s_interp, 'pchip');
wr_new     = interp1(s_raw, wr_data, s_interp, 'linear');
wl_new     = interp1(s_raw, wl_data, s_interp, 'linear');

% —— 新的索引序列
s_data = (1:numel(x_data_new))';
M          = numel(s_data);   % 记得更新 M，和之前一致
x_data     = x_data_new;
y_data     = y_data_new;
wr_data    = wr_new;
wl_data    = wl_new;

% 把赛道中心线和导数都用 CasADi 插值封装

X_ref   = interpolant('X_ref',   'linear', {s_data}, x_data');
Y_ref   = interpolant('Y_ref',   'linear', {s_data}, y_data');

% % % 先算出离散导数，并做插值
dX_data = gradient(x_data,s_data);
dY_data = gradient(y_data,s_data) ;
phi_data = atan2(dY_data, dX_data);

dXdu_f  = interpolant('dXdu_ref','linear', {s_data}, dX_data');
dYdu_f  = interpolant('dYdu_ref','linear', {s_data}, dY_data');
phi_ref = interpolant('phi_ref', 'linear', {s_data}, phi_data');
wr_ref  = interpolant('wr_ref',  'linear', {s_data}, wr_data');
wl_ref  = interpolant('wl_ref',  'linear', {s_data}, wl_data');

%% === 1. 分段参数 & 总段数设定 ===
% numSegments = 6;     % 把跑道分成段
% N_sub       = 70;    % 每段内部离散步数（自行调整）;
% 均匀把 [1, M] 分成 num段，得到每段的起点 s_start、终点 s_end（整数索引）
segBounds = round(linspace(1, M, numSegments+1));
% 比如 segBounds = [1, 450, 900, 1350, M]


% 假设分成 4 段
segTime    = zeros(1, numSegments);      % 存每段用时
segSstart  = zeros(1, numSegments);      % 存每段起始 s
segSend    = zeros(1, numSegments);      % 存每段结束 s
segXstart  = cell(1, numSegments);       % 存每段的起点状态（7×1）
segXend    = cell(1, numSegments);       % 存每段的终点状态（7×1）S
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
    %N_sub = N_sub_array(k);
    s_start = segBounds(k);
    s_end   = segBounds(k+1);

    if k == 1
        X_start_val = [];  % 第一段没有上一段末状态，留空
    else
        X_start_val = X_end_prev;  % 直接把上一段末状态当作本段初始
    end

    % 调用嵌套子函数：解第 k 段
     [Xk_opt, Uk_opt, Tk_opt, ret_stat, X_guess] = solveSegment(k, X_start_val, s_start, s_end, N_sub);
     segXguess_all{k} = X_guess;

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
    
    
    fprintf('== segment %d solved：T_segment = %.3f s， s range [%d → %d]\n', ...
             k, full(Tk_opt), s_start, s_end);
end


fprintf('==== Total accumulated time across all segments T_all = %.3f s ====\n', full(T_all) );
% 【新增】—— 在主循环结束后，统一输出每段的 s_start → s_end 及 return_status
fprintf('\n================ Summary of Exit Status per Segment ================\n');
fprintf(' Segment |  s_start → s_end  |   return_status  \n');
fprintf('------+-------------------+--------------------\n');
for k = 1:numSegments
    fprintf('  %2d   |   %4d  →  %4d   |   %s\n', ...
        k, segBounds(k), segBounds(k+1), exitStatus{k});
end

fprintf('\n====================== Segment Summary ======================\n');
fprintf(' Segment |  s_start → s_end  |  Time [s]  |  Start Speed [km/h]  |  End Speed [km/h]\n');
fprintf(' ----+-------------------+----------------+-------------------+---------------------\n');
for k = 1:numSegments
    v_start_k = segXstart{k}(5)*3.6;   % 把 m/s 换算成 km/h
    v_end_k   = segXend{k}(5)*3.6;
    fprintf(' %2d   |    %4d → %4d    |    %6.3f     |      %6.2f      |      %6.2f\n', ...
            k, segSstart(k), segSend(k), segTime(k), v_start_k, v_end_k );
end
fprintf(' ==================== summary end====================\n');

%% 【1】确定有解的段
valid_segments = find( ...
    cellfun(@(s) ...
        strcmp(s, 'Solve_Succeeded') || ...
        strcmp(s, 'Converged') || ...
        strcmp(s, 'Solved_To_Acceptable_Level'), ...
        exitStatus) );
num_valid = numel(valid_segments);

    %% 【2】分段画图：带赛道边界
    figure('Name','Trajectory & Velocity with Corridor','NumberTitle','off');
    margin = 4;  % 和你建模时一致
    
    for idx = 1:num_valid
        k = valid_segments(idx);
        Xk_here = segXk_all{k};   % 7×(N_sub+1)
        Uk_here = segUk_all{k};
    
        % (1) 轨迹及边界
        subplot(2, num_valid, idx);
        hold on; grid on; axis equal;
    
        % 当前段索引范围
        s_range = segSstart(k):segSend(k);
    
        % 中心线
        X_center = full(X_ref(s_range));
        Y_center = full(Y_ref(s_range));
    
        % 边界法向矢量
        dXu_seg = full(dXdu_f(s_range));
        dYu_seg = full(dYdu_f(s_range));
        Lp_seg  = sqrt(dXu_seg.^2 + dYu_seg.^2) + 1e-6;
        nX = -dYu_seg ./ Lp_seg;
        nY =  dXu_seg ./ Lp_seg;
        % 左右边界
        XcorrL = X_center + nX*margin;
        YcorrL = Y_center + nY*margin;
        XcorrR = X_center - nX*margin;
        YcorrR = Y_center - nY*margin;
    
        % 画轨迹
        plot( Xk_here(1,:), Xk_here(2,:), 'b-', 'LineWidth', 1.5 );
        % 画中心线
        plot( X_center, Y_center, 'k--', 'LineWidth', 1.2 );
        % 画边界
        plot( XcorrL, YcorrL, 'k:', 'LineWidth', 1 );
        plot( XcorrR, YcorrR, 'k:', 'LineWidth', 1 );
    
        xlabel('X [m]'); ylabel('Y [m]');
        title(sprintf('Segment %d trajectory', k));
        legend('Optimal traj','Centerline','Corridor left','Corridor right');
    end
    
    % (2) 速度图同上
    for idx = 1:num_valid
        k = valid_segments(idx);
        Xk_here = segXk_all{k};
        subplot(2, num_valid, num_valid + idx);
        t_local = linspace(0, full(segTime(k)), size(Xk_here,2));
        plot( t_local, Xk_here(5,:) * 3.6, 'b-', 'LineWidth', 1.2 );
        grid on;
        xlabel('Time [s]'); ylabel('Speed [km/h]');
        title(sprintf('Segment %d velocity', k));
    end
    
    sgtitle('Trajectory & Velocity (with Corridor) for Each Feasible Segment');

% === 可视化轨迹：原始轨迹 + 优化实际使用的点 ===
figure('Name','Optimized Points in different segments', 'NumberTitle','off'); hold on; axis equal;
title('Trajectory with Actual Optimization Points');
xlabel('x [m]'); ylabel('y [m]');

% 画原始轨迹
plot(x_data, y_data, 'k--', 'LineWidth', 1.2);  % 原始轨迹

% 每段的实际采样点
colorList = {
    [0.8, 0.9, 1.0];  % very light blue
    [0.6, 0.8, 1.0];  % light blue
    [0.4, 0.7, 1.0];  % medium-light blue
    [0.2, 0.6, 1.0];  % medium blue
    [0.1, 0.4, 0.9];  % darker blue
    [0.0, 0.2, 0.8]   % dark blue
};

for j = 1:numSegments
    Xk = segXk_all{j};  % 该段的状态序列 (7×N_sub)
    plot(Xk(1,:), Xk(2,:), 'o', ...
        'Color', colorList{mod(j-1,length(colorList))+1}, ...
        'MarkerSize', 6, 'DisplayName', ['Segment ', num2str(j)]);
end

% 标记起点终点
plot(segXk_all{1}(1,1), segXk_all{1}(2,1), 'bo', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Start');
plot(segXk_all{end}(1,end), segXk_all{end}(2,end), 'bx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'End');

legend show;

figure('Name','Per-Segment: Init Guess vs Optimized','NumberTitle','off');
for idx = 1:num_valid
    k = valid_segments(idx);
    subplot(1, num_valid, idx); hold on; axis equal; grid on;

    X_guess_k = segXguess_all{k};  % 初始猜测轨迹（7×(N_sub+1)）
    X_opt_k   = segXk_all{k};      % 优化结果轨迹

    plot(X_guess_k(1,:), X_guess_k(2,:), 'kx', 'DisplayName','Initial Guess');
    plot(X_opt_k(1,:),   X_opt_k(2,:),   'bo', 'DisplayName','Optimized');

    title(sprintf('Segment %d', k));
    xlabel('x [m]'); ylabel('y [m]');
    legend;
end
sgtitle('Initial Guess vs Optimized Trajectory (Feasible Segments)');

figure('Name','Full Path: Init Guess vs Optimized','NumberTitle','off');
hold on; axis equal; grid on;

X_guess_all = [];

for j = 1:numSegments
    X_guess_j = segXguess_all{j};  % 获取初始猜测
    if j == 1
        X_guess_all = X_guess_j;
    else
        X_guess_all = [X_guess_all, X_guess_j(:,2:end)];  % 避免重复点
    end
end

% 绘制图像
plot(X_guess_all(1,:), X_guess_all(2,:), 'kx', 'DisplayName','Initial Guess');
plot(X_all(1,:), X_all(2,:), 'bo', 'DisplayName','Optimized');
plot(x_data, y_data, 'k--', 'DisplayName','Centerline');

xlabel('x [m]'); ylabel('y [m]');
title('Initial Guess vs Optimized Trajectory (Full Path)');
legend;


figure('Name','Comparison of Equidistant and Non-equidistant Sampling', 'NumberTitle','off');
% 对比点
h1 = plot(x_data_orig, y_data_orig, 'k--'); hold on;  % 原始轨迹线
h2 = scatter(x_data_orig, y_data_orig, 10, 'k', 'filled');  % 原始采样点
h3 = scatter(x_data_new, y_data_new, 30, 'b');     % 非等距采样点
axis equal;
title('Comparison of Equidistant and Non-equidistant Points');
legend([h1, h2, h3], {'Reference Trajectory', 'Equidistant Points', 'Non-equidistant Points'});

% === 构造密度函数（按曲率归一化）
a1 = 100.0;   % 密度调节系数（可调，越大弯道点越密）
density1 = 1 + a1 * (kappa_raw / max(kappa_raw + 1e-8));

% === 构造“密度弧长”（积分密度得到s曲线）
ds_orig1 = diff(s_raw);
s_cum1 = [0; cumsum(density1(1:end-1) .* ds_orig1)];

% === 在密度弧长上均匀采样，然后反推回原始弧长
s_cum_uniform1 = linspace(0, s_cum1(end), N_sub*numSegments);
s_interp1 = interp1(s_cum1, s_raw, s_cum_uniform1, 'pchip');

% === 用反推出的 s_interp 插值原始轨迹
x_data_new1 = interp1(s_raw, x_data_orig, s_interp1, 'pchip');
y_data_new1 = interp1(s_raw, y_data_orig, s_interp1, 'pchip');
% 如果之前注释了这行，要取消注释或替换
% figure('Name','Comparison of Equidistant and Non-equidistant Sampling:only for visualization', 'NumberTitle','off');
% h1 = plot(x_data, y_data, 'k--'); hold on;  % 原始轨迹线
% h2 = scatter(x_data_orig, y_data_orig, 5, 'k', 'filled');  % 原始采样点
% h3 = scatter(x_data_new1, y_data_new1, 10, 'b');     % 非等距采样点
% 
% axis equal;
% title('Comparison of Equidistant and Non-equidistant Sampling');
% legend([h1, h2, h3], {'Reference Trajectory', 'Equidistant Points', 'Non-equidistant Points'});

% === 主图（完整轨迹对比） ===
figure('Name','Comparison of Equidistant and Non-equidistant Sampling:only for visualization', 'NumberTitle','off');
ax_main = axes('Position', [0.35, 0.2, 0.3, 0.6]);  % 中间主图区域
h1 = plot(x_data, y_data, 'k--'); hold on;
h2 = scatter(x_data_orig, y_data_orig, 5, 'k', 'filled');
h3 = scatter(x_data_new1, y_data_new1, 10, 'b');
axis equal;
title('Comparison of Equidistant and Non-equidistant Sampling');
legend([h1, h2, h3], {'Reference Trajectory', 'Equidistant Points', 'Non-equidistant Points'}, 'Location','southoutside', 'Orientation','horizontal');
xlabel('X [m]'); ylabel('Y [m]');

% === 放大图1 ===
ax_zoom1 = axes('Position', [0.05, 0.2, 0.3, 0.6]);  % 左
h1z1 = plot(x_data, y_data, 'k--'); hold on;
h2z1 = scatter(x_data_orig, y_data_orig, 5, 'k', 'filled');
h3z1 = scatter(x_data_new1, y_data_new1, 10, 'b');
axis equal;
axis([80, 180, 800, 1100]);  % 这里填写第一个放大区域的坐标
title('Zoomed View 1');
set(gca,'XTick',[],'YTick',[]);

% === 放大图2 ===
ax_zoom2 = axes('Position', [0.70, 0.2, 0.3, 0.6]);  % 右
h1z2 = plot(x_data, y_data, 'k--'); hold on;
h2z2 = scatter(x_data_orig, y_data_orig, 5, 'k', 'filled');
h3z2 = scatter(x_data_new1, y_data_new1, 10, 'b');
axis equal;
axis([300, 500, 420, 780]);  % 这里填写第二个放大区域的坐标
title('Zoomed View 2');
set(gca,'XTick',[],'YTick',[]);


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
%margin = 4;


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
title('Whole Speed Profile');
sgtitle('Whole Path Visualization');


figure('Name','Overview with Local Zooms','NumberTitle','off');


% === 主图（中间） ===
ax_main = axes('Position',[0.25, 0.25, 0.5, 0.5]);  % 可调整主图大小位置
hold on; grid on; axis equal;

plot(XcorrL, YcorrL, 'k--');
plot(XcorrR, YcorrR, 'k--');
plot(XbP, YbP, 'k-', 'LineWidth',1.5);
plot(X_all(1,:), X_all(2,:), 'b-', 'LineWidth',1.8);

xlabel('X [m]'); ylabel('Y [m]');
title('Full Trajectory with Corridor');
legend('Corridor Left','Corridor Right','Centerline','Optimized','Location','best');

% === 子图（六段轨迹 + 速度图） ===
num_valid = numel(valid_segments);
%margin = 4;  % 保持与你的建模一致

for idx = 1:num_valid
    k = valid_segments(idx);
    Xk_here = segXk_all{k};
    s_range = segSstart(k):segSend(k);
    
    % 提取参考路径和法向
    X_center = full(X_ref(s_range));
    Y_center = full(Y_ref(s_range));
    dXu_seg  = full(dXdu_f(s_range));
    dYu_seg  = full(dYdu_f(s_range));
    Lp_seg   = sqrt(dXu_seg.^2 + dYu_seg.^2) + 1e-6;
    nX = -dYu_seg ./ Lp_seg;
    nY =  dXu_seg ./ Lp_seg;
    XcorrL = X_center + margin * nX;
    YcorrL = Y_center + margin * nY;
    XcorrR = X_center - margin * nX;
    YcorrR = Y_center - margin * nY;

    % === 轨迹子图（上方） ===
    ax_traj = axes('Position',[0.05 + 0.15*(idx-1), 0.80, 0.12, 0.12]);
    hold on; axis equal; grid on;
    plot(XcorrL, YcorrL, 'k:');
    plot(XcorrR, YcorrR, 'k:');
    plot(X_center, Y_center, 'k--');
    plot(Xk_here(1,:), Xk_here(2,:), 'b-', 'LineWidth',1.2);
    title(sprintf('Segment %d', k));
    set(gca, 'XTick',[], 'YTick',[]);
    
    % === 速度子图（下方） ===
    ax_vel = axes('Position',[0.05 + 0.15*(idx-1), 0.05, 0.12, 0.12]);
    t_local = linspace(0, full(segTime(k)), size(Xk_here,2));
    plot(t_local, Xk_here(5,:)*3.6, 'b-', 'LineWidth',1.2);
    xlabel('t [s]'); ylabel('km/h');
    title(sprintf('v_{%d}(t)', k));
end

figure('Name','Final Layout for PPT','Color','w');
%% === 1. 主图区域（中间） ===
% 全局轨迹图（上）
ax_traj = axes('Position',[0.4, 0.60, 0.20, 0.15]);
hold on; axis equal; grid on;
plot(XcorrL, YcorrL, 'k--');
plot(XcorrR, YcorrR, 'k--');
plot(XbP, YbP, 'k-', 'LineWidth',1.2);
plot(X_all(1,:), X_all(2,:), 'b-', 'LineWidth',1.5);
xlabel('X [m]'); ylabel('Y [m]');
title('Full Trajectory with Corridor');
% legend('Corridor Left','Corridor Right','Centerline','Optimized','Location','southoutside','Orientation','horizontal');

% 全局速度图（下）
ax_speed = axes('Position',[0.4, 0.3, 0.20, 0.15]);
tgrid = linspace(0, full(T_all), size(X_all,2));
plot(tgrid, X_all(5,:)*3.6, 'b-', 'LineWidth',1.5);
xlabel('Time [s]'); ylabel('Speed [km/h]');
title('Overall Speed Profile');
grid on;

%% === 2. 子图：左 3 段 ===
for i = 1:3
    k = valid_segments(i);
    Xk_here = segXk_all{k};
    s_range = segSstart(k):segSend(k);
    Xc = full(X_ref(s_range)); Yc = full(Y_ref(s_range));
    dXu = full(dXdu_f(s_range)); dYu = full(dYdu_f(s_range));
    Lp = sqrt(dXu.^2 + dYu.^2) + 1e-6;
    nX = -dYu ./ Lp; nY = dXu ./ Lp;
    XL = Xc + margin * nX; YL = Yc + margin * nY;
    XR = Xc - margin * nX; YR = Yc - margin * nY;

    posY = 0.65 - (i-1)*0.28;
    axL = axes('Position',[0.05, posY, 0.28, 0.25]);
    hold on; axis equal; grid on;
    plot(XL, YL, 'k:'); plot(XR, YR, 'k:');
    plot(Xc, Yc, 'k--'); plot(Xk_here(1,:), Xk_here(2,:), 'b-', 'LineWidth',1.2);
    title(sprintf('Segment %d', k));
    set(gca,'XTick',[],'YTick',[]);
end

%% === 3. 子图：右 3 段 ===
for i = 4:6
    k = valid_segments(i);
    Xk_here = segXk_all{k};
    s_range = segSstart(k):segSend(k);
    Xc = full(X_ref(s_range)); Yc = full(Y_ref(s_range));
    dXu = full(dXdu_f(s_range)); dYu = full(dYdu_f(s_range));
    Lp = sqrt(dXu.^2 + dYu.^2) + 1e-6;
    nX = -dYu ./ Lp; nY = dXu ./ Lp;
    XL = Xc + margin * nX; YL = Yc + margin * nY;
    XR = Xc - margin * nX; YR = Yc - margin * nY;

    posY = 0.65 - (i-4)*0.28;
    axR = axes('Position',[0.7, posY, 0.28, 0.25]);
    hold on; axis equal; grid on;
    plot(XL, YL, 'k:'); plot(XR, YR, 'k:');
    plot(Xc, Yc, 'k--'); plot(Xk_here(1,:), Xk_here(2,:), 'b-', 'LineWidth',1.2);
    title(sprintf('Segment %d', k));
    set(gca,'XTick',[],'YTick',[]);
end


%% === 自动记录日志 ===
% 设置日志路径（存在则追加，不存在则创建）
logFilePath = 'trajectory_log.txt';
fid = fopen(logFilePath, 'a');

% 当前时间戳
t_now = datetime('now','Format','yyyy-MM-dd HH:mm:ss');

% 写入日志内容
fprintf(fid, '=== Run Time: %s ===\n', string(t_now));
fprintf(fid, 'margin = %.2f,  numSegments = %d,  N_sub = %d,  alpha = %.2f\n', ...
            margin, numSegments, N_sub, a);
fprintf(fid, 'Valid Segments Solved: %d / %d\n', num_valid, numSegments);
fprintf(fid, 'Total Time (s): %.3f\n', full(T_all));
fprintf(fid, 'Segment Status:\n');
for k = 1:numSegments
    fprintf(fid, '  Segment %2d: %s\n', k, exitStatus{k});
end
fprintf(fid, '============================\n\n');
fclose(fid);

%% ===== 嵌套函数：解第 k 段的子问题 =====
function [Xk_opt, Uk_opt, Tk_opt, ret_stat,X_guess_out] = solveSegment(k, X_start_value, s_start, s_end, N_sub)
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
        % opti_sub.subject_to( beta_k(1)== 0 );
        opti_sub.subject_to( v_k(1)   >= 0 );
        % opti_sub.subject_to( r_k(1)   == 0 );
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
    delta_max = pi/2;  beta_max  = pi/2;  psi_max = pi;
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
    g   = 9.81;  mu = 0.9;
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
margin = 4;
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
        s_guess_i = s_start + (s_end - s_start)*(i-1)/N_sub;
        x_ref_i   = full( X_ref(   s_guess_i ) );
        y_ref_i   = full( Y_ref(   s_guess_i ) );
        psi_ref_i = full( phi_ref( s_guess_i ) );

    
        % 计算曲率
        dXu_i = full( dXdu_f(s_guess_i) );
        dYu_i = full( dYdu_f(s_guess_i) );
        ddXu_i= (full( dXdu_f( min(s_guess_i+1,M) ) ) - full( dXdu_f( max(s_guess_i-1,1) ) )) / 2;
        ddYu_i= (full( dYdu_f( min(s_guess_i+1,M) ) ) - full( dYdu_f( max(s_guess_i-1,1) ) )) / 2;
        kappa_i = abs( dXu_i*ddYu_i - dYu_i*ddXu_i ) / ( (dXu_i^2 + dYu_i^2)^(3/2) + 1e-6 );
        if kappa_i < 1e-3
            R_i = 1e4;
        else
            R_i = 1 / kappa_i;
        end

        % 估算该点最大速度 & 合理初始速度
        v_max_i = sqrt(mu * g * R_i);
        v_guess_i = min( v_max_i, 60/3.6 );
    
        % 估算 beta, r
        beta_guess_i = 0;
        if R_i < 1e4
            r_guess_i = v_guess_i / R_i;
        else
            r_guess_i = 0;
        end
    
    
        X_guess(:,i) = [ x_ref_i;    % X
                         y_ref_i;    % Y
                         psi_ref_i;  % psi
                         beta_guess_i;          % beta
                         v_guess_i;          % v
                         r_guess_i;          % r
                         s_guess_i]; % s
    end
    opti_sub.set_initial( Xk, X_guess );
    segXguess_all{k} = X_guess;

    % 构造 Uk_guess (2×N_sub)：先猜 delta=0, Fdr=0
    Uk_guess = zeros(2, N_sub);
    opti_sub.set_initial( Uk, Uk_guess );


    % 给 Tseg 一个粗略猜测：假设平均速度 5 m/s
    T_guess = (s_end - s_start)/5;
    opti_sub.set_initial( Tseg, T_guess );

    % —— 3.11. 本段目标函数 & 求解 IPOPT
    w1 = 0.5;  w2 = 0.5;
    Jsub = w1*Tseg + w2*dt_sub*( sum(delta_k.^2) + sum(Fdr_k.^2) );
    opti_sub.minimize(Jsub);

    opts = struct('print_time',0, 'ipopt',struct('tol',1e-1, 'max_iter',5e4,'acceptable_obj_change_tol', 1e1,'acceptable_tol', 1e-1));
    opti_sub.solver('ipopt', opts);
    X_guess_out = X_guess;  % 把初始猜测也作为输出
    
    try
        sol_sub = opti_sub.solve();
        ret_stat = sol_sub.stats.return_status;
    catch
        sol_sub = opti_sub.debug;
        ret_stat = sol_sub.stats.return_status;
    end

    Xk_opt = sol_sub.value(Xk);
    Uk_opt = sol_sub.value(Uk);
    Tk_opt = sol_sub.value(Tseg);
end
    % —— 嵌套函数结束 —— 
end