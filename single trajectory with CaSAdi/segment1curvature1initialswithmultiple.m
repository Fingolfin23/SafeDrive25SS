function segment1curvature1initialswithmultiple()
import casadi.*

%% === 0. Read Track Data & Construct Interpolation Functions ===
data = readmatrix('Monza2.csv');
x_data = data(:,1);
y_data = data(:,2);
wr_data = data(:,3);
wl_data = data(:,4); 
x_data_orig     = x_data;
y_data_orig     = y_data;


% -- Original arc length (cumulative Euclidean distance)
s_raw  = [0; cumsum( hypot(diff(x_data), diff(y_data)) )];


% -- Compute original curvature (using rate of change of tangent angle)
dx_ds = gradient(x_data, s_raw);
dy_ds = gradient(y_data, s_raw);
theta = atan2(dy_ds, dx_ds);
dtheta_ds = gradient(theta, s_raw);
kappa_raw = abs(dtheta_ds);   % absolute curvature


% -- Density weight (normalized by curvature)
a = 1.8;    % Curvature refinement factor (adjustable; higher -> denser points in curves)
kappa_norm = kappa_raw / max(kappa_raw + 1e-6);  % prevent division by zero
density_weight = 1 + a * kappa_norm;


% -- Weighted length of each segment
ds_orig = diff(s_raw);   % Original length of each segment (M-1)
ds_weighted = ds_orig .* density_weight(1:end-1);
s_dense = [0; cumsum(ds_weighted)];


%% === 1. Segment Parameters & Total Number of Segments ===
numSegments = 6;     % Divide the track into this many segments
N_sub       = 90;    %  Discretization steps within each segment (adjust as needed)


s_uniform = linspace(0, s_dense(end), N_sub*numSegments); 
% -- Inverse mapping: get original s_raw corresponding to s_uniform
s_interp = interp1(s_dense, s_raw, s_uniform, 'pchip');

% -- Interpolate original trajectory data at new sample points
x_data_new = interp1(s_raw, x_data, s_interp, 'pchip');
y_data_new = interp1(s_raw, y_data, s_interp, 'pchip');
wr_new     = interp1(s_raw, wr_data, s_interp, 'linear');
wl_new     = interp1(s_raw, wl_data, s_interp, 'linear');


% -- New index sequence
s_data = (1:numel(x_data_new))';
M          = numel(s_data);   %  Update M to the new total number of points
x_data     = x_data_new;
y_data     = y_data_new;
wr_data    = wr_new;
wl_data    = wl_new;


% Pack track centerline and derivatives into CasADi interpolants
X_ref   = interpolant('X_ref',   'linear', {s_data}, x_data');
Y_ref   = interpolant('Y_ref',   'linear', {s_data}, y_data');

% Pre-compute discrete derivatives and create interpolants
dX_data = gradient(x_data,s_data);
dY_data = gradient(y_data,s_data) ;
phi_data = atan2(dY_data, dX_data);

dXdu_f  = interpolant('dXdu_ref','linear', {s_data}, dX_data');
dYdu_f  = interpolant('dYdu_ref','linear', {s_data}, dY_data');
phi_ref = interpolant('phi_ref', 'linear', {s_data}, phi_data');
wr_ref  = interpolant('wr_ref',  'linear', {s_data}, wr_data');
wl_ref  = interpolant('wl_ref',  'linear', {s_data}, wl_data');


%% === 1. Segment Parameters & Total Number of Segments (continued) ===
% Evenly split [1, M] into numSegments segments to get each segment's start and end indices
segBounds = round(linspace(1, M, numSegments+1));



% Initialize storage for each segment
segTime    = zeros(1, numSegments);      % Time taken in each segment
segSstart  = zeros(1, numSegments);      % Starting s index of each segment
segSend    = zeros(1, numSegments);      % Ending s index of each segment%
segXstart  = cell(1, numSegments);       %% Starting state of each segment 
segXend    = cell(1, numSegments);       %  % Ending state of each segment (7×1)
segXk_all  = cell(1, numSegments);       % Full state trajectory for each segment% 
segUk_all  = cell(1, numSegments);       %  Full control sequence for each segment



% Variables for concatenating full path results
X_all = [];    % (7×total_points) All segment states concatenated% 
U_all = [];    % (2×total_steps) All segment controls concatenated% 
T_all = 0;     % Total time to complete the entire track% 

exitStatus  = cell(numSegments,1);
 % Previous segment's final state (7×1), used as next segment's initial state
X_end_prev = [];  % 7×1 向量
% -----------------------------------------------------------------
% -----------------------------------------------------------------
% Loop through each segment and solve using the nested function solveSegment
for k = 1:numSegments
    %N_sub = N_sub_array(k);
    s_start = segBounds(k);
    s_end   = segBounds(k+1);

    if k == 1
        X_start_val = [];  % First segment has no previous final state
    else
        X_start_val = X_end_prev;  % Use previous segment's final state as this segment's initial
    end

    %  Solve segment k
     [Xk_opt, Uk_opt, Tk_opt, ret_stat, X_guess] = solveSegment(k, X_start_val, s_start, s_end, N_sub);
     segXguess_all{k} = X_guess;

     fprintf('===  Segment %d solved：T_segment = %.3f s， return_status = %s ===\n', ...
                 k, full(Tk_opt), ret_stat);
    
    exitStatus{k} = ret_stat;  % Store IPOPT return status for this segment
     % Store key data for this segment
    segTime(k)   = full(Tk_opt);                  % Time taken for this segment
    segSstart(k) = s_start;                       % Segment starting s index
    segSend(k)   = s_end;                         % Segment ending s index
    segXstart{k} = Xk_opt(:,1);                   % Segment starting state (7×1)
    segXend{k}   = Xk_opt(:,end);                 % Segment ending state (7×1)\
    segXk_all{k} = full(Xk_opt);                  % Complete state trajectory for this segment
    segUk_all{k} = full(Uk_opt);                  % Complete control sequence for this segment
    
    
    
    if k == 1
        % For the first segment, take all N_sub+1 points
        X_all = Xk_opt;          % (7 × (N_sub+1))
        U_all = Uk_opt;          % (2 × N_sub)
        T_all = Tk_opt;          %Total time so far
    else
        % For subsequent segments, avoid duplicating the connecting point
        X_all = [ X_all,           Xk_opt(:,2:end) ];  %#ok<AGROW>
        U_all = [ U_all,           Uk_opt          ];  %#ok<AGROW>
        T_all = T_all + Tk_opt;
    end
    
    % Save this segment's final state for use as next segment's initial state
    X_end_prev = Xk_opt(:, end);
    
    
    fprintf('== segment %d solved：T_segment = %.3f s， s range [%d → %d]\n', ...
             k, full(Tk_opt), s_start, s_end);
end


fprintf('==== Total accumulated time across all segments T_all = %.3f s ====\n', full(T_all) );
% After solving all segments, output a summary of each segment's status
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
    v_start_k = segXstart{k}(5)*3.6;   % Convert m/s to km/h
    v_end_k   = segXend{k}(5)*3.6;
    fprintf(' %2d   |    %4d → %4d    |    %6.3f     |      %6.2f      |      %6.2f\n', ...
            k, segSstart(k), segSend(k), segTime(k), v_start_k, v_end_k );
end
fprintf(' ==================== summary end====================\n');

%% [1] Determine Feasible Segments
valid_segments = find( ...
    cellfun(@(s) ...
        strcmp(s, 'Solve_Succeeded') || ...
        strcmp(s, 'Converged') || ...
        strcmp(s, 'Solved_To_Acceptable_Level'), ...
        exitStatus) );
num_valid = numel(valid_segments);

    %% [2] Plot per Segment: with Track Boundaries
    figure('Name','Trajectory & Velocity with Corridor','NumberTitle','off');
    margin = 4;  % margin consistent with modeling (track half-width)
    
    for idx = 1:num_valid
        k = valid_segments(idx);
        Xk_here = segXk_all{k};   % 7×(N_sub+1)
        Uk_here = segUk_all{k};
    
        % Current segment index range
        subplot(2, num_valid, idx);
        hold on; grid on; axis equal;

        s_range = segSstart(k):segSend(k);
    
        % Centerline for this segment
        X_center = full(X_ref(s_range));
        Y_center = full(Y_ref(s_range));
    
        % Boundary normal vectors
        dXu_seg = full(dXdu_f(s_range));
        dYu_seg = full(dYdu_f(s_range));
        Lp_seg  = sqrt(dXu_seg.^2 + dYu_seg.^2) + 1e-6;
        nX = -dYu_seg ./ Lp_seg;
        nY =  dXu_seg ./ Lp_seg;
        % Left and right boundaries
        XcorrL = X_center + nX*margin;
        YcorrL = Y_center + nY*margin;
        XcorrR = X_center - nX*margin;
        YcorrR = Y_center - nY*margin;
    
        % Plot optimal trajectory for this segment
        plot( Xk_here(1,:), Xk_here(2,:), 'b-', 'LineWidth', 1.5 );
        % Plot centerline
        plot( X_center, Y_center, 'k--', 'LineWidth', 1.2 );
        % Plot boundaries
        plot( XcorrL, YcorrL, 'k:', 'LineWidth', 1 );
        plot( XcorrR, YcorrR, 'k:', 'LineWidth', 1 );
    
        xlabel('X [m]'); ylabel('Y [m]');
        title(sprintf('Segment %d trajectory', k));
        legend('Optimal traj','Centerline','Corridor left','Corridor right');
    end
    
    % Plot velocity profiles for each feasible segment
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
% === Visualize Trajectory: Reference vs Optimized sample points ===
figure('Name','Optimized Points in different segments', 'NumberTitle','off'); hold on; axis equal;
title('Trajectory with Actual Optimization Points');
xlabel('x [m]'); ylabel('y [m]');
% Plot reference trajectory (centerline)
plot(x_data, y_data, 'k--', 'LineWidth', 1.2); 
% Actual sample points in each segment
colorList = {
    [0.8, 0.9, 1.0];  % very light blue
    [0.6, 0.8, 1.0];  % light blue
    [0.4, 0.7, 1.0];  % medium-light blue
    [0.2, 0.6, 1.0];  % medium blue
    [0.1, 0.4, 0.9];  % darker blue
    [0.0, 0.2, 0.8]   % dark blue
};

for j = 1:numSegments
    Xk = segXk_all{j};  % State sequence of this segment (7×N_sub)
    plot(Xk(1,:), Xk(2,:), 'o', ...
        'Color', colorList{mod(j-1,length(colorList))+1}, ...
        'MarkerSize', 6, 'DisplayName', ['Segment ', num2str(j)]);
end

% Mark start and end points
plot(segXk_all{1}(1,1), segXk_all{1}(2,1), 'bo', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Start');
plot(segXk_all{end}(1,end), segXk_all{end}(2,end), 'bx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'End');

legend show;

figure('Name','Per-Segment: Init Guess vs Optimized','NumberTitle','off');
for idx = 1:num_valid
    k = valid_segments(idx);
    subplot(1, num_valid, idx); hold on; axis equal; grid on;

    X_guess_k = segXguess_all{k};  % Initial guess trajectory (7×(N_sub+1))
    X_opt_k   = segXk_all{k};      % Optimized trajectory

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
    X_guess_j = segXguess_all{j}; % Get initial guess for segment j
    if j == 1
        X_guess_all = X_guess_j;
    else
        X_guess_all = [X_guess_all, X_guess_j(:,2:end)];  % Avoid duplicating overlapping points
    end
end

% Plot the graph
plot(X_guess_all(1,:), X_guess_all(2,:), 'kx', 'DisplayName','Initial Guess');
plot(X_all(1,:), X_all(2,:), 'bo', 'DisplayName','Optimized');
plot(x_data, y_data, 'k--', 'DisplayName','Centerline');

xlabel('x [m]'); ylabel('y [m]');
title('Initial Guess vs Optimized Trajectory (Full Path)');
legend;


figure('Name','Comparison of Equidistant and Non-equidistant Sampling', 'NumberTitle','off');
% Comparison of sampling points
h1 = plot(x_data_orig, y_data_orig, 'k--'); hold on;  % Original trajectory line (raw data)
h2 = scatter(x_data_orig, y_data_orig, 10, 'k', 'filled');  % Original equidistant sample points
h3 = scatter(x_data_new, y_data_new, 30, 'b');     %% Non-equidistant sample points
axis equal;
axis equal;
title('Comparison of Equidistant and Non-equidistant Points');
legend([h1, h2, h3], {'Reference Trajectory', 'Equidistant Points', 'Non-equidistant Points'});


% === Construct density function (normalized by curvature) 
a1 = 100.0;   % Density adjustment factor (tunable; higher -> denser in curves)
density1 = 1 + a1 * (kappa_raw / max(kappa_raw + 1e-8));

density1 = 1 + a1 * (kappa_raw / max(kappa_raw + 1e-8));

% === Construct "density arc length" (integrate density to get s curve)
ds_orig1 = diff(s_raw);
s_cum1 = [0; cumsum(density1(1:end-1) .* ds_orig1)];

% === Uniformly sample in density arc length domain, then map back to original arc length
s_cum_uniform1 = linspace(0, s_cum1(end), N_sub*numSegments);
s_interp1 = interp1(s_cum1, s_raw, s_cum_uniform1, 'pchip');

% === Use s_interp1 to interpolate original trajectory
x_data_new1 = interp1(s_raw, x_data_orig, s_interp1, 'pchip');
y_data_new1 = interp1(s_raw, y_data_orig, s_interp1, 'pchip');

% === Main Figure (Full trajectory comparison) ===
figure('Name','Comparison of Equidistant and Non-equidistant Sampling:only for visualization', 'NumberTitle','off');
ax_main = axes('Position', [0.35, 0.2, 0.3, 0.6]);  % 中间主图区域
h1 = plot(x_data, y_data, 'k--'); hold on;
h2 = scatter(x_data_orig, y_data_orig, 5, 'k', 'filled');
h3 = scatter(x_data_new1, y_data_new1, 10, 'b');
axis equal;
title('Comparison of Equidistant and Non-equidistant Sampling');
legend([h1, h2, h3], {'Reference Trajectory', 'Equidistant Points', 'Non-equidistant Points'}, 'Location','southoutside', 'Orientation','horizontal');
xlabel('X [m]'); ylabel('Y [m]');


% === Zoomed View 1 (right) ===
ax_zoom1 = axes('Position', [0.05, 0.2, 0.3, 0.6]); % Left zoomed subplot
h1z1 = plot(x_data, y_data, 'k--'); hold on;
h2z1 = scatter(x_data_orig, y_data_orig, 5, 'k', 'filled');
h3z1 = scatter(x_data_new1, y_data_new1, 10, 'b');
axis equal;
axis([80, 180, 800, 1100]); % Coordinates for the first zoomed region
title('Zoomed View 1');
set(gca,'XTick',[],'YTick',[]);

% === Zoomed View 2 (right) ===
ax_zoom2 = axes('Position', [0.70, 0.2, 0.3, 0.6]);   % Right zoomed subplot
h1z2 = plot(x_data, y_data, 'k--'); hold on;
h2z2 = scatter(x_data_orig, y_data_orig, 5, 'k', 'filled');
h3z2 = scatter(x_data_new1, y_data_new1, 10, 'b');
axis equal;
axis([300, 500, 420, 780]);  % Coordinates for the second zoomed region
title('Zoomed View 2');
set(gca,'XTick',[],'YTick',[]);


%% === 3. Final Visualization: Plot combined X_all, U_all with Track ===
% 3.1. Plot a fine centerline and corridor boundaries
u_plot = linspace(1, M, 400);
XbP    = full( X_ref(u_plot) );
YbP    = full( Y_ref(u_plot) );
dXuP   = full( dXdu_f(u_plot) );
dYuP   = full( dYdu_f(u_plot) );
LpP    = sqrt(dXuP.^2 + dYuP.^2)+1e-6;
nX     = -dYuP ./ LpP;
nY     =  dXuP ./ LpP;
% margin is already defined


XcorrL = XbP + nX*margin;
YcorrL = YbP + nY*margin;
XcorrR = XbP - nX*margin;
YcorrR = YbP - nY*margin;

% 3.2. Plot: one subplot for centerline + corridor + optimized path, another for speed profile
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


% === Main Plot (center) ===
ax_main = axes('Position',[0.25, 0.25, 0.5, 0.5]);  % Adjust main plot size/position if needed
hold on; grid on; axis equal;
hold on; grid on; axis equal;

plot(XcorrL, YcorrL, 'k--');
plot(XcorrR, YcorrR, 'k--');
plot(XbP, YbP, 'k-', 'LineWidth',1.5);
plot(X_all(1,:), X_all(2,:), 'b-', 'LineWidth',1.8);

xlabel('X [m]'); ylabel('Y [m]');
title('Full Trajectory with Corridor');
legend('Corridor Left','Corridor Right','Centerline','Optimized','Location','best');

% === Subplots (six segment trajectories + speed plots) ===
num_valid = numel(valid_segments);
% margin remains as defined (4)

for idx = 1:num_valid
    k = valid_segments(idx);
    Xk_here = segXk_all{k};
    s_range = segSstart(k):segSend(k);
    
    % Extract reference path and normal for this segment
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

    % === Trajectory subplot (top) ===
    ax_traj = axes('Position',[0.05 + 0.15*(idx-1), 0.80, 0.12, 0.12]);
    hold on; axis equal; grid on;
    plot(XcorrL, YcorrL, 'k:');
    plot(XcorrR, YcorrR, 'k:');
    plot(X_center, Y_center, 'k--');
    plot(Xk_here(1,:), Xk_here(2,:), 'b-', 'LineWidth',1.2);
    title(sprintf('Segment %d', k));
    set(gca, 'XTick',[], 'YTick',[]);
    
     % === Velocity subplot (bottom) ===
    ax_vel = axes('Position',[0.05 + 0.15*(idx-1), 0.05, 0.12, 0.12]);
    t_local = linspace(0, full(segTime(k)), size(Xk_here,2));
    plot(t_local, Xk_here(5,:)*3.6, 'b-', 'LineWidth',1.2);
    xlabel('t [s]'); ylabel('km/h');
    title(sprintf('v_{%d}(t)', k));
end

figure('Name','Final Layout for PPT','Color','w');
%% === 1. Main Plot Area (Center) ===
% Global trajectory plot (top)
ax_traj = axes('Position',[0.4, 0.60, 0.20, 0.15]);
hold on; axis equal; grid on;
plot(XcorrL, YcorrL, 'k--');
plot(XcorrR, YcorrR, 'k--');
plot(XbP, YbP, 'k-', 'LineWidth',1.2);
plot(X_all(1,:), X_all(2,:), 'b-', 'LineWidth',1.5);
xlabel('X [m]'); ylabel('Y [m]');
title('Full Trajectory with Corridor');
% legend('Corridor Left','Corridor Right','Centerline','Optimized','Location','southoutside','Orientation','horizontal');

% Global speed plot (bottom)
ax_speed = axes('Position',[0.4, 0.3, 0.20, 0.15]);
tgrid = linspace(0, full(T_all), size(X_all,2));
plot(tgrid, X_all(5,:)*3.6, 'b-', 'LineWidth',1.5);
xlabel('Time [s]'); ylabel('Speed [km/h]');
title('Overall Speed Profile');
grid on;

%% === 2. Subplots: Left 3 Segments ===
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
%% === 3. Subplots: Right 3 Segments ===
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


%% === Automatic Logging ===
% Set log file path (append if exists, create if not)
logFilePath = 'trajectory_log.txt';
fid = fopen(logFilePath, 'a');

% Current timestamp
t_now = datetime('now','Format','yyyy-MM-dd HH:mm:ss');

% Write log content
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

%% ===== Nested Function: Solve subproblem for segment k =====
function [Xk_opt, Uk_opt, Tk_opt, ret_stat,X_guess_out] = solveSegment(k, X_start_value, s_start, s_end, N_sub)
% Explanation:
%   k: which segment number (for logging purposes)
%   X_start_value: previous segment's final state; empty [] if first segment
%   s_start: starting track index for this segment
%   s_end  : ending track index for this segment
%   N_sub  : number of discretization steps within this segment
%
import casadi.*

% -- 3.1. Free final time variable for this segment & time step dt_sub
opti_sub = Opti();
Tseg     = opti_sub.variable();  
dt_sub   = Tseg / N_sub;
opti_sub.subject_to( Tseg > 0 );

% -- 3.2. Decision variables for this segment: state Xk (7×(N_sub+1)), control Uk (2×N_sub)
Xk = opti_sub.variable(7, N_sub+1);
Uk = opti_sub.variable(2, N_sub);

% Split Xk and Uk into components for easier constraint writing
Xp_k   = Xk(1,:);   Yp_k  = Xk(2,:);   psi_k = Xk(3,:);
beta_k = Xk(4,:);   v_k   = Xk(5,:);   r_k   = Xk(6,:);
s_k    = Xk(7,:);

delta_k = Uk(1,:);  Fdr_k = Uk(2,:);

% -- 3.3. Initial/terminal constraints
    if isempty(X_start_value)
        % First segment: enforce starting from track index s_start
        opti_sub.subject_to( s_k(1) == s_start );
        opti_sub.subject_to( Xp_k(1) == X_ref(s_start) );
        opti_sub.subject_to( Yp_k(1) == Y_ref(s_start) );
        opti_sub.subject_to( psi_k(1) == phi_ref(s_start) );
        % opti_sub.subject_to( beta_k(1)== 0 );
        opti_sub.subject_to( v_k(1)   >= 0 );
        % opti_sub.subject_to( r_k(1)   == 0 );
    else
        % Non-first segment: directly use previous segment's final state as initial state
        opti_sub.subject_to( Xk(:,1) == X_start_value );
         % (This fixes s, X, Y, psi, beta, v, r at the start of this segment)
    end
    
    % Enforce end of this segment to reach s = s_end on the track
    opti_sub.subject_to( s_k(N_sub+1) == s_end );
    opti_sub.subject_to( Xp_k(N_sub+1) == X_ref(s_end) );
    opti_sub.subject_to( Yp_k(N_sub+1) == Y_ref(s_end) );
    % (No hard constraints on psi, beta, v, r at the end point; they are free to adjust)

% -- 3.4. Ensure s is non-decreasing & within [s_start, s_end]
    for i = 1:N_sub
        opti_sub.subject_to( s_k(i+1) >= s_k(i) );
    end
    opti_sub.subject_to( s_k >= s_start );
    opti_sub.subject_to( s_k <= s_end );
    
    % Ensure speed v ≥ 0
    opti_sub.subject_to( v_k >= 0 );
    
    % -- 3.5. Control/state bounds
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
    
% -- 3.6. Vehicle dynamics parameters for this segment (same as above)
    m   = 1500;  Jz = 2500;
    lf  = 1.2;   lr = 1.3;
    wf  = 1.5;   wr_ = 1.5;
    g   = 9.81;  mu = 0.9;
    Fz_f = (m*g)*(lr/(lf+lr))/2;
    Fz_r = (m*g)*(lf/(lf+lr))/2;
    C1 = 1.2; C2 = 2.0; C3 = 0.3;
    eps = 1e-6;
    
    % -- 3.7. Define dynamic function f_dyn_sub(x,u) -> xdot for this segment
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
    
   
% Rotate front wheel forces to vehicle coordinate frame
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
    
    % ds/dt = V * cosr / Lp，where cosr = (cos(Psi+B)*dXu + sin(Psi+B)*dYu)/Lp
    dXu  = dXdu_f(s_par);   dYu  = dYdu_f(s_par);
    Lp   = sqrt(dXu.^2 + dYu.^2) + eps;
    cosr = cos(Psi+B)*(dXu/Lp) + sin(Psi+B)*(dYu/Lp);
    ds   = V * cosr / Lp;
    
    xdot_sub = [dXpos; dYpos; dPsi; dB; dV; dR; ds];
    f_dyn_sub = Function('f_dyn_sub',{x_sub,u_sub},{xdot_sub});

% -- 3.8. Add Euler discretization dynamic constraints
    for i = 1:N_sub
        xki  = Xk(:,i);
        uki  = Uk(:,i);
        xki1 = xki + dt_sub * f_dyn_sub(xki, uki);
        opti_sub.subject_to( Xk(:,i+1) == xki1 );
    end

% -- 3.9. Hard corridor constraints for this segment
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

% -- 3.10. Initial guess for each grid point (to help solver)
% Construct X_guess (7×(N_sub+1)): place each grid point on the track centerline
X_guess = zeros(7, N_sub+1);


    for i = 1:(N_sub+1)
        s_guess_i = s_start + (s_end - s_start)*(i-1)/N_sub;
        x_ref_i   = full( X_ref(   s_guess_i ) );
        y_ref_i   = full( Y_ref(   s_guess_i ) );
        psi_ref_i = full( phi_ref( s_guess_i ) );

    
        % Calculate curvature at this point
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

        % Estimate max speed at this point & reasonable initial speed guess
        v_max_i = sqrt(mu * g * R_i);
        v_guess_i = min( v_max_i, 60/3.6 );
    
        % Estimate beta and r for initial guess
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

    % Construct Uk_guess (2×N_sub): initial guess delta = 0, Fdr = 0
    Uk_guess = zeros(2, N_sub);
    opti_sub.set_initial( Uk, Uk_guess );


    % Provide a rough initial guess for Tseg: assume average speed ~5 m/s
    T_guess = (s_end - s_start)/5;
    opti_sub.set_initial( Tseg, T_guess );

   % -- 3.11. Objective function for this segment & solve with IPOPT
    w1 = 0.5;  w2 = 0.5;
    Jsub = w1*Tseg + w2*dt_sub*( sum(delta_k.^2) + sum(Fdr_k.^2) );
    opti_sub.minimize(Jsub);

    opts = struct('print_time',0, 'ipopt',struct('tol',1e-1, 'max_iter',5e4,'acceptable_obj_change_tol', 1e1,'acceptable_tol', 1e-1));
    opti_sub.solver('ipopt', opts);
    X_guess_out = X_guess;  % Also return the initial guess for analysis

    
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
    % End of nested function solveSegment
end
% End of main function