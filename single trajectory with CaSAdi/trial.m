% CasADi Trajectory Optimization with Double-Track Model and Bezier Path
import casadi.*   % Import CasADi (make sure CasADi is installed)

%% Problem Parameters and Reference Path (Example)
N = 15;                      % number of segments (discretization intervals)
L = 2.5;                     % vehicle wheelbase (meters)
% Reference path (Bezier or given as discrete points):
%% Reference Path (via cubic Bézier, discretized into N+1 points)
% assume N is already defined

% control points
P0 = [0,   0];
P1 = [50,  0];
P2 = [100, 20];
P3 = [100,100];

% symbolic Bézier functions (you may have these already)
u_sym = SX.sym('u',1);
Xb = (1-u_sym)^3*P0(1) + 3*(1-u_sym)^2*u_sym*P1(1) + ...
     3*(1-u_sym)*u_sym^2*P2(1) + u_sym^3*P3(1);
Yb = (1-u_sym)^3*P0(2) + 3*(1-u_sym)^2*u_sym*P1(2) + ...
     3*(1-u_sym)*u_sym^2*P2(2) + u_sym^3*P3(2);
X_ref_fun = Function('X_ref_fun',{u_sym},{Xb});
Y_ref_fun = Function('Y_ref_fun',{u_sym},{Yb});

% discretize path
u_grid = linspace(0,1,N+1);
x_ref  = full( X_ref_fun(u_grid) );   % 1×(N+1)
y_ref  = full( Y_ref_fun(u_grid) );   % 1×(N+1)

% optionally package as ref_points
ref_points = [ x_ref; y_ref ];        % 2×(N+1)


% Initial and final states from reference:
x0   = x_ref(1);             % initial X position
y0   = y_ref(1);             % initial Y position
psi0 = atan2(y_ref(2)-y_ref(1), x_ref(2)-x_ref(1));  % initial orientation (tangent of path)
v0   = 0.5;                  % initial speed (small >0 to avoid stagnation)
x_final = x_ref(end);        % final target X position
y_final = y_ref(end);        % final target Y position

% Allowed deviation from reference and obstacle definitions
d_max = 1.5;                 % max lateral deviation (meters) from reference path
% Example obstacle (if any):
obs_x   = []; 
obs_y   = [];  % array of obstacle center coordinates (can be multiple)
obs_rad = [];                % corresponding obstacle radii
safety_margin = 0.1;         % safety margin around obstacles

% % Physical limits
% v_max   = 15.0;              % max speed (m/s)
% a_max   = 3.0;               % max acceleration (m/s^2)
% a_min   = -6.0;              % max deceleration (m/s^2) (negative accel)
% delta_max = 0.5;             % max steering angle (rad)

%% Set up optimization variables in one Opti instance
opti = Opti();                             % create Opti problem
X = opti.variable(7, N+1);                 % state trajectory [x; y; psi; v] over N+1 points
U = opti.variable(2, N);                   % control trajectory [accel; steer] over N segments
T = opti.variable();                       % total time (scalar)

% Split state components for convenience
X_pos = X(1:2, :);                         % (x, y) positions
X_psi = X(3, :);                           % heading angles
X_v   = X(4, :);                           % speeds
X_beta = X(5, :);                         % (x, y) positions
X_r = X(6, :);                           % heading angles
X_s = X(7, :);                           % speeds

%% Define dynamics function (double-track model)
%% 5. Double-Track + Magic-Formula Dynamics
% Symbolic state and input
x    = SX.sym('x',7);     % [Xpos; Ypos; psi; beta; v; r; s]
u_in = SX.sym('u',2);     % [delta; Fdr]

% Unpack for clarity
Xpos  = x(1);  Ypos = x(2);
psi   = x(3);  beta = x(4);
v     = x(5);  r    = x(6);
s     = x(7);
delta = u_in(1);
Fdr   = u_in(2);

% Vehicle & tire parameters (make sure these are in scope)
% m,Jz,lf,lr,wf,wr,g,Fz_f,Fz_r,C1,C2,C3,eps must already be defined

% 1) Body velocities
vx = v*cos(beta);
vy = v*sin(beta);

% 2) Wheel local velocities
vx_fl = vx - r*(wf/2);  vy_fl = vy + r*lf;
vx_fr = vx + r*(wf/2);  vy_fr = vy + r*lf;
vx_rl = vx - r*(wr/2);  vy_rl = vy - r*lr;
vx_rr = vx + r*(wr/2);  vy_rr = vy - r*lr;

% 3) Slip angles
alpha_fl = delta   - atan2(vy_fl, vx_fl+eps);
alpha_fr = delta   - atan2(vy_fr, vx_fr+eps);
alpha_rl = 0       - atan2(vy_rl, vx_rl+eps);
alpha_rr = 0       - atan2(vy_rr, vx_rr+eps);

% 4) Equivalent slip magnitudes
Sfl = sqrt(max(2-2*cos(alpha_fl),0)+eps);
Sfr = sqrt(max(2-2*cos(alpha_fr),0)+eps);
Srl = sqrt(max(2-2*cos(alpha_rl),0)+eps);
Srr = sqrt(max(2-2*cos(alpha_rr),0)+eps);

% 5) Degraded friction μ
mu_fl = C1*(1-exp(-C2*Sfl)) - C3*Sfl;
mu_fr = C1*(1-exp(-C2*Sfr)) - C3*Sfr;
mu_rl = C1*(1-exp(-C2*Srl)) - C3*Srl;
mu_rr = C1*(1-exp(-C2*Srr)) - C3*Srr;

% 6) Lateral tire forces
Fyf_fl = (sin(alpha_fl)/(Sfl+eps)) * mu_fl * Fz_f;
Fyf_fr = (sin(alpha_fr)/(Sfr+eps)) * mu_fr * Fz_f;
Fyf_rl = (sin(alpha_rl)/(Srl+eps)) * mu_rl * Fz_r;
Fyf_rr = (sin(alpha_rr)/(Srr+eps)) * mu_rr * Fz_r;

% 7) Longitudinal tire forces (drive)
Fxf_fl = Fdr;  Fxf_fr = Fdr;
Fxf_rl = Fdr;  Fxf_rr = Fdr;

% 8) Rotate into vehicle frame
Fxf_fl_c = Fxf_fl*cos(delta) - Fyf_fl*sin(delta);
Fyf_fl_c = Fxf_fl*sin(delta) + Fyf_fl*cos(delta);
Fxf_fr_c = Fxf_fr*cos(delta) - Fyf_fr*sin(delta);
Fyf_fr_c = Fxf_fr*sin(delta) + Fyf_fr*cos(delta);
Fxf_rl_c = Fxf_rl;  Fyf_rl_c = Fyf_rl;
Fxf_rr_c = Fxf_rr;  Fyf_rr_c = Fyf_rr;

% Sum total forces
Fxf_tot = Fxf_fl_c + Fxf_fr_c + Fxf_rl_c + Fxf_rr_c;
Fyf_tot = Fyf_fl_c + Fyf_fr_c + Fyf_rl_c + Fyf_rr_c;

% 9) Kinematic derivatives
dXpos = v * cos(psi+beta);
dYpos = v * sin(psi+beta);
dpsi   = r;

% 10) Side-slip rate
dbeta = -r + 1/(m*(v+eps))*( ...
    (Fyf_fl_c+Fyf_fr_c)*cos(delta-beta) + ...
    (Fyf_rl_c+Fyf_rr_c)*cos(-beta) - Fdr*sin(beta) );

% 11) Yaw acceleration
dr = 1/Jz*( ...
    Fyf_fl_c*( lf*cos(delta)- (wf/2)*sin(delta) ) + ...
    Fyf_fr_c*( lf*cos(delta)+ (wf/2)*sin(delta) ) + ...
    Fyf_rl_c*(-lr) + Fyf_rr_c*(-lr) + ...
    Fxf_fl_c*( lf*sin(delta) ) + Fxf_fr_c*( lf*sin(delta) ) );

% 12) Longitudinal acceleration
dv = 1/m*( ...
    (Fyf_fl_c+Fyf_fr_c)*sin(beta-delta) + ...
    (Fyf_rl_c+Fyf_rr_c)*sin(beta) + Fdr*cos(beta) );

% 13) Path-parameter rate
dX = dXdu_fun(s);  % your dXdu_fun
dY = dYdu_fun(s);  % your dYdu_fun
Lp = sqrt(dX^2 + dY^2) + eps;
cosr = cos(psi+beta)*(dX/Lp) + sin(psi+beta)*(dY/Lp);
ds = v * cosr / Lp;

% 14) Assemble f(x,u)
xdot = [dXpos; dYpos; dpsi; dbeta; dv; dr; ds];

% 15) Wrap into a CasADi function
f_dyn = Function('f_dyn',{x,u_in},{xdot});



%% —— 10. Dynamics consistency via Explicit Midpoint (full 7-state f_dyn) —— 
dt = T / N;   % your segment duration

for k = 1:N
    xk    = X(:, k);       % 7×1 state at node k
    uk    = U(:, k);       % 2×1 control at k

    % 1) half‐step prediction
    k1    = f_dyn(xk, uk);            % 7×1
    x_mid = xk + (dt/2) * k1;         % midpoint state

    % 2) compute midpoint slope
    k2    = f_dyn(x_mid, uk);         % 7×1

    % 3) full‐step update
    x_next = xk + dt * k2;            % 7×1

    % 4) enforce causality
    opti.subject_to( X(:, k+1) == x_next );
end



% %% Path Following Constraints
% for k = 1:N+1
%     % Keep each state within track
%     opti.subject_to( (X(1,k) - x_ref(k))^2 + (X(2,k) - y_ref(k))^2 <= d_max^2 );
% end
for k = 1:N+1
    sk = s_v(k);                             % path parameter
    xr = X_ref_fun(sk);                      % reference point
    yr = Y_ref_fun(sk);
    dxu = dXdu_fun(sk); dyu = dYdu_fun(sk);
    Lp  = sqrt(dxu^2 + dyu^2) + eps;
    sinp = dyu / Lp;  cosp = dxu / Lp;
    
    ek = -(X(1,k)-xr)*sinp + (X(2,k)-yr)*cosp;  % lateral projection
    opti.subject_to( -d_max <= ek <= d_max );  % stay in corridor
end
%% Obstacle Avoidance Constraints (if obstacles are defined)
if ~isempty(obs_x)
    for i = 1:length(obs_x)
        for k = 1:N+1
            % Maintain a safe distance from obstacle i at all times
            opti.subject_to( (X(1,k) - obs_x(i))^2 + (X(2,k) - obs_y(i))^2 >= (obs_rad(i) + safety_margin)^2 );
        end
    end
end

%% Boundary and Physical Constraints
% Initial state fixed:
opti.subject_to( X(1,1) == x0 );
opti.subject_to( X(2,1) == y0 );
opti.subject_to( X(3,1) == psi0 );
opti.subject_to( X(4,1) == v0 );
% Final position fixed to reference end:
opti.subject_to( X(1,N+1) == x_final );
opti.subject_to( X(2,N+1) == y_final );
% (Final orientation can be constrained if needed, here left free)
% Speed bounds (nonnegativity and max):
opti.subject_to( 0 <= X_v <= v_max );
% Control bounds (acceleration and steering limits):
opti.subject_to( a_min <= U(1,:) <= a_max );
opti.subject_to( -delta_max <= U(2,:) <= delta_max );
% Time bound:
opti.subject_to( T >= 0 );

%% Objective: minimize total time + control effort
% Weighing factor for control effort:
alpha = 2;
% Sum of squared controls:
control_cost = alpha * ( sum(U(1,:).^2) + sum(U(2,:).^2) );
opti.minimize( T + control_cost );

% %% Initial Guess for faster convergence
% % Guess state trajectory as reference path with small speed
% psi_ref = zeros(1, N+1);
% % Compute an approximate heading for each ref segment (for initial guess)
% for k = 1:N
%     psi_ref(k) = atan2( y_ref(k+1) - y_ref(k),  x_ref(k+1) - x_ref(k) );
% end
% psi_ref(N+1) = psi_ref(N);
% % Compute path length for time guess
% dist = sqrt(diff(x_ref).^2 + diff(y_ref).^2);
% path_length = sum(dist);
% T_guess = path_length / 5.0;   % assume ~5 m/s average speed
% % Speed guess: ramp from 0 to some value (e.g., 5 m/s) along the path
% v_guess = linspace(v0, 5.0, N+1);
% 
% % Set initial values in opti
% opti.set_initial(X(1,:), x_ref);
% opti.set_initial(X(2,:), y_ref);
% opti.set_initial(X(3,:), psi_ref);
% opti.set_initial(X(4,:), v_guess);
% opti.set_initial(U, zeros(2, N));       % no accel, no steer initially
% opti.set_initial(T, T_guess);

%% Solver settings
opts = struct;
opts.ipopt.tol = 1e-2;
opts.ipopt.print_level = 5;
opti.solver('ipopt', opts);


%% Solve and handle failures
try
    sol = opti.solve();
catch err
    disp('Solver failed!');
    % Extract the current iterate using debug mode
    X_dbg = opti.debug.value(X);
    U_dbg = opti.debug.value(U);
    T_dbg = opti.debug.value(T);

    % Plot debug trajectory vs reference
    figure; hold on;
    plot(x_ref, y_ref, 'r--', 'DisplayName', 'Reference');
    plot(X_dbg(1,:), X_dbg(2,:), 'b.-', 'DisplayName', 'Current Traj');
    legend; axis equal; grid on;
    return;
end




%% Extract solution
X_opt = sol.value(X);
U_opt = sol.value(U);
T_opt = sol.value(T);
% (sol.value returns numeric values for the variables at optimum)

% Plot optimized trajectory vs reference
figure; hold on;
plot(x_ref, y_ref, 'r--', 'LineWidth',2, 'DisplayName','Reference Path');
plot(X_opt(1,:), X_opt(2,:), 'b-o', 'LineWidth',2, 'DisplayName','Optimized Trajectory');
if ~isempty(obs_x)
    % plot obstacle areas
    theta = linspace(0, 2*pi, 50);
    for i = 1:length(obs_x)
        xp = obs_x(i) + (obs_rad(i)+safety_margin)*cos(theta);
        yp = obs_y(i) + (obs_rad(i)+safety_margin)*sin(theta);
        patch(xp, yp, 'r', 'FaceAlpha',0.3, 'EdgeColor','none', 'DisplayName', sprintf('Obstacle %d', i));
    end
end
legend('Location','best');
xlabel('X [m]'); ylabel('Y [m]'); grid on;
title(sprintf('Optimized Trajectory (Total time = %.2f s)', T_opt));

% Plot speed profile over time
time = linspace(0, T_opt, N+1);
figure; 
plot(time, X_opt(4,:), 'm-o', 'LineWidth',2);
xlabel('Time [s]'); ylabel('Speed [m/s]'); grid on;
title('Optimized Speed Profile');
