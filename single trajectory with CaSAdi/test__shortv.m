% Clear environment
clc;
clear;
import casadi.*;

% Define symbolic variables (state and control)
x = SX.sym('x');     % Vehicle x-position (global frame)
y = SX.sym('y');     % Vehicle y-position (global frame)
psi = SX.sym('psi'); % Yaw angle (orientation)
vx = SX.sym('vx');   % Longitudinal velocity
vy = SX.sym('vy');   % Lateral velocity
omega = SX.sym('omega'); % Yaw rate

% Control inputs (acceleration and steering angles)
a = SX.sym('a');         % Total acceleration (drive force)
delta_f = SX.sym('delta_f'); % Front wheel steering angle
delta_r = SX.sym('delta_r'); % Rear wheel steering angle

% System parameters
m = 1500;      % Vehicle mass (kg)
Jzz = 2500;    % Vehicle yaw inertia (kg·m^2)
Lf = 1.2;      % Distance from CG to front axle (m)
Lr = 1.6;      % Distance from CG to rear axle (m)
Cf = 80000;    % Front tire cornering stiffness (N/rad)
Cr = 80000;    % Rear tire cornering stiffness (N/rad)
dt = 0.5;     % Simulation time step (s)
N = 200;        % Prediction horizon

% Tire force model (linear)
FyT_fl = Cf * (delta_f - atan((vy + Lf * omega) / max(vx, 0.1)));
FyT_fr = Cf * (delta_f - atan((vy + Lf * omega) / max(vx, 0.1)));
FyT_rl = Cr * (-atan((vy - Lr * omega) / max(vx, 0.1)));
FyT_rr = Cr * (-atan((vy - Lr * omega) / max(vx, 0.1)));

% Double-track model dynamics
ax = -(FyT_fl + FyT_fr) * sin(delta_f) - (FyT_rl + FyT_rr) * sin(delta_r) + a;
ay = (FyT_fl + FyT_fr) * cos(delta_f) + (FyT_rl + FyT_rr) * cos(delta_r);
yaw_dot = (Lf * (FyT_fl + FyT_fr) - Lr * (FyT_rl + FyT_rr)) / Jzz;

% State equations (nonlinear dynamics)
x_dot = vx * cos(psi) - vy * sin(psi); % x-position rate
y_dot = vx * sin(psi) + vy * cos(psi); % y-position rate
vx_dot = ax / m;                      % longitudinal acceleration
vy_dot = ay / m;                      % lateral acceleration
omega_dot = yaw_dot;                  % yaw rate change

% State and control vectors
state = [x; y; psi; vx; vy; omega];
control = [a; delta_f; delta_r];

% CasADi function for dynamics (symbolic model)
f = Function('f', {state, control}, {x_dot, y_dot, vx_dot, vy_dot, omega_dot});

% Setup MPC optimization problem
opti = Opti();
X = opti.variable(6, N+1); % State variables for the prediction horizon
U = opti.variable(3, N);   % Control variables [a, delta_f, delta_r]

% Initial state (starting position)
x0 = [0; 0; 0; 5; 2; 0];
opti.subject_to(X(:, 1) == x0); % Initial state constraint

% Define Bezier control points
P0 = [0, 0];       % Start point
P1 = [5, 5];       % First control point
P2 = [10, 0];     % Second control point
P3 = [10, 10];      % End point

% Generate Bezier curve (track centerline)
t_values = linspace(0, 1, N+1);
track_center_x = (1 - t_values).^3 * P0(1) + ...
                 3 * (1 - t_values).^2 .* t_values * P1(1) + ...
                 3 * (1 - t_values) .* t_values.^2 * P2(1) + ...
                 t_values.^3 * P3(1);
             
track_center_y = (1 - t_values).^3 * P0(2) + ...
                 3 * (1 - t_values).^2 .* t_values * P1(2) + ...
                 3 * (1 - t_values) .* t_values.^2 * P2(2) + ...
                 t_values.^3 * P3(2);

% Adjusted Cost function with RK4 (Runge-Kutta Integration)
J = 0;
target_speed = 22; % Target speed (m/s)
for k = 1:N
    % Runge-Kutta 4th Order Integration (RK4)
    k1 = f(X(:, k), U(:, k));
    k2 = f(X(:, k) + dt/2 * k1, U(:, k));
    k3 = f(X(:, k) + dt/2 * k2, U(:, k));
    k4 = f(X(:, k) + dt * k3, U(:, k));
    
    % RK4 integration
    x_next = X(:, k) + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    opti.subject_to(X(:, k+1) == x_next);

    % Total speed: sqrt(vx^2 + vy^2)
    total_speed = sqrt(X(4, k)^2 + X(5, k)^2);
    % Cost function: maximize speed, minimize distance to track center
    J = J - sum(X(4, k) )+ 0.1* ((X(1, k) - track_center_x(k))^2 + (X(2, k) - track_center_y(k))^2);
    % Set objective
    opti.minimize(J);
end


% Control input constraints (realistic steering angles and acceleration)
opti.subject_to(U(1, :) >= -10000);  % Minimum drive force (braking)
opti.subject_to(U(1, :) <= 10000);   % Maximum drive force (acceleration)
opti.subject_to(U(2, :) >= -0.8);   % Front steering angle range
opti.subject_to(U(2, :) <= 0.8);
opti.subject_to(U(3, :) >= -0.6);   % Rear steering angle range
opti.subject_to(U(3, :) <= 0.6);

% Initial guess (improves solver stability)
U_init = zeros(3, N); % Initial guess for control inputs
X_init = repmat(x0, 1, N+1); % Initial guess for state trajectory
opti.set_initial(X, X_init);
opti.set_initial(U, U_init);

% Set solver (CasADi SQP method)
opti.solver('ipopt');
sol = opti.solve();

% Visualization of Results (Path and Speed)
figure;
subplot(2, 1, 1); % First subplot for path
plot(sol.value(X(1, :)), sol.value(X(2, :)), 'b.-'); % Predicted path
hold on;
plot(track_center_x, track_center_y, 'g--'); % Track centerline
title('Double-Track Model + Speed Optimization + Track Centering');
legend('Predicted Path', 'Track Centerline');
xlabel('X position');
ylabel('Y position');

% Second subplot for speed profile
subplot(2, 1, 2);
plot(0:dt:dt*(N-1), sol.value(X(4, 1:N)), 'r.-'); % Speed (vx)
title('Vehicle Speed (vx) Over Time');
xlabel('Time (s)');
ylabel('Speed (m/s)');
grid on;

