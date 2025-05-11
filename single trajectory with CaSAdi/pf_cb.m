% CasADi Optimization in MATLAB - Maximize Speed with Safe Path Tracking on Bezier Curve
clc;
clear;
import casadi.*;

% Initialize CasADi optimizer
opti = Opti();
T = 10; % Total time (seconds)
N = 100; % Number of discretization steps

% Time step
dt = T / N;

% State variables (x, y, v, psi)
x = opti.variable(N+1, 1);
y = opti.variable(N+1, 1);
v = opti.variable(N+1, 1);
psi = opti.variable(N+1, 1);

% Control variables (acceleration and steering angle)
a = opti.variable(N, 1);
delta = opti.variable(N, 1);

% Bezier Curve Path (Cubic Bezier)
P0 = [0, 0];
P1 = [50, 0];
P2 = [50, 50];
P3 = [100, 50];

% Generate Bezier curve points
t_vals = linspace(0, 1, N+1);
x_target = (1 - t_vals).^3 * P0(1) + 3 * (1 - t_vals).^2 .* t_vals * P1(1) + 3 * (1 - t_vals) .* t_vals.^2 * P2(1) + t_vals.^3 * P3(1);
y_target = (1 - t_vals).^3 * P0(2) + 3 * (1 - t_vals).^2 .* t_vals * P1(2) + 3 * (1 - t_vals) .* t_vals.^2 * P2(2) + t_vals.^3 * P3(2);
margin = 1.0; % Allowed deviation from the path (safety margin)

% Vehicle Dynamics Model
L = 2.5; % Vehicle wheelbase
for k = 1:N
    % Euler integration
    opti.subject_to(x(k+1) == x(k) + v(k) * cos(psi(k)) * dt);
    opti.subject_to(y(k+1) == y(k) + v(k) * sin(psi(k)) * dt);
    opti.subject_to(v(k+1) == v(k) + a(k) * dt);
    opti.subject_to(psi(k+1) == psi(k) + (v(k) / L) * tan(delta(k)) * dt);

    % Path Constraints (Stay within the track)
    opti.subject_to((x(k) - x_target(k))^2 + (y(k) - y_target(k))^2 <= margin^2);
end

% Objective Function: Maximize Speed and Track Path
w_tracking = 0.1; % Weight for path tracking
w_control = 0.0001;  % Weight for control smoothness
J = -sum(v) + w_tracking * sum((x - (x_target)').^2 + (y - (y_target)').^2) + w_control * (sum(a.^2) + sum(delta.^2));
opti.minimize(J);

% Initial State
x0 = [0; 0; 10; 0];
opti.subject_to(x(1) == x0(1));
opti.subject_to(y(1) == x0(2));
opti.subject_to(v(1) == x0(3));
opti.subject_to(psi(1) == x0(4));

% Constraints on Steering Angle
delta_max = pi / 3;
opti.subject_to(-delta_max <= delta <= delta_max);

% Solver settings
opts = struct;
opts.ipopt.print_level = 0;
opts.ipopt.max_iter = 1000;
opts.ipopt.tol = 1e-6;
opti.solver('ipopt', opts);

% Solve optimization problem
sol = opti.solve();

% Extract solution
x_sol = sol.value(x);
y_sol = sol.value(y);
v_sol = sol.value(v);
psi_sol = sol.value(psi);
a_sol = sol.value(a);
delta_sol = sol.value(delta);

% Plot results
figure;
subplot(2, 1, 1);
plot(x_sol, y_sol, 'b'); hold on;
plot(x_target, y_target, 'r--');
legend('Optimized Path', 'Bezier Target Path');
title('Optimized Trajectory on Bezier Curve');
xlabel('x (m)'); ylabel('y (m)');

subplot(2, 1, 2);
plot(v_sol, 'g'); hold on;
plot(a_sol, 'b');
plot(delta_sol, 'r');
legend('Velocity (m/s)', 'Acceleration (m/s^2)', 'Steering Angle (rad)');
title('Optimized Control Inputs');
