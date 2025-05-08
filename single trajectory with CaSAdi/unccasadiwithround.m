% CasADi Optimization in MATLAB - Unconstrained Nonlinear Optimization
clc;
clear;
import casadi.*;

% Define CasADi variables
opti = Opti();
T = 10; % Time horizon (seconds)
N = 100; % Number of discretization steps
dt = T / N;

% State variables (x, y, v, psi)
x = opti.variable(N+1, 1);
y = opti.variable(N+1, 1);
v = opti.variable(N+1, 1);
psi = opti.variable(N+1, 1);

% Control variables (acceleration and steering angle)
a = opti.variable(N+1, 1);
delta = opti.variable(N+1, 1);

% Initial conditions
x0 = [0; 0; 10; 0]; % x, y, v, psi (start at origin with 10 m/s)

% Define curved path (arc)
R = 20; % Radius of the curve (meters)
theta = linspace(0, pi/2, N+1);
x_target = (R * theta)';
y_target = (R * (1 - cos(theta)))';

% Dynamic model equations with turning
L = 2.5; % Vehicle wheelbase
for k = 1:N
    % Euler integration for each step
    opti.subject_to(x(k+1) == x(k) + v(k) * cos(psi(k)) * dt);
    opti.subject_to(y(k+1) == y(k) + v(k) * sin(psi(k)) * dt);
    opti.subject_to(v(k+1) == v(k) + a(k) * dt);
    opti.subject_to(psi(k+1) == psi(k) + (v(k) / L) * tan(delta(k)) * dt);
end

% Objective function: path tracking and smooth driving
w_tracking = 100;
J = sum(w_tracking * (y - y_target).^2 + a.^2 + delta.^2);
opti.minimize(J);

% Set initial state
opti.subject_to(x(1) == x0(1));
opti.subject_to(y(1) == x0(2));
opti.subject_to(v(1) == x0(3));
opti.subject_to(psi(1) == x0(4));

% Constraints on steering angle
delta_max = pi/6;
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
a_sol = sol.value(a);
delta_sol = sol.value(delta);

% Plot results
figure;
subplot(2, 1, 1);
plot(x_sol, y_sol, '-b');
hold on;
plot(x_target, y_target, '--r');
legend('Optimized Path', 'Target Path');
title('Optimized Trajectory in Curve');
xlabel('x (m)'); ylabel('y (m)');

subplot(2, 1, 2);
plot(a_sol); hold on;
plot(delta_sol);
legend('Acceleration (m/s^2)', 'Steering Angle (rad)');
title('Optimized Control Inputs');
