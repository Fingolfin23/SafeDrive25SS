% CasADi Optimization in MATLAB - Minimum Path Distance Tracking
clc;
clear;
import casadi.*;

% Initialize CasADi optimizer
opti = Opti();

% Path Distance (fixed length)
L_total = 50; % Total path length (meters)
N = 100; % Number of discretization steps

% Parameterized Path Definition (Bezier or Polynomial)
s = linspace(0, 1, N+1); % Normalized path parameter
x_target = L_total * s; % Linear path as an example
y_target = 10 * sin(pi * s); % Sine wave as target path

% State variables (x, y, v, psi)
x = opti.variable(N+1, 1);
y = opti.variable(N+1, 1);
v = opti.variable(N+1, 1);
psi = opti.variable(N+1, 1);

% Control variables (acceleration and steering angle)
a = opti.variable(N, 1);
delta = opti.variable(N, 1);

% Total time is now an optimization variable
T = opti.variable();

% Time step is now dependent on T
dt = T / N;

% Vehicle Dynamics Model
L = 2.5; % Vehicle wheelbase
for k = 1:N
    % Euler integration
    opti.subject_to(x(k+1) == x(k) + v(k) * cos(psi(k)) * dt);
    opti.subject_to(y(k+1) == y(k) + v(k) * sin(psi(k)) * dt);
    opti.subject_to(v(k+1) == v(k) + a(k) * dt);
    opti.subject_to(psi(k+1) == psi(k) + (v(k) / L) * tan(delta(k)) * dt);

    % Path Constraints (Stay close to the target path)
    opti.subject_to((x(k) - x_target(k))^2 + (y(k) - y_target(k))^2 <= 1);
end

% Objective Function: Minimize Total Time and Maintain Control Smoothness
% Objective Function: Maximize Speed and Track Path
w_tracking = 1; % Weight for path tracking
w_control = 0;  % Weight for control smoothness
J = -sum(v) + w_tracking * sum((x - (x_target)').^2 + (y - (y_target)').^2) + w_control * (sum(a.^2) + sum(delta.^2));
opti.minimize(J);

% Initial State
x0 = [0; 0; 10; 0];
opti.subject_to(x(1) == x0(1));
opti.subject_to(y(1) == x0(2));
opti.subject_to(v(1) == x0(3));
opti.subject_to(psi(1) == x0(4));

% Constraints on Steering Angle
delta_max = pi / 6;
opti.subject_to(-delta_max <= delta <= delta_max);

% Set an initial guess for T
opti.set_initial(T, 10); % Initial guess of total time
opti.subject_to(T > 0); % T must be positive

% Solver settings
opts = struct;
opts.ipopt.print_level = 0;
opts.ipopt.max_iter = 1000;
opts.ipopt.tol = 1e-6;
opti.solver('ipopt', opts);

% Solve optimization problem
sol = opti.solve();

% Extract solution
T_sol = sol.value(T);
x_sol = sol.value(x);
y_sol = sol.value(y);
v_sol = sol.value(v);
psi_sol = sol.value(psi);
a_sol = sol.value(a);
delta_sol = sol.value(delta);

% Display total time
fprintf('Optimal Total Time: %.4f seconds\n', T_sol);

% Plot results
figure;
subplot(2, 1, 1);
plot(x_sol, y_sol, 'b'); hold on;
plot(x_target, y_target, 'r--');
legend('Optimized Path', 'Target Path');
title('Optimized Trajectory in Minimum Distance');
xlabel('x (m)'); ylabel('y (m)');

subplot(2, 1, 2);
plot(v_sol, 'g'); hold on;
plot(a_sol, 'b');
plot(delta_sol, 'r');
legend('Velocity (m/s)', 'Acceleration (m/s^2)', 'Steering Angle (rad)');
title('Optimized Control Inputs');
