% newversion.m
% Trajectory Optimization (Euler + Double‐Track Model + nion-linear Tires + Obstacles + Free Final Time)

import casadi.*

%% 1. Problem setup
N    = 25;                % number of intervals
opti = Opti();            % create optimization problem
T    = opti.variable();   % free final time
dt   = T / N;             % time step

% require final time positive
opti.subject_to(T > 0);

%% 2. Reference path (cubic Bézier)
P0 = [0,   0];  P1 = [50,  0];
P2 = [100, 20]; P3 = [100,100];
u  = SX.sym('u');
Xb = (1-u)^3*P0(1) + 3*(1-u)^2*u*P1(1) + 3*(1-u)*u^2*P2(1) + u^3*P3(1);
Yb = (1-u)^3*P0(2) + 3*(1-u)^2*u*P1(2) + 3*(1-u)*u^2*P2(2) + u^3*P3(2);
dXdu = jacobian(Xb,u); 
dYdu = jacobian(Yb,u);

X_ref   = Function('X_ref',{u},{Xb});
Y_ref   = Function('Y_ref',{u},{Yb});
dXdu_f  = Function('dXdu',{u},{dXdu});
dYdu_f  = Function('dYdu',{u},{dYdu});
phi_ref = Function('phi_ref',{u},{atan2(dYdu,dXdu)});

%% 3. Decision variables
X = opti.variable(7, N+1);   % state: [X; Y; psi; beta; v; r; s]
U = opti.variable(2, N);     % controls: [delta; Fdr]

% split for readability
X_p  = X(1,:); Y_p = X(2,:);
psi_v= X(3,:); beta_v = X(4,:);
v_v  = X(5,:); r_v    = X(6,:);
s_v  = X(7,:);

delta_v = U(1,:);
Fdr_v   = U(2,:);

%% 4. Initial / boundary conditions
% start at u=0 on the spline
opti.subject_to( X_p(1) == P0(1) );
opti.subject_to( Y_p(1) == P0(2) );
opti.subject_to( s_v(1) == 0 );
opti.subject_to( psi_v(1) == phi_ref(0) );
opti.subject_to( beta_v(1)== 0 );
opti.subject_to( v_v(1)   == 5 );
opti.subject_to( r_v(1)   == 0 );
% —— 强制到达路径末端 u=1
opti.subject_to( s_v(N+1) == 1 ); 
opti.subject_to( X_p(N+1) == X_ref(1) );
opti.subject_to( Y_p(N+1) == Y_ref(1) );

% split for readability
X_p  = X(1,:); Y_p = X(2,:);
psi_v= X(3,:); beta_v = X(4,:);
v_v  = X(5,:); r_v    = X(6,:);
s_v  = X(7,:);

delta_v = U(1,:);
Fdr_v   = U(2,:);


%% 5. Model parameters
m   = 1500;   Jz = 2500;
lf  = 1.2;    lr = 1.3;
wf  = 1.5;    wr = 1.5;
g   = 9.81;
Fz_f = m*g*(lr/(lf+lr))/2;
Fz_r = m*g*(lf/(lf+lr))/2;
C1 = 1.2; C2 = 2.0; C3 = 0.3;
eps = 1e-6;

%% 6. Dynamics function
x    = SX.sym('x',7);
u_in = SX.sym('u',2);
Xpos = x(1); Ypos = x(2); Psi = x(3); B = x(4);
V    = x(5); R    = x(6); s = x(7);
delta = u_in(1); Fdr = u_in(2);

% lateral and longitudinal speeds
vx = V*cos(B);  vy = V*sin(B);
% wheel-local velocities
vx_fl = vx - R*(wf/2); vy_fl = vy + R*lf;
vx_fr = vx + R*(wf/2); vy_fr = vy + R*lf;
vx_rl = vx - R*(wr/2); vy_rl = vy - R*lr;
vx_rr = vx + R*(wr/2); vy_rr = vy - R*lr;

% slip angles
alpha_fl = delta   - atan2(vy_fl, vx_fl+eps);
alpha_fr = delta   - atan2(vy_fr, vx_fr+eps);
alpha_rl =       - atan2(vy_rl, vx_rl+eps);
alpha_rr =       - atan2(vy_rr, vx_rr+eps);

% sliding S
Sfl = sqrt(max(2-2*cos(alpha_fl),0)+eps);
Sfr = sqrt(max(2-2*cos(alpha_fr),0)+eps);
Srl = sqrt(max(2-2*cos(alpha_rl),0)+eps);
Srr = sqrt(max(2-2*cos(alpha_rr),0)+eps);

% friction μ
mu_fl = C1*(1-exp(-C2*Sfl)) - C3*Sfl;
mu_fr = C1*(1-exp(-C2*Sfr)) - C3*Sfr;
mu_rl = C1*(1-exp(-C2*Srl)) - C3*Srl;
mu_rr = C1*(1-exp(-C2*Srr)) - C3*Srr;

% lateral forces
Fyf_fl = (sin(alpha_fl)/(Sfl+eps))*mu_fl*Fz_f;
Fyf_fr = (sin(alpha_fr)/(Sfr+eps))*mu_fr*Fz_f;
Fyf_rl = (sin(alpha_rl)/(Srl+eps))*mu_rl*Fz_r;
Fyf_rr = (sin(alpha_rr)/(Srr+eps))*mu_rr*Fz_r;

% driving forces
Fxf_fl = Fdr; Fxf_fr = Fdr;
Fxf_rl = Fdr; Fxf_rr = Fdr;

% resolve into vehicle frame
Fxf_fl_c = Fxf_fl*cos(delta) - Fyf_fl*sin(delta);
Fyf_fl_c = Fxf_fl*sin(delta) + Fyf_fl*cos(delta);
Fxf_fr_c = Fxf_fr*cos(delta) - Fyf_fr*sin(delta);
Fyf_fr_c = Fxf_fr*sin(delta) + Fyf_fr*cos(delta);
Fxf_rl_c = Fxf_rl;  Fyf_rl_c = Fyf_rl;
Fxf_rr_c = Fxf_rr;  Fyf_rr_c = Fyf_rr;

% total forces
Fxf_tot = Fxf_fl_c + Fxf_fr_c + Fxf_rl_c + Fxf_rr_c;
Fyf_tot = Fyf_fl_c + Fyf_fr_c + Fyf_rl_c + Fyf_rr_c;

% kinematics & dynamics
dXpos = V*cos(Psi+B);
dYpos = V*sin(Psi+B);
dPsi   = R;
dB     = - R + 1/(m*(V+eps))*((Fyf_fl_c+Fyf_fr_c)*cos(delta-B) + (Fyf_rl_c+Fyf_rr_c)*cos(-B) - Fdr*sin(B));
dR = 1/Jz*( ...
    Fyf_fl_c*(lf*cos(delta)-(wf/2)*sin(delta)) + ...
    Fyf_fr_c*(lf*cos(delta)+(wf/2)*sin(delta)) + ...
    Fyf_rl_c*(-lr)           + ...
    Fyf_rr_c*(-lr)           + ...
    Fxf_fl_c*sin(delta)*lf   + ...
    Fxf_fr_c*sin(delta)*lf );
dV = 1/m*((Fyf_fl_c+Fyf_fr_c)*sin(B-delta) + (Fyf_rl_c+Fyf_rr_c)*sin(B) + Fdr*cos(B));

% progress along the spline
dXs = dXdu_f(s); dYs = dYdu_f(s);
Lp  = sqrt(dXs^2 + dYs^2) + eps;
cosr= cos(Psi+B)*(dXs/Lp) + sin(Psi+B)*(dYs/Lp);
ds  = V*cosr / Lp;


xdot = [dXpos; dYpos; dPsi; dB; dV; dR; ds];
f_dyn = Function('f_dyn',{x,u_in},{xdot});

%% 7. Dynamics constraints (Euler)
for k=1:N
  xk  = X(:,k);
  uk  = U(:,k);
  xk1 = xk + dt * f_dyn(xk,uk);
  opti.subject_to( X(:,k+1) == xk1 );
end

%% 8. Hard “stay in corridor” constraint
margin = 2.0;  % lateral bound
for k=1:N+1
  sk = s_v(k);
  xr = X_ref(sk); yr = Y_ref(sk);
  dXu = dXdu_f(sk); dYu = dYdu_f(sk);
  Lp  = sqrt(dXu^2 + dYu^2) + eps;
  sinp= dYu/Lp; cosp = dXu/Lp;
  ek  = -(X_p(k)-xr)*sinp + (Y_p(k)-yr)*cosp;
  opti.subject_to( -margin <= ek <= margin );
end

% %% 9. Obstacle avoidance
% obs_X = [51,79]; 
% obs_Y = [11,27]; 
% obs_R = [0.5,0.5]; 
% r_safe = 1.0;
% M = numel(obs_X);
% for k=1:N+1
%   for j=1:M
%     dx = X_p(k)-obs_X(j);
%     dy = Y_p(k)-obs_Y(j);
%     opti.subject_to( dx^2 + dy^2 >= (obs_R(j)+r_safe)^2 );
%   end
% end

%% 10. Objective: trade‐off final time & control effort
w1 = 1.0;    % weight on time
w2 = 1.5;    % weight on effort
J  = w1*T + w2*dt*( sum(delta_v.^2) + sum(Fdr_v.^2) );
opti.minimize(J);

%% 11. Solve
opti.solver('ipopt',struct('print_time',0,'ipopt',struct('tol',1e-2,'max_iter',1e6)));
sol = opti.solve();

% %% 12. Extract & visualize
% Xopt = sol.value(X);
% Uopt = sol.value(U);
% Topt = sol.value(T);
% 
% % reference spline
% u_plot = linspace(0,1,200);
% XbP = full(X_ref(u_plot));  YbP = full(Y_ref(u_plot));
% 
% % time grid
% t = linspace(0,Topt,N+1);
% 
% figure('Name','Trajectory & Speed','NumberTitle','off');
% subplot(2,1,1);
% plot(XbP,YbP,'r--','LineWidth',1.5); hold on;
% plot(Xopt(1,:),Xopt(2,:),'b-','LineWidth',2);
% axis equal; grid on;
% xlabel('X [m]'); ylabel('Y [m]');
% title('Reference vs. Optimized Trajectory');
% legend('Reference','Optimized','Location','best');
% 
% subplot(2,1,2);
% plot(t, Xopt(5,:),'-o','LineWidth',1.5,'MarkerSize',4);
% grid on;
% xlabel('Time [s]'); ylabel('Speed v [m/s]');
% title('Speed Profile');



%%12. 提取 & 可视化
Xopt = sol.value(X);    % 7 x (N+1)
vopt = sol.value(v_v);  % 1 x (N+1)
Topt = sol.value(T);

% 构造中心线和法向
u_plot = linspace(0,1,200);
XbP = full(X_ref(u_plot));
YbP = full(Y_ref(u_plot));
dXuP = full(dXdu_f(u_plot));
dYuP = full(dYdu_f(u_plot));
LpP  = sqrt(dXuP.^2 + dYuP.^2) + eps;
% 法向 (左 & 右)
nX = -dYuP./LpP;
nY =  dXuP./LpP;

% 障碍物（示例）
obs_X = [51, 79];
obs_Y = [11, 27];
obs_R = [0.5, 0.5];
r_safe = 1.0;

% 时间网格
tgrid = linspace(0, Topt, N+1);

figure('Name','Trajectory & Speed','NumberTitle','off');

%% (1) 上：走廊 + 中心线 + 障碍 + 优化轨迹
ax1 = subplot(2,1,1);
hold(ax1,'on'); grid(ax1,'on'); axis(ax1,'equal');
% 走廊边界
plot(ax1, XbP + nX*margin, YbP + nY*margin, 'k--');
plot(ax1, XbP - nX*margin, YbP - nY*margin, 'k--');

% 中心线
plot(ax1, XbP, YbP, 'k-', 'LineWidth',1.5);
% 障碍物
viscircles([obs_X(:), obs_Y(:)], obs_R(:), 'EdgeColor','r','LineWidth',1);
% 优化后轨迹
plot(ax1, Xopt(1,:), Xopt(2,:), 'b-', 'LineWidth',2);
xlabel(ax1,'X [m]'); ylabel(ax1,'Y [m]');
title(ax1,'Centerline, Corridor, Obstacles & Optimized Path');
legend(ax1, {'Corridor1','Corridor2','Centerline','','Optimized','Obstacles'}, 'Location','best');

%% (2) 下：速度剖面 (km/h)
ax2 = subplot(2,1,2);
v_kmh = vopt * 3.6;  % m/s -> km/h
plot(ax2, tgrid, v_kmh, 'b-o', 'LineWidth',1.5, 'MarkerSize',4);
grid(ax2,'on');
xlabel(ax2,'Time [s]');
ylabel(ax2,'Speed [km/h]');
title(ax2,'Speed Profile');