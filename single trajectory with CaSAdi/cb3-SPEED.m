% Trajectory Optimization (Euler + Double‐Track Model +  Tires)
import casadi.*
%%%%%%%%%time fixed: obj velocity
%% 1. 
N  = 30;    % 
dt = 0.5;   %  [s]

%% 2. cubic Bezier path
P0 = [0,   0];
P1 = [50,  0];
P2 = [100, 20];
P3 = [100,100];
u = SX.sym('u',1);
Xb = (1-u)^3*P0(1) + 3*(1-u)^2*u*P1(1) + 3*(1-u)*u^2*P2(1) + u^3*P3(1);
Yb = (1-u)^3*P0(2) + 3*(1-u)^2*u*P1(2) + 3*(1-u)*u^2*P2(2) + u^3*P3(2);
dXdu = jacobian(Xb,u); dYdu = jacobian(Yb,u);
phi_b = atan2(dYdu,dXdu);
X_ref = Function('X_ref',{u},{Xb});
Y_ref = Function('Y_ref',{u},{Yb});
phi_ref = Function('phi_ref',{u},{phi_b});
dXdu_f = Function('dXdu',{u},{dXdu});
dYdu_f = Function('dYdu',{u},{dYdu});

%% 3. notation
% x = [X; Y; psi; beta; v; r; s]
Xpos = SX.sym('Xpos'); Ypos = SX.sym('Ypos');
psi  = SX.sym('psi');  beta = SX.sym('beta');
v    = SX.sym('v');    r    = SX.sym('r');
s    = SX.sym('s');
x    = [Xpos;Ypos;psi;beta;v;r;s];
% u = [delta; Fdr]
delta = SX.sym('delta');
Fdr   = SX.sym('Fdr');
u_in  = [delta;Fdr];

%% 4. car model parameter
m   = 1500; Jz = 2500;
lf  = 1.2;  lr = 1.3;
wf  = 1.5;  wr = 1.5;
g   = 9.81;
% load
Fz_f = m*g*(lr/(lf+lr))/2;
Fz_r = m*g*(lf/(lf+lr))/2;
% non linear tire model
C1 = 1.2; C2 = 2.0; C3 = 0.3;
eps = 1e-6;

%% 5.  f(x,u)
vx    = v*cos(beta);
vy    = v*sin(beta);
vx_fl = vx - r*(wf/2); vy_fl = vy + r*lf;
vx_fr = vx + r*(wf/2); vy_fr = vy + r*lf;
vx_rl = vx - r*(wr/2); vy_rl = vy - r*lr;
vx_rr = vx + r*(wr/2); vy_rr = vy - r*lr;

alpha_fl = delta   - atan2(vy_fl, vx_fl+eps);
alpha_fr = delta   - atan2(vy_fr, vx_fr+eps);
alpha_rl = 0       - atan2(vy_rl, vx_rl+eps);
alpha_rr = 0       - atan2(vy_rr, vx_rr+eps);

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

% 
Fxf_fl_c = Fxf_fl*cos(delta) - Fyf_fl*sin(delta);
Fyf_fl_c = Fxf_fl*sin(delta) + Fyf_fl*cos(delta);
Fxf_fr_c = Fxf_fr*cos(delta) - Fyf_fr*sin(delta);
Fyf_fr_c = Fxf_fr*sin(delta) + Fyf_fr*cos(delta);
Fxf_rl_c = Fxf_rl;  Fyf_rl_c = Fyf_rl;
Fxf_rr_c = Fxf_rr;  Fyf_rr_c = Fyf_rr;
%total force
Fxf_tot = Fxf_fl_c + Fxf_fr_c + Fxf_rl_c + Fxf_rr_c;
Fyf_tot = Fyf_fl_c + Fyf_fr_c + Fyf_rl_c + Fyf_rr_c;

dXpos = v*cos(psi+beta);
dYpos = v*sin(psi+beta);
dpsi   = r;
% dbeta  = -r + Fyf_tot/(m*(v+eps));
% dv     = Fxf_tot/m;
% dr     = (Fyf_fl_c*lf + Fyf_fr_c*lf - Fyf_rl_c*lr - Fyf_rr_c*lr)/Jz;
% 3) debeta 6)
dbeta = - r + 1/(m*(v+eps))*( ...
    (Fyf_fl_c + Fyf_fr_c)*cos(delta - beta) ...
      + (Fyf_rl_c + Fyf_rr_c)*cos(    - beta) ...
      - (Fdr)*sin(beta) ...
    );

% 4)  Ψ̈ (7)
dr = 1/Jz * ( ...
    Fyf_fl_c*( lf*cos(delta) -  (wf/2)*sin(delta)) ...
  + Fyf_fr_c*( lf*cos(delta) +  (wf/2)*sin(delta)) ...
  + Fyf_rl_c*(-lr*cos(0 ) -  (wr/2)*sin(0)) ...  % δ_r = 0
  + Fyf_rr_c*(-lr*cos(0 )+  (wr/2)*sin(0)) ...
  + Fxf_fl_c*(           sin(delta)*lf) ...         % 
  + Fxf_fr_c*(           sin(delta)*lf) ...
);

% 5)  v̇ (formula 8)
dv = 1/m*( ...
    (Fyf_fl_c + Fyf_fr_c)*sin(beta - delta) ...
  + (Fyf_rl_c + Fyf_rr_c)*sin(beta      ) ...
  + (Fdr)*cos(beta) ...
);

% path parameter
dX = dXdu_f(s); dY = dYdu_f(s);
Lp = sqrt(dX^2+dY^2)+eps;
cosr = cos(psi+beta)*(dX/Lp) + sin(psi+beta)*(dY/Lp);
ds   = v*cosr / Lp;


xdot = [dXpos; dYpos; dpsi; dbeta; dv; dr; ds];

f_dyn = Function('f_dyn',{x,u_in},{xdot});

%% 6. Euler 
xk = SX.sym('xk',7);
uk = SX.sym('uk',2);
xnext = xk + dt*f_dyn(xk,uk);
euler_step = Function('euler_step',{xk,uk},{xnext});

%% 7. Opti
opti = Opti();
Xvar = opti.variable(7, N+1);
Uvar = opti.variable(2, N);

X_p = Xvar(1,:);
Y_p = Xvar(2,:);
psi_v= Xvar(3,:);
beta_v= Xvar(4,:);
v_v  = Xvar(5,:);
r_v  = Xvar(6,:);
s_v  = Xvar(7,:);

%% 8. initial
opti.subject_to(X_p(1)==P0(1));
opti.subject_to(Y_p(1)==P0(2));
opti.subject_to(s_v(1)==0);
% opti.subject_to(psi_v(1)==phi_ref(0));
% opti.subject_to(beta_v(1)==0);
% % opti.subject_to(v_v(1)==10);
% opti.subject_to(r_v(1)==0);
opti.subject_to(v_v >= 0);

%% 9. dynamics (Euler)
for k=1:N
    xk = Xvar(:,k);
    uk = Uvar(:,k);
    xk1 = euler_step(xk,uk);
    opti.subject_to(Xvar(:,k+1)==xk1);
end

%%10. assure s=1 end status）
opti.subject_to(s_v(N+1)==1);
for k = 1:N
    opti.subject_to(s_v(k+1) >= s_v(k));
end

margin = 4;  % [m] max. derivations
for k = 1:N+1
    % 1) 
    s_k = s_v(k);
    xr  = X_ref(s_k); 
    yr  = Y_ref(s_k);
    % 2) 
    dXu = dXdu_f(s_k);  dYu = dYdu_f(s_k);
    Lp  = sqrt(dXu^2 + dYu^2) + eps;
    sinp = dYu/Lp;  cosp = dXu/Lp;
    % 3)  e_k
    e_k = -(X_p(k)-xr)*sinp + (Y_p(k)-yr)*cosp;
    % 4) force |e_k| ≤ margin
    opti.subject_to( abs(e_k) <= margin );
end

%11. obj max sum v ⇒ min  -sum(v)
OBJ = 0;
for k=1:N+1
    OBJ = OBJ - v_v(k);
end
opti.minimize(OBJ/N);


%% 12. solve
opti.solver('ipopt',struct('print_time',0,'ipopt',struct('tol',1e-1,'max_iter',1e5)));
sol = opti.solve();

% %% 13. visualization
Xopt = sol.value(Xvar);    % 7×(N+1)，1  X， 2  Y， 5  v
% 
u_plot = linspace(0,1,100);
XbP = full(X_ref(u_plot));
YbP = full(Y_ref(u_plot));

% === corridor visualization ===
n_plot = length(u_plot);
X_corr_left = zeros(1,n_plot);
Y_corr_left = zeros(1,n_plot);
X_corr_right = zeros(1,n_plot);
Y_corr_right = zeros(1,n_plot);

for i = 1:n_plot
    s_i = u_plot(i);
    xr = full(X_ref(s_i));
    yr = full(Y_ref(s_i));
    dXu = full(dXdu_f(s_i));
    dYu = full(dYdu_f(s_i));
    Lp = sqrt(dXu^2 + dYu^2) + 1e-6;
    sinp = dYu / Lp;
    cosp = dXu / Lp;
    
    % 
    X_corr_left(i)  = xr - margin*sinp;
    Y_corr_left(i)  = yr + margin*cosp;
    X_corr_right(i) = xr + margin*sinp;
    Y_corr_right(i) = yr - margin*cosp;
end


figure('Name','Trajectory & Speed','NumberTitle','off');

subplot(2,1,1);
plot(XbP, YbP, 'k--','LineWidth',1.5); hold on;              % ref 
plot(Xopt(1,:), Xopt(2,:), 'b-','LineWidth',2);             % opti trajectory
plot(X_corr_left,  Y_corr_left,  'k--','LineWidth',1);      % laeft boundary
plot(X_corr_right, Y_corr_right, 'k--','LineWidth',1);      % right boundary
axis equal; grid on;
xlabel('X [m]'); ylabel('Y [m]');
title('Reference vs. Optimized Trajectory');
legend('Reference Bezier','Optimized Path','Corridor Boundaries','Location','best');


subplot(2,1,2);
t = 0:dt:N*dt;        % time N+1
v_opt = Xopt(5,:);    % velocity
plot(t, v_opt, 'b-o','LineWidth',1.5,'MarkerSize',4);
grid on;
xlabel('Time [s]'); ylabel('Speed v [m/s]');
title('Speed Profile when maximize speed with fixed time step');