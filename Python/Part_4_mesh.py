import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

R_input = "sin"


_rad2deg = 180/np.pi
deg2rad = lambda deg: deg*np.pi/180
rad2deg = lambda rad: rad*180/np.pi
r_sin = lambda t: deg2rad(10)*np.sin(t*0.1) + deg2rad(5)*np.sin(t*1) + deg2rad(5)*np.cos(t*0.5-np.pi/2)
r_step = lambda t, T, A: 0 if(t < T) else deg2rad(A)

t_start = 0
t_end = 20
dt = 0.005
N_time = int((1/dt)*(t_end-t_start))
# time_vec = np.arange(t_start, t_end+dt, dt)
time_vec = np.linspace(t_start, t_end, N_time, endpoint=True)
# time_vec = np.linspace(t_start, t_end, 100)

r_step_T0 = 2
r_step_A = 20
if R_input == "step":
    r_input = np.zeros(time_vec.size)
    r_input[time_vec > r_step_T0] = deg2rad(r_step_A)
elif R_input == "sin":
    r_input = r_sin(time_vec)
else:
    print("Error R_input")
    exit()

N_mesh = 5
gamma = np.linspace(5, 100, N_mesh, endpoint=True)
lambada = np.linspace(5, 100, N_mesh, endpoint=True)
Kz_mesh = np.linspace(10, 200, N_mesh, endpoint=True)
Kz = 200
c1 = 4
c2 = 0.4
k = 2
M = 2
g = 9.81
L = 1.5
I = M*L**2
MgL = M*g*L
h = I/L
k1, k2, k3 = 2, 10, 10

# x_init = [deg2rad(0), 0, deg2rad(0), 0, 5, 5, 5, 5, 5]
x_init = [deg2rad(0), 0, deg2rad(0), 0, 0,0,0,0,0]

# def mydiff(X, t, lambada, Kz):
def mydiff_mesh(X, t, gamma, lambada, Kz):
    global u_vec_full, t_vec_full


    x = [X[0], X[1]]
    x_m = [X[2], X[3]]

    h_hat, c1_hat, c2_hat, k_hat, MgL_hat = X[4:]
    # h_hat_sgn = h_hat/abs(h_hat)

    dxm1dt = x_m[1]
    # dxm2dt = -k1 * x_m[1] - k2 * x_m[0] + k3 * r_step(t, r_step_T, r_step_A)
    if R_input == "step":
        dxm2dt = -k1 * x_m[1] - k2 * x_m[0] + k3 * r_step(t, r_step_T0, r_step_A)
    elif R_input == "sin":
        dxm2dt = -k1 * x_m[1] - k2 * x_m[0] + k3 * r_sin(t)

    e = x[0] - x_m[0]
    dedt = x[1] - x_m[1]

    # dxmdt = [dxm1dt, dxm2dt]
    ddxrdt = dxm2dt - lambada*dedt
    z = dedt + lambada * e

    h_hat_sgn = 1
    dh_hat   = -gamma*h_hat_sgn*z * ddxrdt
    dc1_hat  = -gamma*h_hat_sgn*z * x[1]
    dc2_hat  = -gamma*h_hat_sgn*z * x[1]**3
    dk_hat   = -gamma*h_hat_sgn*z * x[0]
    dMgL_hat = -gamma*h_hat_sgn*z * np.sin(x[0])

    u = h_hat*ddxrdt - Kz*z + c1_hat*x[1] + c2_hat*x[1]**3 + k_hat*x[0] - MgL_hat*np.sin(x[0])
    if abs(u) > 20:
        u = 20*u/abs(u)

    dx1dt = x[1]
    dx2dt = (u - (c1*x[1] + c2*x[1]**3 + k*x[0] - MgL*np.sin(x[0])))/h
    # dxdt = [dx1dt, dx2dt]

    u_vec_full.append(u)
    e_vec_full.append(e)
    t_vec_full.append(t)

    return [dx1dt, dx2dt, dxm1dt, dxm2dt, dh_hat, dc1_hat, dc2_hat, dk_hat, dMgL_hat]

# Solve ODE


# // ------------------ Check lambada -----------------------
gamma_now = 10
u_mesh = []
e_mesh = []
t_mesh = []

for i in range(N_mesh):
    u_vec_full = []
    e_vec_full = []
    t_vec_full = []
    x = odeint(mydiff_mesh, x_init, time_vec, args=(gamma_now, lambada[i], Kz))

    u_mesh.append(u_vec_full)
    e_mesh.append(e_vec_full)
    t_mesh.append(t_vec_full)


_fontsize = 20
_linewidth = 3
fig1, axs1 = plt.subplots(2, sharex=True)
fig1.suptitle(r"System Error and control output: $Kz={:.0f}, \gamma={:.1f}$".format(Kz, gamma_now), fontsize=_fontsize)
for i in range(N_mesh):
    axs1[0].plot(t_mesh[i], np.asarray(e_mesh[i])*_rad2deg, label=r"$\lambda$  = {}".format(lambada[i]), linewidth=_linewidth)
    axs1[0].grid()
    axs1[0].legend(fontsize=_fontsize)
    axs1[0].set_ylabel(r"Position error $e_{\theta}$ (deg)", fontsize=_fontsize)
    axs1[0].tick_params(axis='y', labelsize=_fontsize)
    axs1[0].tick_params(axis='x', labelsize=_fontsize)

    axs1[1].plot(t_mesh[i], u_mesh[i], label=r"$\lambda$  = {}".format(lambada[i]), linewidth=_linewidth)
    axs1[1].grid()
    axs1[1].legend()
    axs1[1].set_xlabel("Time (sec)", fontsize=_fontsize)
    axs1[1].set_ylabel("Control output $F$ (N)", fontsize=_fontsize)
    axs1[1].tick_params(axis='y', labelsize=_fontsize)
    axs1[1].tick_params(axis='x', labelsize=_fontsize)

# // ------------------ Check gamma -----------------------
lamda_now = 20
u_mesh = []
e_mesh = []
t_mesh = []
for i in range(N_mesh):
    u_vec_full = []
    e_vec_full = []
    t_vec_full = []
    x = odeint(mydiff_mesh, x_init, time_vec, args=(gamma[i], lamda_now, Kz))

    u_mesh.append(u_vec_full)
    e_mesh.append(e_vec_full)
    t_mesh.append(t_vec_full)

fig2, axs2 = plt.subplots(2, sharex=True)
fig2.suptitle(r"System Error and control output: $Kz={:.0f}, \lambda ={:.1f}$".format(Kz, lamda_now), fontsize=_fontsize)
for i in range(N_mesh):
    axs2[0].plot(t_mesh[i], np.asarray(e_mesh[i])*_rad2deg, label=r"$\gamma$ = {}".format(gamma[i]), linewidth=_linewidth)
    axs2[0].grid()
    axs2[0].legend(fontsize=_fontsize)
    axs2[0].set_ylabel(r"Position error $e_{\theta}$ (deg)", fontsize=_fontsize)
    axs2[0].tick_params(axis='y', labelsize=_fontsize)
    axs2[0].tick_params(axis='x', labelsize=_fontsize)

    axs2[1].plot(t_mesh[i], u_mesh[i], label=r"$\gamma$ = {}".format(gamma[i]), linewidth=_linewidth)
    axs2[1].grid()
    axs2[1].legend(fontsize=_fontsize)
    axs2[1].set_xlabel("Time (sec)", fontsize=_fontsize)
    axs2[1].set_ylabel("Control output $F$ (N)", fontsize=_fontsize)
    axs2[1].tick_params(axis='y', labelsize=_fontsize)
    axs2[1].tick_params(axis='x', labelsize=_fontsize)

# // ------------------ Check Kz -----------------------
lamda_now = 20
gamma_now = 10
u_mesh = []
e_mesh = []
t_mesh = []
for i in range(N_mesh):
    u_vec_full = []
    e_vec_full = []
    t_vec_full = []
    x = odeint(mydiff_mesh, x_init, time_vec, args=(gamma_now, lamda_now, Kz_mesh[i]))

    u_mesh.append(u_vec_full)
    e_mesh.append(e_vec_full)
    t_mesh.append(t_vec_full)

fig3, axs3 = plt.subplots(2, sharex=True)
fig3.suptitle(r"System Error and control output: $\gamma={:.0f}, \lambda ={:.1f}$".format(gamma_now, lamda_now), fontsize=_fontsize)
for i in range(N_mesh):
    axs3[0].plot(t_mesh[i], np.asarray(e_mesh[i])*_rad2deg, label=r"$K_z$ = {}".format(Kz_mesh[i]), linewidth=_linewidth)
    axs3[0].grid()
    axs3[0].legend(fontsize=_fontsize)
    axs3[0].set_ylabel(r"Position error $e_{\theta}$ (deg)", fontsize=_fontsize)
    axs3[0].tick_params(axis='y', labelsize=_fontsize)
    axs3[0].tick_params(axis='x', labelsize=_fontsize)

    axs3[1].plot(t_mesh[i], u_mesh[i], label=r"$K_z$ = {}".format(Kz_mesh[i]), linewidth=_linewidth)
    axs3[1].grid()
    axs3[1].legend(fontsize=_fontsize)
    axs3[1].set_xlabel("Time (sec)", fontsize=_fontsize)
    axs3[1].set_ylabel("Control output $F$ (N)", fontsize=_fontsize)
    axs3[1].tick_params(axis='y', labelsize=_fontsize)
    axs3[1].tick_params(axis='x', labelsize=_fontsize)
# fig4 = plt.figure(11)
# for i in range(N_mesh):
#     plt.plot(gamma[i], max([abs(j) for j in e_mesh[i]]), 'b*')
# plt.grid()
# plt.legend()
# x = odeint(mydiff, x_init, time_vec, args=(lambada, Kz,))

x1 = x[:, 0]
x2 = x[:, 1]
x_m1 = x[:, 2]
x_m2 = x[:, 3]

#  Plot
fig4, axs4 = plt.subplots(2, 2, sharex=True)
axs4[0, 0].plot(time_vec, x1*_rad2deg, label=r"System $\theta$", linewidth=_linewidth)
axs4[0, 0].plot(time_vec, x_m1*_rad2deg, '--', label=r"Reference System $\theta_m$", linewidth=_linewidth)
axs4[0, 0].plot(time_vec, r_input*_rad2deg, label=r"Input signal $r$", linewidth=_linewidth)
axs4[0, 0].set_ylabel(r"Position $\theta$ (deg)", fontsize=_fontsize)
axs4[0, 0].set_xlim([t_start, t_end])
axs4[0, 0].legend()
axs4[0, 0].grid()
axs4[0, 0].set_title("System response", fontsize=_fontsize)
axs4[0, 0].tick_params(axis='y', labelsize=_fontsize)
axs4[0, 0].tick_params(axis='x', labelsize=_fontsize)

axs4[1, 0].plot(time_vec, x2*_rad2deg, label=r"System $\theta$", linewidth=_linewidth)
axs4[1, 0].plot(time_vec, x_m2*_rad2deg, '--', label=r"Reference System $\theta_m$", linewidth=_linewidth)
axs4[1, 0].set_ylabel(r"Velocity $\dot{\theta} (\frac{deg}{sec})$", fontsize=_fontsize)
axs4[1, 0].set_xlabel("Time (sec)", fontsize=_fontsize)
axs4[1, 0].legend()
axs4[1, 0].grid()
axs4[1, 0].set_title("System response", fontsize=_fontsize)
axs4[1, 0].tick_params(axis='y', labelsize=_fontsize)
axs4[1, 0].tick_params(axis='x', labelsize=_fontsize)

axs4[0, 1].plot(time_vec, (x1-x_m1)*_rad2deg, linewidth=_linewidth)
axs4[0, 1].set_ylabel(r"Position error $e_{\theta}$ (deg)", fontsize=_fontsize)
axs4[0, 1].set_xlim([t_start, t_end])
axs4[0, 1].grid()
axs4[0, 1].set_title("Position error", fontsize=_fontsize)
axs4[0, 1].tick_params(axis='y', labelsize=_fontsize)
axs4[0, 1].tick_params(axis='x', labelsize=_fontsize)

A = np.array([t_vec_full, u_vec_full])
i = np.argsort(A)[0, :]
A = A[:, i]
t_vec_full = A[0, :]
u_vec_full = A[1, :]
u_vec_full = savgol_filter(u_vec_full, 10, 3)
axs4[1, 1].plot(t_vec_full, u_vec_full, linewidth=_linewidth)
axs4[1, 1].set_xlabel("Time (sec)", fontsize=_fontsize)
axs4[1, 1].set_ylabel("Force (N)", fontsize=_fontsize)
axs4[1, 1].grid()
axs4[1, 1].set_title("Control output", fontsize=_fontsize)
axs4[1, 1].tick_params(axis='y', labelsize=_fontsize)
axs4[1, 1].tick_params(axis='x', labelsize=_fontsize)

fig5 = plt.figure(5)
plt.plot(time_vec, np.ones(time_vec.size)*h, 'b', label=r"$\hat{h}$", linewidth=_linewidth)
plt.plot(time_vec, x[:, 4], 'b--', label=r"h", linewidth=_linewidth)
plt.plot(time_vec, np.ones(time_vec.size)*c1, 'k', label=r"$\hat{c}_1$", linewidth=_linewidth)
plt.plot(time_vec, x[:, 5], 'k--', label=r"c_1", linewidth=_linewidth)
plt.plot(time_vec, np.ones(time_vec.size)*c2, 'r', label=r"$\hat{c}_2$", linewidth=_linewidth)
plt.plot(time_vec, x[:, 6], 'r--', label=r"c_2", linewidth=_linewidth)
plt.plot(time_vec, np.ones(time_vec.size)*k, 'g', label=r"$\hat{k}$", linewidth=_linewidth)
plt.plot(time_vec, x[:, 7], 'g--', label=r"k", linewidth=_linewidth)
plt.plot(time_vec, np.ones(time_vec.size)*MgL, 'm', label=r"$\hat{MgL}$", linewidth=_linewidth)
plt.plot(time_vec, x[:, 8], 'm--', label=r"MgL", linewidth=_linewidth)
plt.xlabel("Time (sec)", fontsize=_fontsize)
plt.xticks(fontsize=_fontsize)
plt.yticks(fontsize=_fontsize)
plt.grid()

plt.show()

