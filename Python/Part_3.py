import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

R_input = "sin"

_rad2deg = 180/np.pi
deg2rad = lambda deg: deg*np.pi/180
rad2deg = lambda rad: rad*180/np.pi
r_sin = lambda t: deg2rad(10)*np.sin(t*0.1) + deg2rad(5)*np.sin(t*0.2) + deg2rad(5)*np.cos(t*0.5-np.pi/2)
r_step = lambda t, T0, A: 0 if(t < T0) else deg2rad(A)

t_start = 0
t_end = 40
dt = 0.005
time_vec = np.arange(t_start, t_end+dt, dt)
# time_vec = np.linspace(t_start, t_end, 100)
r_step_T0 = 5
r_step_A = 10
if R_input == "step":
    r_input = np.zeros(time_vec.size)
    r_input[time_vec > r_step_T0] = deg2rad(r_step_A)
elif R_input == "sin":
    r_input = r_sin(time_vec)
else:
    print("Error R_input")
    exit()
lambada = 100
Kz = 10
c1 = 4
c2 = 0.4
k = 2
M = 2
g = 9.81
L = 1.5
I = M*L**2

h = I/L
k1, k2, k3 = 2, 10, 10

x_init = [deg2rad(0), 0, deg2rad(0), 0]


u_vec_full = []
t_vec_full = []

# y_m1 = np.zeros(time_vec.size)
# y_m1[time_vec >= 10] = deg2rad(30)
# y_m2 = 0.5*np.sin(time_vec*0.1) + 0.2*np.cos(time_vec*0.5)


# def mydiff(X, t, lambada, Kz):
def calc_z_ddxr(t, x, x_m):
    e = x[0] - x_m[0]
    dedt = x[1] - x_m[1]

    if R_input == "step":
        dxm2dt = -k1 * x_m[1] - k2 * x_m[0] + k3 * r_step(t, r_step_T0, r_step_A)
    elif R_input == "sin":
        dxm2dt = -k1 * x_m[1] - k2 * x_m[0] + k3 * r_sin(t)

    ddxrdt = dxm2dt - lambada * dedt
    z = dedt + lambada * e
    return z, ddxrdt, dxm2dt

def mydiff(X, t):
    global u_vec_full, t_vec_full

    x = [X[0], X[1]]
    x_m = [X[2], X[3]]

    z, ddxrdt, dxm2dt = calc_z_ddxr(t, x, x_m)
    u = h*ddxrdt - Kz*z + c1*x[1] + c2*x[1]**3 + k*x[0] - M*g*L*np.sin(x[0])

    dxm1dt = x_m[1]
    # if R_input == "step":
    #     dxm2dt = -k1 * x_m[1] - k2 * x_m[0] + k3 * r_step(t, r_step_T0, r_step_A)
    # elif R_input == "sin":
    #     dxm2dt = -k1 * x_m[1] - k2 * x_m[0] + k3 * r_sin(t)

    dx1dt = x[1]
    dx2dt = (u - (c1*x[1] + c2*x[1]**3 + k*x[0] - M*g*L*np.sin(x[0])))/h


    u_vec_full.append(u)
    t_vec_full.append(t)

    return [dx1dt, dx2dt, dxm1dt, dxm2dt]

# Solve ODE
x = odeint(mydiff, x_init, time_vec)
# x = odeint(mydiff, x_init, time_vec, args=(lambada, Kz,))

x1 = x[:, 0]
x2 = x[:, 1]
x_m1 = x[:, 2]
x_m2 = x[:, 3]
# u = x[:, 4]
#  Plot

_fontsize = 22
_linewidth = 5
fig, axs = plt.subplots(2, 2, sharex=True)

axs[0, 0].plot(time_vec, x1*_rad2deg, label=r"System $\theta$", linewidth=_linewidth)
axs[0, 0].plot(time_vec, x_m1*_rad2deg, '--', label=r"Reference System $\theta_m$", linewidth=_linewidth)
axs[0, 0].plot(time_vec, r_input*_rad2deg, label=r"Input signal $r$", linewidth=_linewidth, zorder=0)
axs[0, 0].set_ylabel(r"Position $\theta$ (deg)", fontsize=_fontsize)
axs[0, 0].set_xlim([t_start, t_end])
axs[0, 0].legend()
axs[0, 0].grid()
axs[0, 0].set_title("System response", fontsize=_fontsize)
axs[0, 0].tick_params(axis='y', labelsize=_fontsize)
axs[0, 0].tick_params(axis='x', labelsize=_fontsize)

axs[1, 0].plot(time_vec, x2*_rad2deg, label=r"System $\theta$", linewidth=_linewidth)
axs[1, 0].plot(time_vec, x_m2*_rad2deg, '--', label=r"Reference System $\theta_m$", linewidth=_linewidth)
axs[1, 0].set_ylabel(r"Velocity $\dot{\theta} (\frac{deg}{sec})$", fontsize=_fontsize)
axs[1, 0].set_xlabel("Time (sec)", fontsize=_fontsize)
axs[1, 0].legend()
axs[1, 0].grid()
axs[1, 0].set_title("System response", fontsize=_fontsize)
axs[1, 0].tick_params(axis='y', labelsize=_fontsize)
axs[1, 0].tick_params(axis='x', labelsize=_fontsize)

axs[0, 1].plot(time_vec, (x1-x_m1)*_rad2deg, linewidth=_linewidth)
axs[0, 1].set_ylabel(r"Position error $e_{\theta}$ (deg)", fontsize=_fontsize)
axs[0, 1].set_xlim([t_start, t_end])
axs[0, 1].grid()
axs[0, 1].set_title("Position error", fontsize=_fontsize)
axs[0, 1].tick_params(axis='y', labelsize=_fontsize)
axs[0, 1].tick_params(axis='x', labelsize=_fontsize)

A = np.array([t_vec_full, u_vec_full])
i = np.argsort(A)[0, :]
A = A[:, i]
t_vec_full = A[0, :]
u_vec_full = A[1, :]
u_vec_full = savgol_filter(u_vec_full, 10, 3)
axs[1, 1].plot(t_vec_full, u_vec_full, linewidth=_linewidth)
axs[1, 1].set_xlabel("Time (sec)", fontsize=_fontsize)
axs[1, 1].set_ylabel("Force (N)", fontsize=_fontsize)
axs[1, 1].grid()
axs[1, 1].set_title("Control output", fontsize=_fontsize)
axs[1, 1].tick_params(axis='y', labelsize=_fontsize)
axs[1, 1].tick_params(axis='x', labelsize=_fontsize)
# fig2 = plt.figure(2)
# plt.plot(time_vec, x1-x_m1)
# plt.ylabel(r"Position error $e_{\theta}$ (deg)")
# plt.xlabel("Time (sec)")
# plt.xlim([t_start, t_end])
# plt.grid()

plt.show()

