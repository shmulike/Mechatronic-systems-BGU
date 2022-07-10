import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

input_wave = "sin"

deg2rad = lambda deg: deg*np.pi/180
rad2deg = lambda rad: rad*180/np.pi
r_sin = lambda t: deg2rad(30)*np.sin(t*0.1) + deg2rad(10)*np.sin(t*0.5) + deg2rad(0)*np.cos(t*0.1-np.pi/2)

r_step_T = 15
r_step_A = deg2rad(20)
r_step = lambda t: 0 if(t <r_step_T) else r_step_A

t_start = 0
t_end = 40
f_control = 195     # Hz
f_sys = 1000     # Hz
T_control = 1/f_control     # sec
T_sys = 1/f_sys     # sec
# dt = 0.01
time_sys = np.arange(t_start, t_end+T_sys, T_sys)
time_control = np.arange(t_start, t_end+T_control, T_control)
# time_vec = np.linspace(t_start, t_end, 100)

if input_wave == "step":
    r_input = np.zeros(time_sys.size)
    r_input[time_sys > r_step_T] = r_step_A
    gamma, lambada, Kz = 150, 1, 20
elif input_wave == "sin":
    r_input = r_sin(time_sys)
    gamma, lambada, Kz = 10, 20, 20


c1 = 4
c2 = 2
k = 2
M = 2
g = 9.81
L = 1.5
I = M*L**2
MgL = M*g*L
h = I/L
k1, k2, k3 = 6, 20, 20

x_init = [deg2rad(0), 0]

u_vec_full = []
u_disc_vec_full = []
X_vec_full = np.asarray(x_init)
def mydiff(X, t, u_disc, t0):
    global u_disc_vec_full
    dx1dt = X[1]
    dx2dt = (u_disc - (c1*X[1] + c2*X[1]**3 + k*X[0] - MgL*np.sin(X[0])))/h
    return [dx1dt, dx2dt]

# Solve ODE
u_disc = 0
time_vec_temp = np.arange(time_control[0], time_control[3]+1/f_sys, 1/f_sys)
X_vec_full = odeint(mydiff, x_init, time_vec_temp, args=(u_disc, 0))
x_init = X_vec_full[-1, :]
# X_vec_full = x
u_disc_vec_full = [u_disc] * time_vec_temp.size
x_m_vec_full = [0] * time_vec_temp.size

x_init = X_vec_full[-1, :]

h_hat, c1_hat, c2_hat, k_hat, MgL_hat = 2,2,2,2,2

def get_z_ddxrdt_disc(X_vec_full, x_m):
    theta_m_km2, theta_m_km1, theta_m_k = x_m[-3:]
    theta_km1, theta_k = X_vec_full[-2:, 0]

    e_disc_k = theta_k - theta_m_k
    e_disc_km1 = theta_km1 - theta_m_km1
    de_disc_k_dt = (e_disc_k - e_disc_km1) / T_control

    z_disc = de_disc_k_dt + lambada * e_disc_k
    ddxrdt_disc = (theta_m_k - 2 * theta_m_km1 + theta_m_km2) / (T_control ** 2)

    return z_disc, ddxrdt_disc

x_m = [0, 0]
for i in range(2, time_control.size-1):

    theta_km1, theta_k = X_vec_full[-2:, 0]
    dtheta_k_dt = (theta_k-theta_km1)/T_control

    x_m_km2, x_m_km1 = x_m[-2:]
    if input_wave == "step":
        x_m_k = ((x_m_km1 * (2 + k1 * T_control) - x_m_km2 + k3 * r_step(time_control[i]) * T_control ** 2)) / (
                1 + k2 * T_control ** 2 + k1 * T_control)
    elif input_wave == "sin":
        x_m_k = ((x_m_km1 * (2 + k1 * T_control) - x_m_km2 + k3 * r_sin(time_control[i]) * T_control ** 2)) / (
            1 + k2 * T_control ** 2 + k1 * T_control)
    x_m.append(x_m_k)

    z_disc, ddxrdt_disc = get_z_ddxrdt_disc(X_vec_full, x_m)

    # Update parameters
    h_hat   = (-gamma * z_disc * ddxrdt_disc)*T_control + h_hat
    c1_hat  = (-gamma * z_disc * dtheta_k_dt)*T_control + c1_hat
    c2_hat  = (-gamma * z_disc * dtheta_k_dt**3)*T_control + c2_hat
    k_hat   = (-gamma * z_disc * theta_k)*T_control + k_hat
    MgL_hat = (-gamma * z_disc * np.sin(theta_k))*T_control + MgL_hat

    # Calculate the control
    u_disc = h_hat * ddxrdt_disc - \
             Kz * z_disc + \
             c1_hat * (theta_k - theta_km1) / T_control + \
             c2_hat * ((theta_k - theta_km1) / T_control) ** 3 + \
             k_hat * theta_k - \
             MgL_hat * np.sin(theta_k)
    # u_disc = 0

    # Create the time vector for current step
    time_vec_temp = np.arange(0, T_control+1/f_sys, 1/f_sys)

    x = odeint(mydiff, x_init, time_vec_temp, args=(u_disc, time_control[i], ))
    x_init = x[-1, :]
    X_vec_full = np.vstack((X_vec_full, x[1:, :]))


    u_disc_vec_full.extend([u_disc]*(time_vec_temp.size-1))
    x_m_vec_full.extend([x_m_k]*(time_vec_temp.size-1))
    # u_disc_vec_full.append(u_disc)
# x = odeint(mydiff, x_init, time_vec, args=(lambada, Kz,))

# u_disc_vec_full.append(u_disc)
# u_disc_vec_full.append(u_disc)
# time_vec = time_vec[:-1]
# r_input = r_input[:-1]
x = X_vec_full[:time_sys.size, :]
# time_sys = time_sys[:X_vec_full.__len__()]
# r_input = r_input[:X_vec_full.__len__()]

x1 = x[:, 0]
x2 = x[:, 1]
x_m = x_m_vec_full[:time_sys.size]
# x_m2 = x_m_vec_full[:, 1]

#  Plot
fig, axs = plt.subplots(2, 2, sharex=True)
axs[0, 0].plot(time_sys, x1, label=r"System $\theta$")
axs[0, 0].plot(time_sys, x_m, '--', label=r"Reference System $\theta_m$")
axs[0, 0].plot(time_sys, r_input, label=r"Input signal $r$")
axs[0, 0].set_ylabel(r"Position $\theta$ (rad)")
axs[0, 0].set_xlim([t_start, t_end])
axs[0, 0].legend()
axs[0, 0].grid()
axs[0, 0].set_title("System response")

# axs[1, 0].plot(time_sys, x2, label=r"System $\theta$")
# axs[1, 0].plot(time_sys, x_m2, '--', label=r"Reference System $\theta_m$")
# axs[1, 0].set_ylabel(r"Velocity $\dot{\theta} (\frac{deg}{sec})$")
# axs[1, 0].set_xlabel("Time (sec)")
# axs[1, 0].legend()
# axs[1, 0].grid()
# axs[1, 0].set_title("System response")

axs[0, 1].plot(time_sys, x1-x_m)
axs[0, 1].set_ylabel(r"Position error $e_{\theta}$ (rad)")
axs[0, 1].set_xlim([t_start, t_end])
axs[0, 1].grid()
axs[0, 1].set_title("Position error")

# A = np.array([t_vec_full, u_vec_full])
# i = np.argsort(A)[0, :]
# A = A[:, i]
# t_vec_full = A[0, :]
# u_vec_full = A[1, :]
# u_vec_full = savgol_filter(u_vec_full, 10, 3)
# axs[1, 1].plot(time_control[:-1], u_disc_vec_full, '*-')
axs[1, 1].plot(time_sys, u_disc_vec_full[:time_sys.size])
axs[1, 1].set_xlabel("Time (sec)")
axs[1, 1].set_ylabel("Force (N)")
axs[1, 1].grid()
axs[1, 1].set_title("Control output")


# fig2 = plt.figure(2)
# plt.plot(time_control, np.ones(time_control.size)*h, 'b', label=r"$\hat{h}$")
# plt.plot(time_control, x[:, 4], 'b--', label=r"h")
# plt.plot(time_control, np.ones(time_vec.size)*c1, 'k', label=r"$\hat{c}_1$")
# plt.plot(time_control, x[:, 5], 'k--', label=r"c_1")
# plt.plot(time_control, np.ones(time_vec.size)*c2, 'r', label=r"$\hat{c}_2$")
# plt.plot(time_control, x[:, 6], 'r--', label=r"c_2")
# plt.plot(time_control, np.ones(time_vec.size)*k, 'g', label=r"$\hat{k}$")
# plt.plot(time_control, x[:, 7], 'g--', label=r"k")
# plt.plot(time_control, np.ones(time_vec.size)*MgL, 'm', label=r"$\hat{MgL}$")
# plt.plot(time_control, x[:, 8], 'm--', label=r"MgL")
# plt.grid()

plt.show()

