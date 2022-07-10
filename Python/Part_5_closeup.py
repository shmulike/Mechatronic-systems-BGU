import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from Part_5_class import ADAPTIVE_CONTROL

input_type = "sin"
_rad2deg = 180 / np.pi
sim = ADAPTIVE_CONTROL()
sim.update_input_type(input_type)
init_freq = 100
init_Kz = 50
init_gamma = 50
init_lambada = 50

sim.update_control_laws(init_Kz, init_gamma, init_lambada)
# sim.set_t_end(10)
t_end = 1
fig1 = plt.figure(1)
fig2 = plt.figure(2)
fig3 = plt.figure(3)
_fontsize = 20
_fontsize_legend = 12
_linewidth = 3

# [2,3,4,5,6,10,12,15,20,25,30,50,60,75,100,125,150,250,300,375,500,750]
f_control_vec = np.array([10, 15, 20, 25, 50, 100])
f_control_vec = np.array([10, 15, 25])
f_control_vec = np.flip(f_control_vec)
color_plot = ['b', 'g', 'm', 'y', 'c']

# Descrete
for i in range(f_control_vec.size):
    # f_now = f_control_vec[i]
    print(f_control_vec[i])
    sim.update_control_freq(f_control_vec[i])
    time_sys, x, x_m, u_disc_vec_full = sim.solve()

    plt.figure(1)
    plt.plot(time_sys, x[:, 0] * _rad2deg, '-', color=color_plot[i], label=r"System response $\theta$ f={}Hz".format(f_control_vec[i]),
             linewidth=_linewidth)
    plt.plot(time_sys, np.asarray(x_m) * _rad2deg, '--', color=color_plot[i],
             label=r"Reference System $\theta_m$ f={}Hz".format(f_control_vec[i]), linewidth=_linewidth)

    plt.figure(2)
    plt.plot(time_sys, np.asarray(x_m - x[:, 0]) * _rad2deg, label=r"Position error  f={}Hz".format(f_control_vec[i]),
             linewidth=_linewidth)

    plt.figure(3)
    plt.plot(time_sys, u_disc_vec_full, label=r"Control signal f={}Hz".format(f_control_vec[i]), linewidth=_linewidth)

# Continius
sim.update_control_freq(1500)
time_sys, x, x_m, u_disc_vec_full = sim.solve()

plt.figure(1)
plt.plot(time_sys, x[:, 0] * _rad2deg, 'k-', label=r"Continues system response $\theta$", linewidth=_linewidth)
plt.plot(time_sys, np.asarray(x_m) * _rad2deg, 'k--', label=r"Continues reference System $\theta_m$",
         linewidth=_linewidth)

plt.figure(2)
plt.plot(time_sys, np.asarray(x_m - x[:, 0]) * _rad2deg, 'k', label=r"Continues position error", linewidth=_linewidth)

plt.figure(3)
plt.plot(time_sys, u_disc_vec_full, 'k', label=r"Continues control signal", linewidth=_linewidth)

# generacnfurations
plt.figure(1)
plt.plot(time_sys, sim.r_input * _rad2deg, label=r"Input signal $r$")
plt.ylabel(r"Position $\theta$ (deg)", fontsize=_fontsize)
plt.xlabel("Time (sec)", fontsize=_fontsize)
plt.xlim([0, t_end])
plt.ylim([-2, 8])
plt.legend(fontsize=_fontsize)
plt.grid()
plt.title("System response", fontsize=_fontsize)
plt.xticks(fontsize=_fontsize)
plt.yticks(fontsize=_fontsize)

plt.figure(2)
plt.title("Error", fontsize=_fontsize)
plt.xlabel("Time (sec)", fontsize=_fontsize)
plt.ylabel(r"Position error $e_{\theta}$ (deg)", fontsize=_fontsize)
plt.legend(fontsize=_fontsize)
plt.grid()
plt.xlim([0, t_end])
plt.ylim([-2, 2])
plt.xticks(fontsize=_fontsize)
plt.yticks(fontsize=_fontsize)

plt.figure(3)
plt.title("Control signal", fontsize=_fontsize)
plt.xlabel("Time (sec)", fontsize=_fontsize)
plt.ylabel(r"Control signal $F$ (N)", fontsize=_fontsize)
plt.legend(fontsize=_fontsize)
plt.grid()
plt.xlim([0, t_end])
plt.ylim([-1, 25])
plt.xticks(fontsize=_fontsize)
plt.yticks(fontsize=_fontsize)

plt.show()
