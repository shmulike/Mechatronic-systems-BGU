import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from Part_5_class import ADAPTIVE_CONTROL

input_type = "sin"
_rad2deg = 180/np.pi
sim = ADAPTIVE_CONTROL()
sim.update_input_type(input_type)
init_freq = 100
init_Kz = 10
init_gamma = 10
init_lambada = 10

def f(feq, Kz, gamma, lambada):
    sim.update_control_freq(feq)
    sim.update_control_laws(Kz, gamma, lambada)
    return sim.solve()

time_sys, x, x_m, u = f(init_freq, init_Kz, init_gamma, init_lambada)
fig1, ax1 = plt.subplots()
line1, = plt.plot(time_sys, x[:, 0]*_rad2deg, label=r"System $\theta$")
line2, = plt.plot(time_sys, np.asarray(x_m)*_rad2deg, '--', label=r"Reference System $\theta_m$")
line3, = plt.plot(time_sys, sim.r_input*_rad2deg, label=r"Input signal $r$")
ax1.set_ylabel(r"Position $\theta$ (rad)")
ax1.set_xlabel('Time (sec)')
ax1.grid()
ax1.legend()
plt.subplots_adjust(left=0.25, bottom=0.25)

# plt.xlim([sim.t_start, sim.t_end])


# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0, 0.55, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Controller Frequency [Hz]',
    valmin=1,
    valmax=750,
    # valstep=[1, 2, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000],
    valstep=[2,3,4,5,6,10,12,15,20,25,30,50,60,75,100,125,150,250,300,375,500,750],
    valinit=init_freq,
)

axkz = plt.axes([0.25, 0.05, 0.55, 0.03])
kz_slider = Slider(
    ax=axkz,
    label='$K_z$',
    valmin=1,
    valmax=50,
    valstep = 1,
    valinit=init_Kz,
)

axgamma = plt.axes([0.25, 0.1, 0.55, 0.03])
gamma_slider = Slider(
    ax=axgamma,
    label=r"$\gamma$ Gamma",
    valmin=1,
    valmax=200,
    valstep=1,
    valinit=init_gamma,
)

axlambada = plt.axes([0.25, 0.15, 0.55, 0.03])
lambada_slider = Slider(
    ax=axlambada,
    label=r"$\lambda$ Lamda",
    valmin=1,
    valmax=200,
    valstep=1,
    valinit=init_lambada,
)
# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.03])
button = Button(resetax, 'Reset', hovercolor='0.975')

fig2, ax2 = plt.subplots()
line4, = plt.plot(time_sys, x[:, 0]-x_m)
ax2.set_ylabel(r"Position error $e_{\theta}$ (deg)")
ax2.set_xlabel('Time (sec)')
ax2.grid()

fig3, ax3 = plt.subplots()
line5, = plt.plot(time_sys, u)
ax3.set_ylabel(r"Control value $F$ (N)")
ax3.set_xlabel('Time (sec)')
ax3.grid()

# The function to be called anytime a slider's value changes
def update(val):
    # fff2 = lambda sys, cont: sys - sys % cont
    # f_sys_new = fff2(sim.f_sys, freq_slider.val)
    # sim.update_system_freq(f_sys_new)
    # sim.update_input_type(input_type)

    # fff = lambda sys, cont: cont - sys % cont
    # freq_slider.eventson = False
    # freq_slider.set_val(fff(sim.f_sys, val))
    # freq_slider.eventson = True

    time_sys, x, x_m, u = f(freq_slider.val, kz_slider.val, gamma_slider.val, lambada_slider.val)
    line1.set_ydata(x[:, 0]*_rad2deg)
    line2.set_ydata(np.asarray(x_m)*_rad2deg)
    fig1.canvas.draw_idle()

    line4.set_ydata(np.asarray(x[:, 0]-x_m)*_rad2deg)
    fig2.canvas.draw_idle()

    line5.set_ydata(u)
    fig3.canvas.draw_idle()

# register the update function with each slider
freq_slider.on_changed(update)
kz_slider.on_changed(update)
gamma_slider.on_changed(update)
lambada_slider.on_changed(update)



def reset(event):
    freq_slider.reset()
    kz_slider.reset()
    gamma_slider.reset()
    gamma_slider.reset()

button.on_clicked(reset)
# time_sys, x, x_m, u_disc_vec_full = sim.solve()
# fig1 = plt.figure(1)
# plt.plot(time_sys, x[:, 0], label=r"System $\theta$")
# plt.plot(time_sys, x_m, '--', label=r"Reference System $\theta_m$")
# plt.plot(time_sys, sim.r_input, label=r"Input signal $r$")
# plt.ylabel(r"Position $\theta$ (rad)")
# plt.xlim([sim.t_start, sim.t_end])
# plt.legend()
# plt.grid()
# plt.title("System response")



plt.show()



