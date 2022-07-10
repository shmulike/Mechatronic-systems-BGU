import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


deg2rad = lambda deg: deg*np.pi/180
rad2deg = lambda rad: rad*180/np.pi
r_sin = lambda t: deg2rad(30)*np.sin(t*0.1) + deg2rad(10)*np.sin(t*0.5) + deg2rad(0)*np.cos(t*0.1-np.pi/2)

class ADAPTIVE_CONTROL:
    def __init__(self):

        self.input_wave = "sin"
        self.r_step_T = 15
        self.r_step_A = deg2rad(20)
        self.r_step = lambda t: 0 if(t <self.r_step_T) else self.r_step_A

        self.t_start = 0
        self.t_end = 40
        self.f_control = 50     # Hz
        self.T_control = 1/self.f_control     # sec
        self.f_sys = 1500     # Hz
        self.T_sys = 1/self.f_sys     # sec
        # dt = 0.01
        self.time_sys = np.arange(self.t_start, self.t_end+self.T_sys, self.T_sys)
        self.time_control = np.arange(self.t_start, self.t_end+self.T_control, self.T_control)
        # time_vec = np.linspace(t_start, t_end, 100)

        # self.gamma, self.lambada, self.Kz = 150, 1, 20
        if self.input_wave == "step":
            self.r_input = np.zeros(self.time_sys.size)
            self.r_input[self.time_sys > self.r_step_T] = self.r_step_A
            self.gamma, self.lambada, self.Kz = 150, 1, 20
        elif self.input_wave == "sin":
            self.r_input = r_sin(self.time_sys)
            self.gamma, self.lambada, self.Kz = 10, 20, 20


        self.c1 = 100
        self.c2 = 1
        self.k = 100
        M = 5
        g = 9.81
        L = 1.5
        I = M*L**2
        self.MgL = M*g*L
        self.h = I/L
        self.k1, self.k2, self.k3 = 6, 20, 20

    def update_control_freq(self, f_control):
        self.f_control = f_control  # Hz
        self.T_control = 1 / self.f_control  # sec
        self.time_control = np.arange(self.t_start, self.t_end + self.T_control, self.T_control)

    def update_system_freq(self, f_sys):
        self.f_sys = f_sys
        self.T_sys = 1 / self.f_sys  # sec
        self.time_sys = np.arange(self.t_start, self.t_end + self.T_sys, self.T_sys)
        # self.time_sys = np.linspace(self.t_start, self.t_end, f_sys*self.t_end, endpoint=True)

    def update_control_laws(self, Kz, gamma, lambada):
        self.Kz = Kz
        self.gamma = gamma
        self.lambada = lambada

    def update_input_type(self, input_type):
        self.input_wave = input_type
        if self.input_wave == "step":
            self.r_input = np.zeros(self.time_sys.size)
            self.r_input[self.time_sys > self.r_step_T] = self.r_step_A
            # self.gamma, self.lambada, self.Kz = 150, 1, 20
        elif self.input_wave == "sin":
            self.r_input = r_sin(self.time_sys)
            # self.gamma, self.lambada, self.Kz = 10, 20, 20

    def get_z_ddxrdt_disc(self, X_vec_full, x_m):
        theta_m_km2, theta_m_km1, theta_m_k = x_m[-3:]
        theta_km1, theta_k = X_vec_full[-2:, 0]

        e_disc_k = theta_k - theta_m_k
        e_disc_km1 = theta_km1 - theta_m_km1
        de_disc_k_dt = (e_disc_k - e_disc_km1) / self.T_control

        z_disc = de_disc_k_dt + self.lambada * e_disc_k
        ddxrdt_disc = (theta_m_k - 2 * theta_m_km1 + theta_m_km2) / (self.T_control ** 2)

        return z_disc, ddxrdt_disc

    def solve(self):
        x_init = [deg2rad(0), 0]

        u_vec_full = []
        u_disc_vec_full = []
        X_vec_full = np.asarray(x_init)
        def mydiff(X, t, u_disc, t0):
            global u_disc_vec_full
            dx1dt = X[1]
            dx2dt = (u_disc - (self.c1*X[1] + self.c2*X[1]**3 + self.k*X[0] - self.MgL*np.sin(X[0])))/self.h
            return [dx1dt, dx2dt]

        # Solve ODE
        u_disc = 0
        time_vec_temp = np.arange(self.time_control[0], self.time_control[3]+1/self.f_sys, 1/self.f_sys)
        X_vec_full = odeint(mydiff, x_init, time_vec_temp, args=(u_disc, 0))
        x_init = X_vec_full[-1, :]
        # X_vec_full = x
        u_disc_vec_full = [u_disc] * time_vec_temp.size
        x_m_vec_full = [0] * time_vec_temp.size

        x_init = X_vec_full[-1, :]
        h_hat, c1_hat, c2_hat, k_hat, MgL_hat = 2, 20, 2, 2, 2

        x_m = [0, 0]
        for i in range(2, self.time_control.size-1):

            theta_km1, theta_k = X_vec_full[-2:, 0]
            dtheta_k_dt = (theta_k-theta_km1)/self.T_control

            x_m_km2, x_m_km1 = x_m[-2:]
            if self.input_wave == "step":
                x_m_k = ((x_m_km1 * (2 + self.k1 * self.T_control) - x_m_km2 + self.k3 * self.r_step(self.time_control[i]) * self.T_control ** 2)) / (
                        1 + self.k2 * self.T_control ** 2 + self.k1 * self.T_control)
            elif self.input_wave == "sin":
                x_m_k = ((x_m_km1 * (2 + self.k1 * self.T_control) - x_m_km2 + self.k3 * r_sin(self.time_control[i]) * self.T_control ** 2)) / (
                    1 + self.k2 * self.T_control ** 2 + self.k1 * self.T_control)
            x_m.append(x_m_k)

            z_disc, ddxrdt_disc = self.get_z_ddxrdt_disc(X_vec_full, x_m)

            # Update parameters
            h_hat   = (-self.gamma * z_disc * ddxrdt_disc) * self.T_control + h_hat
            c1_hat  = (-self.gamma * z_disc * dtheta_k_dt) * self.T_control + c1_hat
            c2_hat  = (-self.gamma * z_disc * dtheta_k_dt**3) * self.T_control + c2_hat
            k_hat   = (-self.gamma * z_disc * theta_k) * self.T_control + k_hat
            MgL_hat = (-self.gamma * z_disc * np.sin(theta_k)) * self.T_control + MgL_hat

            # Calculate the control
            u_disc = h_hat * ddxrdt_disc - \
                     self.Kz * z_disc + \
                     c1_hat * (theta_k - theta_km1) / self.T_control + \
                     c2_hat * ((theta_k - theta_km1) / self.T_control) ** 3 + \
                     k_hat * theta_k - \
                     MgL_hat * np.sin(theta_k)

            # Create the time vector for current step
            time_vec_temp = np.arange(0, self.T_control+1/self.f_sys, 1/self.f_sys)

            x = odeint(mydiff, x_init, time_vec_temp, args=(u_disc, self.time_control[i], ))
            x_init = x[-1, :]
            X_vec_full = np.vstack((X_vec_full, x[1:, :]))


            u_disc_vec_full.extend([u_disc]*(time_vec_temp.size-1))
            x_m_vec_full.extend([x_m_k]*(time_vec_temp.size-1))



        x = X_vec_full[:self.time_sys.size, :]
        # x = np.vstack(self.time_sys, x)

        x_m = x_m_vec_full[:self.time_sys.size]



        return self.time_sys, x, x_m, u_disc_vec_full[:self.time_sys.size]

    def set_t_end(self, t_end):
        self.t_end = t_end


