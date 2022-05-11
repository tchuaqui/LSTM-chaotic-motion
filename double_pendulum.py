from scipy.integrate import odeint 
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import pandas as pd

class Pendulum():
    def __init__(self, t_final, N_t, M, L, g, import_from_csv=None, initial_conditions=None, solution=None):
        """
        Assign properties to double pendulum problem

        initial_conditions: list of size 4 containing positions (rad) and velocities (rad/s) of the masses
        e.g. [theta1, omega1, theta2, omega2]

        t_final: total duration of simulation

        N_t: number of time increments

        M: list of size 2 containing masses (kg)
        e.g. [m1, m2]

        L: list of size 2 containing lengths of connecting rods
        e.g. [l1, l2]

        g: gravitational acceleration (m/s**2)
        """
        self.t_final = t_final
        self.N_t = N_t
        self.M = M
        self.L = L
        self.g = g
        self.t = np.linspace(0, t_final, N_t)
        if import_from_csv is not None:
            # Import existing solution from csv file
            df = pd.read_csv(import_from_csv, index_col=0)
            self.solution = df.to_numpy()
            self.initial_conditions = self.solution[0, :].tolist()
        else:
            if initial_conditions is not None:
                self.initial_conditions = initial_conditions
                self.solution = None
            else:
                if solution is not None:
                    self.solution = solution
                else:
                    raise Exception('Error - Provide either csv file to import existing solution, solution array or list with initial conditions')


    def ode_pendulum(u, t, m1, m2, l1, l2, g):
        """
        Physical model - ordinary differential equations describing the system.
        u - variables
        du - time derivatives of u
        t - time
        """

        du = np.zeros(4)
        c = np.cos(u[0]-u[2])
        s = np.sin(u[0]-u[2])
        du[0] = u[1]   # d(theta1)
        du[1] = ( m2*g*np.sin(u[2])*c - m2*s*(l1*c*u[1]**2 + l2*u[3]**2) - (m1+m2)*g*np.sin(u[0]) ) /( l1 *(m1+m2*s**2) )
        du[2] = u[3]   # d(theta2)
        du[3] = ((m1+m2)*(l1*u[1]**2*s - g*np.sin(u[2]) + g*np.sin(u[0])*c) + m2*l2*u[3]**2*s*c) / (l2 * (m1 + m2*s**2))

        return du

    def solve_physical_model(self):
        """
        Solver of physical model.
        Solution is (N_t + 1)x4 matrix containing positions and velocities of the masses
        e.g. theta1 = solution[:,0], omega1 = solution[:,1]
             theta2 = solution[:,2], omega2 = solution[:,3]
        """
        self.solution = odeint(Pendulum.ode_pendulum, self.initial_conditions, self.t,
                               args=(self.M[0], self.M[1], self.L[0], self.L[1], self.g))

    def save_to_csv(self, path_to_file):
        """
        Save solution to csv file
        """
        if self.solution is None:
            raise Exception('Error - No solution to physical model. Solve physical model first.')
        else:
            df = pd.DataFrame(self.solution)
            df.columns = ['theta1', 'omega1', 'theta2', 'omega2']
            df.to_csv(path_to_file)

    def animation_gen(self, title=None):
        """
        Generate animation (based on https://github.com/zaman13/Double-Pendulum-Motion-Animation)
        """
        if self.solution is None:
            raise Exception('Error - No solution to physical model. Solve physical model first.')
        else:
            # Retrieve positions and velocities from solution
            theta1 = self.solution[:,0]
            omega1 = self.solution[:,1]
            theta2 = self.solution[:,2]
            omega2 = self.solution[:,3]
            t = self.t

            # Map from polar to cartesian coordinate system
            x1 = self.L[0]*np.sin(theta1)
            y1 = -self.L[0]*np.cos(theta1)
            x2 = x1 + self.L[1]*np.sin(theta2)
            y2 = y1 - self.L[1]*np.cos(theta2)

            # Generate plt figure
            fig = plt.figure()
            plt.title(title)
            ax = plt.axes(xlim=(-1.1*(self.L[0]+self.L[1]), 1.1*(self.L[0]+self.L[1])), ylim=(-1.1*(self.L[0]+self.L[1]), 0.5))
            point1, = ax.plot([], [], 'o', color = 'k', markersize = 10)
            point2, = ax.plot([], [], 'o-',color = 'r',markersize = 12, markerfacecolor = 'r',lw=2, markevery=1000, markeredgecolor = 'k')   
            point3, = ax.plot([], [], 'o', color='k', markersize = 10)
            line1, = ax.plot([], [], color='k', linestyle='-', linewidth=2)
            line2, = ax.plot([], [], color='k', linestyle='-', linewidth=2)
            time_template = 'Time = %.1f s'
            time_string = ax.text(0.05, 0.9, '', transform=ax.transAxes)
            # Hide x and y axes ticks
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            def init():
                """
                Initialisation function: plot the background of each frame.
                """
                point1.set_data([], [])
                point2.set_data([], [])
                point3.set_data([], [])
                line1.set_data([], [])
                line2.set_data([], [])
                time_string.set_text('')

                return point1, point2, point3, line1, line2, time_string

            def animate_function(i):
                """
                Animation function.
                Motion trail sizes. Defined in terms of indices. Length will vary with the time step,
                dt. E.g. 5 indices will span a lower distance if the time step is reduced.
                """
                # length of motion trail of weight 2
                trail2 = 100
                # time step
                dt = t[2]-t[1]
                
                point1.set_data([x1[i], x1[i]], [y1[i], y1[i]])
                # marker + rod of the second weight
                point2.set_data(x2[i:max(1,i-trail2):-1], y2[i:max(1,i-trail2):-1])
                point3.set_data([0, 0], [0, 0])
                # rod connecting weight 2 to weight 1
                line1.set_data([x1[i], x2[i]], [y1[i], y2[i]])
                # rod connecting origin to weight 1
                line2.set_data([x1[i], 0], [y1[i],0])
                
                time_string.set_text(time_template % (i*dt))
                return point1, point2,point3, line1, line2, time_string


            anim = animation.FuncAnimation(fig, animate_function, init_func=init,
                                           frames=self.N_t, interval=1000*(t[2]-t[1])*0.8, blit=True)
            return anim
