from prol_spheroid import ProlateSpheroid
from IterativeSolvers import BiCGSTABSolver, ILUPreconditioner
import os


class UserDefinedSettings:

    def __init__(self):
        self.prefix = 'user'        # just a name to identify the output files
        # media properties
        self.ro_w = 1027            # the density of the surrounding media [kg/m^3]
        self.ro_s = 1100            # the density of the spheroid [kg/m^3]
        self.c_w = 1500             # the sound speed in the surrounding media [m/s]
        self.c_s = 1600             # the sound speed in the spheroid [m/s]

        # geometrical properties
        self.a = 0.05               # the length of the semi-major axis [m]
        self.b = 0.005               # the length of the semi-minor axis [m]

        # frequencies
        self.delta_f = 2000         # the distance between frequencies [Hz]
        self.min_freq = 50000       # the start frequency [Hz]
        self.max_freq = 56100       # the end frequency [Hz]

        #incident angle
        self.theta_i_deg = 60       # the incidence angle [degrees]

        # precision
        self.precision_fbs = 1e-6


if __name__ == '__main__':
    # if the path to the gfortran compiler is not already in the PATH environment variable, it can be added here
    os.environ['PATH'] += os.pathsep + os.path.abspath(r'C:\bin\mingw64\bin')

    user_defined_settings = UserDefinedSettings()
    solver = BiCGSTABSolver(ILUPreconditioner())
    ts_file_name = 'ts_vs_freq_loop_{}_a_{}_b_{}_f1_{}_f2_{}_rhos_{:.2f}_IncAngle_{}_{}.csv'.format(user_defined_settings.prefix,
                                                                                                    user_defined_settings.a, user_defined_settings.b,
                                                                                                    int(user_defined_settings.min_freq / 1000),
                                                                                                    int(user_defined_settings.max_freq / 1000),
                                                                                                    user_defined_settings.ro_s, user_defined_settings.theta_i_deg,
                                                                                                    solver.solver_name())

    spheroid = ProlateSpheroid(user_defined_settings, solver)
    spheroid.run(ts_file_name)
