from prol_spheroid_II import ProlateSpheroid
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
        self.a = 0.06               # the length of the semi-major axis [m]
        self.b = 0.02               # the length of the semi-minor axis [m]

        # frequencies
        self.delta_f = 2000         # the distance between frequencies [Hz]
        self.min_freq = 50000       # the start frequency [Hz]
        self.max_freq = 56100       # the end frequency [Hz]

        #incidence angle
        self.theta_i_deg = 60       # the incidence angle [degrees]


if __name__ == '__main__':
    # if the path to the gfortran compiler is not already in the PATH environment variable, it can be added here
    os.environ['PATH'] += os.pathsep + os.path.abspath(r'C:\bin\mingw64\bin')

    spheroid = ProlateSpheroid(UserDefinedSettings(), BiCGSTABSolver(ILUPreconditioner()))
    spheroid.run()
