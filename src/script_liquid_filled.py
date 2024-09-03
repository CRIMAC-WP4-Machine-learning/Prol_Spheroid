from prol_spheroid_II import ProlateSpheroid
from settings import LiquidFilledSettings
from IterativeSolvers import IterativeRefinement
import os


if __name__ == '__main__':
    # if the path to the gfortran compiler is not already in the PATH environment variable, it can be added here
    os.environ['PATH'] += os.pathsep + os.path.abspath(r'C:\bin\mingw64\bin')

    spheroid = ProlateSpheroid(LiquidFilledSettings(), IterativeRefinement('LU'))
    spheroid.run()
