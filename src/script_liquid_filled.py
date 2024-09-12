from prol_spheroid import ProlateSpheroid
from settings import LiquidFilledSettings
from IterativeSolvers import IterativeRefinement
import os


if __name__ == '__main__':
    # if the path to the gfortran compiler is not already in the PATH environment variable, it can be added here
    os.environ['PATH'] += os.pathsep + os.path.abspath(r'C:\bin\mingw64\bin')

    liquid_filled_settings = LiquidFilledSettings()
    solver = IterativeRefinement('LU')
    ts_file_name = 'ts_vs_freq_loop_{}_a_{}_b_{}_f1_{}_f2_{}_rhos_{:.2f}_IncAngle_{}_{}_test.csv'.format(liquid_filled_settings.prefix,
                                                                                                    liquid_filled_settings.a, liquid_filled_settings.b,
                                                                                                    int(liquid_filled_settings.min_freq / 1000),
                                                                                                    int(liquid_filled_settings.max_freq/ 1000),
                                                                                                    liquid_filled_settings.ro_s, liquid_filled_settings.theta_i_deg,
                                                                                                    solver.solver_name())
    spheroid = ProlateSpheroid(liquid_filled_settings, solver)
    spheroid.run(ts_file_name)
