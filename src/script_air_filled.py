from prol_spheroid_vectorized import ProlateSpheroid
from settings import AirFilledSettings
from IterativeSolvers import IterativeRefinement
import os


if __name__ == '__main__':
    # if the path to the gfortran compiler is not already in the PATH environment variable, it can be added here
    os.environ['PATH'] += os.pathsep + os.path.abspath(r'C:\bin\mingw64\bin')

    air_filled_settings = AirFilledSettings()
    solver = IterativeRefinement('LU')
    ts_file_name = 'ts_vs_freq_loop_{}_a_{}_b_{}_f1_{}_f2_{}_rhos_{:.2f}_IncAngle_{}_{}.csv'.format(air_filled_settings.prefix,
                                                                                                    air_filled_settings.a, air_filled_settings.b,
                                                                                                    int(air_filled_settings.min_freq / 1000),
                                                                                                    int(air_filled_settings.max_freq/ 1000),
                                                                                                    air_filled_settings.ro_s, air_filled_settings.theta_i_deg,
                                                                                                    solver.solver_name())
    spheroid = ProlateSpheroid(air_filled_settings, solver)
    spheroid.run(ts_file_name)
