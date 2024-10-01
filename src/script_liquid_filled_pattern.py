from prol_spheroid import ProlateSpheroid
from settings import LiquidFilledSettings
from IterativeSolvers import IterativeRefinement
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


def plot_pattern(file, settings):
    data = pd.read_csv(file)
    scale = 0.005
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.plot(np.pi * data['angle'] / 180, data['pattern'])
    pattern_max = np.max(data['pattern'])
    if scale is None:
        scale = pattern_max
    arrow_length = scale * 0.3
    ax.arrow(x=np.pi * settings.theta_i_deg / 180 + np.pi, y=scale * 0.9, dx=0, dy=-arrow_length,
             width=0.01,
             head_length=arrow_length * 0.1, length_includes_head=True, color='k')
    ax.set_ylim(0, scale)
    plt.show()


def main(output_file, settings, solver, freq_Hz):
    # if the path to the gfortran compiler is not already in the PATH environment variable, it can be added here
    os.environ['PATH'] += os.pathsep + os.path.abspath(r'C:\bin\mingw64\bin')

    spheroid = ProlateSpheroid(settings, solver)
    angles, pattern = spheroid.compute_far_field_pattern(freq_Hz, 300)
    data = pd.DataFrame({'angle':angles, 'pattern':pattern})
    data.to_csv(output_file, index=False)


if __name__ == '__main__':
    liquid_filled_settings = LiquidFilledSettings()
    solver = IterativeRefinement('LU')
    parentdir=os.path.split(os.getcwd())[0]

    freq_Hz = 100000
    file = os.path.join(parentdir, 'temp', 'scatter_pattern_{}_a_{}_b_{}_f_{}_rhos_{:.2f}_IncAngle_{}_{}.csv'.format(
        liquid_filled_settings.prefix,
        liquid_filled_settings.a, liquid_filled_settings.b,
        int(freq_Hz / 1000),
        liquid_filled_settings.ro_s, liquid_filled_settings.theta_i_deg,
        solver.solver_name()))

    main(file, liquid_filled_settings, solver, freq_Hz)
    plot_pattern(file, liquid_filled_settings)
