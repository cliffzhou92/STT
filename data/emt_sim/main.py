import numpy as np
import matplotlib.pyplot as plt
import functions_emt as f
from params_emt import *
import time
import pandas as pd




def main():

    tm = 10000.  # simulation time (dt=0.1 is set in the time_course_emt() function )
    b = 1.

    np.random.seed(100)

    # load initial conditions for E, E/M, and M states
    ics = np.loadtxt('ics_tristable_0.8.txt')
    epi_ic, hyb_ic, mes_ic = ics[0], ics[1], ics[2]
    E_epi, E_hyb, E_mes = epi_ic[-2], hyb_ic[-2], mes_ic[-2]

    # sample multiple values of TGFB inducer
    ax_list = [231, 232, 233, 234, 235, 236]
    tgf_list = [0.6, 0.8, 1, 1.2, 1.4, 1.6]

    fig = plt.figure(figsize=(12, 8))
    for i in range(len(tgf_list)):
        Tu, su, Su, R3u, zu, Zu, R2u, Eu, Nu, T, s, S, R3, z, Z, R2, E, N = f.time_course_emt(b, tgf_list[i], tm, epi_ic, sigma=0.01)
        ax = plt.subplot(ax_list[i])
        f.plot_traj(ax, tm, E, E_epi, E_hyb, E_mes)
        plt.title('TGFB=' + str(tgf_list[i]))

    plt.tight_layout()
    plt.savefig('traj_sample.pdf', format='pdf', dpi=300)


if __name__=='__main__':
    main()


