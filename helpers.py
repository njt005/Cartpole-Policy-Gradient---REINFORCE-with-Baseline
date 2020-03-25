import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import mahalanobis


def forward_euler(dxdt, tstep, tend, xbeg):
    t = np.arange(0, tend + tstep, tstep)
    x = np.zeros((4, len(t)))

    x[:, 0] = xbeg
    dxdt0 = dxdt[0]
    dxdt1 = dxdt[1]
    dxdt2 = dxdt[2]
    dxdt3 = dxdt[3]

    for k in range(1, len(t)):
        x[0, k] = x[0, k - 1] + tstep * dxdt0(x[0, k - 1], x[1, k - 1], x[2, k - 1], x[3, k - 1])
        if x[0, k] > np.pi:
            x[0, k] = x[0, k] - 2 * np.pi
        elif x[0, k] <= -np.pi:
            x[0, k] = x[0, k] + 2 * np.pi

        x[1, k] = x[1, k - 1] + tstep * dxdt1(x[0, k - 1], x[1, k - 1], x[2, k - 1], x[3, k - 1])
        if x[1, k] > 10:
            x[1, k] = 10
        elif x[1, k] <= -10:
            x[1, k] = -10

        x[2, k] = x[2, k - 1] + tstep * dxdt2(x[0, k - 1], x[1, k - 1], x[2, k - 1], x[3, k - 1])
        if x[2, k] >= 6:
            x[2, k] -= 12
        elif x[2, k] <= -6:
            x[2, k] += 12

        x[3, k] = x[3, k - 1] + tstep * dxdt3(x[0, k - 1], x[1, k - 1], x[2, k - 1], x[3, k - 1])
        if x[3, k] >= 10:
            x[3, k] = 10
        elif x[3, k] <= -10:
            x[3, k] = -10
    return x, t


def simulator(s, a, dur_a, dt):
    # s = np.reshape(s,4)
    # w is the derivative of theta, v is the derivative of x
    # s = [theta;w;x;v]

    # consts
    m1 = 0.5
    m2 = 0.5
    l = 0.6
    g = -9.81
    b = 0.1

    j_target = np.array([0, 0, 1])
    def j(x, z):
        return np.array([x, np.sin(z), np.cos(z)])
    T_in = np.array([[1, l, 0],
                    [l, (l*l), 0],
                    [0, 0, (l*l)]])

    # a in range(-10, 10); it's the force acting on the cart
    F = a

    dthetadt = lambda theta, w, x, v: w
    dwdt = lambda theta, w, x, v: (-3 * m2 * l * w ** 2 * np.sin(theta) * np.cos(theta) - 6 * (m1 + m2) * g * np.sin(theta) - 6 * (F - b * v) * np.cos(theta)) / \
                                  (4 * l * (m1 + m2) - 3 * l * m2 * np.cos(theta) ** 2)
    dxdt = lambda theta, w, x, v: v
    dvdt = lambda theta, w, x, v: (2 * m2 * l * w ** 2 * np.sin(theta) + 3 * m2 * g * np.sin(theta) * np.cos(theta) + 4 * F - 4 * b * v) / \
                                  (4 * (m1 + m2) - 3 * m2 * np.cos(theta) ** 2)
    funcs = [dthetadt, dwdt, dxdt, dvdt]

    snext, _ = forward_euler(funcs, dt, dur_a, s)
    snext = snext[:, -1]
    snext = np.reshape(snext,(4))

    rnext = -(1 - np.exp(-0.5*mahalanobis(j(s[0], s[2]), j_target, T_in)**2))
    
    done = False
#    if (s[2] <= -6 and a < 0) or (s[2] >= 6 and a > 0):
#        F = 0
#        done = True
        
    return snext, rnext, done


def visualization(s):
    xc = s[2]
    yc = 0
    x = np.sin(s[0]) + xc
    y = np.cos(s[0]) + yc

    plt.figure(100)
    plt.plot([xc, x], [yc, y], color='blue', linewidth=2, marker='o', markersize=6)
    plt.xlim(-6.5, 6.5)
    plt.ylim(-2, 2)
    plt.axes().set_aspect('equal')
    plt.pause(1/100)
    plt.clf()


def plot_err(array,color,fig_nr):
    plt.figure(fig_nr)
    plt.plot(array, color=color)
    plt.pause(1/100)
    plt.clf()
