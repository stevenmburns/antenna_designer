from antenna_designer.core import save_or_show

import numpy as np
import scipy

from matplotlib import pyplot as plt

#fn = None
fn = '/dev/null'

def test_sici():

    def f(r):
        return np.exp((0+1j)*r)/r

    def F(r):
        si, ci = scipy.special.sici(r)
        return ci + (0+1j)*si

    def rectangle(a, b):
        return (b-a) * f((a+b)/2)

    def lrectangle(a, b):
        return (b-a) * f(a)

    def rrectangle(a, b):
        return (b-a) * f(b)

    def trapezoidal(a, b, n):
        return (b-a)/n * (f(a)/2 + f(b)/2 + sum(f(a+k*(b-a)/n) for k in range(1,n)))

    def simpsons(a, b):
        return (b-a)/6 * (f(a) + f(b) + 4*f((a+b)/2))

    xs = np.linspace(0.001, 0.10, 101)

    rects = []
    lrects = []
    rrects = []
    t1s = []
    t2s = []
    t4s = []
    t8s = []
    refs = []
    ss = []

    spacing = 1
    for delta in xs:
        a, b = 2*np.pi*(spacing-0.5)*delta, 2*np.pi*(spacing+0.5)*delta

                          
        rects.append(rectangle(a, b))
        lrects.append(lrectangle(a, b))
        rrects.append(rrectangle(a, b))
        t1s.append(trapezoidal(a, b, 1))
        t2s.append(trapezoidal(a, b, 2))
        t4s.append(trapezoidal(a, b, 4))
        t8s.append(trapezoidal(a, b, 8))
        refs.append(F(b) - F(a))
        ss.append(simpsons(a, b))


    delta = 0.01

    fig, ax0 = plt.subplots()
    ax1 = ax0.twinx()

    for spacing in [1,2,3,4]:
        xxs = np.linspace(2*np.pi*(spacing-0.5)*delta, 2*np.pi*(spacing+0.5)*delta, 101)
        fs = f(xxs)
        ax0.plot(xxs, fs.real)
        ax1.plot(xxs, fs.imag)

    plt.xlim(0, None)
    save_or_show(plt, fn)

    rects = np.array(rects)
    lrects = np.array(lrects)
    rrects = np.array(rrects)
    t1s = np.array(t1s)
    t2s = np.array(t2s)
    t4s = np.array(t4s)
    t8s = np.array(t8s)
    refs = np.array(refs)
    ss = np.array(ss)

    

    plt.plot(xs, rects.real, label='rects real')
#    plt.plot(xs, lrects.real, label='lrects real')
#    plt.plot(xs, rrects.real, label='rrects real')
    plt.plot(xs, ss.real, label='ss real')
#    plt.plot(xs, t1s.real, label='t1s real')
    plt.plot(xs, t2s.real, label='t2s real')
    plt.plot(xs, t4s.real, label='t4s real')
    plt.plot(xs, t8s.real, label='t8s real')
    plt.plot(xs, refs.real, label='refs real')

    plt.plot(xs, rects.imag, label='rects imag')
#    plt.plot(xs, lrects.imag, label='lrects imag')
#    plt.plot(xs, rrects.imag, label='rrects imag')
    plt.plot(xs, ss.imag, label='ss imag')
#    plt.plot(xs, t1s.imag, label='t1s imag')
    plt.plot(xs, t2s.imag, label='t2s imag')
    plt.plot(xs, t4s.imag, label='t4s imag')
    plt.plot(xs, t8s.imag, label='t8s imag')
    plt.plot(xs, refs.imag, label='refs imag')

    plt.legend()
    save_or_show(plt, fn)
