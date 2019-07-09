

import numpy as np
import matplotlib.pyplot as plt
from itertools import count, takewhile

import matplotlib.pyplot as plt
from scipy.optimize import leastsq


"""" stolen from stack overflow"""
import numpy, scipy.optimize

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    Fit is a least squares approch with no regularization impleimented'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2.**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])
    print(guess)


    def sinfunc(t, A, w, p, c):
        return A * numpy.sin(w * t + p) + c

    def ComplexSinFunc(t, A, w, p, c, A1, w1, p1, c1):
        return (A * numpy.sin(w * t + p) + c) + (A1 * numpy.sin(w1 * t + p1) + c1)

    #Code from internet was a curve fit using least squares. It didn't seem to fit respratory trace so we write a custom cost
    #function and change to using scipy.optimize.
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    # popt, pcov = scipy.optimize.curve_fit(ComplexSinFunc, tt, yy, p0=numpy.append(guess, guess))
    print(popt)
    A, w, p, c = popt
    # A, w, p, c, A1, w1, p1, c1 = popt
    f = w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}


def fit_cos(tt, yy):
    '''Fit cos to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    Fit is a least squares approch with no regularization impleimented'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2.**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])
    print(guess)


    def cosfunc(t, A, w, p, c):
        return A * numpy.cos(w * t + p) + c

    def ComplexSinFunc(t, A, w, p, c, A1, w1, p1, c1):
        return (A * numpy.sin(w * t + p) + c) + (A1 * numpy.sin(w1 * t + p1) + c1)

    #Code from internet was a curve fit using least squares. It didn't seem to fit respratory trace so we write a custom cost
    #function and change to using scipy.optimize.
    popt, pcov = scipy.optimize.curve_fit(cosfunc, tt, yy, p0=guess)
    # popt, pcov = scipy.optimize.curve_fit(ComplexSinFunc, tt, yy, p0=numpy.append(guess, guess))
    print(popt)
    A, w, p, c = popt
    # A, w, p, c, A1, w1, p1, c1 = popt
    f = w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.cos(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}




def fit_sin_regularized(tt, yy):
    '''Fit more complex regularization functions'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2. ** 0.5
    guess_offset = numpy.mean(yy)
    guess_phase = 0
    guess = numpy.array([guess_amp, 2. * numpy.pi * guess_freq, guess_phase, guess_offset])


    def sinfunc(t, A, w, p, c):
        return A * numpy.sin(w * (t + p)) + c

    def obj(params, time, data):
        A, w, p, c = params[0], params[1], params[2], params[3]
        t = time
        v = sinfunc(t, A, w, p, c)
        #Cost is the sum of the square of the residual
        return(np.sum((v - data))**2)/len(data)

    # res = scipy.optimize.minimize(fun=obj, x0=guess, args=(tt, yy), method='Nelder-Mead')
    res = scipy.optimize.minimize(fun=obj, x0=guess, args=(tt, yy))

    fitfunc = lambda t: res.x[0] * numpy.sin(res.x[1] * (t + res.x[2])) + res.x[3]
    print(res)
    return {"amp": res.x[0], "omega": res.x[1], "phase": res.x[2], "offset": res.x[3], "fitfunc": fitfunc,
            "rawres": res}

def Toy_example():
    #1D optimization to understand how to call
    def f(x):
        return -np.exp(-(x - 0.7) ** 2)

    result = scipy.optimize.minimize_scalar(f)
    print(result.success)
    print(result)


def thetaAtPoint(A, w, p, c, timePoint):
    """THis function is for figuring out given a time T and the parameters of the sin curve
    What the phase is at time T with respect to the last min or max.
    """""
    sine = lambda t: A * numpy.sin(w * t + p) + c
    cosine = lambda t: A * numpy.cos(w * t + p)
    value = sine(timePoint)
    instant_slope = cosine(timePoint)
    period = 2.*numpy.pi / w
    cycle = (timePoint - p)/period
    cycle_frac = cycle - np.floor(cycle)
    print('X value of the point of intrest is' , str(timePoint))
    print("Y value of the point is " + str(value), 'instant slope is ' + str(instant_slope))
    print("period is " + str(period))
    print("cycle num is "  + str(cycle_frac))

    r = numpy.arange(0,timePoint*2,1/250)



    return (value, instant_slope , cycle_frac )



if __name__ == "__main__":
    N = 1000  # number of data points
    t = np.linspace(0, 6 * np.pi, N)
    fine_t = np.arange(0, max(t), 0.1)
    f = 1 # Optional!! Advised not to use
    data = 3.0 * np.sin(f * t + 8*np.pi/4) + 0.5 + np.random.randn(N) / 2  # create artificial data with noise
    # Toy_example()
    ressimple = fit_sin(t, data)
    # resDict = fit_sin_regularized(t,data)
    # res = resDict['rawres']

    A, w, p, c = ressimple['amp'], ressimple['omega'], ressimple['phase'], ressimple['offset']

    timePoint = np.pi + np.pi
    value, slope, cycle_frac = thetaAtPoint(A, w, p, c, timePoint)


    plt.plot(t, data, '.', ms=1, label = "data")

    # plotting test code for thetaAtPoint
    plt.plot(timePoint,value,'.', ms=20 ,label= 'Point of intrest')
    sine = lambda t: A * numpy.sin(w * t + p) + c
    r = numpy.arange(0,timePoint*2,1/250)
    plt.plot(r,sine(r), label = 'fit')
    # plt.plot(r,cosine(r), label = 'derivative')

    plt.legend()
    plt.show()

    # # plt.plot(t, data_first_guess, label='first guess')
    # plt.plot(t, resDict["fitfunc"](t), label='custom cost fitting')
    # plt.plot(t, ressimple['fitfunc'](t), label='least squares fitting')
    # plt.legend()
    # plt.show()





    def TestCode():
        N = 1000 # number of data points
        t = np.linspace(0, 4*np.pi, N)
        f = 1.15247 # Optional!! Advised not to use
        data = 3.0*np.sin(f*t+0.001) + 0.5 + np.random.randn(N)/2 # create artificial data with noise
        f2= 2.6
        data2 = 3.0*np.sin(f*t+0.001) + 0.5 + 1.5*np.sin(f2*t+0.01) + np.random.randn(N)/10

        guess_mean = np.mean(data)
        guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)
        guess_phase = 0
        guess_freq = 1
        guess_amp = 1


        # we'll use this to plot our first estimate. This might already be good enough for you
        data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

        # Define the function to optimize, in this case, we want to minimize the difference
        # between the actual data and our "guessed" parameters
        optimize_func = lambda x:(x[0]*np.sin(x[1]*t+x[2]) + x[3] - data)**2

        # est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]
        est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

        # recreate the fitted curve using the optimized parameters
        data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean

        # recreate the fitted curve using the optimized parameters

        fine_t = np.arange(0,max(t),0.1)
        data_fit=est_amp*np.sin(est_freq*fine_t+est_phase)+est_mean


        res = fit_sin(t,data)
        print(res)

        plt.plot(t, data, '.')
        # plt.plot(t, data_first_guess, label='first guess')
        plt.plot(fine_t, data_fit, label='after fitting')
        plt.plot(fine_t, res['fitfunc'](fine_t),label='alturnative fitting')
        plt.legend()
        plt.show()