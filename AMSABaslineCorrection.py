from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import GazmuriLoader

def AMSA_calc(y=None):
    T = 1 / 60  # spacing of points to time

    #If given no imput, use this funcion to generate data
    if y is None:
        N = int(1000)  # number of points
        x = np.linspace(0.0, N * T, N)
        y = 0 + 2*np.sin(3 * 2.0 * np.pi * x) + 0.5 * np.sin(8 * 2.0 * np.pi * x) + np.random.normal(0, 0.5, x.shape)
    else:
        N = int(len(y))  # number of points
        x = np.linspace(0.0, N * T, N)

    #toy example AMSA

    print(N)

    #calculate fft with
    yf = np.fft.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
    #

    #Graphs
    plt.subplot(3, 1, 1)
    plt.plot(x, y)
    plt.subplot(3, 1, 2)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:int(N / 2)]))
    plt.subplot(3, 1, 3)
    plt.plot(xf[1:], 2.0 / N * np.abs(yf[0:int(N / 2)])[1:])
    plt.show()

def Circshift(x, roll = 2):
    #matlab version of this algo has a parameter for weights, I think that is intresting because you could actually weight the poly fits
    #to the future values such that you would pre-emtively elimate the noise artifacts from the mechanical disruptions.
    #to simulate that, do a circshift of the SG filter to pretend it knows the future!
    return np.roll(x, roll)


def ECGJumpCorrection():

    #Load a sample ECG strip with compressions
    #orignal example
    path = r'C:\Program Files\RI DAS\DATAFILES\EXP #122019\CC-6 04-04-14 at 01 08 18 PM'
    # path = r'C:\Program Files\RI DAS\DATAFILES\EXP #122021\CC-2 04-08-14 at 11 41 26 AM'
    # path = r'C:\Program Files\RI DAS\DATAFILES\EXP #122021\CC-7 04-08-14 at 11 46 26 AM'


    df = GazmuriLoader.LoadLabviewData( path=path , samplingRate=250, headers=2)

    slashes =  ([pos for pos, char in enumerate(path) if char == '\\'])

    cols = df.columns.values
    Range = np.arange(100,2600*5)

    # testing parameter
    offset = 0

    ECG = df[cols[0]]
    Depth = df[cols[6]]

    ECG_seg = ECG[Range] + offset
    Depth_seg = Depth[Range] + (offset*50)
    t1 = time.time()
    #Key line filters the signal by a rolling least squares polynomial fit
    #Implemented in Labview allready
    #http://zone.ni.com/reference/en-XX/help/371361M-01/lvanls/sgfil/

    #window length should be 2 times greater than the frequency of compressions in points. 250 points is 1 second.
    #
    f1 = savgol_filter(ECG_seg, 21, 2, deriv=0)
    t2 = time.time()

    # Real time is important
    print("running on " + str(len(ECG_seg)) + " Points takes " + str(t2 - t1) + " Seconds")

    #Write a test to see if the mean values change in the compressed vs non compressed state. They should get closer if this is working.

    points = np.arange(len(ECG))
    # print(path[slashes[len(slashes)-2]+1 : -1])
    plt.title(path[slashes[len(slashes)-2]+1 : -1])
    plt.plot(ECG_seg, color = 'blue')
    plt.plot(Range,f1, color = 'orange')
    plt.plot(Range, ECG_seg - (f1) + offset , color = 'green' )
    plt.plot(Depth_seg / 50, color='red')
    plt.legend(["ECG data", "SGfit", "Flattened data", "Depth of compressions"],loc = 'upper right')
    plt.show()

    #Without Apriori knowlege of compresisons Can we correct for baseline jumps?
    #Seems like we do need to set the number of points for sav gol apriori, but
    # since we are trying to subtract the CPR artifacts we just have to make it equal to CPR Frequency / Sampling rate.





def toyExample():
    #This is just to understand imputs. Not for actual use.
    np.set_printoptions(precision=2)  # For compact display.

    #x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])
    point = np.arange(0,25,0.1)

    x = np.sin(point) + (np.sin( 10 * point))/2 + 0 + np.random.normal(0,0.3,point.shape)
    #+  np.random.normal(0,0.1,point.shape)

    t1 = time.time()

    f1= savgol_filter(x, 11, 3)
    t2 = time.time()

    print ("running on " + str(len(x)) + " Points takes " + str(t2-t1) + " Seconds" )

    flattened = x - f1

    # p1 = plt.plot(point,x, label = 'Data')
    # p2 = plt.plot(point,f1, label = 'SG Filter')
    # p3 = plt.plot(point,flattened, label = 'Flattened Data')
    # plt.legend([ "Orginal Data", "SGFit","Flattened data"],loc= 'upper right')
    # plt.title('SG filter Toy example')
    # plt.show()

    AMSA_calc()


if __name__ == "__main__":
    #ECGJumpCorrection()


    toyExample()

