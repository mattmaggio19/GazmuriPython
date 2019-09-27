import pandas as pd
import numpy as np
import time
import GazmuriLoader as GL
import matplotlib.pyplot as plt
import SA1DataLoader
import os, pickle



def DeathDetector(df):
    #Scan the dataframe to see if you can find where the animal dies. Start from the end of the dataset, working backwords if the animal is determined to be dead.

    #Death Find when the average endtidal CO2 goes below 10.
    Ao = df.iloc[:, 1]
    Pa = df.iloc[:, 3]
    ETCo2 =df.iloc[:, 10]
    ETCo2_ave = ETCo2.rolling(60*250).mean()
    Time = df['Time']
    AvGradient = (Ao-Pa)
    DeathMask = Time.where(ETCo2_ave < 5)

    if all(DeathMask.isnull())==True:
        print('no Death Event Detected.')
        return False
    else:
        DeathTime = DeathMask.loc[~DeathMask.isnull()].iloc[0]
        print('Death Event Detected at {0} min  '.format(str(DeathTime/60)))
        return DeathTime


def FFTSignalAnalysis(Series):
    #An attempt to see if decomposing a signal into it's freqency domain illiminates if it is phyisology or noise.

    #What happens if we submitt a rolling average

    # SeriesRoll = Series.rolling(3).mean()

    Array = Series.to_numpy()
    n = Series.shape[0]

    fft = np.fft.fft(Array)/n
    fft = fft[range(int(n / 2))]
    samplingRate = 250
    Ts = 1/samplingRate

    time = np.arange(0, n*Ts , Ts)
    k = np.arange(n)
    T = n/samplingRate
    frq = k / T
    frq = frq[range(int(n / 2))]

    print(fft[2:])


    fig, ax = plt.subplots(2, 1)
    # ax[0].plot(time, abs(Array), 'b')
    ax[0].plot(time, abs(Series), 'r')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq[2:], abs(fft[2:]), 'r')
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()



def DenoiseSignal(Series, SignalType = 'Hemodyanmic'):
    #This fucntion should be called after you extract a signal of intrest from the experiment dataframe.
    #we will use different logic for hemodynamic functions than for other functions we aren't going to focus here on average or anything more just censoring data

    #start by calculating the zscore for the whole trace. the standard dev may also be informative.

    #First zero out any data above 150 and below -10, this should tighen our std range significantly. Maybe doesn't work for LV?
    # filter1, filter2,  = Series > -10, Series < 150
    # Series = Series.where(filter1 & filter2)

    #

    # Calculate the mean, and standard dev of the whole trace. This will let us throw outliers out and drop things like flushes.
    SeriesMean = Series.mean()
    SeriesStd = Series.std()

    #Average the data over 5 second windows, this way the flushes should be more obvious in the derivative, as the raw derivative contains hemodynamics.
    print('Overall mean of {0} and std of {1}'.format(SeriesMean, SeriesStd))

    Series_zscore = (Series - SeriesMean) / SeriesStd

    SeriesRoll = Series.rolling(30*250).mean()
    RollDiff = SeriesRoll.diff()
    diffZ = (RollDiff - RollDiff.mean())/RollDiff.std()


    Threshold = 3 # a zscore above 5 usually means a flush. Works pretty well for acute.
    # #I kind of don't like the zscore thing. lets try an approach based on the d.

    filter3 = diffZ.abs() < Threshold
    Series = Series.where(filter3)

    Padding = 20*250
    CensorSeries = Series.copy()

    # Censor any data that is within 10 seconds of a detected bad point.
    for ix, value in Series.items(): #TODO make faster by instead iterating only though the places where we are not below the threshold.
        if np.isnan(value) and not all(CensorSeries.iloc[ix-Padding:ix+Padding-1] == np.nan):
            # print('removed data around the point {0}'.format(ix))
            CensorSeries.iloc[ix-Padding:ix] = np.nan #Apparently it is quite bad to change the thing you are iterating over proformance wise.

    # plt.plot(Series)
    plt.plot(CensorSeries)
    plt.plot((filter3)*SeriesMean)
    plt.show()

    return CensorSeries


def MapBeforeDeathEvent(useCashe = False, Timebefore = 10):

    #trying to answer a question Dr.G had. what is the stable map that the animals have before hemodynamic deterioration.
    # Add some timing stuff.
    t = time.time()

    SA1Raw = GL.SA1standardRAWloadingFunction(useCashe=False)
    # Figure out if a cashe file exists.
    CasheFilename = 'Sa1DeathDetection.pickle'

    CasheExists = os.path.exists(os.path.join('cashe', CasheFilename))

    Deathdict = dict()

    if not useCashe or not CasheExists:
        for expNum in SA1Raw:

            # Sometimes the death detector will fail, in those cases we can either build a better mousetrap or hardcode the values.
            HardCodedDeathEvent = {'2018129': False, '2018162': False}

            Exp = SA1Raw[expNum]
            df = GL.LoadRawExperiment(Exp, TimeTup=(0, 240))  # Load the acute dataset
            print('Finding death event in exp {0}'.format(Exp['metadata']['expNumber']))
            if Exp['metadata']['expNumber'] in HardCodedDeathEvent.keys():
                Deathdict[Exp['metadata']['expNumber']] = HardCodedDeathEvent[Exp['metadata']['expNumber']]
            else:
                Deathdict[Exp['metadata']['expNumber']] = DeathDetector(df)
            print('Been running for ' + str(round(((time.time() - t) / 60), 2)) + 'min ')

        print(Deathdict)  # Might want to cashe this for later use. Takes 15 min to produce, i'd say thats reasonable, We got 2 wrong in the dataset! not bad!

        #Pickle the deathdict variable as a cashe.
        print("Total Dataset Loaded and processed at {0} seconds".format(time.time() - t))

        # Save the dataset to the cashe. (Maybe date the cashes, or that might lead to file inflation.
        print('Cashing dataset to disk.')
        with open(os.path.join('cashe', CasheFilename), 'wb') as f:
            pickle.dump(Deathdict, f)
            print('Cashe dumped to disk  at {0} '.format(time.time() - t))

    elif not CasheExists:
        os.mkdir(os.path.join('cashe')) #hopefully this doesn't delete old cashes. hopefully.
        print('Call loader recursively to refresh Cashe.')
        MapBeforeDeathEvent(useCashe=False)

    else:
        #Load the cashe
        with open(os.path.join('cashe', CasheFilename), 'rb') as f:
            Deathdict = pickle.load(f)
            print('Cashe loaded from disk  in {0} '.format(time.time() - t))


    #build a list of animals that had detected death events.
    DeathLst = []
    #Now that we have the deathdict, take only those that died in the alloted time frame (0,240) in this case.
    for exp in Deathdict:
        if Deathdict[exp] != False:
            DeathLst.append(exp)
    print(DeathLst)

    for ix, exp in enumerate(DeathLst):
        #load the experiment into memory, should load only the files that go upto and contain the death time.
        df = GL.LoadRawExperiment(SA1Raw[exp], TimeTup=(0, Deathdict[exp]))

        #How many minitues before death you want to look at.
        BeforeDeathFrame = 15

        # Extract the features of intrest.
        Deathidx = int(round(Deathdict[exp]) * 250) #Round to the nearest second convert to a point index.
        idx = slice(Deathidx - (BeforeDeathFrame * 60 * 250), Deathidx)

        Ao = df.iloc[:, 1]
        AoDenoised = DenoiseSignal(Ao) #Maybe, Only denoise the signal that we want to use for the thing, faster, but causes us to censor less of the flushes.
        AoAveraged = AoDenoised.rolling(250*60).mean()    #1 min mean
        Etco2 = df.iloc[:, 10].rolling(250*60).mean()  #1 min mean
        Time = df['Time']



        plt.plot(Time.iloc[idx], AoAveraged.iloc[idx])
        plt.plot(Time.iloc[idx], Etco2.iloc[idx])
        plt.show()

        # Graph the Ao signal with a 1 min average MAP, 1 min average end tidal

    print('For refrence, there were {0} Experiments in this analysis'.format(ix+1))



        #Diagnotic code to load the manually produced death times from the excel sheet.
        # Dataset = SA1DataLoader.StandardLoadingFunction(useCashe=False)  # Load the dataset from the master excel sheet.
        #
        # for exp in Dataset:
        #     expNum = exp['experimentNumber']
        #     print('exp number {0} Death time from automatic detection {1}  '
        #           'vs  {2}   from the excel sheet '.format(expNum, Deathdict[expNum] / 60, exp['Survival time']))



def DecomposeArterialSignal(df=None, window=10, sampleingFreq=250, rollingOrWindow = 'window', AoChannel = 1):

    #Given an arterial signal find the systolic, diastolic, and mean after considering the effect of the breathing artifact.
    Ao = df.iloc[:, 1]
    Time = df['Time']

    #Calculate the mean, and standard dev of the whole trace. This will let us throw outliers out and drop things like flushes.
    AoMean = Ao.mean()
    AoStd = Ao.std()

    Ao_zscore =  (Ao - AoMean) / AoStd
    idx = slice(250*60*55, 250*60*240)
    plt.plot(Ao.iloc[idx].index/(250*60), Ao.iloc[idx].values)

    plt.show()
    #As an additional thing, if you can locate the dicrotic notch, calculate the stroke volume.


    #We cannot output these values at every timepoint so there has to be some kind of window at which a new score is determined.



if __name__ == '__main__':

    SA1Raw = GL.SA1standardRAWloadingFunction(useCashe=False)

    df = GL.LoadRawExperiment(SA1Raw['2018106'], TimeTup=(0, 30))

    Ao = df.iloc[:, 1]

    FFTSignalAnalysis(Ao.iloc[300*250:360*250])


    # MapPreDeath = MapBeforeDeathEvent(useCashe = True)
