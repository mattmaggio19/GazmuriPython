import pandas as pd
import numpy as np
import time
import GazmuriLoader as GL
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
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

    #First nan out any data above 150 and below -10, this should tighen our std range significantly. Maybe doesn't work for LV?

    #

    # Calculate the mean, and standard dev of the whole trace. This will let us throw outliers out and drop things like flushes.
    SeriesMean = Series.mean()
    SeriesStd = Series.std()

    #Average the data over 5 second windows, this way the flushes should be more obvious in the derivative, as the raw derivative contains hemodynamics.
    print('Overall mean of {0} and std of {1}'.format(SeriesMean, SeriesStd))

    Series_zscore = (Series - SeriesMean) / SeriesStd

    SeriesRoll = Series.rolling(10*250).mean()
    RollDiff = SeriesRoll.diff()
    diffZ = (RollDiff - RollDiff.mean())/RollDiff.std()


    Threshold = 5 # a zscore above 5 usually means a flush. Works pretty well for acute.
    # #I kind of don't like the zscore thing. lets try an approach based on the d.

    filter3 = diffZ.abs() < Threshold
    Series = Series.where(filter3)

    Padding = 20*250

    filter1, filter2,  = Series > -10, Series < 150
    Series = Series.where(filter1 & filter2)

    CensorSeries = Series.copy()

    # Censor any data that is within 20 seconds of a detected bad point.
    for ix, value in Series.items(): #TODO make faster by instead iterating only though the places where we are not below the threshold.
        if np.isnan(value) and not all(CensorSeries.iloc[ix-Padding:ix-1] == np.nan):
            # print('removed data around the point {0}'.format(ix))
            CensorSeries.iloc[ix-Padding:ix] = np.nan #Apparently it is quite bad to change the thing you are iterating over proformance wise.

    # plt.plot(Series)
    # plt.plot(CensorSeries)
    # plt.plot((filter3)*SeriesMean)
    # plt.show()

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

    fig = plt.subplot(np.ceil(len(DeathLst)/2), 2, 1)

    for ix, exp in enumerate(DeathLst):
        #load the experiment into memory, should load only the files that go upto and contain the death time.
        df = GL.LoadRawExperiment(SA1Raw[exp], TimeTup=(0, Deathdict[exp]))

        #How many minitues before death you want to look at.
        BeforeDeathFrame = 30

        # Extract the features of intrest.
        Deathidx = int(round(Deathdict[exp]) * 250) #Round to the nearest second convert to a point index.
        idx = slice(Deathidx - (BeforeDeathFrame * 60 * 250), Deathidx)

        Ao = df.iloc[:, 1]
        AoDenoised = DenoiseSignal(Ao) #Maybe, Only denoise the signal that we want to use for the thing, faster, but causes us to censor less of the flushes.
        AoAveraged = AoDenoised.rolling(250*10).mean()    #10 sec mean
        Etco2 = df.iloc[:, 10].rolling(250*10).mean()  #1 min mean
        Time = df['Time']

        Ax =  plt.subplot( np.ceil(len(DeathLst) / 2), 2, ix+1, frameon=True)
        Ax.set_title(exp)
        Ax.plot((Time.iloc[idx]-(Deathidx/250))/60, Etco2.iloc[idx], label='Mean Etco2 value')
        Ax.plot((Time.iloc[idx]-(Deathidx/250))/60, AoAveraged.iloc[idx], label='Mean Art pressure')

        Ax.set_yticks(np.arange(0, 60, step=10))
        Ax.set_ylim(0, 60)
        fontP = FontProperties()
        fontP.set_size('small')
        Ax.legend(loc='lower left', bbox_to_anchor=(-0.25, -0.25), prop=fontP)

    plt.tight_layout()
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

def DecomposeEtCo2Signal(ETCO2 , Return = 'Corrected' ):
    #Depending on the user to feed the correct series.
    t = time.time()

    #Run a 1 min rolling average and subtract the Series from it. Then mask out the positive portion, average the positive and negative portion and return the corrected trace.
    AveETCO2 = ETCO2.rolling(250*20).mean()
    ZeroedETCO2 = ETCO2 - AveETCO2

    #This is a clever filter because you use the local maximum as the cutoff to eliminate most of the artifact from the upswing downswing.
    ExhaledETCO2Filter = ZeroedETCO2 > 0.7 * ZeroedETCO2.rolling(250*20).max()
    InhaledETCO2Filer = ZeroedETCO2 < 0.7 * ZeroedETCO2.rolling(250*20).min()

    if Return == 'Corrected':
        #cython approach is almost required for data of this size.
        ExhaledETCO2 = ETCO2.where(ExhaledETCO2Filter).rolling(20 * 250, min_periods=1, center=True).mean()
        InhaledETCO2 = ETCO2.where(InhaledETCO2Filer).rolling(20 * 250, min_periods=1, center=True).mean()
        CorrectedETCO2 = ExhaledETCO2 - InhaledETCO2
        t1 = time.time()
        print('Took {0} to process {1} seconds of data '.format(str(t1 - t), str(ETCO2.shape[0]/250)))

        #Diagnotic plotting code.
        plt.plot(ETCO2, label = 'Raw ETCO2 signal')
        plt.plot(ExhaledETCO2, label = 'Exhaled')
        plt.plot(InhaledETCO2, label = 'Inhaled')
        plt.plot(CorrectedETCO2, color='red', label = 'Corrected')
        # plt.plot(CorrectedETCO2First)
        plt.legend()
        plt.show()
        return CorrectedETCO2

    elif Return == 'filters': #Much faster.
        #If we want just the filters, we are much more likely intrested in a multually exclusive filter that transitions during inhilation and exhilation.
        ExhaledETCO2Filter = ZeroedETCO2 > 0
        InhaledETCO2Filer = ~ExhaledETCO2Filter
        return (ExhaledETCO2Filter, InhaledETCO2Filer)





def MomentOfCCIAnalysis(dataset):
    #TODO This function should parse the dataset and isolate the small period around the CCI impact.
    #We are concerned with the phase of the resp and cardiac cycle that we are in when the impact happens.
    #At some point we might try to see if ICP spike hieght is related to the dura rupture event or if it is independant.
    pass

def DecomposeArterialSignal(Ao, Return = 'all'):
    #Starting work on the arterial signal

    #Start with a moving average of the signal for like a 10 seconds.
    AveAo = Ao.rolling(250 * 10).mean()
    ZeroedAo = Ao - AveAo

    #Count the cumulative number of transitions per time as the heart rate.
    #Convert to numpy for this, the solutions just don't seem as elegant in pandas.
    ZeroedAoArray = ZeroedAo.to_numpy()

    BeatTrace = np.where(np.diff(np.signbit(ZeroedAoArray)))[0] + Ao.index[0] #sync with the experimental time.

    # plt.plot(Ao)
    plt.plot(ZeroedAo)
    plt.vlines(BeatTrace, ymin=-20, ymax=20)
    #detecting the dichrotic notch, need to eliminate points that are too close to other points.
    #High quality Ao signal is needed.

    plt.show()

    # print(BeatTrace)
    # print(len(BeatTrace))
    # print(len(BeatTrace))

    pass


if __name__ == '__main__':
    # MapBeforeDeathEvent(useCashe=True)

    SA1Raw = GL.SA1standardRAWloadingFunction(useCashe=False)

    df = GL.LoadRawExperiment(SA1Raw['2018108'], TimeTup=(0, 60))

    Ao = df.iloc[:, 1]
    # ETCO2 = df.iloc[:, 10]
    # ETCO2Corrected = DecomposeEtCo2Signal(ETCO2.iloc[0:400*250])

    DecomposeArterialSignal(Ao[30*250*60:40*250*60])

    # FFTSignalAnalysis(Ao.iloc[300*250:360*250])


    # MapPreDeath = MapBeforeDeathEvent(useCashe = True)
