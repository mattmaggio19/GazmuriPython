import pandas as pd
import numpy as np
import GazmuriLoader as GL
import matplotlib.pyplot as plt

#Assumption is that we will be getting pandas dataframes with standard channels for different signals.


def DeathDetector(df):
    #Scan the dataframe to see if you can find where the animal died. Start from the end of the dataset, working backwords if the animal is determined to be dead.
    pass


def DecomposeArterialSignal(df=None, window=10, sampleingFreq=250, rollingOrWindow = 'window', AoChannel = 1):

    #Given an arterial signal find the systolic, diastolic, and mean after considering the effect of the breathing artifact.
    Ao = df.iloc[:,1].to_numpy(dtype=float , copy=False)
    Time = df['Time'].to_numpy(dtype=float , copy=False)
    #Calculate the mean, and standard dev of the whole trace. This will let us throw outliers out and drop things like flushes.
    AoMean = Ao.mean()
    AoStd = Ao.std()

    Ao_zscore =  (Ao - AoMean) / AoStd
    idx = slice(250*60*14)
    plt.plot(x= Time[idx]/60, y=Ao[idx])

    plt.show()
    #As an additional thing, if you can locate the dicrotic notch, calculate the stroke volume.


    #We cannot output these values at every timepoint so there has to be some kind of window at which a new score is determined.



if __name__ == '__main__':
    SA1Raw = GL.SA1standardRAWloadingFunction(useCashe=True)
    Exp = SA1Raw['2018130']
    df = GL.LoadRawExperiment(Exp, TimeTup=(0, 60))
    DecomposeArterialSignal(df )
