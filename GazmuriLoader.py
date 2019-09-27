import os, time
from sklearn import linear_model, preprocessing
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from itertools import count, takewhile
from SinosoidalFit import fit_sin, fit_sin_regularized, thetaAtPoint

#copy pasted htis from internet, it just impliments the same function as range, but for float values.
def frange(start, stop, step):
    return takewhile(lambda x: x< stop, count(start, step))


def LoadLabviewData(path=None, samplingRate = 250, headers = 5):
    # This is the most important method here, Gazmuri's labview system dumps the data to a file without timestamps on the datasamples, the headers are what is inside the colunms.
    # We can use this to load to a dataframe, but I never resolved how to stitch them together in a time resolved manner.

    #TODO finish this and make it robust to different experiments.





    #use pandas to load from the text file skiping the first 3 rows (metadata)
    df = pd.read_csv(path, sep='\t', skiprows=[0, 1, 2], header=[1])

    #Create a list of times that corrispond to the length of time from start of file.
    idx = list(frange(0, df.shape[0] / samplingRate, 1 / samplingRate))

    if len(idx) != df.shape[0]:
        #Take steps to make sure that each row has a time assigned to it. Likely this is due to the 1/sampling step size in the list.
        #Get the number they are different and append the idx with as many entries as needed.
        entriesNeeded = df.shape[0] - len(idx)
        if entriesNeeded > 0:
            for i in range(0, entriesNeeded):
                idx.append(idx[-1] + (1/samplingRate))
        else:
            #cut off the last few entries to make them equal.
            idx = idx[:entriesNeeded]

    assert len(idx) == df.shape[0]
    df['Time'] = idx


    #Store the sampling rate for refrence later.
    df.samplingRate = samplingRate
    #return the dataframe
    return df



def PlotTimeSeries(timeseries= None , dataSeries = None, timeRange = (0, 15*250)):
    #timeSeries should be a single series
    #dataSeries is a dictionary of series to be plotted, with names as the keys
    #timerange is the indexs of time you want plotted. Tickrate needs to be known outside this function

    #Pull the time data of requested range
    t = timeseries[timeRange[0]:timeRange[1]]
    #pull the dependent data of requested range
    for key in dataSeries:
        # print(key)
        value = dataSeries[key]
        y = value[timeRange[0]:timeRange[1]]
        plt.plot(t, y)

    plt.legend(dataSeries.keys())
    # plt.show()

def PlotTimeSeriesDataframe(df , signal = 6 , timeRange = (0, 15*250), Series = None ):
    cols = df.columns.values
    #Pull the time data of requested range
    t = df['Time'][timeRange[0]:timeRange[1]]
    #pull the dependent data of requested range
    y = df[cols[signal]][timeRange[0]:timeRange[1]]
    #plot both signals
    plt.plot(t, y)
    #If an extra series is supplied, plot it.
    if Series is not None:
        try:
            if Series.movingAverage:
                #if the series to be plotted is a moving average subtract
                #half the value of time window of the average from the x-value of plot.
                #this isnt mathmatically true but it makes the plot look better.
                phaseShift = True
        except:
            phaseShift = False


        if phaseShift:
            #Todo Do something better than this.
            y1 = Series[timeRange[0]:timeRange[1]].values
            plt.plot(t-(Series.timeWindow/2) , y1)
        else:
            y1 = Series[timeRange[0]:timeRange[1]].values
            plt.plot(t,y1)
    plt.show()



def Moving_average(df, signal = 6, timeWindow = 1 ):
    #Returns a series representing the moving average using the timewindow.
    samplingRate = df.samplingRate
    cols = df.columns.values
    window = round(timeWindow*samplingRate)
    #Use pandas to calc the moving average. It seems to have some lag in it.
    #Maybe we can caluclate
    Mov_ave = df[cols[signal]].rolling(window).mean()
    #flag to let us know its a moving average.
    Mov_ave.movingAverage = True
    #Store the timewindow in sections so we know later what we averaged over.
    Mov_ave.timeWindow = timeWindow
    return Mov_ave


def detectLargePeaks(df, signal = 6, sensitivity = 10 , timeWindow = 1 ):
    #The point here is to take the mean of the entire trace and count the number of peaks that exceed the sensitivity threshold.
    #Then we should be able to return a binary mask that corrisponds to the ICP spike height.
    samplingRate = df.samplingRate
    cols = df.columns.values
    window = round(timeWindow * samplingRate)
    Z = zscore(df[cols[signal]] , window)
    #Return a mask of the place where the zscore exceeds the sensitivity
    return Z.mask(Z<sensitivity)
    
    #Maybe also return a count of the number of nonconsequtive points where that happens? Otherwise we can just assume the first True is the hit.


#Stole from stack overflow, simple rolling zscore calculator
def zscore(x, window):
     r = x.rolling(window=window)
     m = r.mean().shift(1)
     s = r.std(ddof=0).shift(1)
     z = (x - m) / s
     return z

def movingAverageSeries(series, samplingRate =250 , timeWindow = 1 , Center = False ):
    #Returns a series representing the moving average using the timewindow.
    cols = df.columns.values
    window = round(timeWindow*samplingRate)
    #Use pandas to calc the moving average. It seems to have some lag in it.
    #Maybe we can caluclate
    Mov_ave = series.rolling(window, center=Center).mean()
    #flag to let us know its a moving average.
    Mov_ave.movingAverage = True
    #Store the timewindow in sections so we know later what we averaged over.
    Mov_ave.timeWindow = timeWindow
    return Mov_ave

def findFirstNonNan(series):
    #Given a pandas mask, find the first nonzero entry and return the index.
    # inelegant way to find the first nonzero entry in the mask.
    for idx, item in series.items():
        # print(idx, item)
        if not np.isnan(item):
            return idx
        else:
            continue

    print("No nonnan's found")
    return False

def NormalizeMinStrat(time, series):
    #normalize to min value = 0 , max value  = 100
    series_norm = 100* (series - series.min()) / (series.max() - series.min())
    #figure out if the value is going up or down in the last 20 points
    stepBack = 100
    diff = series.diff()
    ave_velo = diff[len(diff)-stepBack:len(diff)-1].mean()
    print(ave_velo, series_norm[series_norm.index[-1]])
    return (ave_velo, series_norm[series_norm.index[-1]])

    # plt.plot(time,series_norm)
    # plt.show()



def differentateSeries(series):
    # calc the derviative point by point and return that as a series.
    # Assumes that points are evenly sampled.
    return series.diff()

def ICPSpikeAnalysis(df):
    cols = df.columns.values
    signal , sensitivity = (6, 6)
    Zscore_mask = detectLargePeaks(df, timeWindow= 0.5 ,  signal=signal, sensitivity=sensitivity)
    if findFirstNonNan(Zscore_mask) is False:
        #try other signal channel
        signal = 9
        Zscore_mask = detectLargePeaks(df, timeWindow= 0.5, signal=signal, sensitivity=sensitivity)
        #If that doesn't produce a result go back to the first signal.
        if findFirstNonNan(Zscore_mask) is False:
            signal = 6
            Zscore_mask = detectLargePeaks(df, timeWindow=0.5, signal=signal, sensitivity=sensitivity)

    #Calculate the moving average of the ICP using 0.5 a second to filter out the heartbeat and focus on the breathing artifact
    ICPMoving = movingAverageSeries(df[df.columns.values[signal]], timeWindow=0.5, Center = False)

    ICP_spike = findFirstNonNan(Zscore_mask)

    # if not ICP_spike:
    #     print("No peaks observed above sensitivity threshold")
    # else:
    #     print(ICP_spike)

    #use a linear regression on the data just before the ICP spike, slope of the regression will be the result.
    #Issue here is avoiding contamination by the icp spike itself, which if we use it to fit the slope it will always be highly positive.
    #While still remaining quantitative
    skip_points_before = 20
    regression_points = 350

    x= df["Time"][ICP_spike-skip_points_before-regression_points:ICP_spike-skip_points_before]
    # y= df[cols[signal]][ICP_spike-skip_points_before-regression_points:ICP_spike-skip_points_before]
    y = ICPMoving[ICP_spike - skip_points_before - regression_points:ICP_spike - skip_points_before]


    # #Take the average of the whole trace compared to the last 100 points before the hit.
    # Average = df[cols[signal]][0:ICP_spike-skip_points_before].mean()
    # print(Average , df[cols[signal]][ICP_spike-skip_points_before-regression_points:ICP_spike-skip_points_before].mean()  )
    # if df[cols[signal]][ICP_spike-skip_points_before-regression_points:ICP_spike-skip_points_before].mean() > Average:
    #     print("Final 400 ms before impact are ABOVE the mean of the trace")
    # else:
    #     print("Final 400 ms before impact are BELOW the mean of the trace")

    #TODO use this fit function to determine the place in the respratory cycle
    # res = fit_sin(df["Time"][ICP_spike-1000:ICP_spike], ICPMoving[ICP_spike-1000:ICP_spike])

    slope, percentIntoCycle = NormalizeMinStrat(df["Time"][ICP_spike-1000:ICP_spike], ICPMoving[ICP_spike-1000:ICP_spike])


    #Adding a custom cost function
    # res = fit_sin_regularized(df["Time"][ICP_spike - 1000:ICP_spike], ICPMoving[ICP_spike - 1000:ICP_spike])

    #I guess sci-kit has some issues understanding pandas series as inputs. Cast them as numpy arrays to fix it.
    # TrendLine = linear_model.LinearRegression()
    # TrendLine.fit(np.array(x).reshape(-1,1), np.array(y).reshape(-1,1) )

    #Plot the regression against the smoothed data.
    PlotTimeSeries(df["Time"], timeRange=(0 * 250, ICP_spike-skip_points_before), dataSeries= {'ICP': df[df.columns.values[signal]], 'ICPMoving': ICPMoving}  )
    # plt.plot(x, TrendLine.predict(np.array(x).reshape(-1,1)))
    # plt.plot(df["Time"][ICP_spike-1000:ICP_spike+skip_points_before+10],res['fitfunc'](df["Time"][ICP_spike-1000:ICP_spike+skip_points_before+10]))
    # plt.plot(df["Time"][ICP_spike],res['fitfunc'](df["Time"][ICP_spike]),'.', ms=20 ,label= 'Point of intrest')
    plt.title(str(df.Experiment_name) + '   ICP spike = ' + str(df.ICPSpikeHeight) + '   BrainDamage % =   ' + str(df.BrainDamage))
    plt.xlabel("Time (seconds)")
    plt.ylabel("ICP pressure (mmHg)")

    #From the fitted parameters we can deduce the place in the cycle that the hit takes place during.
    #Since the derivate of sin is cos we can figure out if the sinusoid is going up or down at the time of the hit.
    #Then I guess we can calculate the theta at the ICP spike?
    # res['cycle_frac'] = thetaAtPoint(res['amp'], res['omega'], res['phase'], res['offset'] , df["Time"][ICP_spike])
    # return res
    return (slope, percentIntoCycle)


def Parse_experiment_dir(dirpath, StartFile = None):
    #Todo Basically I want this function to return a collection of the paths to the raw data files in the directory.
    # If we return a list, it should be a tuple of two lists with the first as the experiment time the file starts, and the second as a list of the paths.
    # However, it's probably better to use an ordered dict. The order of the dict is therefore the order of the
    # date modified of the files and then the key is the time point the file is assosated with.
    # 9/19/2019 Coming back to this, I think it might be better to just have all of the files in the experimental directory in order of when they were made.
    # Then we can find a sensable start, end, and death times for the acute data. If these can't be found automatically we can find them manually.

    #Return ordered Dict. look for information under ['metadata']
    dirLst = getfiles(dirpath)
    od = OrderedDict()
    od['metadata'] = dict()
    od['metadata']['expPath'] = dirpath
    Dir = os.path.basename(os.path.normpath(dirpath))
    od['metadata']['expNumber'] = ''    #Get the experiment number as a string.
    for char in Dir:
        if char.isnumeric():
            od['metadata']['expNumber'] +=char

    # od['metadata']['expNumber'] = int(od['metadata']['expNumber']) #Cast into an integer? #maybe remove, I think I use strings elsewhere.

    #Access the events file, take it out of the dir
    if 'EVENTS' in dirLst:
        od['metadata']['Events'] = dirLst.pop(dirLst.index('EVENTS'))

    #Read the experiment log and take it out of the dir.
    if 'Experiment log' in dirLst:
        od['metadata']['ExpLog'] = dirLst.pop(dirLst.index('Experiment log'))

    #Attempt to find the start of the experiment.
    #Lame, this doesn't exactly work every time. We have times when we start over the aquistion to get data from the after 240 time.
    #TODO Need to get the exp start time from something... We could always refer to the excel sheet, it's inelagant, but it should work.
    if StartFile is not None:
        od['metadata']['ExpStart'] = StartFile #user supplies the start file.
    else:
        #If no user input see if there is more than 1 candidate.
        od['metadata']['ExpStart'] = None
        candidates = []
        for filename in dirLst:
            if 'Time 0-10' in filename:
                candidates.append(filename)

        if len(candidates) == 1:
            #Single Candidate, make that the exp start file.
            od['metadata']['ExpStart'] = candidates[0]
        elif len(candidates) > 1:
            #If there is more than 1 possible start file, check if the file is in the correct ball park for length, most of our false starts are only a few seconds of data.
            for c in candidates:
                if os.path.getsize(os.path.join(dirpath, c)) < 10000000:
                    P = candidates.pop(candidates.index(c))

            #
            od['metadata']['ExpStart'] = candidates[0]  # always use the earliest starting file, this should be correct if we eliminate the false starts above.

    #Same logic to get the latest baseline.
    od['metadata']['Base'] = None
    candidates = []
    for filename in dirLst:
        if 'Time 0-10' in filename:
            candidates.append(filename)
    if len(candidates) == 1:
        od['metadata']['Baseline'] = candidates[0]




    #Return the dir list entirely. This is all the labview data sorted by when the file was created.
    od['dataLst'] = dirLst
    return od


def LoadRawExperiment(expdict = None, TimeTup = (0, 15)):
    #Ok, so we have all the labview data, and we know when the experiment started,
    #next step is to actually load the dataset from multiple files. use a seperate method to get basline.
    Path = expdict['metadata']['expPath']
    dataLst =expdict['dataLst']

    #if TimeTup is 0, get the first file loaded.
    if TimeTup[0] == 0:
        LoadIdx = dataLst.index(expdict['metadata']['ExpStart'])
        print('starting from the begining of the experiment {0}  {1}  '.format(expdict['metadata']['expNumber'], dataLst[LoadIdx]))
    else:
        #TODO Figure out how far into the exp you need to start if not the begining
        pass
    #Load the first exp.
    df = LoadLabviewData(path=os.path.join(Path, dataLst[LoadIdx]), samplingRate = 250, headers = 5)
    df.set_index('Time')
    upToTime = df['Time'].iloc[-1]

    while round(upToTime/60) < TimeTup[1]:
        LoadIdx += 1
        print('loading next file, ' + dataLst[LoadIdx])
        df1 = LoadLabviewData(path=os.path.join(Path, dataLst[LoadIdx]), samplingRate=250, headers=5)

        #Add the upToTime to the new time col in the new dataframe, then merge them vertically.
        df1['Time'] = df1['Time'] + upToTime
        df = pd.concat([df, df1], axis=0, ignore_index=True)

        upToTime = df['Time'].iloc[-1] #update the time we are up to in seconds.


        #Add a check if there is even another file to load in the next iter of the while loop.
        if not round(upToTime/60) < TimeTup[1]:
            print('Succesfully loaded the full dataset from {0} to {1} min'.format(str(TimeTup[0]), str(TimeTup[1])))
            break
        elif LoadIdx+1 > len(dataLst)-1:
            print('could not load the full amount of time on exp {0} stopped at time {1}'.format(expdict['metadata']['expNumber'], str(upToTime/60)))
            break
        elif 'BL' in dataLst[LoadIdx+1]:
            print('For some reason there was a Baseline file in the middle of our experiment. weird right?')
            break

    return df

def getFileLabel(String, fileType = 'exp'):
    # Takes the experimental filename and produces the best label for it. May not actually need a function for this. This might get annoying.
    #return a tuple of the string to write as the key for the OD from above and the datetime object for the thingy.
    if fileType == 'exp':
        split = String.split(' ')
        if split[1] == '10-30':
            format = '%m-%d-%y %I:%M:%S %p'
            date = split[2] + ' ' + split[4] + ':' + split[5] + ':' + split[6] + ' ' + split[7]
            return ('20-30', datetime.datetime.strptime(date,format))
        else:
            format = '%m-%d-%y %I:%M:%S %p'
            date = split[2] + ' ' + split[4] + ':' + split[5] + ':' + split[6]+ ' ' + split[7]
            return (split[1], datetime.datetime.strptime(date,format))
    elif fileType == 'Baseline':
        pass


def getfiles(dirpath): #Pulled from stack overflow, returns files sorted by last time modified.
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return a

def SA1standardRAWloadingFunction(useCashe = False):
    # So the same logic applies as in the standard loading function. We put code here that loads the entire labview dataset.
    # I do wonder how long it would be and if we can cashe it. Probably can.



    #What is the point? well now we can start to write automated analysis code that decomposes the signals down or time averages or filters them.
    # Timing function
    t1, timeTotal = time.time(), time.time()
    # Figure out if a cashe file exists.
    CasheFilename = 'Sa1RAWataset.pickle'

    CasheExists = os.path.exists(os.path.join('cashe', CasheFilename))

    if not useCashe or not CasheExists:
        print('loading dataset fresh from on disk ')

        WD = os.getcwd()
        CashePath = os.path.join(WD, 'cashe')
        # Keep the standard loading code here so we can easily call it other places.
        experiment_lst = np.arange(2018104, 2018168 + 1)  # Create a range of the experiment lists
        censor = np.isin(experiment_lst, [2018112, 2018120, 2018123, 2018153,
                                          2018156])  # Create a boolean mask to exclude censored exps from the lst.

        censor = [not i for i in censor]  # Invert the boolean mask
        experiment_lst = experiment_lst[censor]  # Drop censored exp numbers
        experiment_lst = list(map(str, experiment_lst))  # Convert the list to strings

        MasterPath = r'C:\Program Files\RI DAS\DATAFILES' #where all the RIDAS data lives.

        RawDataSet = dict()

        for exp in experiment_lst:
            expDir = Parse_experiment_dir(os.path.join(MasterPath, 'EXP #' + str(exp)))
            if expDir['metadata']['ExpStart'] is not None:

                print('Loading experiment: {0}  Total of  {1} Files found'.format(str(exp), str(len(expDir['dataLst']))))

                #TODO Load 0-240 min, file it under raw and let the RANDOM ACCESS MEMORY (RAM) FLOW THROUGH YOU.
                #TODO Ok that was a spectacular failure. Next strat is to cashe each experiment and then leave a path to the cashe in the dict.
                # #only makes sense if we think that accessing from pickle is faster than loading from disk. It probably is.
                #
                #TODO Cashe each experiment raw data to disk JSON and leave the full filename in the experiment directory.
                # RawDataCashe = LoadRawExperiment(expDir, TimeTup=(0, 240))
                # CasheExpFilePath = os.path.join(CashePath, 'exp '+ expDir['metadata']['expNumber'] + '.pickle')
                # RawDataCashe.to_json(CasheExpFilePath)
                # expDir['metadata']['RawDataCashe'] = CasheExpFilePath
                #TODO Find a casheing strategy that works, we are actually about 50% less space  writing to JSON objects.
                #Right now, Casheing doesn't really work, so everytime you need raw data from on disk, you have to go load it.
                RawDataSet[expDir['metadata']['expNumber']] = expDir

        print("Total Dataset Loaded and processed at {0} seconds".format(time.time() - timeTotal))

        # Save the dataset to the cashe. (Maybe date the cashes, or that might lead to file inflation.
        print('Cashing dataset to disk.')
        with open(os.path.join('cashe', CasheFilename), 'wb') as f:
            pickle.dump(RawDataSet, f)
            print('Cashe dumped to disk  at {0} '.format(time.time() - timeTotal))

    elif not CasheExists:
        os.mkdir(os.path.join('cashe'))
        print('Call loader recursively to refresh Cashe.')
        SA1standardRAWloadingFunction(useCashe=False)

    else:
        # Load the dataset from disk.
        with open(os.path.join('cashe', CasheFilename), 'rb') as f:
            RawDataSet = pickle.load(f)
            print('Cashe loaded from disk  in {0} '.format(time.time() - timeTotal))

    return RawDataSet




if __name__ == "__main__":


    RawDataSet = SA1standardRAWloadingFunction()



    # for exp in RawDataSet:
    #     expDict = RawDataSet[exp]
    # for exp in RawDataSet:
    #     expDict = RawDataSet[exp]


        # plt.plot(df['Time'].iloc[:], df.iloc[:, 11])
        # plt.title(expDict['metadata']['expNumber'])
        # plt.show()

    # exppath = r'C:\Program Files\RI DAS\DATAFILES\EXP #2018142'
    #
    # expdict = Parse_experiment_dir(exppath)
    #
    # print(expdict['metadata']['ExpStart'])
    #
    # df = LoadRawExperiment(expdict, TimeTup=(0, 60))
    #
    # #Plot the weight by time. Works properly.
    # plt.plot(df['Time'].iloc[:], df.iloc[:, 11])
    #     # plt.show()






    #
    # #Old attempt to look at the data from just the impact. Not a crazy thing to do, but not at the moment.
    # #Get an excel sheet with the paths to all the ICP spike height data.
    # DataSetPath = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\Brain_injury_dataset.xlsx"
    # Dataset = pd.read_excel(DataSetPath, header=[0])
    # print(Dataset.head())
    # Results_lst = []
    # #Iterate through the dataset, performing the analysis.
    # for row in Dataset.itertuples():
    #     print(row[1])
    #     if bool(row[4]) is True:
    #         df= LoadLabviewData(os.path.join(row[2],row[3]))
    #         #This is the fiber optic channel, it's empty in all the experiments after # 08,
    #         # but what it does pick up on is the signal we sent to the soilinoid
    #         # PlotTimeSeries(timeseries=df['Time'], dataSeries={'empty channel': df[df.columns.values[9]]} )
    #         # plt.show()
    #         #Add in some information from the Dataset, Name of experiment for example.
    #         df.Experiment_name = row[1]
    #         #Add the ICP Spike Height and ammount of brain damage found
    #         df.ICPSpikeHeight = row[6]
    #         df.BrainDamage = row[5]
    #         Results = ICPSpikeAnalysis(df)
    #         Results_lst.append([Results[0], Results[1],  df.ICPSpikeHeight ])
    #         # print(Results[0][0])
    #
    #         # plt.show()
    # plt.show()
    #
    # #Plot the results of the analysis here.
    # split1, split2 , split3 = zip(*Results_lst)
    # sign = lambda x: (1, -1)[x < 0]
    # lst = []
    # for idx, item in enumerate(split3):
    #     lst.append(split2[idx] * sign(split1[idx]))
    # plt.scatter(split3,tuple(lst))
    # plt.title("Phase vs ICP spike height")
    # plt.xlabel("ICP spike height")
    # plt.ylabel("cycle of the resp signal")
    # plt.show()

    #Form the dataset of processed experiments into a dataframe and output it to excel
    




    #output the analysis.

    #Test code for the functions in this file.
    # path = 'C:\Program Files\RI DAS\DATAFILES\EXP #2018122\Time 0-10 10-30-18 at 02 47 40 PM'
    # path2 = 'C:\Program Files\RI DAS\DATAFILES\EXP #2018121\Time 0-10 10-29-18 at 01 44 34 PM'
    # path3 = 'C:\Program Files\RI DAS\DATAFILES\EXP #2018118\Time 0-10 10-02-18 at 01 26 46 PM'
    # path4 = 'C:\Program Files\RI DAS\DATAFILES\EXP #2018116\Time 0-10 09-25-18 at 12 40 17 PM'
    #
    # df = LoadLabviewData(path)
    # ICPSpikeAnalysis(df)
    # df2 = LoadLabviewData(path2)
    # ICPSpikeAnalysis(df2)
    # df3 = LoadLabviewData(path3)
    # ICPSpikeAnalysis(df3)
    # df4 = LoadLabviewData(path4)
    # ICPSpikeAnalysis(df4)
    #
    # plt.legend(['High','Low','Low','High'])
    # plt.show()

