""""
This file loads the excel data from the survival HS/TBI experiment. It parses the Excel sheet into a dataframe and has
output functions, subselection functions and plotting functions that should be general enough to use conversationally.
Matt Maggio
4/12/19
"""
import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter as xlwrite
import time, re, datetime, winsound
import lifelines, sys, pdb

def Parse_excel(path=None, Experiment_lst = ["2018124"]):
    #There were many formating decisions made to make the sheet more human readable.
    #basically this function is gathering the data back into a machine readable form.
    #Output is going to be a dict for each experiment that has key value pairs and pd.series of data as outputs.
    #later I think I can pile all of this into a giant dataframe or I can have a subselection function to go through and pull the data I need using a loop.

    t1, timeTotal = time.time() , time.time()
    xls = pd.ExcelFile(path)
    df = pd.read_excel(xls, sheet_name=experiment_lst, header= None)

    print("loaded dataset from excel takes {0} seconds ".format(time.time() - t1))


    Output_lst = []

    #Iterate through all the sheets, get the sheet as a dataframe, reformating until it's sensible.
    for exp in Experiment_lst:
        print("Loading experiment {0}".format(exp))
        t1 = time.time()
        df_exp = df[exp]
        #Get the Unique var names
        slice_lst = [slice(3,21), slice(23,30)]
        UVar = pd.concat([df_exp.iloc[slice_lst[0],1], df_exp.iloc[slice_lst[1],1]])
        UVar = Drop_units(UVar) # Shorten the names to take the units out.
        UVal = pd.concat([df_exp.iloc[slice_lst[0],4], df_exp.iloc[slice_lst[1],4]])
        Udict = dict(zip(UVar, UVal))

        # Time values are known a priori
        Time = np.concatenate([np.arange(0, 20, 5), np.arange(30, 240+15, 15), np.arange(8 * 60, (72+4) * 60, 4 * 60)])

        #Get the Repeditive var names and data. Similar to above.
        indexX = [slice(33,63), slice(64, 74), slice(86, 104), slice(105, 123),  slice(128, 137), slice(138, 147), slice(158, 179), slice(180, 197) ]
        indexY = slice(5, 41)

        RVar = list()
        RVal = list()
        for i in range(0,len(indexX)):   #Run a loop to make this less verbose
            RVar.append(df_exp.iloc[indexX[i], 1])
            RVal.append(df_exp.iloc[indexX[i], indexY])
        RVar = pd.concat(RVar, axis=0)
        RVar = Drop_units(RVar) # Shorten the names to take the units out.
        RVal = pd.concat(RVal, axis=0)

        RDict = dict()
        #Loop through RVar, adding key and value pairs adding the series to the dict
        for index, key in enumerate(RVar):
            RDict[key] = RVal.iloc[index, :]

        ExpDict = {**Udict, **RDict}
        #Add a few more fields and then stack up into a list to complete the experiment loading
        ExpDict["experimentNumber"] = exp
        ExpDict["Time"] = Time
        print("Experiment {0} Loaded and Parsed taking {1}".format(exp, time.time()-t1))
        Output_lst.append(ExpDict)
        print("done")
    print("Total Dataset processed in {0} seconds".format(time.time() - timeTotal))
    return Output_lst

def Drop_units(series):
    #Remove everthing that is between (), remove all commas
    #Learning regex. Be Gentle.
    output = list()
    for entry in series:
        entry = re.sub('\(.*?\)','', entry)#Remove all text between ()'s
        entry = re.sub(r',',' ', entry) #Remove commas replace with white space
        entry = re.sub(r'\s\s+', ' ', entry)  # Remove double white space
        output.append(entry.strip())
        # print(entry)
    return pd.Series(output)

def Randomize_groups(Dataset):
    #This function should only be used to randomly assign groups to the data before the code has been broken and to test other code.
    Group_original = ["A", "B", "C", "D"]
    Group = Group_original.copy() #Draw each group with replacement
    for exp in Dataset:
        if exp["Intervention"] == "XXX":
            exp["Intervention"] = np.random.choice(Group)
            Group.remove(exp["Intervention"])
            if len(Group) == 0: #If we have drawn all 4 groups, replace for the next loop.
                Group = Group_original.copy()
    #return Dataset #Intentionally broken to prevent dumb mistakes.


def selectData(Dataset, Key = None, groupBy = None, averageGroups= True, returnLists = True):
    #This function should iterate through all the data in the list and subselect the key of intrest and return it.
    #Organize by group, and then return a list of the data by group. If groupby is a key and not None then attempt to group by the ordinal values suggested by the input.
    #this should work on blocks for example. If averageGroups is False we will return a dataframe instead of a series.
    #Key should be a list of the fields you are intrested in. We will return a list of the same size as Key containing the data either as a series for each group.
    #TODO Start with a single Key. Eventually it would be SWEET to take Key as a list and call this function recursively!
    #Clarifying the output. We need to have a dict output. [Group A: Data A, Group B: Data B, Group C: Data C] Groups should either be dataframes or series depending on requested output
    #Dataframe seems like a bad fit here. Resorting to np.array().
    groupNames = []
    data = dict()
    if Key == None:
        Key = 'Ao systolic'  #Resonable default

    if groupBy == None:
        GroupField = 'Intervention'  #Resonable default
    else:
        GroupField = groupBy
    for exp in Dataset:
        if not exp[GroupField] in groupNames:  # If our treatment group hasn't been seen yet, add it.
            groupNames.append(exp[GroupField]) # Add to group name since we haven't seen it before whatever type it is doesn't matter. y/n, int, catag
            data[str((exp[GroupField]))] = []  # Create an empty list
            # print(groupNames)
        # print(str(exp['Intervention']) , '  ', str(exp[Key]) )
        data[str((exp[GroupField]))].append(exp[Key])  #Add the data by appending it to the list that is the value of the dict.

    #TODO Average the data together and send the mean and stdev out, need another layer of dict?

    if returnLists == False:
        for key in data.keys():
            #let numpy reform the data into an array. See what happens.
            data[key] = np.array(data[key], dtype=np.float64)
        return data
    else:
        return data

def BoxPlot(Data):
    #Call with preselected data using select data function, The returnLists param should be set to True
    fig1, ax1 = plt.subplots()
    boxes = []
    for key in Data.keys(): #Add other types and sizes of data for sensible plotting.
        if isinstance(Data[key][0], datetime.time):
            box = []
            for point in Data[key]:
                box.append((point.hour*60) + point.minute)
            boxes.append(box)
    ax1.boxplot(boxes)

    return (fig1, ax1) # This is a plotting function, so it returns the plot objects.




def LinearPlot(Data, xData, ylabel, averageGroups= False, Groups = None ):
    # Groups = ["A"] #Test code, remove If groups is == None use all the keys in the dict as the groups
    color_lst = ['blue', 'purple', 'red', 'green', 'cyan']
    fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
    plt.title("Group {0} vs Time".format(str(Groups)))
    if Groups == None:
        Groups = Data.keys()
    for index, group in enumerate(Groups):
        for exp in Data[group]:
            if not len(Groups) == 1: #If more than one group is plotted, keep colors the same per group.
                ax.plot(xData, exp, color=color_lst[index])
                ax2.plot(xData, exp, color=color_lst[index])
            else:
                ax.plot(xData,exp)
                ax2.plot(xData,exp)
    #This is just complex plotting code to make the xscaling work.
    ax.set_xlim(0, 240)
    ax2.set_xlim(240 + (3*60), 72*60)
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()
    d = .015
    #This stuff I don't understand. Some clever guy on stack overflow is getting credit for it. https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib/32186074#32186074
    #TODO Consider this package: https://github.com/bendichter/brokenaxes
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    plt.xlabel("Time (min)") #TODO Usually true. Work on this later
    # ax.set_xticks(xData) #TODO Cut the distance between points in the survival phase so we get equal weights.
    # ax.set_xticklabels(xData)
    ax2.ylabel(ylabel)
    return (fig, (ax, ax2)) # This is a plotting function, so it returns the plot objects.

def survivalPlot(Dataset=None): #TODO Implement.

    from lifelines.datasets import load_waltons
    from lifelines.statistics import logrank_test
    import distutils
    if Dataset == None:  #Test data
        df = load_waltons()  # returns a Pandas DataFrame
        groups = df['group']
        Treatment = 'miR-137'
        print(df.head())
    else:
        #Collect the survival data and groups in a manner similar to above
        # Survival time
        # Survival 240 minutes
        # Survival 72 hours
        Event = selectData(Dataset, Key='Survival 72 hours', returnLists=True)
        Event = [distutils.util.strtobool(e) for e in Event] #FIX THIS
        Time = selectData(Dataset, Key='Survival time', returnLists=False )
        Groups = Event.keys()
        (Earray, Tarray, Garray) = (np.empty(0), np.empty(0), np.empty(0))
        for group in Groups:
            Earray = np.concatenate(Earray, Event[group])
            Tarray = np.concatenate(Tarray, Time[group])
            Garray = np.concatenate(Garray, np.array(((str(group)+' ')*len(Event[group]).split(' '))))
        df = pd.DataFrame({'E':Earray, 'T':Tarray,'groups': Garray})
        print(df)
    """
        T  E    group
    0   6  1  miR-137
    1  13  1  miR-137
    2  13  1  miR-137
    3  13  1  miR-137
    4  19  1  miR-137
    """

    T = df['T']
    E = df['E']
    ix = (groups == Treatment)

    kmf = lifelines.KaplanMeierFitter()
    kmf.fit(T, event_observed=E)  # or, more succinctly, kmf.fit(T, E)
    kmf.survival_function_
    kmf.cumulative_density_
    kmf.median_
    res = logrank_test(T[~ix], T[ix], E[~ix], E[ix], alpha=0.99)
    res.print_summary()
    kmf.plot_survival_function()

    # kmf.plot_cumulative_density()

    # groups = df['group']
    # ix = (groups == 'miR-137')
    #
    # kmf.fit(T[~ix], E[~ix], label='control')
    # ax = kmf.plot()
    #
    # kmf.fit(T[ix], E[ix], label='miR-137')
    # ax = kmf.plot(ax=ax)
def cloneStringToList(string, N):
    lst = ((string + ',') *N).split(',')
    lst.pop(-1)
    return lst


def SPSSExport(Dataset = None): #TODO. Fix the reapeat colunms. Annoying that it keeps breaking.
    #SPSS has weird needs for data, but It shouldn't be that hard to output in a form. I quoted dr.G 4 hours. I can probably do it in one if I get my ass in gear.
    #4 Hours later. Still plugging holes.
    outpath = r'C:\Users\mattm\PycharmProjects\ \Export\\'
    path = outpath +  'SPSSExport.xlsx'
    workbook = xlwrite.Workbook(path, {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet()
    #Get all fields from the first dataset.
    fields = Dataset[0].keys()

    #Develop a list of interventions.
    groups = list(selectData(Dataset).keys())
    # Develop a dict that has the indexes as a list for each group.
    groupIdx = {key: [] for key in groups}
    for idx, data in enumerate(Dataset):
        groupIdx[str(data['Intervention'])].append(idx)

    # Write ALL the headers
    repeatFields = ['Time', 'experimentNumber', 'Intervention', 'Block']
    for idx, field in enumerate(repeatFields):  # Write the repeated fields as headers
        worksheet.write(0, idx, field)




    (row, col) = (1, 0)  # Where to start counting from for the data, if you are going to have repeats like time and exp num they have to be factored in.

    # exp = Dataset[3]    #Start with 2018107
    for ix, expReal in enumerate(Dataset):

        exp = expReal.copy() #Apparently pop removes it from the actual object in memory. LAME.

        if ix == 0:
            for idx, field in enumerate(fields):  # Write the rest of the headers fields as headers
                worksheet.write(0, idx+len(repeatFields), field)

        #Format the repeating fields here.  #Make this a loop, for now just add reapeat fields here.
        Time = exp['Time']
        expNum = cloneStringToList(exp['experimentNumber'], len(Time))
        intervention = cloneStringToList(exp['Intervention'], len(Time))
        Block = cloneStringToList(str(exp['Block']), len(Time))
        array = list((Time, expNum, intervention, Block))

        for field in repeatFields: #Repeat fields should be popped from the experiment dict so we don't have duplicates.
            exp.pop(field)

        fields = exp.keys() #Re get the keys after we remove the repeat headers.
        if ix == 0: #Only on the first experiment are we worried about headers.
            for idx, field in enumerate(fields):  # Write the rest of the headers fields as headers
                worksheet.write(0, idx+len(repeatFields), field)

        fields = exp.keys()
        for idx, field in enumerate(fields):
                array.append(exp[field])

        #lets just get an array of the values and do a write_col at the end.
        for col, data in enumerate(array):
            # print(type(data))
            if isinstance(data, (type(pd.Series()), type(list()), type(np.array(1)))):
                worksheet.write_column(row, col, data)
            else:
                worksheet.write(row, col, data)
        row += len(Time)

    workbook.close()
    #Reload the data into a pandas dataframe and then replace the nans. This is much easier than trying to figure out if a nan is going to print. GOOD CLUGE
    df = pd.read_excel(path)
    df.to_excel(path)


def SigmaPlotExport(Dataset=None):
    outpath = r'C:\Users\mattm\PycharmProjects\GazmuriDataLoader\Export\\'
    path = outpath + 'SigmaPlotExport.xlsx'
    workbook = xlwrite.Workbook(path, {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet()

    # Write ALL the headers
    # repeatFields = ['Time'] #Other repeated cols don't make sense here Unless i'm bad at sigma plot. I probably am.
    Time = Dataset[0]['Time']
    (row, col) = (1, 0)
    Fields = ['Ao systolic', 'Ao systolic','Heart rate LV', 'ICP']
    # exp = Dataset[3] #2018107 for testing. #silly me, I can't go exp by exp because I need summary stats.

    array = np.empty(len(Time)*4, 1+(2*len(Fields))) * np.nan #Take that 4 out of there.

    for ix, field in enumerate(Fields):
        Data = selectData(Dataset, Key=field, returnLists=False)
        #Average across group. Then stack them and write the entire column out #Get the standard error too for error bars.
        Groups=[]
        Times=[]
        for group in Data.keys():
            if ix == 0: #First field we have to do all the groups in one col, all the times in another and all the data in the third.
                intervention = (cloneStringToList(group, len(Time))) #append the treatment group to the list first.
                data = np.nanmean(Data[group], 0)
                stdev = np.nanstd(Data[group], 0)/ Data['A'].shape[0]
                # array[0:] # and then on not the first field we do the same thing with the rest of the averaged data.


    #lets just get an array of the values and do a write_col at the end.
    for col, data in enumerate(array):
        worksheet.write_column(row, col, data)

    workbook.close()

def DescriptivesExportLinked(Dataset): #TODO Dr.G wants descriptives that are interactive so if you change the sheet it will update the sumarry table and summarry stats.
    pass






def MSCPlots(Dataset, makePlots = None):
    #This is where i'll keep code to reproduce plots I like. TODO Move all plotting code elsewhere when we functionalize this loader.
    outpath = r'C:\Users\mattm\PycharmProjects\GazmuriDataLoader\Figures\\'
    plotNum = 2 #How many plots are in the chart Add to this num if you want them to process automatically.
    if makePlots == None:
        makePlots = list(np.arange(1, plotNum+1))


    # These are the "Tableau 20" colors as RGB.  Default colors suck
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    for num in makePlots:
        if num == 1:
            #Plot the prep time vs block.
            Data = selectData(Dataset, Key='Preparation time', groupBy='Block')
            plt.rc('font', family='serif')
            fig, ax = BoxPlot(Data)
            fig.suptitle("Preparation Time by Block")
            ax.set_ylabel("Time (min)")

            ax.set_xlabel("Block number")

            ax.clip_on = False
            ax.spines['left'].set_position(('outward', 25))
            # ax.spines['bottom'].set_position(('outward', 25))
            # ax.spines['left'].set_smart_bounds(True)

            #standard stuff, should probably make a function to do this.
            fig.figsize =(12, 14)
            ax.spines["top"].set_visible(False)
            # ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # ax.spines["left"].set_visible(False)
            # Ensure that the axis ticks only show up on the bottom and left of the plot.
            # Ticks on the right and top of the plot are generally unnecessary chart junk.
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            # Make sure your axis ticks are large enough to be easily read.
            # You don't want your viewers squinting to read your plot.
            axLim = ax.get_ybound()

            ticks = 50
            plt.yticks(range(0, int(np.around(axLim[1]*1.2, decimals=-1)), ticks), [str(x) for x in range(0,  int(np.around(axLim[1]*1.2,decimals=-1)), ticks)], fontsize=10)
            plt.xticks(fontsize=12)
            ax.set_ylim(bottom=0, top=np.around(axLim[1]*1.2, decimals=-1)) #This is gonna change plot to plot.
            plt.rc('text', usetex=True)
            fig.savefig(outpath + 'Preparation Time.png', dpi=600, transparent=False, bbox_inches='tight')


        if num == 2:
            print("plotting the dura rip frequency")
            Data = selectData(Dataset, Key='Dural tear', groupBy='Block')
            DepthData = selectData(Dataset, Key='Impactor depth', groupBy='Block')

            Data['13'] = ['Y', 'N', 'N', 'Y']
            print(Data)
            plotData = dict.fromkeys(Data)
            for key in plotData:
                lst = Data[key]
                rip = 0
                nonrip = 0
                for item in lst:
                    if item == 'N':
                        nonrip += 1
                    else:
                        rip += 1
                plotData[key] = rip / (rip + nonrip)
            x, y = list(plotData.keys()), list(plotData.values())
            fig = plt.scatter(x, y)
            plt.ylabel("Dura Rip frequency")
            plt.xlabel("Block number")
            plt.title('Dura Rip Frequency by blocks')
            print(DepthData)
            fig.show()
            #TODO. Save the figure, just do it you lazy ass.

        if num == 3:
            # more plots
            print("Plotting Differences between MS animals and Non MS animals")
            Data = selectData(Dataset, Key='Lactate Ao', groupBy='Block')
            print(Data)

            #TODO. Save the figure, just do it you lazy ass.



def Parse_experiment_dir(dirpath):
    #Todo Basically I want this function to return a collection of the paths to the raw data files in the directory.
    # If we return a list, it should be a tuple of two lists with the first as the experiment time the file starts, and the second as a list of the paths.
    # However, it's probably better to use an ordered dict. The order of the dict is therefore the order of the
    # date modified of the files and then the key is the time point the file is assosated with.
    #Logic to decide based on filename what it is.
    #Return ordered Dict. look for information under ['metadata']
    dirLst = getfiles(dirpath)
    od = OrderedDict()
    od['metadata'] = dict()
    od['metadata']['expPath'] = dirpath
    #need a few flags for logical reasons.
    BaselineN = 0
    ExpStart = False

    for file in dirLst:
        if 'BL' in file: #Must be a baseline file
            print('Baseline found  '  + file)

        elif 'Time' in file: # must be an Experimental file
            if 'Time 0-10' in file: # must be the first experimental data file, unless we had a false start experiment thing that happened once or twice. use a flag to mark.
                if ExpStart is False:
                    (key, filetime) = getFileLabel(file)
                    od[key] = file #Store the file under the key for experiment time.
                    ExpStart = True
                    od['metadata']['ExperimentStart'] = filetime
                elif True: #TODO Determine if we had a false start or we restart the exp after 240 to keep recording during second surgery. Take corrective action

                    pass
            else:
                (key, timestamp) = getFileLabel(file)
                od[key] = file

            print('Experiment file found  '  + file)
        elif 'Experiment' in file:
            print('Experiment Log found '  + file)
        elif 'EVENTS' in file:
            print('EVENT Log found  '  + file)

    #Test code for display
    for key in od.keys():
        print(key)
        print(od[key])
        for field in od['metadata']:
            print('Printing Metadata')
            print(field)
            print(od['metadata'][field])

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




if __name__ == "__main__":


    # parse a raw labview experiment for what data is availble.
    exppath = r"C:\Program Files\RI DAS\DATAFILES\EXP #2018149"
    expData = Parse_experiment_dir(exppath)


    ##Created the experiment list
    experiment_lst = np.arange(2018104,2018168) #Create a range of the experiment lists #use +1 for the top number because we count from zero
    censor = np.isin(experiment_lst,[2018112, 2018120, 2018123, 2018153, 2018156]) #Create a boolean mask to exclude censored exps from the lst.
    censor = [not i for i in censor] #Invert the boolean mask
    experiment_lst = experiment_lst[censor] #Drop censored exp numbers
    experiment_lst = list(map(str, experiment_lst)) #Convert the list to strings


    #Older Master workbooks
    # path = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SA-1 Survival Phase (Master  Workbook) April 12, 2019 (masked).xlsx" #old data before groups became public.
    # path = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SA-1 Survival Phase (Master  Workbook) April 23, 2019 (Check Values Fixed).xlsx"
    # path = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SA-1 Survival Phase (Master  Workbook) May 2, 2019 (Check Values Fixed).xlsx"


    #Load the latest master workbook
    path = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SA-1 Survival Phase (Master  Workbook) July 5, 2019.xlsx"
    # print(experiment_lst)
    Dataset = Parse_excel(path=path, Experiment_lst=experiment_lst)
    Ao = selectData(Dataset, returnLists=False)


    # Run for KM analysis
    # survivalPlot(Dataset)
    #Dataset = Randomize_groups(Dataset)

    #Run when you want to reproduce figures. I have a whole function set up to store those!
    # MSCPlots(Dataset,[3])
    #Run to produce an xlsx file for spss to import.
    # SPSSExport(Dataset)
    #Run to produce an xlsx file for Sigma plot. Right now just do it for Ao.
    # SigmaPlotExport(Dataset)

    #Produce the descriptives sheet for easy identification of outlier values.
    DescriptivesExport(Dataset)


    #Make a sound when program finsishes so I know it's done!
    duration = 500  # milliseconds
    freq = 820  # Hz
    # winsound.Beep(freq, duration) #Make a noise to signal end of run.
    print("Program finished succesfully, that IS what you wanted right?")

"""
This is a list of the fields we are using currently.

Date
Series
Block
Intervention
Weight
Time ketamine injection
Time LL/TBI
Preparation time
Survival time
Survival 240 minutes
Survival 72 hours
Liver Lacerations
Estimated blood loss
Impactor depth
Impactor speed
Impactor dwell time
HES Infused
HES
Dural tear
Left hemisphere volume
Left hemisphere injury
Right hemisphere volume
Right hemisphere injury
EKG Heart Rate
Ao systolic
Ao diastolic
RA systolic
RA diastolic
PA systolic
PA diastolic
LV systolic
LV end-diastolic
Heart rate LV
LV dP/dt max
LV dP/dt min
ICP
CCO
PetCO2
PiCO2
Respiratory Rate
Temperature PA
Blood Removed
Temperature X
R Ao
K Ao
Angle Ao
MA Ao
PMA Ao
G Ao
EPL Ao
A Ao
CI Ao
LY30 Ao
R Ven
K Ven
Angle Ven
MA Ven
PMA Ven
G Ven
EPL Ven
A Ven
CI Ven
LY30 Ven
tHg Ao
O2Hb Ao
COHb Ao
MetHb Ao
O2Ct Ao
O2Cap Ao
sO2 Ao
pH Ao
pCO2 Ao
pO2 Ao
BE Ao
tCO2 Ao
HCO3 Ao
stHCO3 Ao
tHB Ao
SO2 Ao
HCT Ao
Lactate Ao
tHg PA
O2Hb PA
COHb PA
MetHb PA
O2Ct PA
O2Cap PA
sO2 PA
pH PA
pCO2 PA
pO2 PA
BE PA
tCO2 PA
HCO3 PA
stHCO3 PA
tHB PA
SO2 PA
HCT PA
Lactate PA
Na+ Ao
K+ Ao
Cl- Ao
Ca++ Ao
AnGap Ao
nCa++ Ao
pH Ven
tCO2 Ven
HCO3 Ven
Na+ Ven
K+ Ven
Cl- Ven
Ca++ Ven
AnGap Ven
nCa++ Ven
BSA
Hufner's Number
Ao mean
RA mean
PA mean
Cerebral PP Sys-ICP
Cerebral PP MAP-ICP
PetCO2 End Tidal Corrected
CaO2
CvO2
CCI
Stroke Volume Index
Left Ventricular Stroke Work Index
Cardiac Stroke Work Index
Right Ventricular Stroke Work Index
SVRI
PVRI
DO2I
VO2I
VO2/ DO2
Neurological deficit exam score part 1
Neurological deficit exam score part 2
Neurological deficit exam score total
Food acquisition test time 1
Food acquisition test error score 1
Food acquisition test time 2
Food acquisition test error score 2
Food acquisition test time 3
Food acquisition test error score 3
Food acquisition test time 4
Food acquisition test error score 4
Food acquisition test time 5
Food acquisition test error score 5
Novel object discrimination index
experimentNumber
Time

"""

