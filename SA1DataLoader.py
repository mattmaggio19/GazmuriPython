""""
This file loads the excel data from the survival HS/TBI experiment. It parses the Excel sheet into a dataframe and has
output functions, subselection functions and plotting functions that should be general enough to use conversationally.
Matt Maggio
4/12/19

This file has grown signficantly since I started it. It now houses a lot of functionality related to the SA-1 Experiments.
Need to possibly break this file out into a class calling the loader functions here.
6/7/19

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter as xlwrite
import time, re, datetime, winsound
import lifelines, sys, pdb
import SA1DataManipulation
import pickle
import os


def Parse_excel(path=None, Experiment_lst = ["2018124"]):
    #There were many formating decisions made to make the sheet more human readable.
    #basically this function is gathering the data back into a machine readable form.
    #Output is going to be a dict for each experiment that has key value pairs and pd.series of data as outputs.
    #later I think I can pile all of this into a giant dataframe or I can have a subselection function to go through and pull the data I need using a loop.


    xls = pd.ExcelFile(path)
    df = pd.read_excel(xls, sheet_name=Experiment_lst, header= None)

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

        # Special one off to get the hespand delievered in a time resolved manner.
        HesDelivered = df_exp.iloc[19, slice(5, 6 + 35)]


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
        ExpDict['HESDelivered'] = HesDelivered
        # print("Experiment {0} Loaded and Parsed taking {1}".format(exp, time.time()-t1))
        Output_lst.append(ExpDict)
        print("done")

    return Output_lst

def Drop_units(series):
    #Remove everthing that is between (), remove all commas
    #Learning regex. Be Gentle.
    output = list()
    for entry in series:
        # Find text between () that matches either OPTI or AVOX, it appears that we are overwriting some fields.
        TechniqueLst = ['(OPTI)', '(AVOX)', '(OPTI ELYTE)']
        Technique = re.search( '\(.*?\)', entry) #A match object for the first matching between pattern and the entry.

        entry = re.sub('\(.*?\)','', entry)#Remove all text between ()'s
        entry = re.sub(r',',' ', entry) #Remove commas replace with white space
        entry = re.sub(r'\s\s+', ' ', entry)  # Remove double white space
        if not Technique is None:
            if Technique.group(0) in TechniqueLst:
                # print('adding in the tech ()' + Technique.group(0) )
                output.append(entry.strip()+' '+ Technique.group(0) )
            else:
                output.append(entry.strip())
        else:
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


def selectData(Dataset, Key = 'Ao systolic', KeyType = None, groupBy = 'Intervention', returnForm = 'array'):
    #This function should iterate through all the data in the list and subselect the key of intrest and return it.
    #Organize by group, and then return a list of the data by group. If groupby is a key and not None then attempt to group by the ordinal values suggested by the input.
    #this should work on blocks for example. If averageGroups is False we will return a dataframe instead of a series.
    #Key should be a list of the fields you are intrested in. We will return a list of the same size as Key containing the data either as a series for each group.
    #TODO Start with a single Key. Eventually it would be SWEET to take Key as a list and call this function recursively!
    #Clarifying the output. We need to have a dict output. [Group A: Data A, Group B: Data B, Group C: Data C] Groups should either be dataframes or series depending on requested output
    #Dataframe seems like a bad fit here. Resorting to np.array().

    #returnForm has 3 accepted values.
    # 'means' means give just the mean value,
    # 'array' means give back a numpy array that the user can do w/e with.
    # 'standardError' is give back the standard error of each time point within the dataset.
    # 'N' is give back the length of the array along the dim.

    groupNames = []
    data = dict()

    #If user doesn't spesifcy a type we should expect the data to be we have to check on one of the experiments to find the type it is.
    # if KeyType == None:
    #     sample = Dataset[1][Key]
    #     samplenotnana = np.invert(np.isnan(sample))
    #     # KeyType = type(sample.iloc[sampleIdx[]])

    for exp in Dataset:
        if not exp[groupBy] in groupNames:  # if our treatment group hasn't been seen yet, add it.
            groupNames.append(exp[groupBy]) #Add to group name since we haven't seen it before whatever type it is doesn't matter. y/n, int, catag
            data[str((exp[groupBy]))] = []  # Create an empty list to placehold for the array.
            # print(groupNames)
        # print(str(exp['Intervention']) , '  ', str(exp[Key]) )
        data[str((exp[groupBy]))].append(exp[Key])  #Add the data by appending it to the list that is the value of the dict.



    try:
        for group in data.keys():
            #Let numpy Cast the data into an array.
            data[group] = np.array(data[group], dtype=np.float64) #Failure here means we shouldn't output anything. No number
            if  returnForm == 'means':
                #return means across the experiments.
                data[group] = np.nanmean(data[group], axis=0)
            elif returnForm == 'N':
                #return the counts of observations across the group.
                data[group] = np.sum(np.invert(np.isnan(data[group])), axis=0)
            elif returnForm == 'standardError':
                #Return the standard error of the mean. Sample std means ddof=1 for nanstd
                data[group] = np.nanstd(data[group], axis=0,ddof=1) / np.sqrt(np.sum(np.invert(np.isnan(data[group])), axis=0))
        return data #Send out the array if user doesn't want other ways of display.
    except:
        print('Data could not be accessed correctly.')
        return None


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

def DateTimesToStr(Dataset = None):
    #This function is simple, it changes datatimes to simple strings to export to other programs, If they are ever important we can modify this.
    for exp in Dataset:
        for field in exp.keys():
            data = exp[field]
            # print(type(data))
            if type(data) == type(datetime.datetime.now()):
                print('found a datetime obj ' + field + ' of type '+ str(type(data)))
                exp[field] = str(data.strftime('%d/%m/%Y %H:%M:%S')) #Cast into a string var
            elif type(data) == type(datetime.time()):
                print('found a datetime obj ' + field + ' of type '+ str(type(data)))
                exp[field] = str(data.strftime('%H:%M:%S')) #Cast into a string var
    return Dataset

def SPSSExport(Dataset = None):
    #Export all data to SPSS form, for repeated measurements they must be broken out into rows such that each animal and timepoint has a row or case.
    #Initially it was unclear which varibles need to be repeated in each case belonging to an animal, The basic survival and mixed models can be run only copying
    #The experiment number and Intervention so that each animal and group can be kept distinct. According to Dr.Miller, to use the cox proportional hazard model, the covariate needs to be copied into each case row for each animal.
    #Now this isn't a huge deal, but considering that we want to extract each of the baseline values for use as a fixed factor or predictive of surival, it is going to lead to a huge inflation of the spss dataset.
    #
    outpath = r'C:\Users\mattm\PycharmProjects\GazmuriDataLoader\Export\\'
    path = outpath +  'SPSSExport.xlsx'
    workbook = xlwrite.Workbook(path, {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet()

    #Get all fields from the first dataset.
    fields = Dataset[0].keys()

    #Use the groups in the same order as the other export functions.
    groups = ['NS', 'TLP', 'POV', 'AVP']

    # # Develop a list of interventions. More general, don't control the order though.
    # groups = list(selectData(Dataset).keys())
    # # Develop a dict that has the indexes as a list for each group.
    # groupIdx = {key: [] for key in groups}
    # for idx, data in enumerate(Dataset):
    #     groupIdx[str(data['Intervention'])].append(idx)

    # Write ALL the headers
    repeatFields = ['Time', 'experimentNumber', 'Intervention', 'Block']
    for idx, field in enumerate(repeatFields):  # Write the repeated fields as headers
        worksheet.write(0, idx, field)

    (row, col) = (1, 0)  # Where to start counting from for the data, if you are going to have repeats like time and exp num they have to be factored in.

    # exp = Dataset[3]    #Start with 2018107
    for ix, expReal in enumerate(Dataset):

        exp = expReal.copy() #Apparently pop removes it from the actual object in memory. LAME but nessasary.

        if ix == 0:
            for idx, field in enumerate(fields):  # Write the rest of the headers fields as headers
                worksheet.write(0, idx+len(repeatFields), field)

        #Format the repeating fields here.  #Make this a loop, for now just add reapeat fields here one at a time. turns out this isn't super nessasary, kinda nice to have these first though.
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
            if type(exp[field]) == type(float()):  #For all single measurements, turn them into repeating measurements here.
                array.append(cloneStringToList(str(round(exp[field],2)), len(Time)))
            elif not type(exp[field]) == type(pd.Series()):
                array.append(cloneStringToList(str(exp[field]),len(Time)))
            else:
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

    #

    #7/22/2019 Finally got the format that Dr.G wants for the sigmaplot data.
    #He wants the time, mean of a group, SEM of a group, n of a group, repeat for all groups. Do not include time points that are nans.

    (Row, Col) = (0,0)

    groups = ['NS', 'TLP', 'POV', 'AVP'] #Dr.G made a totally arbitrary desision, but it important to remain consistant.

    # Arbitrary = selectData(Dataset)
    # groups = Arbitrary.keys()

    #TODO Export the data for survival anaylsis first.

    Col += 2

    # Export the HES admin datax
    Time = Dataset[0]['Time']
    DoseRatio = SA1DataManipulation.ResolvedHESAdministration(Dataset, output='ratio', graph = False)
    DoseRatioN = SA1DataManipulation.ResolvedHESAdministration(Dataset, output='possibleDoses', graph = False)
    CheckTimes = [30, 120, 240, 8 * 60, 12 * 60, 16 * 60, 20 * 60, 24 * 60]
    worksheet.write_string(row=Row, col=Col, string=('Time'))
    worksheet.write_column(row=Row + 1, col=Col, data=CheckTimes)

    Bothsets = set(CheckTimes).intersection(Time)
    indices = [list(Time).index(x) for x in Bothsets]
    Col += 1
    for group in groups:
        worksheet.write_string(row=Row, col=Col, string=('HES ratio ' + '-' + group ))
        worksheet.write_column(row=Row + 1, col=Col, data=DoseRatio[group][sorted(indices)])
        worksheet.write_string(row=Row, col=Col + 1, string=('BLANK  ' + '-' + group ))
        worksheet.write_column(row=Row + 1, col=Col + 1, data=[])
        worksheet.write_string(row=Row, col=Col + 2, string=('HES doses possible' + '-' + group + ' N'))
        worksheet.write_column(row=Row + 1, col=Col + 2, data=DoseRatioN[group][sorted(indices)])
        Col += 3
    Col += 1

    #Export the data from the rest of the fields. from the groups in order.
    for field in Dataset[0].keys():
        means = selectData(Dataset, Key=field, returnForm='means')
        N = selectData(Dataset, Key=field, returnForm='N')
        stdErr = selectData(Dataset, Key=field, returnForm='standardError')

        if means is not None:
            BoolArray = np.invert(np.isnan(means[groups[3]]))
            if len(BoolArray.shape) == 0:
                #Write out the single value field.
                continue #Skip for now.
            else:
                Time = Dataset[0]['Time']
                #write time gap, time, list the field, then go through the groups
                Col += 1 #Gap column
                worksheet.write_string(row=Row, col=Col, string='Time (min)')
                worksheet.write_column(row=Row+1, col=Col, data=Time[BoolArray])
                worksheet.write_string(row=Row, col=Col+1, string=field)
                Col += 2

                for group in groups:
                    worksheet.write_string(row=Row, col=Col, string=(field + '-' + group +' means' ))
                    worksheet.write_column(row=Row+1, col=Col, data=means[group][BoolArray])
                    worksheet.write_string(row=Row, col=Col + 1, string=(field + '-' + group + ' StdError'))
                    worksheet.write_column(row=Row+1, col=Col+1, data=stdErr[group][BoolArray])
                    worksheet.write_string(row=Row, col=Col+2, string=(field + '-' + group + ' N'))
                    worksheet.write_column(row=Row+1, col=Col+2, data=N[group][BoolArray])
                    Col += 3
    # TODO Export the Neurological test data last, some reformatting is required.


    workbook.close()


    #See above, this was a first attempt I'll clean up when I have a working version.
    # # Write ALL the headers
    # # repeatFields = ['Time'] #Other repeated cols don't make sense here Unless i'm bad at sigma plot. I probably am.
    # Time = Dataset[0]['Time']
    # (row, col) = (1, 0)
    # Fields = ['Ao systolic', 'Ao systolic','Heart rate LV', 'ICP']
    # # exp = Dataset[3] #2018107 for testing. #silly me, I can't go exp by exp because I need summary stats.
    #
    # array = np.empty(len(Time)*4, 1+(2*len(Fields))) * np.nan #Take that 4 out of there.
    #
    # for ix, field in enumerate(Fields):
    #     Data = selectData(Dataset, Key=field, returnLists=False)
    #     #Average across group. Then stack them and write the entire column out #Get the standard error too for error bars.
    #     Groups=[]
    #     Times=[]
    #     for group in Data.keys():
    #         if ix == 0: #First field we have to do all the groups in one col, all the times in another and all the data in the third.
    #             intervention = (cloneStringToList(group, len(Time))) #append the treatment group to the list first.
    #             data = np.nanmean(Data[group], 0)
    #             stdev = np.nanstd(Data[group], 0)/ Data['A'].shape[0]
    #             # array[0:] # and then on not the first field we do the same thing with the rest of the averaged data.
    #
    #
    # #lets just get an array of the values and do a write_col at the end.
    # for col, data in enumerate(array):
    #     worksheet.write_column(row, col, data)
    #
    # workbook.close()

def DescriptivesExportLinked(Dataset): #TODO Dr.G wants descriptives that are interactive so if you change the sheet it will update the sumarry table and summarry stats.
    pass


def DescriptivesExport(Dataset):
    #Dr.G Likes to visually page through the data to look for outliers and strange values.
    #He uses a copy and paste macro I can't replicate. This should work though.
    outpath = r'C:\Users\mattm\PycharmProjects\GazmuriDataLoader\Export\\'
    workbook = xlwrite.Workbook(outpath + 'Descriptives.xlsx', {'nan_inf_to_errors': False})
    worksheet = workbook.add_worksheet()
    #Add formats here
    FormulaFormat = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#FFA500'})
    FieldFormat = workbook.add_format({'text_wrap': True })
    #Row, colunm, data
    # worksheet.write(0, 2, "TEst test yo world")

    #Get all fields from the first dataset.
    fields = list(Dataset[0].keys())

    #Develop a list of interventions.
    groups = ['NS', 'TLP', 'POV', 'AVP']
    # groups = list(selectData(Dataset).keys())
    # groups.sort() #Sort alphabetical. might change when we unblind. TODO.
    TypeList = []
    #Develop a dict that has the indexes as a list for each group.
    groupIdx = {key: [] for key in groups}
    for idx, data in enumerate(Dataset):
        groupIdx[str(data['Intervention'])].append(idx)

    (row, col) = (0, 0) #Where to start counting from.

    for index, field in enumerate(fields):
        if index == 0:  # write the first 2 columns that Dr.G has set up.
            col1 = (r'Exp.#,Intervention,General,BL,' + ('LL/TBI,' * 35)).split(',')
            col1.pop(-1)  # Drop the last entry in the list
            col2a = np.concatenate((np.arange(0, 15 + 5, 5), np.arange(30, 240 + 15, 15)))
            col2b = np.arange(8, 72 + 4, 4)
            col2 = []
            for num in col2a:
                col2.append(str(num) + ' min')
            for num in col2b:
                col2.append(str(num) + ' h')
            for i, entry in enumerate(col1):
                worksheet.write(i + 2, col, entry)
                if i < len(col2):
                    worksheet.write(i + 5, col + 1, col2[i])
            col += 2  # Move over 2 cols
        groupcounter = [] #Reset counter each time we go to a new field, it will be the same each time but why not?
        for i, group in enumerate(groups):
            #here we enter the data in for each group starting with the field, then the exp num, then the intervention, then the data.
            groupData = [Dataset[index] for index in groupIdx[group]]
            groupcounter.append(0)
            for exp in groupData:
                data = exp[field]

                worksheet.write(row+1, col, field, FieldFormat) #Write the field name we are intrested in.
                worksheet.write(row+2, col, exp['experimentNumber'], FieldFormat)  # Write the experiment number for this col.
                worksheet.write(row+3, col, group, FieldFormat)  # Write the treatment

                if isinstance(data, (type(pd.Series()), type(list()), type(np.array(1)))): #Need to discriminate between single valued fields and array based fields.
                    for i, datum in enumerate(data):
                        if not np.isnan(datum):
                            try:
                                worksheet.write(row+5+i, col, datum)
                            except:
                                print("an error from the first write statement happened")
                elif isinstance(data, (type(datetime.datetime.now()), type(str()), type(int()), type(float()), type(datetime.time()))): #Many types are loaded from excel data.
                    try:
                        worksheet.write(row + 4, col, data)
                    except:
                        print("an error from the Second write statement happened")
                else:
                    if type(data) not in TypeList:
                        TypeList.append(type(data))

                col += 1 #move over 1 col every exp.
                groupcounter[-1] += 1 #count the number of experiments in each group

        #Between each group we need 5 rows to calc the sumamary statstics. First figure out the excel addresses for the start:stop points.
        #write the headers for the rows #Dr.G wants the summary statistics at the end of all four groups. Each of the groups can have a different 5 colunm summary stats.
        for ix, group in enumerate(groups):

            worksheet.write_string(row+1, col, group, FormulaFormat)
            worksheet.write_string(row+1, col+1, group, FormulaFormat)
            worksheet.write_string(row+1, col + 2, group, FormulaFormat)
            worksheet.write_string(row+1, col + 3, group, FormulaFormat)
            worksheet.write_string(row+1, col + 4, group, FormulaFormat)

            worksheet.write_string(row+2, col, "Mean", FormulaFormat)
            worksheet.write_string(row+2, col+1, "SEM", FormulaFormat)
            worksheet.write_string(row+2, col + 2, "n", FormulaFormat)
            worksheet.write_string(row+2, col + 3, "Min", FormulaFormat)
            worksheet.write_string(row+2, col + 4, "Max", FormulaFormat)

            #Put in the group name for each of the stats cols. Merge these cells later
            worksheet.write_string(row + 3, col, group, FormulaFormat)
            worksheet.write_string(row + 3, col + 1, group, FormulaFormat)
            worksheet.write_string(row + 3, col + 2, group, FormulaFormat)
            worksheet.write_string(row + 3, col + 3, group, FormulaFormat)
            worksheet.write_string(row + 3, col + 4, group, FormulaFormat)

            for statrow in range(row+4, len(col2)+6):

                ncol = str(xlwrite.utility.xl_col_to_name(col + 2)) + str(statrow)
                slide = 5 * ix #Slide over enough to not count the previous summary columns.
                startcol = str(xlwrite.utility.xl_col_to_name(col - slide - (sum(groupcounter[ix:len(groups)])) )) + str(statrow)
                stopcol = str(xlwrite.utility.xl_col_to_name(col - 1 - slide - (sum(groupcounter[ix+1:len(groups)]))  )) + str(statrow)

                # ncol = str(xlwrite.utility.xl_col_to_name(col+2)) + str(statrow)
                # startcol = str(xlwrite.utility.xl_col_to_name(col-(len(groupData)*len(groups)))) + str(statrow)
                # stopcol = str(xlwrite.utility.xl_col_to_name(col-1)) + str(statrow)



                worksheet.write_formula(statrow-1, col, '=IF('+ ncol+'=0,"",AVERAGE('+ startcol + ':' + stopcol + '))', FormulaFormat)
                worksheet.write_formula(statrow-1, col+1, '=IF(' + ncol + '=0,"",STDEV(' + startcol + ':' + stopcol + ')/SQRT(' + ncol + '))', FormulaFormat)
                worksheet.write_formula(statrow-1, col+2, '=COUNT('+ startcol + ':' + stopcol + ')', FormulaFormat) #NCol
                worksheet.write_formula(statrow-1, col+3, '=IF(' + ncol + '=0,"",MIN(' + startcol + ':' + stopcol + '))', FormulaFormat)
                worksheet.write_formula(statrow-1, col+4, '=IF(' + ncol + '=0,"",MAX(' + startcol + ':' + stopcol + '))', FormulaFormat)

            #Dr.G wants a max of max and min of min col to help look for outliers.
            startrow = str(xlwrite.utility.xl_col_to_name(col + 0) + str(4))
            stoprow = str(xlwrite.utility.xl_col_to_name(col + 0) + str(len(col2)+4))
            worksheet.write_formula(len(col2)+6, col + 0,
                                    'AVERAGE(' + startrow + ':' + stoprow + ')', FormulaFormat)
            startrow = str(xlwrite.utility.xl_col_to_name(col + 3) + str(4))
            stoprow = str(xlwrite.utility.xl_col_to_name(col + 3) + str(len(col2)+4))
            worksheet.write_formula(len(col2)+6, col + 3,
                                    'MIN(' + startrow + ':' + stoprow + ')', FormulaFormat)
            startrow = str(xlwrite.utility.xl_col_to_name(col + 4) + str(4))
            stoprow = str(xlwrite.utility.xl_col_to_name(col + 4) + str(len(col2) + 4))
            worksheet.write_formula(len(col2)+6, col + 4,
                                    'MAX(' + startrow + ':' + stoprow + ')', FormulaFormat)
            col += 5  # move over 5 col after every group/treatment. Start the next field output.

    #xlsxwriter.utility.xl_col_to_name(index) #use to figure out cols from numbers.

    print(TypeList) #Some of the time obj are being rejected. find out why.
    workbook.close()



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
            #Plot the prep time vs block.
            Data = selectData(Dataset, Key='Dura Rip Frequency', groupBy='Block')
            plt.rc('font', family='serif')
            fig, ax = BoxPlot(Data)
            fig.suptitle("Preparation Time by Block")
            ax.set_ylabel("Dura Rip frequency")

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
            # more plots
            pass



def generalizedExcelLoader(path, dataset = None, Integrate_on  = ("Experiment Number", 'experimentNumber') , AddFields = ['field1'] ):
    #This function simply takes an excel sheet and reads it in. (defaulting to the first sheet)
    #Then we extract the data into a dataframe and iterate through it, adding a new field to the dataset if there is a match between the experiment and integrate on.
    #Integrate on is the holder for the field names we want to match between dataframes, it's a tuple with 2 values,
    # the first is what the search field is called in the dataset, the second is the colunm from the data you want to match with
    # TODO One issue is we are only able to match on experiment number here, but I don't expect to need to insert data on any other catagory

    ExcelMatchField = Integrate_on[0]
    DatasetMatchField = Integrate_on[1]

    if dataset != None:
        xls = pd.ExcelFile(DeathsClass)
        df = pd.read_excel(xls)

        for i, num in enumerate(df[ExcelMatchField]):
            if not np.isnan(num):
                match = []
                for j, exp in enumerate(dataset):
                    if int(exp[DatasetMatchField]) == int(num):
                        match.append(j) #Store the index in the dataset list.

                if len(match) == 1:
                    # print("found match")
                    #Now that we have a match, we can add the fields to the dict for the exp.
                    for field in AddFields:
                        #TODO Make this more general by letting the user spesfic if we are bool or continous, but if data is nan, we can just leave that.
                        if not pd.isnull(df[field].iloc[i]):
                            dataset[match[0]][field] = 'Y'
                        else:
                            dataset[match[0]][field] = np.nan
                else:
                    print('No match found, either 0 or 2+')

    else:
        print("No dataset input to pair with.")
        return  None  #This should tell the user something is V wrong.
    return dataset

def DataFrameExport():
    #TODO Reform the data as a pandas dataframe. Small project.
    pass

def ArterialVenusAveraged(dataset=None, fields = ['R', 'K', 'Angle', 'MA', 'PMA', 'G', 'EPL', 'A', 'CI', 'LY30' ], VenOrPA = 'Ven', Technique = ''):
    #This function replicates the part of the excel sheet that Dr G put in to choose the blood gas and TEG data. Much of that data doesnt care if it's Ao or Venous  so we average both together.
    #Just put the symbols that are outside of

    #Need to keep the technique that the measurement came from straight.
    if Technique != '':
        TechniqueSpace = ' ' + Technique
    else:
        TechniqueSpace = ''

    for exp in dataset:
        for field in fields:
            AoData = exp[field + ' ' + 'Ao' + TechniqueSpace ]
            VenData = exp[field + ' ' + VenOrPA + TechniqueSpace]
            Outlist = []
            for ix, Ao in AoData.iteritems():
                if np.isnan(AoData[ix]) and np.isnan(VenData[ix]):
                    Outlist.append(np.nan)
                elif not np.isnan(AoData[ix]) and np.isnan(VenData[ix]):
                    Outlist.append(AoData[ix])
                elif np.isnan(AoData[ix]) and not np.isnan(VenData[ix]):
                    Outlist.append(VenData[ix])
                else:
                    Outlist.append((AoData[ix] + VenData[ix])/2)
            exp[field + ' ' + 'Ao' + ' or ' + VenOrPA + TechniqueSpace ] = pd.Series(Outlist)

    return dataset

def StandardLoadingFunction(useCashe = False):

    # Timing function
    t1, timeTotal = time.time(), time.time()
    #Figure out if a cashe file exists.
    CasheFilename = 'Sa1Dataset.pickle'

    CasheExists = os.path.exists(os.path.join('cashe',CasheFilename))

    if not useCashe or not CasheExists:
        print('loading dataset fresh from on disk ')


        #Keep the standard loading code here so we can easily call it other places.
        experiment_lst = np.arange(2018104, 2018168 + 1)  # Create a range of the experiment lists
        censor = np.isin(experiment_lst, [2018112, 2018120, 2018123, 2018153,
                                          2018156])  # Create a boolean mask to exclude censored exps from the lst.

        censor = [not i for i in censor]  # Invert the boolean mask
        experiment_lst = experiment_lst[censor]  # Drop censored exp numbers
        experiment_lst = list(map(str, experiment_lst))  # Convert the list to strings

        # path = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SA-1 Survival Phase (Master  Workbook) April 12, 2019 (masked).xlsx" #old data before groups became public.
        # path = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SA-1 Survival Phase (Master  Workbook) April 23, 2019 (Check Values Fixed).xlsx"
        # path = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SA-1 Survival Phase (Master  Workbook) May 2, 2019 (Check Values Fixed).xlsx"
        # path = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SA-1 Survival Phase (Master  Workbook) July 12 2019.xlsx"
        path = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SA-1 Survival Phase (Master  Workbook Final) July 15 2019.xlsx"
        path = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SA-1 Survival Phase (Master  Workbook FINAL) Aug 1 2019.xlsx"
        print(' Loading from ' + path)

        # print(experiment_lst)


        Dataset = Parse_excel(path=path, Experiment_lst=experiment_lst)

        # Convert all datetimes to an int or string, they are annoying.
        Dataset = DateTimesToStr(Dataset=Dataset)

        # Calculate the pCO2 Ratio from the aortic to the venus.
        # Dataset = SA1DataManipulation.ProduceRatios(Dataset, fieldNum='pCO2 Ao (OPTI)', fieldDenom='pCO2 PA (OPTI)', ratio = True,
        #                                           OutfieldName='PCO2 PV Ratio')

        #Alternate calc for the pCO2 gradient across arterial and venus samples.
        Dataset = SA1DataManipulation.ProduceSums(Dataset, field1='pCO2 PA (OPTI)', field2='pCO2 Ao (OPTI)', add=False,
                                                  OutfieldName='PCO2 PV difference')

        Dataset = SA1DataManipulation.ProduceSums(Dataset, field1='pCO2 Ao (OPTI)', field2='PetCO2', add=False,
                                                  OutfieldName='pCO2 Art-ETCo2 difference')

        Dataset = SA1DataManipulation.ProduceRatios(Dataset, fieldNum= 'pCO2 Art-ETCo2 difference', fieldDenom='pCO2 Ao (OPTI)', ratio=True,
                                                  OutfieldName='pCO2 A-ETCo2 %')
        for exp in Dataset:
            exp['pCO2 A-ETCo2 %'] = exp['pCO2 A-ETCo2 %'] * 100 #I'll MAKE A % OUT OF YOUUUUUUUUUU

        SingleIntrestParameters = [('Ao mean', 30), ('Ao mean', 60), ('Ao mean', 240), ('Lactate Ao (OPTI)', 240), ('PetCO2 End Tidal Corrected', 30),
                                   ('PetCO2 End Tidal Corrected', 30), ('VO2/ DO2', 30), ('VO2/ DO2', 240),
                                   ('LV end-diastolic', 30), ('LV end-diastolic', 240)] #List of tuples of the form (field, timepoint to loop through for the timepoints of intrest.)
        for Parameter in SingleIntrestParameters:
            #Extract single timepoints of intrest for COX hazard analysis.
            Dataset = SA1DataManipulation.SingleTimePointExtraction(Dataset=Dataset, field=Parameter[0], TimePoint=Parameter[1])

        # List of tuples of the form (field, Max or min) to loop through for the max or min values, False gives you the minimum
        MaxMinParameters = [('Ao mean', False), ('PetCO2 End Tidal Corrected', False), ('Heart rate LV', True), ('VO2/ DO2', True), ('LV end-diastolic', False)]
        for Parameter in MaxMinParameters:
            #Find the mins of the following parameters per animal for the cox proportional hazard model. Ao mean
            Dataset = SA1DataManipulation.ProduceMinMaxValues(Dataset=Dataset, field=Parameter[0], FindMax=Parameter[1])


        print("Total Dataset Loaded and processed at {0} seconds".format(time.time() - timeTotal))

        print('Taking averages for the bloodwork data where Ao and Venus data exist, combining where they are exclusive.')
        # Default settings are for the TEG data, take either or.
        Dataset = ArterialVenusAveraged(Dataset)
        # Default settings are for the TEG data. #Had to change the strings to match the new (OPTI) field format.
        Dataset = ArterialVenusAveraged(Dataset, fields=['tHg', 'O2Hb', 'COHb', 'MetHb', 'O2Ct', 'O2Cap', 'sO2'],
                                        VenOrPA='PA', Technique='(AVOX)')
        Dataset = ArterialVenusAveraged(Dataset,
                                        fields=['pH', 'pCO2', 'pO2', 'BE', 'tCO2', 'HCO3', 'stHCO3', 'tHB', 'SO2', 'HCT',
                                                'Lactate'], VenOrPA='PA', Technique='(OPTI)')
        Dataset = ArterialVenusAveraged(Dataset,
                                        fields=['pH', 'tCO2', 'HCO3', 'Na+', 'K+', 'Cl-', 'Ca++', 'AnGap', 'nCa++'],
                                        Technique='(OPTI ELYTE)')
        print("Bloodwork averages done at at {0} seconds".format(time.time() - timeTotal))

        # Save the dataset to the cashe. (Maybe date the cashes, or that might lead to file inflation.
        print('Cashing dataset to disk.')
        with open(os.path.join('cashe', CasheFilename), 'wb') as f:
            pickle.dump(Dataset, f)
            print('Cashe dumped to disk  at {0} '.format(time.time() - timeTotal))

    elif not CasheExists:
        os.mkdir(os.path.join('cashe'))
        print('Call loader recursively to refresh Cashe.')
        StandardLoadingFunction(useCashe=False)

    else:
        #Load the dataset from disk.
        with open(os.path.join('cashe',CasheFilename), 'rb') as f:
                Dataset = pickle.load(f)
                print('Cashe loaded from disk  in {0} '.format(time.time() - timeTotal))

    return Dataset

if __name__ == "__main__":

    #Run the anaylsis starting from the master Excel sheets.
    Dataset = StandardLoadingFunction(useCashe=False)

    #Run the standard loader using the cashe, good for testing new functions that don't depend on changes in input data.
    # Dataset = StandardLoadingFunction(useCashe=True)



    Ao = selectData(Dataset)


    # # for group in Ao.keys():
    #     frameLst.append(pd.DataFrame(data= np.transpose(Ao[group]), index=Time))
    # dfTot = pd.join(frameLst, keys= Ao.keys())


    # Run for KM analysis
    # survivalPlot(Dataset)

    #This was for before we had the data divided into groups
    #Dataset = Randomize_groups(Dataset)

    # integrate the deaths classifier
    DeathsClass = r"C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\Experimental Deaths Classification July 12.xlsx"


    Dataset = generalizedExcelLoader(dataset=Dataset, path=DeathsClass, AddFields=['Hemodynamic', 'Neurologic'])

    #Add the ratio of blood withdrawn by wieght in KG and after the 30 min mark, use the estimated blood withdrawn.
    Dataset = SA1DataManipulation.BloodWithdrawnPerKg(Dataset)



    #Run when you want to reproduce figures. I have a whole function set up to store those!
    # MSCPlots(Dataset)
    #Run to produce an xlsx file for spss to import.
    SPSSExport(Dataset)
    #Run to produce an xlsx file for Sigma plot.
    SigmaPlotExport(Dataset)
    #Produce the descriptives sheet for easy identification of outlier values.
    DescriptivesExport(Dataset)

    #Make a sound when program finsishes so I know it's done!
    duration = 500  # milliseconds
    freq = 450  # Hz
    winsound.Beep(freq, duration)
    print("Program finished succesfully, that IS what you wanted right?")



"""
This is a list of the fields we are using currently. 7/22/19
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
tHg Ao (AVOX)
O2Hb Ao (AVOX)
COHb Ao (AVOX)
MetHb Ao (AVOX)
O2Ct Ao (AVOX)
O2Cap Ao (AVOX)
sO2 Ao (AVOX)
pH Ao (OPTI)
pCO2 Ao (OPTI)
pO2 Ao (OPTI)
BE Ao (OPTI)
tCO2 Ao (OPTI)
HCO3 Ao (OPTI)
stHCO3 Ao (OPTI)
tHB Ao (OPTI)
SO2 Ao (OPTI)
HCT Ao (OPTI)
Lactate Ao (OPTI)
tHg PA (AVOX)
O2Hb PA (AVOX)
COHb PA (AVOX)
MetHb PA (AVOX)
O2Ct PA (AVOX)
O2Cap PA (AVOX)
sO2 PA (AVOX)
pH PA (OPTI)
pCO2 PA (OPTI)
pO2 PA (OPTI)
BE PA (OPTI)
tCO2 PA (OPTI)
HCO3 PA (OPTI)
stHCO3 PA (OPTI)
tHB PA (OPTI)
SO2 PA (OPTI)
HCT PA (OPTI)
Lactate PA (OPTI)
pH Ao (OPTI ELYTE)
tCO2 Ao (OPTI ELYTE)
HCO3 Ao (OPTI ELYTE)
Na+ Ao (OPTI ELYTE)
K+ Ao (OPTI ELYTE)
Cl- Ao (OPTI ELYTE)
Ca++ Ao (OPTI ELYTE)
AnGap Ao (OPTI ELYTE)
nCa++ Ao (OPTI ELYTE)
pH Ven (OPTI ELYTE)
tCO2 Ven (OPTI ELYTE)
HCO3 Ven (OPTI ELYTE)
Na+ Ven (OPTI ELYTE)
K+ Ven (OPTI ELYTE)
Cl- Ven (OPTI ELYTE)
Ca++ Ven (OPTI ELYTE)
AnGap Ven (OPTI ELYTE)
nCa++ Ven (OPTI ELYTE)
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
R Ao or Ven
K Ao or Ven
Angle Ao or Ven
MA Ao or Ven
PMA Ao or Ven
G Ao or Ven
EPL Ao or Ven
A Ao or Ven
CI Ao or Ven
LY30 Ao or Ven
tHg Ao or PA
O2Hb Ao or PA
COHb Ao or PA
MetHb Ao or PA
O2Ct Ao or PA
O2Cap Ao or PA
sO2 Ao or PA
pH Ao or PA
pCO2 Ao or PA
pO2 Ao or PA
BE Ao or PA
tCO2 Ao or PA
HCO3 Ao or PA
stHCO3 Ao or PA
tHB Ao or PA
SO2 Ao or PA
HCT Ao or PA
Lactate Ao or PA
pH Ao or Ven
tCO2 Ao or Ven
HCO3 Ao or Ven
Na+ Ao or Ven
K+ Ao or Ven
Cl- Ao or Ven
Ca++ Ao or Ven
AnGap Ao or Ven
nCa++ Ao or Ven
"""