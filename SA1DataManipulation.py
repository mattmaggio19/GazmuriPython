
import SA1DataLoader
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#This file is for functions that actually change or calculate other parameters from the SA-1 study.
# Hopefully these funciton are general enough to be useful.

def FoodAqPlots():
    #This func should process the food aq data into binary form.
    pass

def PCo2Ratio(Dataset = None , graph = True):
    Time = Dataset[0]['Time']

    pCo2Ao = SA1DataLoader.selectData(Dataset, Key='pCO2 Ao (OPTI)')
    pCo2PA = SA1DataLoader.selectData(Dataset, Key='pCO2 PA (OPTI)')
    EtCo2 = SA1DataLoader.selectData(Dataset, Key='PetCO2 End Tidal Corrected')
    CCI = SA1DataLoader.selectData(Dataset, Key='CCI')

    cO2Ratio = dict()
    if graph == True:
        plt.subplot(2, 2, 1)
    for ix, group in enumerate(pCo2Ao.keys()):

        cO2Ratio[group] = np.divide(pCo2Ao[group], pCo2PA[group])
        cO2RatioMean = np.nanmean(cO2Ratio[group], axis=0)
        CCIMean = np.nanmean(CCI[group], axis=0)
        EtCo2Mean = np.nanmean(EtCo2[group], axis=0)

        if graph == True:
            plt.subplot(2, 2, ix + 1)
            plt.scatter(x=Time[~np.isnan(CCIMean)], y=CCIMean[~np.isnan(CCIMean)]/CCIMean[0],
                            label=str('CCI'))
            plt.scatter(x=Time[~np.isnan(EtCo2Mean)], y=EtCo2Mean[~np.isnan(EtCo2Mean)]/EtCo2Mean[0],
                        label=str('EtCo2'))
            plt.scatter(x=Time[~np.isnan(cO2RatioMean)], y=cO2RatioMean[~np.isnan(cO2RatioMean)] / cO2RatioMean[0],
                        label=str(' cO2 AV ratio'))
            # plt.xlabel('Time (min)')
            plt.title(group)
            plt.ylim((0.25,1.2)) #This is the data normalized to baseline, change if you can switch back to actual values.
            plt.legend(loc='lower center')
            # plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
            # ax.scatter(x=Time[~np.isnan(cO2RatioMean)], y=cO2RatioMean[~np.isnan(cO2RatioMean)],
            #             label=group + ' cO2 AV ratio')

    if graph==True:
        plt.show()
    # print(cO2Ratio)





def SingleTimePointExtraction(Dataset=None, field='', TimePoint=0):
    # This function does a simple thing, It finds the value of the field at the time given by the input, then returns it to the dataset as a new measurement.
    for exp in Dataset:
        if type(exp[field]) == type(pd.Series()):
            if not field == '':
                Time = exp['Time']
                TimeIdx = np.where(np.isin(Time, TimePoint))[0][0] #Weirdly complex expression, could be made better.
                exp[field + ' Time ' + str(TimePoint)] = exp[field].iloc[TimeIdx]
            else:
                print('No Target field given')
        else:
            print('Input value is not a repeated measure.')

    return Dataset


def ProduceMinMaxValues(Dataset=None, field='', FindMax=True ):
    #This function does a simple thing, It finds the min or max value in a series parameter and then returns that back to the dataset under a new field which is really the old field plus max or min.
    #Since our repeated measurements are stored as a pandas.Series object, we should check for type first.

    for exp in Dataset:
        if type(exp[field]) == type(pd.Series()):
            if not field == '':
                if FindMax:
                    exp[field+' Max'] = exp[field].max()
                else:
                    exp[field + ' Min'] = exp[field].min()
            else:
                print('No Target field given')
        else:
            print('Input value is not a repeated measure.')
    return Dataset

def ProduceRatios(Dataset=None, fieldNum='', fieldDenom='', ratio = True, OutfieldName = None):
    #This is for multiplying/ dividing traces to calculate new Series. Calculated measurements like SVRI should be possible with this tool.
    #If Ratio is set to false we will instead multiply the traces. Because multiplication is comutitive, we can call this function multiple times.
    # if OutfieldName is not given we will construct a string from the fields we put together, if given, use that as the name of the dict variable.
    for exp in Dataset:
        if type(exp[fieldNum]) == type(pd.Series()) and type(exp[fieldDenom]) == type(pd.Series()) :
            if not fieldNum == '' and not fieldDenom == '':
                if ratio:
                    #Take the Ratio of fieldNum to fieldDenom
                    Outdata = exp[fieldNum].divide(exp[fieldDenom], fill_value=np.nan)
                    Outfield = 'Ratio of ' + fieldNum + ' to' + fieldDenom
                else:
                    # Take the product of fieldNum and fieldDenom
                    Outdata = exp[fieldNum].multiply(exp[fieldDenom], fill_value=np.nan)
                    Outfield = 'Product of ' + fieldNum + ' to' + fieldDenom
                if OutfieldName is None:
                    exp[Outfield] =Outdata
                else:
                    exp[OutfieldName] = Outdata
            else:
                print('No Target fields given')
        else:
            print('Input value is not a repeated measure.') #This isn't technically true, a general version of this function could take a single value as the numerator or denominator.
    return Dataset



def ProduceSums(Dataset=None, field1='', field2='', add = True, OutfieldName = None):
    #This is for multiplying/ dividing traces to calculate new Series. Calculated measurements like SVRI should be possible with this tool.
    #If Ratio is set to false we will instead multiply the traces. Because multiplication is comutitive, we can call this function multiple times.
    # if OutfieldName is not given we will construct a string from the fields we put together, if given, use that as the name of the dict variable.
    for exp in Dataset:
        if type(exp[field1]) == type(pd.Series()) and type(exp[field2]) == type(pd.Series()) :
            if not field1 == '' and not field2 == '':
                if add:
                    #Take the Ratio of fieldNum to fieldDenom
                    Outdata = exp[field1].add(exp[field2], fill_value=np.nan)
                    Outfield = 'Ratio of ' + field1 + ' to' + field2
                else:
                    # Take the product of fieldNum and field Denom
                    Outdata = exp[field1].subtract(exp[field2], fill_value=np.nan)
                    Outfield = 'Product of ' + field1 + ' to' + field2
                if OutfieldName is None:
                    exp[Outfield] =Outdata
                else:
                    exp[OutfieldName] = Outdata
            else:
                print('No Target fields given')
        else:
            print('Input value is not a repeated measure.') #This isn't technically true, a general version of this function could take a single value as the numerator or denominator.
    return Dataset


def extractSurvivalCurve(Dataset = None, groupBy = 'Intervention', graph = True):
    #Extract and build a step function that can be plotted to show the survival curves in certain domains.
    groups = ['NS', 'TLP', 'POV', 'AVP']
    surv = dict.fromkeys(groups)
    MaxTime = 72*60
    Time = np.arange(0, MaxTime, 1)#function to define the time bins, real version is more sophisticated.
    for group in groups:
        surv[group] = []
    for exp in Dataset:
        surv[exp['Intervention']].append(exp['Survival time'])

    cumSurvival = dict.fromkeys(groups)
    for group in groups:
        y = []
        for t in Time:
            y.append(sum(surv[group]> t)/len(surv[group]))

        if graph == True:
            plt.step(Time, y, where='post', label=group)
            plt.ylim((0,1.1))

        cumSurvival[group] = y #Store the cumSurvival, can assume user knows the units of time are in min.
    if graph == True:
        plt.title('Overall Survival Curves.')
        plt.legend(loc='upper right')
        plt.show()
    return cumSurvival

def RelationshipBetweenAoSystolicHESandSurvival(Dataset = None, graph = True):
    #This is a plot Sal proposed that would let us look at the relationship between HES administration pressure and survival.
    #Basically we can seperate the animals by who got the last dose of hespand and who did not get the last dose of hespand
    #Main y axis would be systolic Ao pressure. (though techincally we used the LV in the acute phase because it was more accurate.)
    #Secondary y axis would be survival to see if the survival curve declines after missing a dose, we can code the deaths by their last checktime

    CheckTimes = [30, 120, 240, 8*60, 12*60, 16*60, 20*60, 24*60]
    CheckTimesArray = np.array(CheckTimes)
    groups = ['NS', 'TLP', 'POV', 'AVP']
    groups_Format = []
    Time = Dataset[0]['Time']
    # HES_output = ResolvedHESAdministration(Dataset) #Doesn't answer the question at hand.
    cumSurvival = extractSurvivalCurve(Dataset, graph=False)

    DeathScatterPlot = dict.fromkeys(groups) #Initialize the list for storing the survival time (x) and cum_haz level (y)
    # and if each point is in the group that last got HES or did not get HES at the last checktime. This is the crux of the plot.
    SystolicPressure = dict.fromkeys(groups)
    for group in DeathScatterPlot:
        DeathScatterPlot[group] = [] #Init empty list, going to put a 3 member Tuple. in for each animal.
        SystolicPressure[group] = dict()
        SystolicPressure[group]['GotHES'] = np.ones(Time.shape, dtype=float) * np.nan #Going to put in 2 np.arrays as tuples one with the data blocked out when the animal misses/gets a does of hespand. nanmean across animals.
        SystolicPressure[group]['MissedHES'] = np.ones(Time.shape, dtype=float) * np.nan
    for exp in Dataset:
        group = exp['Intervention']

        TimeArray = np.array(Time)
        HES = exp['HESDelivered']
        HESArray = np.array(HES)
        AoSys = np.array(exp['Ao systolic'])
        LVSys = np.array(exp['LV systolic'])
        Acute = (TimeArray <= 240)

        CombinedSystolic = LVSys.copy() #used LV in the acute phase as it was more accurate than the side ports in most cases.
        CombinedSystolic[~Acute] = AoSys[~Acute]

        #Bothsets = set(CheckTimes).intersection(Time)
        #indices = [list(Time).index(x) for x in Bothsets]

        Last_HESDose = Time[np.where(~HES.isnull())[0][-1]]
        Last_Checktime = CheckTimes[np.where(np.array(CheckTimes) <= exp['Survival time'])[0][-1]]
        if Last_HESDose < Last_Checktime:
            DeathScatterPlot[group].append((exp['Survival time'], cumSurvival[group][exp['Survival time']-1], False )) #Missed a dose of HES
        else:
            DeathScatterPlot[group].append((exp['Survival time'], cumSurvival[group][exp['Survival time']-1], True)) #Got last dose of HES

        GotHES = CombinedSystolic.copy()
        MissedHes = CombinedSystolic.copy() * np.nan
        for t in CheckTimesArray:
            prevHesDosesix = np.where(CheckTimesArray <= t)[0]
            HESix = np.where(TimeArray == CheckTimesArray[prevHesDosesix[-1]])[0]

            if not np.isnan(HESArray[HESix[-1]]): #Test if the last check time was given or not.
                #Dose has been gotten. Do nothing.
                pass
            else:
                # Previous dose has been missed. nan out the GotHes array between this does and the last one.
                #if we are at the last HES checkpoint, add to MissedHES then nan out the rest GotHes
                if t == CheckTimesArray[-1]:
                    MissedHes[HESix[0]:] = GotHES[HESix[0]:]
                    GotHES[HESix[0]:] = np.nan
                else:
                    HESNext = np.where(TimeArray == CheckTimesArray[prevHesDosesix[-1] + 1])[0]
                    MissedHes[HESix[0]:HESNext[0]] = GotHES[HESix[0]:HESNext[0]]
                    GotHES[HESix[0]:HESNext[0]] = np.nan

        SystolicPressure[group]['GotHES'] = np.vstack((SystolicPressure[group]['GotHES'], GotHES))
        SystolicPressure[group]['MissedHES'] = np.vstack((SystolicPressure[group]['MissedHES'], MissedHes))


    for group in groups:
        #Plotting code for survival curves that indicate if the last dose of HES was missed or gotten.
        x, y, z  = zip(*DeathScatterPlot[group])
        X, Y, Z = np.array(x), np.array(y), np.array(z, dtype='bool') #Recast as np.arrays, easier to slice.
        plt.scatter(X[Z], Y[Z], label="Got last checked dose of HES")
        plt.scatter(X[~Z], Y[~Z], label="Missed last checked dose of HES")
        plt.step(np.arange(len(cumSurvival[group])), np.array(cumSurvival[group]), where='post', label='Cum Survival')
        plt.legend(loc='upper right')
        plt.title('Survival Curve, Treatment = ' + group)
        plt.show()

        #Plotting code for the Systolic pressure traces.
        plt.plot(np.array(Time), np.nanmean(SystolicPressure[group]['GotHES'], dtype=float, axis=0))
        plt.plot(np.array(Time), np.nanmean(SystolicPressure[group]['MissedHES'], dtype=float, axis=0))
        plt.step(np.arange(len(cumSurvival[group])), np.array(cumSurvival[group])*100, where='post', label=group)
        plt.title('Systolic Pressure, Treatment = ' + group)
        plt.show()







def HESvsSurvival(Dataset = None, graph = True ):
    groups = ['NS', 'TLP', 'POV', 'AVP']
    CheckTimes = [30, 120, 240, 8*60, 12*60, 16*60, 20*60, 24*60]
    MaxHesVol = (np.arange(1, len(CheckTimes)+1) * 250) - 250
    SubplotCoordVert, SubplotCoordHoriz = [0, 0, 1, 1], [0, 1, 0, 1] #useful lists for getting plots in a square.

    # Plot to show the relationship of  survival time to total HES administered. Overall scatterplot.
    HESTotal = dict.fromkeys(groups)
    SurvivalTime = dict.fromkeys(groups)
    ExperimentID = dict.fromkeys(groups)
    Surv72 = dict.fromkeys(groups)
    for group in groups: #TODO I should write a function to abstract this dumb process... If only!
        HESTotal[group] = []
        SurvivalTime[group] = []
        ExperimentID[group] = []
        Surv72[group] = []
    for exp in Dataset:
        HESTotal[exp['Intervention']].append(exp['HES'])
        SurvivalTime[exp['Intervention']].append(exp['Survival time'])
        ExperimentID[exp['Intervention']].append(exp['experimentNumber'])
        Surv72[exp['Intervention']].append(exp['Survival 72 hours'])

        #Convert the Y//N to Bool.
        if Surv72[exp['Intervention']][-1] == 'Y':
            Surv72[exp['Intervention']][-1] = True
        else:
            Surv72[exp['Intervention']][-1] = False

    if graph:
        for ix, group in enumerate(groups):
            plt.scatter(
                y=np.array(HESTotal[group]) + np.random.normal(loc=1, scale=10, size=len(np.array(HESTotal[group]))),
                x=np.array(SurvivalTime[group]) + np.random.normal(loc=1, scale=10, size=len(np.array(HESTotal[group]))),
                label=group)  # Dither the groups a bit to make overlap less likely..

            for i, item in enumerate(ExperimentID[group]): #Print the data for manual exploration.
                print(ExperimentID[group][i], HESTotal[group][i], SurvivalTime[group][i], Surv72[group][i], group)
        plt.step(CheckTimes, MaxHesVol, where='pre', label = 'Maximum possible HES')
        plt.legend(loc='upper left')
        plt.ylabel('Total HES administered (ml)')
        plt.xlabel('Survival Time (min)')
        plt.title('HES administration vs Survival Time')
        plt.show()



        fig, axs = plt.subplots(2, 2)
        plt.suptitle('HES administration among survivors and non survivors')
        RUNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for ix, group in enumerate(groups):
            surv = np.array(HESTotal[group])[np.where(np.array(Surv72[group]))]
            fail = np.array(HESTotal[group])[np.where(np.invert(np.array(Surv72[group])))]
            if fail.shape[0] > surv.shape[0]:
                surv = np.concatenate((surv, np.full((fail.shape[0]-surv.shape[0]), np.nan))) #add nans so they can be joined.
                x = np.stack((surv, fail), axis=1)
            elif fail.shape[0] < surv.shape[0]:
                fail = np.concatenate((fail, np.full((surv.shape[0]-fail.shape[0]), np.nan))) #add nans so they can be joined.
                x = np.stack((surv, fail), axis=1)
            else:
                x = np.stack((surv, fail), axis=1) #They must be the same shape!

            print(group, x)
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].hist(x, 8, histtype='bar', stacked=True, fill = True, label= ['Survivors', 'Non survivors'])
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_title(group)
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].legend(loc = 'best')
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_yticks(RUNS)

        for ax in axs.flat:
            ax.set(xlabel='HES Admin', ylabel='Animal counts')
        # # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()

        plt.tight_layout()
        plt.show()



def ResolvedHESAdministration(Dataset = None, output = 'ratio', graph = True ):
    #Purpose of this function is to go through the time resolved dataset and extract the HES data per time point. We also will need to know the survival time for each animal.
    #Develop a count of doses of HES given to the animal / the number of possible doses per timepoint. Obviously since it's a binary choice between give and no give per animal, the fraction only makes sense on the group.
    #Output should then be in the form of a binary matrix that can be summed across the timepoints. That way if we need that we can use it later.
    groups = ['NS', 'TLP', 'POV','AVP']  # Dr.G made a totally arbitrary desision on the ordering, but it important to remain consistant.
    CheckTimes = [30, 120, 240, 8*60, 12*60, 16*60, 20*60, 24*60] #Times at which we checked systolic pressure to determine if animal got saline. suppose we don't have to use becasue there should be no stray doses.
    HES_Given = dict()
    HES_Possible = dict()
    for group in groups: #Create an empty list for each group
        HES_Given[group] = []
        HES_Possible[group] = []
    for exp in Dataset:
        HES_Data = np.array(exp['HESDelivered'].values, dtype =float) #Cast into np.array for easy access
        HES_Delivered_raw = np.invert(np.isnan([i if i is not np.nan else None for i in HES_Data])) #Convert to boolean.

        Time = exp['Time']
        Survival_Time = exp['Survival time']
        Doses_possible = Time <= Survival_Time
        # print(Survival_Time)
        # Limit each series to checktimes, could do this later but it's easier to do here.
        for i, dose in enumerate(Doses_possible):
            if not Time[i] in CheckTimes:
                # print(Time[i])
                Doses_possible[i] = False
            else:
                pass
                # print(type(Time[i]), type(CheckTimes))


        # HES_Given[exp['Intervention']].append(HES_Delivered_checked) #TODO Make just the times that we check and may admin.
        # HES_Possible[exp['Intervention']].append(Doses_possible_checked)

        HES_Given[exp['Intervention']].append(HES_Delivered_raw)
        HES_Possible[exp['Intervention']].append(Doses_possible)

        # print(HES_Delivered)
        # print(Doses_possible)

    HES_output = dict()
    for group in groups:
        np.errstate(divide='ignore', invalid='ignore')
        #Go through the HES_Given group, if a checktime is nan, switch it to 0.
        SumDose = np.sum(np.array(HES_Given[group]), axis=0)
        SumPossible = np.sum(np.array(HES_Possible[group]), axis=0)
        Ratio_Time = np.divide(SumDose, SumPossible)
        Ratio_Time[np.isnan(Ratio_Time)] = 0

        # Ratio_CheckTimes = Ratio_Time[~np.isnan(Ratio_Time)]

        # print('Treatment group  ' + group)
        # print(Time, Ratio_Time)
        # print(Time.shape, Ratio_Time.shape)
        Bothsets = set(CheckTimes).intersection(Time)
        indices = [list(Time).index(x) for x in Bothsets]
        if graph == True:
            plt.scatter(x=Time[indices], y=Ratio_Time[indices], label=group)

        if output == 'ratio': #Output the ratio of given doses / max possible doses per unit time. Don't filter for checktime at this point.
            HES_output[group] = Ratio_Time
        elif output == 'possibleDoses':
            HES_output[group] = SumPossible
        elif output == 'givenDoses':
            HES_output[group] = SumDose
        else:
            print(' HES Output nonstandard, no output returned.')
            return None

    if graph == True:
        plt.legend()
        plt.show()
    return HES_output

def BloodWithdrawnPerKg(Dataset= None):
    # I actually ended up doing this in the spss syntax.
    #adds a new field to each experiment. 'BloodWithdrawnPerKg'
    if Dataset != None:
        for exp in Dataset:
            # print('experiment number ' + str(exp['experimentNumber']))
            BlWithEst = exp['Estimated blood loss']
            BlTrace = exp['Blood Removed'].copy()
            Time = exp['Time']
            Weight = exp['Weight']

            for ix, Bl in enumerate(BlTrace):
                if Time[ix] > 35: #Replace values after 35 min with the estimated blood volume r
                    # print('replaced value in new thing.' + str(ix) + '  is the index,  ' + str(Time[ix]) + '  Is the time')
                    BlTrace.iloc[ix] = BlWithEst * 1.06 #Convert back to weight by multiplying by the density of blood
                else:
                    BlTrace.iloc[ix] = exp['Blood Removed'].iloc[ix]
            exp['BloodWithdrawnMLPerKg'] = BlTrace / (Weight * 1.06) #Divide by weight and the density of blood.
    return Dataset


def PV_loop():
    #TODO Technically this isn't SA-1 material, but if I ever get around to doing this I'll troubleshoot the data here.
    pass

if __name__ == '__main__':
    #Testing code goes here.
    Dataset = SA1DataLoader.StandardLoadingFunction(useCashe=True)
    # HES = ResolvedHESAdministration(Dataset, output='ratio', graph = True)

    PCo2Ratio(Dataset, graph=False)

    HESvsSurvival(Dataset)

    # extractSurvivalCurve(Dataset, graph=False)

    # RelationshipBetweenAoSystolicHESandSurvival(Dataset, graph=True)