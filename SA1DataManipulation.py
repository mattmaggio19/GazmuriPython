
import SA1DataLoader
import numpy as np
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
    plt.subplot(2, 2, 1)
    for ix, group in enumerate(pCo2Ao.keys()):
        plt.subplot(2, 2, ix+1)
        cO2Ratio[group] = np.divide(pCo2Ao[group], pCo2PA[group])
        cO2RatioMean = np.nanmean(cO2Ratio[group], axis=0)
        CCIMean = np.nanmean(CCI[group], axis=0)
        EtCo2Mean = np.nanmean(EtCo2[group], axis=0)

        if graph == True:
            plt.scatter(x=Time[~np.isnan(cO2RatioMean)], y=cO2RatioMean[~np.isnan(cO2RatioMean)]/cO2RatioMean[0], label=str(' cO2 AV ratio'))
            plt.scatter(x=Time[~np.isnan(CCIMean)], y=CCIMean[~np.isnan(CCIMean)]/CCIMean[0],
                            label=str('CCI'))
            plt.scatter(x=Time[~np.isnan(EtCo2Mean)], y=EtCo2Mean[~np.isnan(EtCo2Mean)]/EtCo2Mean[0],
                        label=str('EtCo2'))
            # plt.xlabel('Time (min)')
            plt.title(group)
            plt.legend(loc='lower center')
            # plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
            # ax.scatter(x=Time[~np.isnan(cO2RatioMean)], y=cO2RatioMean[~np.isnan(cO2RatioMean)],
            #             label=group + ' cO2 AV ratio')

    if graph==True:
        plt.show()
    # print(cO2Ratio)


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

    cumHaz = dict.fromkeys(groups)
    for group in groups:
        y = []
        for t in Time:
            y.append(sum(surv[group]> t)/len(surv[group]))

        if graph == True:
            plt.step(Time, y, where='post', label=group)
            plt.ylim((0,1))

        cumHaz[group] = (Time, y)
    if graph == True:
        plt.title('Overall Survival Curves.')
        plt.legend(loc='upper right')
        plt.show()
    return cumHaz






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

    # PCo2Ratio(Dataset)

    extractSurvivalCurve(Dataset, graph = True)