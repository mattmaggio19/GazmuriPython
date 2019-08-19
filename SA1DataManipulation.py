
import SA1DataLoader
import numpy as np
import subprocess, os
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec

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
        plt.axvline(240, color='red', label='End of acute phase')
        plt.title('Overall Survival Curves.')
        plt.legend(loc='upper right')
        plt.show()
    return cumSurvival

def RelationshipBetweenAoSystolicHESandSurvival(Dataset = None, graph = False):
    #This is a plot Sal proposed that would let us look at the relationship between HES administration pressure and survival.
    #Basically we can seperate the animals by who got the last dose of hespand and who did not get the last dose of hespand
    #Main y axis would be systolic Ao pressure. (though techincally we used the LV in the acute phase because it was more accurate.)
    #Secondary y axis would be survival to see if the survival curve declines after missing a dose, we can code the deaths by their last checktime

    CheckTimes = [30, 120, 240, 8*60, 12*60, 16*60, 20*60, 24*60]
    CheckTimesArray = np.array(CheckTimes)
    groups = ['NS', 'TLP', 'POV', 'AVP']
    groups_Format = []
    SubplotCoordVert, SubplotCoordHoriz = [0, 0, 1, 1], [0, 1, 0, 1]  # useful lists for getting plots in a square.
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
        SystolicPressure[group]['CombinedSystolic'] = np.ones(Time.shape, dtype=float) * np.nan
        SystolicPressure[group]['Survival'] = []
        SystolicPressure[group]['Survival'].append(np.nan)

    for exp in Dataset:
        group = exp['Intervention']

        TimeArray = np.array(Time)
        HES = exp['HESDelivered']
        HESArray = np.array(HES)
        AoSys = np.array(exp['Ao systolic'])
        LVSys = np.array(exp['LV systolic'])
        Acute = (TimeArray <= 240)
        Survival = exp['Survival 72 hours']

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
        SystolicPressure[group]['CombinedSystolic'] = np.vstack((SystolicPressure[group]['CombinedSystolic'], CombinedSystolic))
        SystolicPressure[group]['GotHES'] = np.vstack((SystolicPressure[group]['GotHES'], GotHES))
        SystolicPressure[group]['MissedHES'] = np.vstack((SystolicPressure[group]['MissedHES'], MissedHes))
        if Survival == 'Y':
            SystolicPressure[group]['Survival'].append(True)
        else:
            SystolicPressure[group]['Survival'].append(False)

    if graph:
        fig, axs = plt.subplots(2, 2)
        plt.suptitle('Systolic Pressure Per group ')
        Percentconversion = 100
    for ix, group in enumerate(groups):

        #CombinedSystolic
        # axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].step(np.arange(len(cumSurvival[group])), np.array(cumSurvival[group]), where='post', label='Cum Survival')

        # axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].title('Survival Curve, Treatment = ' + group)
        # axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].show()

        # #Plotting code for the Systolic pressure traces.
        # plt.plot(np.array(Time), np.nanmean(SystolicPressure[group]['GotHES'], dtype=float, axis=0))
        # plt.plot(np.array(Time), np.nanmean(SystolicPressure[group]['MissedHES'], dtype=float, axis=0))
        # plt.step(np.arange(len(cumSurvival[group])), np.array(cumSurvival[group])*100, where='post', label=group)

        # Plotting code for the combined Systolic pressure traces.

        #The mean systolic pressure for determining if an animal got HES.
        if graph:
            size = 4
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].plot(np.array(Time), np.nanmean(SystolicPressure[group]['CombinedSystolic'], dtype=float, axis=0), label='Group mean of Systolic Pressure', color = 'blue' )
            for col in np.arange(1, SystolicPressure[group]['CombinedSystolic'].shape[0]-1):
                if col == 1:
                    axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(-10, -10, s=size, color = 'blue', marker = 'x', label= 'Survivors' )
                    axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(-10, -10, s=size, color='red', marker='x',
                                                                             label='NonSurvivors')
                if SystolicPressure[group]['Survival'][col]:
                    axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(np.array(Time), SystolicPressure[group]['CombinedSystolic'][col][:], s=size, color = 'blue', marker = 'x', label= None)
                else:
                    axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(np.array(Time),
                                                                             SystolicPressure[group]['CombinedSystolic'][
                                                                                 col][:], s=size, color='red', marker='x', label= None)

            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axhline(80, color='red', label='HES Admin Threshold')
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axvline(28, color='Cyan', label='Hes Admin times')
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axvline(118, color='Cyan', label=None)
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axvline(235, color='Cyan', label=None)
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axvline(8*60 -20, color='Cyan', label=None)
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axvline(8 * 60 - 20, color='Cyan', label=None)
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axvline(12 * 60 - 20, color='Cyan', label=None)
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axvline(16 * 60 - 20, color='Cyan', label=None)
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axvline(20 * 60 - 20, color='Cyan', label=None)
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axvline(24 * 60 - 20, color='Cyan', label=None)
            # axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].axvline(240, color='red', label='End of acute phase')

            # # #Draw the survival curve on the plot. Get's a little busy.
            # axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(X[Z], Y[Z]*Percentconversion, label="Got last checked dose of HES", color= 'Firebrick')
            # axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(X[~Z], Y[~Z]*Percentconversion, label="Missed last checked dose of HES", color= 'Cyan')
            # axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].step(np.arange(len(cumSurvival[group])), np.array(cumSurvival[group])*Percentconversion, where='post', color='darkorange', label=(group + ' % Cumulative Survival'))

            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].legend(loc='lower right', prop={'size': 8})
            # axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_xlim([-10, 300])
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_xlim([-10, 800])
            # axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_xlim([-10, 72*60]) #The whole dataset.
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_ylim([30, 130])
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_title(group)
    if graph:
        plt.tight_layout()
        plt.show()

def GeneralizedPlotter(XData, YData, Field=None, ScatterCatagories = False, ScatterArray=None, SeperateTreatmentplots = False, TreatmentArray=None ):
    #For now assume that x is always time. Can test later.

    ColorArray = ['Blue', 'FireBrick']
    Treatmentgroups = ['NS', 'TLP', 'POV', 'AVP']
    SubplotCoordVert, SubplotCoordHoriz = [0, 0, 1, 1], [0, 1, 0, 1]
    size = 4
    FudgeFactor = 3
    AxesBreakListX = ((0, 240), (8*60, 72*60))

    suptitle = Field + ' grouped by '
    if isinstance(ScatterCatagories, type(list())):
        suptitle = suptitle + ScatterCatagories[0]
    if isinstance(SeperateTreatmentplots, type(list())):
        suptitle = suptitle + 'and ' + 'Treatment '

    #Detect if we need to use broken axes
    PlotIdxBroke = np.nonzero(np.sum(np.invert(np.isnan(YData)), axis=0))
    if XData[PlotIdxBroke][-1] > 250:
        BreakAxes = True
        print('brakeing the axes')
    else:
        BreakAxes = False


    if isinstance(SeperateTreatmentplots, type(list())):
        if BreakAxes:
            fig = plt.figure(constrained_layout=False)
            sps = GridSpec(2, 2, figure=fig)
        else:
            fig, axs = plt.subplots(2, 2)

        for ix, group in enumerate(SeperateTreatmentplots):
            if BreakAxes == False:
                for i, cat in enumerate(ScatterCatagories):
                    Index = np.where(np.logical_and(ScatterArray == cat, TreatmentArray == group))[0]
                    print(ColorArray[i], cat, group, len(Index))
                    for col in np.arange(0, Index.shape[0]):
                        # Scatter the data from each animal.
                        if col == 0:
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(XData + (i * FudgeFactor),
                                                                                     YData[Index[col]][:], s=size,
                                                                                     color=ColorArray[i],
                                                                                     label=ScatterCatagories[i])
                        else:
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(XData + (i * FudgeFactor),
                                                                                     YData[Index[col]][:], s=size,
                                                                                     color=ColorArray[i],
                                                                                     label=None)
                        # Plot the group means
                        if np.any(np.isnan(YData[Index])):  # Plot group means correctly even when data has nans.
                            PlotIdx = np.nonzero(np.sum(np.invert(np.isnan(YData[Index])), axis=0))
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].plot(XData[PlotIdx],
                                                                                  np.nanmean(YData[Index], axis=0)[
                                                                                      PlotIdx],
                                                                                  color=ColorArray[i], label=None)
                        else:
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].plot(XData,
                                                                                  np.nanmean(YData[Index], axis=0),
                                                                                  color=ColorArray[i], label=None)

                        axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_title(group)
                        axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].legend(loc='best', prop={'size': 8})
            else:

                # #Custom Implementation of a broken axis plot taking some inspiration.
                # gs = GridSpec.GridSpecFromSubplotSpec(subplot_spec=sps[ix])
                # big_ax = plt.Subplot(fig, sps[ix])
                # [sp.set_visible(False) for sp in big_ax.spines.values()]
                # big_ax.set_xticks([])
                # big_ax.set_yticks([])
                # big_ax.patch.set_facecolor('none')
                # for igs in gs:
                #     ax = plt.Subplot(fig, igs)
                #     fig.add_subplot(ax)
                #     axs.append(ax)
                # fig.add_subplot(big_ax)


                #broken Axes was unsuccessful at doing the job out of the box.
                bax = brokenaxes(xlims=AxesBreakListX, subplot_spec=sps[ix])
                bax.standardize_ticks(xbase = 10)
                for i, cat in enumerate(ScatterCatagories):
                    Index = np.where(np.logical_and(ScatterArray == cat, TreatmentArray == group))[0]
                    print(ColorArray[i], cat, group, len(Index))

                    for col in np.arange(0, Index.shape[0]):
                        # Scatter the data from each animal. break the axes
                        if col == 0:
                            bax.scatter(XData + (i * FudgeFactor), YData[Index[col]][:], s=size, color=ColorArray[i],label=ScatterCatagories[i])
                        else:
                            bax.scatter(XData + (i * FudgeFactor), YData[Index[col]][:], s=size, color=ColorArray[i], label=None)
                        # Plot the group means
                        if np.any(np.isnan(YData[Index])):  # Plot group means correctly even when data has nans.
                            PlotIdx = np.nonzero(np.sum(np.invert(np.isnan(YData[Index])), axis=0))
                            bax.plot(XData[PlotIdx], np.nanmean(YData[Index], axis=0)[PlotIdx], color=ColorArray[i], label=None)
                        else:
                            bax.plot(XData, np.nanmean(YData[Index], axis=0), color=ColorArray[i], label=None)


                        bax.set_title(group)
                        bax.legend(loc='best', prop={'size': 8})

        #
        # if BreakAxes == True: #Reformat the axes breaks.
        #     for j, ax in enumerate(fig.axes):
        #         if j%3 == 0: #Left of the break
        #             ax.set_xlim(0, 250)
        #         elif j%3 == 1: #Right of the break
        #             ax.set_xlim(280, 72*60)
        #         elif j%3 == 2: #This axis contains the other 2. For some reason.
        #             pass


        plt.suptitle(suptitle, size=10)
        plt.tight_layout()
        plt.show()

        return fig

    else:
        fig, axs = plt.subplots(1, 1)



    pass

def GroupedPlots(Dataset= None, Field=None, groupBy = 'HES120', graph=False):
    # Prurpose of this function is to construct a plot to see if at the whole experiment level animals that got fluid at 120 had better hemodynamics.
    # To that end, seperate the animals into Got HES at 120 min or didn't get HES at 120 and then graph the group means and of the field in question.
    # Hes120
    Time = Dataset[0]['Time']
    TimeArray = np.array(Time)
    ColorArray = ['Blue', 'FireBrick']
    Treatmentgroups = ['NS', 'TLP', 'POV', 'AVP']
    SubplotCoordVert, SubplotCoordHoriz = [0, 0, 1, 1], [0, 1, 0, 1]

    size = 4
    FudgeFactor = 3
    for ix, exp in enumerate(Dataset):
        Data = np.array(exp[Field], dtype=float)
        HES = np.array(exp['HESDelivered'])
        Treatment = exp['Intervention']
        Survival = exp['Survival 72 hours']
        if groupBy == 'HES120':
            #For grouping by Categories Needs to be deterministic.
            categories = ['Got HES at 120', 'Missed HES at 120']
            if np.isnan(HES[10]):
                categ = categories[1]
            else:
                categ = categories[0]

        elif groupBy == 'Treatment':
            categories = Treatmentgroups
            categ =None

        elif groupBy == 'Survival&Treatment':
            categories = ['Survived to 72', 'Non-survivors']
            if Survival == 'N':
                categ = categories[1]
            else:
                categ = categories[0]

        elif groupBy == 'HES120&Treatment':
            categories = ['Got HES at 120', 'Missed HES at 120']
            if np.isnan(HES[10]):
                categ = categories[1]
            else:
                categ = categories[0]


        #elif more groupBy Conditions here.


        if ix == 0:
            DataArray = Data
            categArray = np.ones(0)
            categArray= np.append(categArray, categ)
            TreatmentArray = np.ones(0)
            TreatmentArray = np.append(TreatmentArray, Treatment)
        else:
            DataArray = np.vstack((DataArray, Data))
            categArray= np.append(categArray, categ)
            TreatmentArray = np.append(TreatmentArray, Treatment)

    if graph:

        if groupBy == 'HES120':
            #This is for the whole dataset grouped by if the animal got HES120
            for i, cat in enumerate(categories):
                print(ColorArray[i], cat)
                Index = np.where(categArray == cat)[0]
                for col in np.arange(0, Index.shape[0]):

                    #Scatter the data from each animal.
                    if col == 0:
                        plt.scatter(TimeArray+(i*FudgeFactor), DataArray[Index[col]][:], s=size, color=ColorArray[i], label=categories[i])
                    else:
                        plt.scatter(TimeArray+(i*FudgeFactor), DataArray[Index[col]][:], s=size, color=ColorArray[i], label=None)

                #Plot the group means

                if np.any(np.isnan(DataArray[Index])):  # Plot group means correctly even when data has nans.
                    PlotIdx = np.nonzero(np.sum(np.invert(np.isnan(DataArray[Index])), axis=0))
                    plt.plot(TimeArray[PlotIdx], np.nanmean(DataArray[Index], axis=0)[PlotIdx], color=ColorArray[i], label=None)
                else:
                    plt.plot(TimeArray, np.nanmean(DataArray[Index], axis=0),color=ColorArray[i], label=None)

            plt.legend(loc='best', prop={'size': 8})
            plt.title(Field +' from whole Dataset, grouped by ' + groupBy )
            plt.tight_layout()
            plt.show()

        elif groupBy == 'Treatment':
            fig, axs = plt.subplots(2, 2)
            plt.suptitle(Field + ' grouped by ' + groupBy)
            for ix, group in enumerate(Treatmentgroups):
                # print(ColorArray[i], cat)
                Index = np.where(TreatmentArray == group)[0]
                for col in np.arange(0, Index.shape[0]):
                    # Scatter the data from each animal.
                    if col == 0:
                        axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(TimeArray,
                                                                                 DataArray[Index[col]][:], s=size,
                                                                                 color=ColorArray[0],
                                                                                 label=Treatmentgroups[ix])
                    else:
                        axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(TimeArray ,
                                                                                 DataArray[Index[col]][:], s=size,
                                                                                 color=ColorArray[0], label=None)
                    # Plot the group means
                    if np.any(np.isnan(DataArray[Index])):  # Plot group means correctly even when data has nans.
                        PlotIdx = np.nonzero(np.sum(np.invert(np.isnan(DataArray[Index])), axis=0))
                        axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].plot(TimeArray[PlotIdx],
                                                                              np.nanmean(DataArray[Index], axis=0)[
                                                                                  PlotIdx],
                                                                              color=ColorArray[0], label=None)
                    else:
                        axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].plot(TimeArray,
                                                                              np.nanmean(DataArray[Index], axis=0),
                                                                              color=ColorArray[0], label=None)
                    axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_title(group)
                    axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].legend(loc='best', prop={'size': 8})

            plt.tight_layout()
            plt.show()

        elif groupBy == 'Survival&Treatment':
            fig, axs = plt.subplots(2, 2)
            plt.suptitle(Field +' grouped by ' + groupBy)
            for ix, group in enumerate(Treatmentgroups):
                for i, cat in enumerate(categories):
                    # print(ColorArray[i], cat)
                    Index = np.where(np.logical_and( categArray == cat , TreatmentArray == group ))[0]
                    for col in np.arange(0, Index.shape[0]):
                        # Scatter the data from each animal.
                        if col == 0:
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(TimeArray + (i * FudgeFactor), DataArray[Index[col]][:], s=size,
                                        color=ColorArray[i], label=categories[i])
                        else:
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(TimeArray + (i * FudgeFactor), DataArray[Index[col]][:], s=size,
                                        color=ColorArray[i], label=None)
                        # Plot the group means
                        if np.any(np.isnan(DataArray[Index])): #Plot group means correctly even when data has nans.
                            PlotIdx = np.nonzero(np.sum(np.invert(np.isnan(DataArray[Index])), axis=0))
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].plot(TimeArray[PlotIdx], np.nanmean(DataArray[Index], axis=0)[PlotIdx],
                                     color=ColorArray[i], label=None)
                        else:
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].plot(TimeArray, np.nanmean(DataArray[Index], axis=0),
                                     color=ColorArray[i], label=None)
                        axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_title(group)
                        axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].legend(loc = 'best', prop={'size': 8})
            plt.tight_layout()
            plt.show()

        elif groupBy == 'HES120&Treatment':
            try:
                GeneralizedPlotter(XData=TimeArray, YData=DataArray, Field=Field, ScatterCatagories=categories, ScatterArray=categArray,
                                   SeperateTreatmentplots=Treatmentgroups, TreatmentArray=TreatmentArray)
            except:
                print('generalized code still borked.')
            #Developing a generalized plotting func

            fig, axs = plt.subplots(2, 2)
            plt.suptitle(Field + ' grouped by ' + groupBy)
            for ix, group in enumerate(Treatmentgroups):
                for i, cat in enumerate(categories):
                    # print(ColorArray[i], cat)
                    Index = np.where(np.logical_and(categArray == cat, TreatmentArray == group))[0]
                    for col in np.arange(0, Index.shape[0]):
                        # Scatter the data from each animal.
                        if col == 0:
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(TimeArray + (i * FudgeFactor),
                                                                                     DataArray[Index[col]][:], s=size,
                                                                                     color=ColorArray[i],
                                                                                     label=categories[i])
                        else:
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].scatter(TimeArray + (i * FudgeFactor),
                                                                                     DataArray[Index[col]][:], s=size,
                                                                                     color=ColorArray[i], label=None)
                        # Plot the group means
                        if np.any(np.isnan(DataArray[Index])):  # Plot group means correctly even when data has nans.
                            PlotIdx = np.nonzero(np.sum(np.invert(np.isnan(DataArray[Index])), axis=0))
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].plot(TimeArray[PlotIdx],
                                                                                  np.nanmean(DataArray[Index], axis=0)[
                                                                                      PlotIdx],
                                                                                  color=ColorArray[i], label=None)
                        else:
                            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].plot(TimeArray,
                                                                                  np.nanmean(DataArray[Index], axis=0),
                                                                                  color=ColorArray[i], label=None)
                        axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_title(group)
                        axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].legend(loc='best', prop={'size': 8})

            plt.tight_layout()
            plt.show()


        # elif more groupBy Conditions for more plots here.







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
        plt.axvline(240, color= 'red' , label='End of acute period')
        plt.legend(loc='upper left')
        plt.ylabel('Total HES administered (ml)')
        plt.xlabel('Survival Time (min)')
        plt.title('HES administration vs Survival Time')
        plt.tight_layout()
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
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].hist(x, 8, histtype='bar', stacked=True, fill = True, label=['Survivors', 'Non survivors'])
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].set_title(group)
            axs[SubplotCoordVert[ix], SubplotCoordHoriz[ix]].legend(loc='best')
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

def TimeInvariantScatterPlot(Dataset=None, xfield=None,yfield=None, seperateTreatement=False ):
    x, y, Treatment = [], [], []
    groups = ['NS', 'TLP', 'POV', 'AVP']

    for exp in Dataset:
        x.append(exp[xfield])
        y.append(exp[yfield])
        Treatment.append(exp['Intervention'])
    if seperateTreatement:
        x = np.array(x)
        y = np.array(y)
        Treatment = np.array(Treatment)
        for group in groups:
            xdata = x[np.where(Treatment == group)[0]]
            xdata[np.where(xdata == 72*60)[0]] = xdata[np.where(xdata == 72*60)[0]] + np.random.normal(size=xdata[np.where(xdata == 72*60)[0]].shape, loc = 0, scale = 80)#Dither the x where animals make it to 72*60 min.
            plt.scatter(xdata, y[np.where(Treatment == group)[0]], label=group)

        plt.legend(loc='best', prop={'size': 8})

    else:
        plt.scatter(x, y)

    plt.title(yfield + ' by ' + xfield)
    plt.xlabel(xfield)
    plt.ylabel(yfield)
    plt.show()

def SaveFigureAsEPS(Fig, filepath=None):
    inkscape_path = "C://Program Files//Inkscape//inkscape.exe"\

    if filepath is not None:
        path, filename = os.path.split(filepath)
        filename, extension = os.path.splitext(filename)

        svg_filepath = os.path.join(path, filename + '.svg')
        emf_filepath = os.path.join(path, filename + '.emf')

        Fig.savefig(svg_filepath, format='svg')

        subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
        # os.remove(svg_filepath)

def PV_loop():
    #TODO Technically this isn't SA-1 material, but if I ever get around to doing this I'll troubleshoot the data here.
    pass

if __name__ == '__main__':
    #Testing code goes here.
    Dataset = SA1DataLoader.StandardLoadingFunction(useCashe=True)

    # GroupedPlots(Dataset, Field='LV systolic', groupBy='HES120&Treatment', graph=True)

    # Plot that identifies the HES administered per group against survival time and then graphs the histogram of HES admin per group with survivors vs non survivors
    # HESvsSurvival(Dataset, graph=True)

    # GroupedPlots(Dataset, Field='PetCO2 End Tidal Corrected', groupBy='Survival&Treatment', graph=True)

    TimeInvariantScatterPlot(Dataset, yfield='BloodLossByKg', xfield='Ao mean Time 30', seperateTreatement=True)

    TimeInvariantScatterPlot(Dataset, yfield='BloodLossByKg', xfield='Survival time', seperateTreatement=True)

    TimeInvariantScatterPlot(Dataset, yfield='BloodLossByKg', xfield='Liver Lacerations', seperateTreatement=True)



    RelationshipBetweenAoSystolicHESandSurvival(Dataset, graph=True)

    # HES = ResolvedHESAdministration(Dataset, output='ratio', graph=True)  PA systolic

    GroupedPlots(Dataset, Field='Heart rate LV', groupBy='HES120&Treatment', graph=True)

    # GroupedPlots(Dataset, Field='HESDelivered', groupBy='Survival&Treatment', graph=True)

    GroupedPlots(Dataset, Field='HCT Ao (OPTI)', groupBy='HES120&Treatment', graph=True)

    GroupedPlots(Dataset, Field='CCI', groupBy='Survival&Treatment', graph=True)

    GroupedPlots(Dataset, Field='PA systolic', groupBy='Survival&Treatment', graph=True)

    GroupedPlots(Dataset, Field='LV dP/dt min', groupBy='Survival&Treatment', graph=True)

    GroupedPlots(Dataset, Field='SVRI', groupBy='Survival&Treatment', graph=True)

    GroupedPlots(Dataset, Field='Lactate Ao (OPTI)', groupBy='HES120&Treatment', graph=True)

    GroupedPlots(Dataset, Field='Lactate Ao (OPTI)', groupBy='Survival&Treatment', graph=True)





    # GroupedPlots(Dataset, Field='Ao systolic', groupBy='Survival&Treatment', graph=True)

    # GroupedPlots(Dataset, Field='LV systolic', groupBy='Treatment', graph=True)

    GroupedPlots(Dataset, Field='LV systolic',  groupBy='HES120&Treatment', graph=False)

    GroupedPlots(Dataset, Field='VO2/ DO2', groupBy='Survival&Treatment', graph=True)

    #



    GroupedPlots(Dataset, Field='SVRI', groupBy='HES120&Treatment', graph=True)

    GroupedPlots(Dataset, Field='LV systolic', graph=True)


    # extractSurvivalCurve(Dataset, graph=True)

    # PCo2Ratio(Dataset, graph=False)



    # DO2I
    # VO2I
    # VO2/ DO2

    # 'Lactate Ao or PA'