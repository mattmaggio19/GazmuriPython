import time
import numpy as np
import pandas as pd
import SA1DataLoader

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
        slice_lst = [slice(4, 23)]
        UVar = pd.concat([df_exp.iloc[slice_lst[0], 1]])
        UVar = SA1DataLoader.Drop_units(UVar) # Shorten the names to take the units out.
        UVal = pd.concat([df_exp.iloc[slice_lst[0], 5]])
        Udict = dict(zip(UVar, UVal))




        # Special one off to get the hespand delievered in a time resolved manner.
        # HesDelivered = df_exp.iloc[19, slice(5, 6 + 35)]

        #Get the Intervention group as if they got ZN or EPO
        ZN, EPO = df_exp.iloc[6, 3], df_exp.iloc[6, 4]

        if ZN == 'X' and EPO == 'X':
            Intervention = 'Blinded'
        elif ZN == 'Y' and EPO == 'Y':
            Intervention = 'ZN+/EPO+'
        elif  ZN == 'Y' and EPO == 'N':
            Intervention = 'ZN+/EPO-'
        elif  ZN == 'N' and EPO == 'Y':
            Intervention = 'ZN-/EPO+'
        elif  ZN == 'N' and EPO == 'N':
            Intervention = 'Neg Control'
        else:
            Intervention = 'Unknown'


        #Get the Repeditive var names and data. Similar to above.
        indexX = [slice(26, 43), slice(46, 64), slice(65, 73), slice(75, 93), slice(94, 99)]
        indexY = slice(5, 46)


        # Time values are known a priori, Time here is in segments of VF and CC and resp.
        Time = df_exp.iloc[0, indexY]

        #df_exp.iloc[slice(26,90),1]
        RVar = list()
        RVal = list()
        for i in range(0, len(indexX)):   #Run a loop to make this less verbose
            RVar.append(df_exp.iloc[indexX[i], 1])
            # print(df_exp.iloc[indexX[i], 1])
            RVal.append(df_exp.iloc[indexX[i], indexY])
        RVar = pd.concat(RVar, axis=0)
        RVar = SA1DataLoader.Drop_units(RVar) # Shorten the names to take the units out.
        RVal = pd.concat(RVal, axis=0)

        RDict = dict()
        #Loop through RVar, adding key and value pairs adding the series to the dict
        for index, key in enumerate(RVar):
            RDict[key] = RVal.iloc[index, :]

        ExpDict = {**Udict, **RDict}
        #Add a few more fields and then stack up into a list to complete the experiment loading
        ExpDict["experimentNumber"] = exp
        ExpDict["Time"] = Time
        ExpDict['Intervention'] = Intervention
        # ExpDict['HESDelivered'] = HesDelivered
        # print("Experiment {0} Loaded and Parsed taking {1}".format(exp, time.time()-t1))
        Output_lst.append(ExpDict)
        print("done")

    return Output_lst


def StandardLoadingFunction(useCashe=False):
    path = r'C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\Master Workbook (ZNEPO).xlsx'
    Dataset = Parse_excel(path=path, Experiment_lst=["2019058"])
    SA1DataLoader.DescriptivesExport(Dataset, OutName='RatZnEPO_Descriptives', groups=None)
    #Quick piece of code to fix the first two colunms being hard coded.


if __name__ == "__main__":

    #Run the anaylsis starting from the master Excel sheets.
    Dataset = StandardLoadingFunction(useCashe=False)