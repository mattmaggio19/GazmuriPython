

import numpy as np
import pandas as pd
import csv

def loadPlateReaderTxt(path):
    #More general loader.
    f = csv.reader(open(path, "r"), delimiter='\t')
    metadata = []
    datalst = []
    for i, row in enumerate(f):
        if i < 3:
            metadata.append(row)
        else:
            if len(row) < 16:
                for j in range(0, 16-len(row)):
                    row.append('')
            row = [np.nan if R in ['#Sat', '', '\n'] else R for R in row]
            datalst.append(row)

    DataArray = np.array(datalst, dtype=float)
    Emissions = DataArray[np.where(~np.isnan(DataArray[:, 0]))[0], 0]
    for Emission in np.where(~np.isnan(DataArray[:, 0]))[0]:
        #TODO Fold the results from different emissions into the 3rd dim of the array.
        pass

    # DataLst, EmissionLst = [], []
    # count = 0
    # NewEmission, EmissionWavelenth = True, None
    # Data2D = []
    # with open(path) as f:
    #     # x = f.read()
    #     # print(x)
    #     lines = f.readlines()
    #     for line in lines:
    #         split = line.split('\t')
    #         print(split)
    #         # print(count, ": ", repr(line))
    #         count += 1
    #
    #         if split[0] == '~End': #No mo data.
    #             print('Reached the end of the file.')
    #             break
    #
    #         if np.char.isnumeric(split[0]) and NewEmission: #Handle a new Emissions instance
    #             Data2D = [] #Init a new 2D array the size of the plate.
    #             EmissionWavelenth = int(split[0])
    #             # Data2D = np.empty(shape=(8, 16)) * np.nan #Init a new 2D array the size of the plate.
    #             NewEmission = False
    #
    #         if not NewEmission:
    #             #append the data into the Data2D array.
    #             #Replace empty and saturated with nans.
    #             split = [np.nan if R in['#Sat', '', '\n'] else R for R in split]
    #             #pop the first 2 nans out, pop the last nan out because it was a /n
    #             # split.pop[]
    #             Data2D.append(split) #Convert to numpy array later.
    #
    #
    #         if all(elem == np.nan for elem in split[:-1]):
    #             if EmissionWavelenth is not None:
    #                 EmissionLst.append(EmissionWavelenth)
    #                 DataLst.append(Data2D)
    #             NewEmission, EmissionWavelenth = True, None


            # if np.char.isnumeric(next(f).split('\t')[0]):  #Reset for a new Emissions instance. Store data here and reset. Annoying that it triggers the first time.



        #Pile the 2D data into a 3D numpy array.


def loadPlateReaderTxtOLD(path, sampleNames, skipCol, dataType ='Spectrum'):
    #What was I thinking when I wrote this?

    f = open(path)
    next(f)  # Skip first line
    if dataType == 'Spectrum':  # logic should only work for spectra data
        expLst = [] #Set up initial experiment
        expLst.append({sample : [] for sample in sampleNames})
        expLst[-1]['Emissions'] = [] #Add a key value for the emissions. Excitation you have to get from the metadata or a priori.
        expNum = 0
        for idx, line in enumerate(f):
                    data = line.split('\t')
                    if data[0] == '~End': #Next experiment make a new entry in the exp list
                        # print('Making a new experiment')
                        expLst.append({sample : [] for sample in sampleNames})
                        expLst[-1]['Emissions'] = []  # Add a key value for the emissions. Excitation you have to get from the metadata or a priori.
                        expNum += 1
                    # elif data[1]
                    elif data[0] == 'Plate:': #do nothing
                        pass
                    elif data[0] == 'Wavelength(nm)':  #do nothing
                        pass
                    elif data[0] == r'~End\\n':
                        pass
                    elif data[0] == 'Copyright ? 2004 Molecular Devices. All rights reserved.':  #do nothing
                        pass
                    elif bool(data[0]): #then we have a new emissions data for the plate.
                        sample = 0
                        expLst[expNum]['Emissions'].append(int(data[0])) #file the emissions
                        Row = data[2 + skipCol:-1]
                        Row = [i for i in Row if i]  # remove empty string values
                        Row = [np.nan if R == '#Sat' else R for R in Row ]  # Replace saturated values with nan
                        expLst[expNum][sampleNames[sample]].append(Row)  # #Put the first one in.

                        sample += 1
                    elif not data[0]: #add another entry to the dataset.
                        if line.replace('\t', '').replace('\n', ''): #If list is empty we will get an out of range error
                            Row = data[2+skipCol:-1]
                            Row = [i for i in Row if i] #remove empty string values
                            Row = [np.nan if R == '#Sat' else R for R in Row ]  # Replace saturated values with nan
                            expLst[expNum][sampleNames[sample]].append(Row)  # Put the rest of the data in. one in.
                            sample += 1
        # expLst.pop() #Remove the last exp, the text file terminates in a ~end

        #Clean up the sample 2 D lists into a numpy array.
        Output = []
        for num, exp in enumerate(expLst):
            Output.append(dict.fromkeys(sampleNames, []))
            Output[num]['Emissions'] = expLst[num]['Emissions']  # Get the Emissions
            for Sample in sampleNames:
                Output[num][Sample] = np.array(expLst[num][Sample],dtype=float) #cast into numpy
    print('Dataset Loaded')
    return Output

def plotSpectra(Dataset, exp = [0], title = ['450 nm Excitation']):
    pass #Maybe implement here for easy calling?



if __name__ == "__main__":


    #
    path = r'C:\Users\mattm\Documents\Gazmuri analysis\Microsphere Spectroscopy\Plate reader data\Microspheres spectra data_blue and green.txt'
    # sampleNames = ['Empty', 'DI', "Yellow MS", 'Purple MS', 'Coral MS', 'Pink MS']

    path = r'C:\Users\mattm\Documents\Gazmuri analysis\Microsphere Spectroscopy\Plate reader data\Microspheres boiled 485 ex spectra.txt'
    sampleNames = ["Yellow MS", 'Purple MS', 'Coral MS', 'Pink MS']

    # Dataset = loadPlateReaderTxtOLD(path, sampleNames, skipCol=1)
    Dataset = loadPlateReaderTxt(path)

    plotSpectra(Dataset, exp=[0], title=['485 nm Excitation'])