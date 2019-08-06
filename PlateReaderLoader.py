

import numpy as np


def loadPlateReaderTxt(path, sampleNames, skipCol, dataType ='Spectrum'):
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
        expLst.pop() #Remove the last exp, the text file terminates in a ~end

        #Clean up the sample 2 D lists into a numpy array.
        Output = []
        for num, exp in enumerate(expLst):
            Output.append(dict.fromkeys(sampleNames, []))
            Output[num]['Emissions'] = expLst[num]['Emissions']  # Get the Emissions
            for Sample in sampleNames:
                Output[num][Sample] = np.array(expLst[num][Sample],dtype=float) #cast into numpy
    print('Dataset Loaded')
    return Output

def plotSpectra(Dataset, exp = [0], titles = ['450 nm Excitation']):
    pass #Maybe implement here for easy calling?



if __name__ == "__main__":
    path = r'C:\Users\mattm\Documents\Gazmuri analysis\Microsphere Spectroscopy\Plate reader data\Microspheres spectra data_blue and green.txt'
    sampleNames = ['Empty', 'DI', "Yellow MS", 'Pink MS', 'Purple MS', 'Coral MS']

    Dataset = loadPlateReaderTxt(path, sampleNames, skipCol=1)
    plotSpectra(Dataset, exp=[0], titles=['450 nm Excitation'])