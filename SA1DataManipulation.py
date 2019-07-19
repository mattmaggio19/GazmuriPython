


#This file is for functions that actually change or calculate other parameters from the SA-1 study.
# Hopefully these funciton are general enough to be useful.

def BloodWithdrawnPerKg(Dataset= None):
    #adds a new field to each experiment. 'BloodWithdrawnPerKg'
    if Dataset != None:
        for exp in Dataset:
            print('experiment number ' + str(exp['experimentNumber']))
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
    pass

if __name__ == '__main__':
    #Testing code goes here.
    pass