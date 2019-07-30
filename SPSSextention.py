
import os
import re
import string
import SA1DataLoader

#Code to help deal with some of the syntax issues, DR.G wants to run the mixed model on every single rep parameter



def GeneralizedCopyFunction(Dataset, Path, ReplaceString = 'Aomean' , ExcludeFields = ['Date', 'Series', 'Time', 'Block', 'Intervention', 'Weight',
                                                                                       'Time ketamine injection', 'Time LL / TBI', 'Preparation time','Survival time','Survival 240 minutes',
                                                                                       'Survival 72 hours', 'Liver Lacerations', 'Estimated blood loss', 'Impactor depth', 'Impactor speed',
                                                                                       'Impactor dwell time','HES Infused', 'Dural tear', 'Left hemisphere volume',  'Left hemisphere injury',
                                                                                       'Right hemisphere volume','Right hemisphere injury', 'Temperature X', 'Food acquisition test time 1',
                                                                                       'Food acquisition test error score 1','Food acquisition test time 2','Food acquisition test error score 2',
                                                                                       'Food acquisition test time 3','Food acquisition test error score 3','Food acquisition test time 4',
                                                                                       'Food acquisition test error score 4','Food acquisition test time 5','Food acquisition test error score 5',
                                                                                       'experimentNumer']):





    #Function to generate a text file with copied syntax for each parameter.
    (OutPath, Infile) = os.path.split(Path)
    Outfile = open(os.path.join(OutPath,  Infile + '_Copied' + '.txt'), 'w')

    #Open the text file
    file = open(Path)
    text = file.read()

    SPSSRemovedCharicters = '   ()-+/'

    exp = Dataset[20] #Random dataset, they should have the same fields RIIIIGHT?
    for field in exp.keys():
        if field not in ExcludeFields: # Skip fields in Exclude fields.
            var = ''.join( c for c in field if c not in SPSSRemovedCharicters ) #Remove all the spss forbiden from fieldnames to make them the same as SPSS var syntax.
            print(var)
            Outfile.write(text.replace(ReplaceString, var))
            Outfile.write('\n')
            Outfile.write('\n')
            Outfile.write('\n') #3 newlines, just cus.

    file.close()
    Outfile.close()




if __name__ == '__main__':
    #Testing code goes here.
    Dataset = SA1DataLoader.StandardLoadingFunction()
    path = r'C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SPSS\07262019\MixedModelScaleUp.txt'
    Dataset = GeneralizedCopyFunction(Dataset, Path= path)



