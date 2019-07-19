
import os
import SA1DataLoader

#Code to help deal with some of the syntax issues, DR.G wants to run the mixed model on every single rep parameter


def GeneralizedCopyFunction(Dataset, Path, ReplaceString = 'Aomean' , ExcludeFields = ['Date', 'Series', 'Time', 'Block']):
    #Function to generate a text file with copied syntax for each parameter.
    (OutPath, Infile) = os.path.split(Path)
    Outfile = open(os.path.join(OutPath, 'CopySyntax.txt'), 'w')

    #Open the text file
    file = open(Path)
    text = file.read()

    exp = Dataset[20] #Random dataset, they should have the same fields RIIIIGHT?
    for field in exp.keys():
        if field not in ExcludeFields: # Skip fields in Exclude fields.
            print(field)
            Outfile.write(text.replace(ReplaceString, field.replace(' ', '') )) #Remove all spaces from fieldnames to make them the same as SPSS syntax.
            Outfile.write('\n')
            Outfile.write('\n')
            Outfile.write('\n') #3 newlines, just cus.

    file.close()
    Outfile.close()




if __name__ == '__main__':
    #Testing code goes here.
    Dataset = SA1DataLoader.StandardLoadingFunction()
    path = r'C:\Users\mattm\Documents\Gazmuri analysis\SA1 Analysis\SPSS\07152018\LinearMixedCode.txt'
    Dataset = GeneralizedCopyFunction(Dataset, Path= path)



