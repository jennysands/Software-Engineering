'''Create the files needed for Insect images'''

import pandas as pd
import os
import sklearn
from sklearn.model_selection import train_test_split

print(os.getcwdb())

dataframe = {"Animal name": [], "Path": [], "ID": []}
dataframe = pd.DataFrame(dataframe)

ID = 450
p = 0
directory = '/home/u668954/Insect_classifier/database/'
for foldername in os.listdir(directory):
    f = os.path.join(directory, foldername)

    name = f.replace("/home/u668954/Insect_classifier/database/", "")
    ID_str = str(ID)
    new_row = {"Animal name":name , "Path":f, "ID":ID_str}
    dataframe = dataframe.append(new_row, ignore_index=True)
    p += 1
    ID += 1

dataframe.to_csv("Insects.csv", index=False)


insectinfo = {"Animal name": [], "Path": [], "ID": []}
insectinfo = pd.DataFrame(insectinfo)

i= -1
for folder in dataframe["Path"]:
  i += 1
  for file in os.listdir(folder):
    insectname = (dataframe['Animal name'].loc[[i]])
    insectname = insectname.to_string(index=False, header=False)

    insectid = (dataframe['ID'].loc[[i]])
    insectid = insectid.to_string(index=False, header=False)



    new_row = {"Animal name":insectname, "Path":folder+"/"+file, "ID":insectid}
    insectinfo = insectinfo.append(new_row, ignore_index=True)
  
insectinfo.to_csv("Full insects.csv", index = False)
  

# labels = (df["labels"].loc[[p]])
# labels = labels.to_string(index=False, header=False)
# class_ID = (df["class id"].loc[[p]])
# class_ID = class_ID.to_string(index=False, header=False)
# new_row = {"Bird name":labels , "Path":i, "ID":class_ID}
# dataframe = dataframe.append(new_row, ignore_index=True)


'''Train test split'''
insectinfo = pd.read_csv (r'/home/u668954/Full insects.csv')


train, validation = train_test_split(insectinfo, test_size = 0.2, stratify=insectinfo["ID"])
validation, test = train_test_split(validation, test_size = 0.5, stratify=validation["ID"])


train.to_csv("insects train.csv", index = False)
validation.to_csv("insects validation.csv", index = False)
test.to_csv("insects test.csv", index = False)
