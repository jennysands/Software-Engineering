import pandas as pd
import matplotlib.pyplot as plt

'''Create dataframe for Training data'''
dataframe = {"Animal name": [], "Path": [], "ID": []}
dataframe = pd.DataFrame(dataframe)


p = 0
df = pd.read_csv (r'/home/u668954/Bird_classifier/birds.csv')
for i in df["filepaths"]:
  if i.startswith("tr"):
    labels = (df["labels"].loc[[p]])
    labels = labels.to_string(index=False, header=False)
    class_ID = (df["class id"].loc[[p]])
    class_ID = class_ID.to_string(index=False, header=False)
    new_row = {"Animal name":labels , "Path":("/home/u668954/Bird_classifier/"+str(i)), "ID":class_ID}
    dataframe = dataframe.append(new_row, ignore_index=True)
    p+=1
    continue
  else:
    break

print(dataframe)
dataframe.to_csv("Trainingfiles_bird.csv", index=False)

'''Create dataframe for Validation data'''
dataframe = {"Animal name": [], "Path": [], "ID": []}
dataframe = pd.DataFrame(dataframe)


p = 0
df = pd.read_csv (r'/home/u668954/Bird_classifier/birds.csv')
for i in df["filepaths"]:
  if i.startswith("v"):
    labels = (df["labels"].loc[[p]])
    labels = labels.to_string(index=False, header=False)
    class_ID = (df["class id"].loc[[p]])
    class_ID = class_ID.to_string(index=False, header=False)
    new_row = {"Animal name":labels , "Path":("/home/u668954/Bird_classifier/"+str(i)), "ID":class_ID}
    dataframe = dataframe.append(new_row, ignore_index=True)
    p+=1
    continue
  else:
    p+=1
    continue

print(dataframe)
dataframe.to_csv("Validationfiles_bird.csv", index=False)

'''Create dataframe for Testing data'''
dataframe = {"Animal name": [], "Path": []}
dataframe = pd.DataFrame(dataframe)

p = 0
df = pd.read_csv (r'/home/u668954/Bird_classifier/birds.csv')
for i in df["filepaths"]:
  if i.startswith("te"):
    labels = (df["labels"].loc[[p]])
    labels = labels.to_string(index=False, header=False)
    class_ID = (df["class id"].loc[[p]])
    class_ID = class_ID.to_string(index=False, header=False)
    new_row = {"Animal name":labels , "Path":("/home/u668954/Bird_classifier/"+str(i)), "ID": class_ID}
    dataframe = dataframe.append(new_row, ignore_index=True)
    p+=1
    continue
  else:
    p+=1
    continue

print(dataframe)
dataframe.to_csv("Testingfiles_bird.csv", index=False)



