import pandas as pd

'''Combine bird and insect training dataframes'''
insect_trained = pd.read_csv (r'/home/u668954/insects train.csv')
bird_trained = pd.read_csv(r'/home/u668954/Trainingfiles_bird.csv')

frames = [bird_trained, insect_trained]

final = pd.concat(frames)
final.to_csv("Merged_train.csv", index = False)



'''Combine bird and insect validation dataframes'''
insect_validation = pd.read_csv (r'/home/u668954/insects validation.csv')
bird_validation = pd.read_csv(r'/home/u668954/Validationfiles_bird.csv')

frames = [bird_validation, insect_validation]

final1 = pd.concat(frames)
final1.to_csv("Merged_validation.csv", index = False)



'''Combine bird and insect testing dataframes'''
insect_testing = pd.read_csv (r'/home/u668954/insects test.csv')
bird_testing = pd.read_csv(r'/home/u668954/Testingfiles_bird.csv')

frames = [bird_testing, insect_testing]

final2 = pd.concat(frames)
final2.to_csv("Merged_testing.csv", index = False)

