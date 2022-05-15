import pickle

comment = "Erster Durchlauf mit 4 Agenten, vollständig dezentral"
toChangeComment=[1]
path = "data/unabhaengig/data{}.pkl"

for i in toChangeComment:
    with open(path.format(i), "rb") as a_file:
        argsDict = pickle.load(a_file)

    argsDict['params']['comment'] = comment

    with open(path.format(i), "wb") as a_file: 
        pickle.dump(argsDict,a_file)
    