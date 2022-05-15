import pickle,Plot

toPlot=[0,1,2,3,4,5,6,7,8,9] #
plotScatterPlots = False
plotSeparateRuns = False
plotSeparateRunsWithManualLabels = False
manualLabels = ['job type 0','_','job type 1','_','job type 2'] 
plotAverageAndSD = False
plot2AveragesAndSDs = False
plotSeveralMetricsOfMultipleRuns = False
plotOneMetricOfMultipleRunsOfMultipleFolders = False
plotSeveralJobMetricsOfMultipleRunsOfMultipleFolders = True
ylabel = "Average price"
value = "prices"
values = ['0','1','2']
printInfo = False
location = 'upper center'
jobKind = 0
jobKinds = [0,1,2]  #
bottom = 0
top = 8
nCol = 3
smooth= 20
variable = None # 'hardcodedAgents'
if variable is not None:
    path = "C:/Users/lenna/Desktop/Experiment 5/nCommRew/4 Kerne/data/data{}.pkl"
if variable is None:
    path = 'C:/Users/lenna/Desktop/marl-scheduling/data neu/Experiment 1 mit Trading/2 Agenten/voll agg/data{}.pkl'#

paths = ['C:/Users/lenna/Desktop/marl-scheduling/data neu/Experiment 5/3 Jobs/commRew/data{}.pkl']

'''
'C:/Users/lenna/Desktop/marl-scheduling/data neu/Experiment 1 mit Trading/4 Agenten 12k e/divided/data{}.pkl',
        'C:/Users/lenna/Desktop/marl-scheduling/data neu/Experiment 1 mit Trading/4 Agenten 12k e/divided lok PS/data{}.pkl',
        'C:/Users/lenna/Desktop/marl-scheduling/data neu/Experiment 1 mit Trading/4 Agenten 12k e/halb agg/data{}.pkl'

'C:/Users/lenna/Desktop/marl-scheduling/data/Experimente/Experiment 5/3 Jobs/commRew/data/data{}.pkl'

    "C:/Users/lenna/Desktop/Experiment 1/2 Agenten/voll-aggregiert/data/data{}.pkl",
        "C:/Users/lenna/Desktop/Experiment 1/2 Agenten/halb-aggregiert/data/data{}.pkl",
        "C:/Users/lenna/Desktop/Experiment 1/2 Agenten/dezentral/data/data{}.pkl"

"C:/Users/lenna/Desktop/marl-scheduling_Neuanfang/data/Test3/data{}.pkl",
        "C:/Users/lenna/Desktop/marl-scheduling_Neuanfang/data/Test2/data{}.pkl",
        "C:/Users/lenna/Desktop/marl-scheduling_Neuanfang/data/Test1/data{}.pkl",

"C:/Users/lenna/Desktop/Experiment 1/2 Agenten/voll-aggregiert fix/data/data{}.pkl",
"C:/Users/lenna/Desktop/Experiment 1/2 Agenten/halb-aggregiert fix/data/data{}.pkl",
"C:/Users/lenna/Desktop/Experiment 1/2 Agenten/dezentral/data/data{}.pkl"

"C:/Users/lenna/Desktop/Experiment 2/2 Agenten/globales PS/data/data{}.pkl",
        "C:/Users/lenna/Desktop/Experiment 2/2 Agenten/lokales PS/data/data{}.pkl"

"C:/Users/lenna/Desktop/Experiment 1/4 Agenten/dezentral/data/data{}.pkl",
        "C:/Users/lenna/Desktop/Experiment 1/4 Agenten/halb-aggregiert/data/data{}.pkl"

"C:/Users/lenna/Desktop/Experiment 3/2 Kerne/halb-aggregiert/data/data{}.pkl",
    "C:/Users/lenna/Desktop/Experiment 3/2 Kerne/aufgeteilt/data/data{}.pkl",
    "C:/Users/lenna/Desktop/Experiment 3/2 Kerne/lokPM/data/data{}.pkl"

"C:/Users/lenna/Desktop/Experiment 1/4 Agenten/dezentral/data/data{}.pkl",
        "C:/Users/lenna/Desktop/Experiment 2/4 Agenten/lokales PS/data/data{}.pkl",
        "C:/Users/lenna/Desktop/Experiment 2/4 Agenten/globales PS/data/data{}.pkl"

"C:/Users/lenna/Desktop/Experiment 4/6 Jobs/nCommRew/data/data{}.pkl",
"C:/Users/lenna/Desktop/Experiment 4/6 Jobs/commRew/data/data{}.pkl"

"C:/Users/lenna/Desktop/Vollständig Dezentral/neuAngRew/data/data{}.pkl",  
        "C:/Users/lenna/Desktop/Experiment 2/4 Agenten/lokales PS/data/data{}.pkl",
        "C:/Users/lenna/Desktop/Experiment 1/2 Agenten/dezentral/data/data{}.pkl",
        'C:/Users/lenna/Desktop/Experiment 2/2 Agenten/globales PS/data/data{}.pkl'
'''


if plotScatterPlots is True:
    for i in toPlot:
        a_file = open(path.format(i), "rb")
        argsDict = pickle.load(a_file)
        FREEPRICES = argsDict['params']['freePrices']

        if FREEPRICES:
            Plot.plotFreePricesResult(argsDict)
        else:
            Plot.plotFixPricesResult(argsDict)
        a_file.close()

if (plotScatterPlots is False)&(plotOneMetricOfMultipleRunsOfMultipleFolders is False)&(plotSeveralJobMetricsOfMultipleRunsOfMultipleFolders is False):
    argsDicts = []
    for i in toPlot:
        a_file = open(path.format(i), "rb")
        argsDict = pickle.load(a_file)
        argsDicts.append(argsDict)
    #if variable is not None:
        #argsDicts.sort(key=lambda _: _['params'][variable])
    if plotAverageAndSD is True:
        Plot.plotAverageAndSDOfMultipleRuns(argsDicts,value,ylabel,variable,smooth)
    if plotSeparateRuns is True:
        Plot.plotSeparateRuns(argsDicts,value,variable,smooth)
    if plotSeparateRunsWithManualLabels is True:
        Plot.plotSeparateRunsWithManualLabels(argsDicts,values,ylabel,manualLabels,bottom,top,smooth,location,printInfo)
    if plot2AveragesAndSDs is True:
        Plot.plotTwoPrioritiesAverageAndSDOfMultipleRuns(argsDicts,value,smooth,bottom,top,printInfo,location)
    if plotSeveralMetricsOfMultipleRuns is True:
        Plot.plotSeveralMetricsOfMultipleRuns(argsDicts,values,ylabel,manualLabels,bottom,top,smooth,printInfo,location)


if plotOneMetricOfMultipleRunsOfMultipleFolders is True:
    listOfArgsDicts = []
    for path in paths:
        argsDicts = []
        for i in toPlot:
            a_file = open(path.format(i), "rb")
            argsDict = pickle.load(a_file)
            argsDicts.append(argsDict)
        listOfArgsDicts.append(argsDicts)
    
    Plot.plotOneMetricOfMultipleRunsOfMultipleFolders(listOfArgsDicts,value,ylabel,manualLabels,bottom,top,smooth,printInfo,location,jobKind)

if plotSeveralJobMetricsOfMultipleRunsOfMultipleFolders is True:
    listOfArgsDicts = []
    for path in paths:
        argsDicts = []
        for i in toPlot:
            a_file = open(path.format(i), "rb")
            argsDict = pickle.load(a_file)
            argsDicts.append(argsDict)
        listOfArgsDicts.append(argsDicts)
    
    Plot.plotSeveralJobMetricsOfMultipleRunsOfMultipleFolders(listOfArgsDicts,value,ylabel,manualLabels,bottom,top,smooth,printInfo,location,jobKinds,nCol)