from matplotlib import pyplot as plt
import numpy as np
from itertools import cycle
import os
import textwrap as tw
from fractions import Fraction as F
import seaborn as sns, pandas as pd, torch
import statistics
from itertools import chain



def plotFixPricesResult(argsDict):
    plotPath = argsDict['plotPath']
    acceptorRewards = argsDict['acceptorRew']
    offerRewards = argsDict['coreChooserRew']
    prices = argsDict['prices'] 
    auctioneerRewards = argsDict['auctioneerRew'] 
    dwellTimes = argsDict['dwellTimes']
    meanJobFraction = argsDict['meanJob']
    agentRewards = argsDict['agentRew']
    acceptionQualities = argsDict['acceptionQuality']
    acceptionAmounts = argsDict['acceptionAmount']
    params = argsDict['params']
    # Zähler um eins erhöht für die Minues 1 JobPrio, Nenner um eins erhöht für den einen Zeitschritt mit Leerstand
    optimisticAverageAcceptorReward = (F(meanJobFraction.numerator,meanJobFraction.denominator + 1)) / params['numberOfAgents']   #numCores wurde oben und unten rausgekürzt
    roughAverageOfferReward = (F(meanJobFraction.numerator + 1,meanJobFraction.denominator + 1)*(params['numberOfCores'])) / (params['numberOfAgents'] * params['collectionLength'])
    fig = plt.figure(1,figsize=(30,22))
    sub1 = fig.add_subplot(331)
    sub1.title.set_text('Average Acceptor Rewards')
    sub1.set_xlabel('episode')
    sub1.set_ylabel('average reward')
    sub1.axhline(y = optimisticAverageAcceptorReward, color = 'r', linestyle = '-')
    sub2 = fig.add_subplot(332)
    sub2.title.set_text('Average Core Chooser Rewards')
    sub2.set_xlabel('episode')
    #sub2.axhline(y = roughAverageOfferReward, color = 'r', linestyle = '-')
    sub3 = fig.add_subplot(333)
    sub3.set_xlabel('episode')
    sub3.title.set_text('Average Agent Rewards')
    sub4 = fig.add_subplot(334)
    sub4.set_xlabel('episode')
    sub4.title.set_text('Average Prices')
    sub5 = fig.add_subplot(335)
    sub5.set_xlabel('episode')
    sub5.title.set_text('Average Auctioneer-Entity Reward')
    sub6 = fig.add_subplot(336)
    sub6.set_xlabel('episode')
    sub6.title.set_text('Average Dwell Time')
    sub7 = fig.add_subplot(337)
    sub7.set_xlabel('episode')
    sub7.title.set_text('Average Non-auctioneer Acception Quality')
    sub8 = fig.add_subplot(339)
    sub8.set_xlabel('episode')
    sub8.title.set_text('Average Non-auctioneer Acception Amount')
    x1 = [float(i) for i in list(range(len(acceptorRewards)))]
    x2 = [float(i) for i in list(range(len(offerRewards)))]
    x5 = [float(i) for i in list(range(len(auctioneerRewards)))]
    coef1 = np.polyfit(x1,acceptorRewards,1)
    coef2 = np.polyfit(x2,offerRewards,1)
    coef5 = np.polyfit(x5,auctioneerRewards,1)
    poly1d_1 = np.poly1d(coef1)
    poly1d_2 = np.poly1d(coef2)
    poly1d_5 = np.poly1d(coef5)
    sub1.plot(x1,acceptorRewards,'o',x1,poly1d_1(x1), '--k')
    #sub1.set_ylim(bottom=0 )
    sub2.plot(x2,offerRewards,'o',x2,poly1d_2(x2), '--k')
    #sub2.set_ylim(bottom=0)
    xAxis = np.arange(len(prices))
    for i in range(len(prices[0])):
        data = np.array([t[i] for t in prices]).astype(np.double)
        mask = np.isfinite(data)
        sub4.plot(xAxis[mask],data[mask], label="Kind: {}".format(i))
    sub4.legend(loc = "best")
    sub5.plot(x5,auctioneerRewards,'o',x5,poly1d_5(x5), '--k')
    for i in range(len(dwellTimes[0])):
        data = np.array([t[i] for t in dwellTimes])
        sub6.plot(data,'o',label="Kind: {} Prio: {}  Length: {}  Ratio: {}".format(i,params['possibleJobPriorities'][i],params['possibleJobLengths'][i],round(params['possibleJobPriorities'][i]/params['possibleJobLengths'][i],2)))
    sub6.legend(loc = "best")
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for i in range(len(agentRewards[0])):
        data = np.array([l[i] for l in agentRewards])
        sub3.plot(data,'o', color=next(colors), label = "Agent: {}".format(i + 1))
    sub3.legend(loc = 'best')
    xAxis1 = np.arange(len(acceptionAmounts))
    qualities1 = np.array(acceptionQualities).astype(np.double)
    #mask1 = np.isfinite(data)
    sub7.plot(xAxis1,qualities1,'o', color = 'k', label = 'acceptionQuality')
    sub7.axhline(y = 0, color = 'r', linestyle = '-')
    sub7.legend(loc = 'best')
    sub8.plot(acceptionAmounts, color = 'm', label = 'acceptionAmount')
    sub8.legend(loc = 'best')
    infotext = ''.join("{}: {}  ".format(k, str(v)) for (k,v) in params.items())
    fig_txt = tw.fill(tw.dedent(infotext.rstrip() ), width = 60)
    plt.figtext(0.35, 0.1, fig_txt, verticalalignment='bottom',fontsize=20)
    fig.subplots_adjust(bottom=0.5)
    fig.tight_layout()
    i = 0
    while os.path.exists(plotPath.format(i)):
        i += 1
    plt.savefig(plotPath.format(i), bbox_inches="tight",dpi=200)
    fig.clear()

def plotFreePricesResult(argsDict):
    plotPath = argsDict['plotPath']
    acceptorRewards = argsDict['acceptorRew']
    coreChooserRewards = argsDict['coreChooserRew']
    priceChooserRewards = argsDict['priceChooserRew']
    prices = argsDict['prices'] 
    auctioneerRewards = argsDict['auctioneerRew'] 
    dwellTimes = argsDict['dwellTimes']
    meanJobFraction = argsDict['meanJob']
    agentRewards = argsDict['agentRew']
    params = argsDict['params']
    # Zähler um eins erhöht für die Minues 1 JobPrio, Nenner um eins erhöht für den einen Zeitschritt mit Leerstand
    optimisticAverageAcceptorReward = (F(meanJobFraction.numerator,meanJobFraction.denominator + 1)) / params['numberOfAgents']   #numCores wurde oben und unten rausgekürzt
    roughAverageOfferReward = (F(meanJobFraction.numerator + 1,meanJobFraction.denominator + 1)*(params['numberOfCores'])) / (params['numberOfAgents'] * params['collectionLength'])
    fig = plt.figure(1,figsize=(20,15))
    sub1 = fig.add_subplot(331)
    sub1.title.set_text('Average Acceptor Rewards')
    sub1.set_xlabel('episode')
    sub1.set_ylabel('average reward')
    sub1.axhline(y = optimisticAverageAcceptorReward, color = 'r', linestyle = '-')
    sub2 = fig.add_subplot(332)
    sub2.title.set_text('Average Core Chooser Rewards')
    sub2.set_xlabel('episode')
    #sub2.axhline(y = roughAverageOfferReward, color = 'r', linestyle = '-')
    sub3 = fig.add_subplot(333)
    sub3.set_xlabel('episode')
    sub3.title.set_text('Average Price Chooser Rewards')
    #sub3.axhline(y = roughAverageOfferReward, color = 'r', linestyle = '-')
    sub4 = fig.add_subplot(334)
    sub4.set_xlabel('episode')
    sub4.title.set_text('Average Prices')
    sub5 = fig.add_subplot(335)
    sub5.set_xlabel('episode')
    sub5.title.set_text('Average Auctioneer-Entity Reward')
    sub6 = fig.add_subplot(336)
    sub6.set_xlabel('episode')
    sub6.title.set_text('Average Dwell Time')
    sub7 = fig.add_subplot(337)
    sub7.set_xlabel('episode')
    sub7.title.set_text('Average Agent Rewards')
    x1 = [float(i) for i in list(range(len(acceptorRewards)))]
    x2 = [float(i) for i in list(range(len(coreChooserRewards)))]
    x3 = [float(i) for i in list(range(len(priceChooserRewards)))]
    x5 = [float(i) for i in list(range(len(auctioneerRewards)))]
    coef1 = np.polyfit(x1,acceptorRewards,1)
    coef2 = np.polyfit(x2,coreChooserRewards,1)
    coef3 = np.polyfit(x2,priceChooserRewards,1)
    coef5 = np.polyfit(x5,auctioneerRewards,1)
    poly1d_1 = np.poly1d(coef1)
    poly1d_2 = np.poly1d(coef2)
    poly1d_3 = np.poly1d(coef3)
    poly1d_5 = np.poly1d(coef5)
    sub1.plot(x1,acceptorRewards,'o',x1,poly1d_1(x1), '--k')
    #sub1.set_ylim(bottom=0 )
    sub2.plot(x2,coreChooserRewards,'o',x2,poly1d_2(x2), '--k')
    #sub2.set_ylim(bottom=0)
    sub3.plot(x3,priceChooserRewards,'o',x3,poly1d_3(x3), '--k')
    #sub3.set_ylim(bottom=0)
    xAxis = np.arange(len(prices))
    for i in range(len(prices[0])):
        data = np.array([t[i] for t in prices]).astype(np.double)
        mask = np.isfinite(data)
        sub4.plot(xAxis[mask],data[mask], label="Kind: {}".format(i))
    sub4.legend(loc = "best")
    sub5.plot(x5,auctioneerRewards,'o',x5,poly1d_5(x5), '--k')
    for i in range(len(dwellTimes[0])):
        data = np.array([t[i] for t in dwellTimes])
        sub6.plot(data,'o',label="Kind: {} Prio: {}  Length: {}  Ratio: {}".format(i,params['possibleJobPriorities'][i],params['possibleJobLengths'][i],round(params['possibleJobPriorities'][i]/params['possibleJobLengths'][i],2)))
    sub6.legend(loc = "best")
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for i in range(len(agentRewards[0])):
        data = np.array([l[i] for l in agentRewards])
        sub7.plot(data,'o', color=next(colors), label = "Agent: {}".format(i + 1))
    sub7.legend(loc = 'best')
    infotext = ''.join("{}: {}  ".format(k, str(v)) for (k,v) in params.items())
    fig_txt = tw.fill(tw.dedent(infotext.rstrip() ), width = 100)
    plt.figtext(0.5, 0.2, fig_txt, verticalalignment='bottom',fontsize=12)
    fig.subplots_adjust(bottom=0.5)
    fig.tight_layout()
    i = 0
    while os.path.exists(plotPath.format(i)):
        i += 1
    plt.savefig(plotPath.format(i), bbox_inches="tight",dpi=200)
    fig.clear()

def plotAverageAndSDOfMultipleRuns(dataDicts,value,ylabel,variable=None,smooth=100):
    plotPath = dataDicts[0]['plotPath']
    acc = {}
    labels1 = []
    if smooth > 1:
        y = np.ones(smooth)
        for i, dataDict in enumerate(dataDicts):
            if value == "dwellTimes":
                #bezieht sich auf die hochpriorisierten Jobs
                rawData = [float(entry[1]) if entry[1] is not None else np.nan for entry in dataDict[value]]
            else:
                rawData = [float(entry) if entry is not None else np.nan for entry in dataDict[value]]
            if variable is not None:
                label = variable+' {}'.format(round(dataDict['params'][variable],2))
            else:
                label = i
            x = np.asarray(rawData)
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same')/np.convolve(z,y,'same')
            acc[label]=smoothed_x
    
    data = pd.DataFrame(acc)
    data["episode"]=data.index
    data_long = pd.melt(data,id_vars=['episode'],value_vars= data.columns[1:10])
    sns.set_style("darkgrid")
    sns.lineplot(data=data_long,x='episode',y='value',ci='sd')
    plt.ylim(bottom=0,top=1)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.tight_layout(pad=0.5)
    i = 0
    while os.path.exists(plotPath.format(i)):
        i += 1
    plt.savefig(plotPath.format(i), bbox_inches="tight",dpi=200)

def plotSeparateRuns(dataDicts,value,variable,smooth=100):
    fig, ax = plt.subplots(1,1)
    plotPath = dataDicts[0]['plotPath']
    acc = {}
    labels1 = []
    if smooth > 1:
        y = np.ones(smooth)
        for dataDict in dataDicts:
            highPJobs = [float(entry[1]) if entry[1] is not None else np.nan for entry in dataDict[value]]
            label = variable+' {}'.format(round(dataDict['params'][variable],2)) if variable != None else 'None'
            labels1.append(label)
            x = np.asarray(highPJobs)
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same')/np.convolve(z,y,'same')
            acc[label]=smoothed_x
    
    data = pd.DataFrame(acc)
    sns.set_style("darkgrid")
    sns.lineplot(data=data,ax=ax)
    plt.legend(bbox_to_anchor=(1.5, 0), loc='lower left', borderaxespad=0.,labels=labels1)
    infotext = ''.join("{}: {}  ".format(k, str(v)) for (k,v) in dataDicts[0]['params'].items())
    fig_txt = tw.fill(tw.dedent(infotext.rstrip() ), width = 60)
    plt.figtext(1.1, 1, fig_txt, verticalalignment='bottom',fontsize=12)

    i = 0
    while os.path.exists(plotPath.format(i)):
        i += 1
    os.makedirs(plotPath, exist_ok=True)
    plt.savefig(plotPath.format(i), bbox_inches="tight",dpi=200)


def plotTwoPrioritiesAverageAndSDOfMultipleRuns(dataDicts,value,smooth,bottom1,top1,printInfo,location):
    num_runs = len(dataDicts)
    fig, ax = plt.subplots(1,1)
    plotPath = dataDicts[0]['plotPath']
    acc1 = {}
    acc2 = {}
    labels1=['niedrige Priorität','_Test','hohe Priorität']
    if smooth > 1:
        y = np.ones(smooth)
        for i, dataDict in enumerate(dataDicts):
            highPJobs = [float(entry[1]) if entry[1] is not None else np.nan for entry in dataDict[value]]
            lowPJobs = [float(entry[0]) if entry[0] is not None else np.nan for entry in dataDict[value]]
            x = np.asarray(highPJobs)
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same')/np.convolve(z,y,'same')
            acc1[i]=smoothed_x
            x1 = np.asarray(lowPJobs)
            z1 = np.ones(len(x1))
            smoothed_x1 = np.convolve(x1,y,'same')/np.convolve(z1,y,'same')
            acc2[i]=smoothed_x1
    
    data = pd.DataFrame(acc1)
    data["episode"]=data.index
    data_long = pd.melt(data,id_vars=['episode'],value_vars= data.columns[1:num_runs])
    data_long.rename({'value':value},axis=1,inplace=True)
    data1 = pd.DataFrame(acc2)
    data1["episode"]=data1.index
    data_long1 = pd.melt(data1,id_vars=['episode'],value_vars= data1.columns[1:num_runs])
    data_long1.rename({'value':value},axis=1,inplace=True)
    sns.set_style("whitegrid")
    sns.lineplot(data=data_long1,x='episode',y=value,ci='sd',ax=ax)
    sns.set_style("whitegrid")
    sns.lineplot(data=data_long,x='episode',y=value,ci='sd',ax=ax)
    plt.ylim(bottom=bottom1,top=top1)
    plt.xlabel("Episode")
    plt.ylabel("normalisierte Verweilzeit")
    plt.legend(loc=location,labels=labels1).set_draggable(True)
    plt.tight_layout(pad=0.5)
    if printInfo:
        infotext = ''.join("{}: {}  ".format(k, str(v)) for (k,v) in dataDicts[0]['params'].items())
        infotext += 'Number Of Runs: {}'.format(num_runs)
        fig_txt = tw.fill(tw.dedent(infotext.rstrip() ), width = 60)
        plt.figtext(1.1, 0.5, fig_txt, verticalalignment='bottom',fontsize=12)
    i = 0
    while os.path.exists(plotPath.format(i)):
        i += 1
    plt.savefig(plotPath.format(i), bbox_inches="tight",dpi=200)


def plotSeparateRunsWithManualLabels(dataDicts,values,ylabel,manualLabels,bottom1,top1,smooth,location,printInfo):
    plotPath = dataDicts[0]['plotPath']
    colorlist = ['y', 'b', 'g', 'r', 'k', 'c', 'sienna','m']
    colors = cycle(colorlist)
    for value in values:
        acc = {}
        if smooth > 1:
            y = np.ones(smooth)
            for i,dataDict in enumerate(dataDicts):
                if value == "dwellTimes":
                        #bezieht sich auf die hochpriorisierten Jobs
                    rawData = [float(entry[1]) if entry[1] is not None else np.nan for entry in dataDict[value]]
                else:
                    rawData = [float(entry) if entry is not None else np.nan for entry in dataDict[value]]
                x = np.asarray(rawData)
                z = np.ones(len(x))
                smoothed_x = np.convolve(x,y,'same')/np.convolve(z,y,'same')
                label=value
                acc[label]=smoothed_x
    
        data = pd.DataFrame(acc)
        sns.set_style("darkgrid")
        sns.lineplot(data=data,palette=[next(colors)])

    plt.ylim(bottom=bottom1,top=top1)
    plt.xlabel("Episode")
    plt.yticks(range(0,8))
    leg = plt.legend(loc=location,labels=manualLabels,ncol=3)
    hl_dict = {manualLabels[i]: handle for i, handle in enumerate(leg.legendHandles)}
    for i, value in enumerate(values):
        hl_dict[manualLabels[i]].set_color(colorlist[i])
    plt.ylabel(ylabel)
    if printInfo:
        infotext = ''.join("{}: {}  ".format(k, str(v)) for (k,v) in dataDicts[0]['params'].items())
        fig_txt = tw.fill(tw.dedent(infotext.rstrip() ), width = 60)
        plt.figtext(1.1, 0.5, fig_txt, verticalalignment='bottom',fontsize=12)
    i = 0
    while os.path.exists(plotPath.format(i)):
        i += 1
    plt.savefig(plotPath.format(i), bbox_inches="tight",dpi=200)

def plotSeveralMetricsOfMultipleRuns(dataDicts,values,ylabel,manualLabels,bottom1,top1,smooth,printInfo,location):
    plotPath = dataDicts[0]['plotPath']
    colors = cycle(['b', 'g', 'b', 'r', 'g', 'y', 'c'])
    for value in values:
        acc = {}
        if smooth > 1:
            y = np.ones(smooth)
            for i, dataDict in enumerate(dataDicts):
                if value == "dwellTimes":
                    #bezieht sich auf die hochpriorisierten Jobs
                    rawData = [float(entry[1]) if entry[1] is not None else np.nan for entry in dataDict[value]]
                else:
                    rawData = [float(entry) if entry is not None else np.nan for entry in dataDict[value]]
                
                label = i
                x = np.asarray(rawData)
                z = np.ones(len(x))
                smoothed_x = np.convolve(x,y,'same')/np.convolve(z,y,'same')
                acc[label]=smoothed_x
        
        data = pd.DataFrame(acc)
        data["episode"]=data.index
        data_long = pd.melt(data,id_vars=['episode'],value_vars= data.columns[1:len(dataDicts)])
        sns.set_style("darkgrid")
        sns.lineplot(data=data_long,x='episode',y='value',ci='sd',color=next(colors))
    plt.ylim(bottom=bottom1,top=top1)
    plt.xlabel("Episode")
    plt.legend(loc=location,labels=manualLabels).set_draggable(True)
    plt.ylabel(ylabel)
    if printInfo:
        fig_txt = 'smooth: {}'.format(smooth)
        plt.figtext(1.01, 0.2, fig_txt, verticalalignment='bottom',fontsize=12)
    i = 0
    while os.path.exists(plotPath.format(i)):
        i += 1
    plt.savefig(plotPath.format(i), bbox_inches="tight",dpi=200)

def plotOneMetricOfMultipleRunsOfMultipleFolders(listOfArgsDicts,value,ylabel,manualLabels,bottom1,top1,smooth,printInfo,location,jobKind):
    plotPath = listOfArgsDicts[0][0]['plotPath']
    colors = cycle(['g','navy', 'c', 'b', 'k', 'c'])
    for dataDicts in listOfArgsDicts:
        acc = {}
        if smooth > 1:
            y = np.ones(smooth)
            for i, dataDict in enumerate(dataDicts):
                if value == "dwellTimes":
                    #bezieht sich auf die hochpriorisierten Jobs
                    rawData = [float(entry[jobKind]) if entry[jobKind] is not None else np.nan for entry in dataDict[value]]
                if value == "agentRew":
                    rawData = [float(statistics.mean(entry)) if entry is not None else np.nan for entry in dataDict[value]]
                if (value != "agentRew") & (value != "dwellTimes"):
                    rawData = [float(entry) if entry is not None else np.nan for entry in dataDict[value]]
                
                label = i
                x = np.asarray(rawData)
                z = np.ones(len(x))
                smoothed_x = np.convolve(x,y,'same')/np.convolve(z,y,'same')
                acc[label]=smoothed_x
        
        data = pd.DataFrame(acc)
        data["episode"]=data.index
        data_long = pd.melt(data,id_vars=['episode'],value_vars= data.columns[1:len(dataDicts)])
        sns.set_style("darkgrid")
        sns.lineplot(data=data_long,x='episode',y='value',ci='sd',color=next(colors))
    plt.ylim(bottom=bottom1,top=top1)
    plt.xlabel("Episode")
    plt.legend(loc=location,labels=manualLabels)
    plt.ylabel(ylabel)
    if printInfo:
        fig_txt = 'smooth: {}'.format(smooth)
        plt.figtext(1.01, 0.2, fig_txt, verticalalignment='bottom',fontsize=12)
    i = 0
    while os.path.exists(plotPath.format(i)):
        i += 1
    plt.savefig(plotPath.format(i), bbox_inches="tight",dpi=200)

def plotSeveralJobMetricsOfMultipleRunsOfMultipleFolders(listOfArgsDicts,value,ylabel,manualLabels,bottom1,top1,smooth,printInfo,location,jobKinds,nCol1):
    plotPath = listOfArgsDicts[0][0]['plotPath']
    colors = cycle(['r','m','g', 'm', 'b', 'c', 'y', 'k', 'c'])   #navy,gold
    for jobKind in jobKinds:
        for dataDicts in listOfArgsDicts:
            acc = {}
            if smooth > 1:
                y = np.ones(smooth)
                for i, dataDict in enumerate(dataDicts):
                    if value == "dwellTimes":
                        #bezieht sich auf die hochpriorisierten Jobs
                        rawData = [float(entry[jobKind]) if entry[jobKind] is not None else np.nan for entry in dataDict[value]]
                    if value == "prices":
                        rawData = [float(entry[jobKind]) if entry[jobKind] is not None else np.nan for entry in dataDict[value]]
                    
                    label = i
                    x = np.asarray(rawData)
                    z = np.ones(len(x))
                    smoothed_x = np.convolve(x,y,'same')/np.convolve(z,y,'same')
                    acc[label]=smoothed_x
            
            data = pd.DataFrame(acc)
            data["episode"]=data.index
            data_long = pd.melt(data,id_vars=['episode'],value_vars = data.columns[1:len(dataDicts)])
            sns.set_style("darkgrid")
            sns.lineplot(data=data_long,x='episode',y='value',ci='sd',color=next(colors))
    plt.ylim(bottom=bottom1,top=top1)
    plt.yticks(range(bottom1,top1+1))  #plt.yticks(list(chain.from_iterable((i, i+0.5) for i in range(bottom1, top1))))
    plt.xlabel("Episode", weight = 'bold', fontsize=13)
    plt.legend(loc=location,labels=manualLabels,ncol=nCol1)
    plt.ylabel(ylabel,weight='bold', fontsize=13)
    if printInfo:
        fig_txt = 'smooth: {}'.format(smooth)
        plt.figtext(1.01, 0.2, fig_txt, verticalalignment='bottom',fontsize=12)
    i = 0
    while os.path.exists(plotPath.format(i)):
        i += 1
    plt.savefig(plotPath.format(i), bbox_inches="tight",dpi=200)