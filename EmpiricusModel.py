from pandas import read_csv

import numpy as np

from sklearn.neighbors import KernelDensity

import scipy.special

import matplotlib.pyplot as plt

import datetime

import time

from bisect import bisect


##############################################################################
#FUNCTIONS
##############################################################################

#-----------------------------------------------------------------------------
#Function: file_reader
#Description: reads the csv file with given name (filename)
#Output: stocks closing price dataframe (from csv to df)
#-----------------------------------------------------------------------------
def file_reader (filename):
    
    file = read_csv(filename)
    
    file = file.set_index('Unnamed: 0')
    
    numberOfStocks = len(file.columns)
    
    return file, numberOfStocks


#-----------------------------------------------------------------------------
#Function: time_interval_data
#Description: locates the time interval of interest in dataframe and counts the
#number of trading days (invalid and valid days)
#Output: slice of dataframe and number of trading days
#-----------------------------------------------------------------------------
def time_interval_data (beginDate, endDate, file):
    
    intervalClosingPrices = file.loc[beginDate:endDate]
    
    totalTradingDays = len(intervalClosingPrices)
    
    return intervalClosingPrices, totalTradingDays


#-----------------------------------------------------------------------------
#Function: closing_price_variation
#Description: calculates the daily closing price variation
#Output: list of daily closing price variation list (each day has a list of all
#stocks closing price variation)
#-----------------------------------------------------------------------------
def closing_price_variation (intervalClosingPrices, totalTradingDays):
    
    closingPriceVariation = []
    
    for i in range (totalTradingDays-1):
        
        previousClosingPrice = intervalClosingPrices.iloc[i]
        
        nextClosingPrice = intervalClosingPrices.iloc[i+1]
                
        closingPriceVariationVariable = nextClosingPrice.subtract(previousClosingPrice)
        
        closingPriceVariation.append(closingPriceVariationVariable.values.tolist())

    return closingPriceVariation


#-----------------------------------------------------------------------------
#Function: positive_days_counter_n_true_days
#Description: counts the number of daily positive stock price variations and 
#the number of valid positive days
#Output: list and array of positive daily variation counts, and number of valid
#positive trading days
#-----------------------------------------------------------------------------

def positive_days_counter_n_true_days (totalTradingDays, numberOfStocks, closingPriceVariation):
    
    roughTradingDaysDifference = totalTradingDays - 1 # Same length as variation list
    
    countPositiveVariationDailyList = []

    for i in range(roughTradingDaysDifference):
        
        closingPriceVariation[i].sort()
        
        count = len(closingPriceVariation[i]) - bisect(closingPriceVariation[i], 0)

        if count!=0.:
            
            countPositiveVariationDailyList.append(int(count))

    validTradingDays = len(countPositiveVariationDailyList)
    
    countPositiveVariation = np.zeros((validTradingDays, 1))
    
    for i in range (validTradingDays):
        
        countPositiveVariation[i, 0] = countPositiveVariationDailyList[i]
    
    return countPositiveVariationDailyList, countPositiveVariation, validTradingDays


#-----------------------------------------------------------------------------
#Function: frequency_distribution_positive
#Description: counts the number positive stocks fraction repetition through the
#period of analysis (anually in the paper)
#Output: list of frequency of positive fraction repetition and list of fractions
#of positive stocks
#-----------------------------------------------------------------------------
def frequency_distribution_positive (numberOfStocks, countPositiveVariationDailyList, validTradingDays):
    
    frequencyOfPositiveFraction = np.zeros((numberOfStocks, 1))  
    
    positiveFractionInteger = np.zeros((numberOfStocks, 1))
    
    for i in range(numberOfStocks):
        
        frequencyOfPositiveFraction[i, 0] = countPositiveVariationDailyList.count(i) / validTradingDays
        
        positiveFractionInteger[i, 0] = i
        
    positiveFraction = positiveFractionInteger / numberOfStocks 

    return frequencyOfPositiveFraction, positiveFraction

def kernel_density_estimation (countPositiveVariation, positiveFraction, numberOfStocks):

    bandWidth = 100
    
    Kernel = 'gaussian'
    
    KDE = KernelDensity(kernel=Kernel, bandwidth=bandWidth).fit(countPositiveVariation)
    
    logDensities = KDE.score_samples(positiveFraction * numberOfStocks)
    
    densities = np.exp(logDensities)
    
    experimentalFrequencyOfPositive = densities * numberOfStocks

    return experimentalFrequencyOfPositive

def experimental_data_mean_std (positiveFraction, frequencyOfPositiveFraction, fixedMeanChoice = 'y'):
    
    if fixedMeanChoice == 'y':
        
        experimentalMean = 0.5
        
    else:
    
        experimentalMean=0.
    
        for i in range(len(positiveFraction)):
        
            experimentalMean += positiveFraction[i,0] * frequencyOfPositiveFraction[i]
    
    differencePositiveFraction = positiveFraction - experimentalMean
    
    variation = 0.    
    
    for i in range(len(positiveFraction)):

        variation += (differencePositiveFraction[i] ** 2) * frequencyOfPositiveFraction[i]
    
    experimentalStd = variation ** 0.5
    
    return experimentalMean, experimentalStd

def U_D_estimation (experimentalMean, experimentalStd, numberOfStocks):
    
    xi = experimentalMean
    
    a = (xi * (1 - xi) - experimentalStd**2) / (experimentalStd**2 - xi * (1- xi) / numberOfStocks)
    
    Uest = a * xi
    
    Dest = a * (1 - xi)
    
    return Uest, Dest


def binomial_model (Uest, Dest, positiveFraction, numberOfStocks):
    
    modelAnswer = []
    
    for i in range (len(positiveFraction)):
    
        N1 = scipy.special.binom(Uest + positiveFraction[i,0]*numberOfStocks - 1, positiveFraction[i,0]*numberOfStocks)
        
        N2 = scipy.special.comb(Dest + numberOfStocks - positiveFraction[i,0]*numberOfStocks - 1, numberOfStocks - positiveFraction[i,0]*numberOfStocks)
        
        D = scipy.special.binom(Uest + Dest + numberOfStocks - 1, numberOfStocks)
        
        modelAnswer.append(((N1 * N2) / D) * numberOfStocks)
        
    return modelAnswer


def interval_static_U_plotter (positiveFraction, experimentalFrequencyOfPositive, modelAnswer, Uest, Dest):
    
    plt.plot(positiveFraction, experimentalFrequencyOfPositive,'r',label='Data')
    
    plt.plot(positiveFraction, modelAnswer, '--b', label = 'Model')
    
    plt.xlabel(r'$k/N$')
    
    plt.ylabel(r'$f$')
    
    plt.grid('on')
    
    plt.legend()
    
    if Uest == Dest:
        
        plt.title('U = D = %.2f'%Uest)
        
    else:
        
        plt.title('U = %.2f | D = %.2f'%(Uest, Dest))
    
    plt.show()
    
    return

    
def static_U_caller (beginDate, endDate, file, numberOfStocks):

    intervalClosingPrices, totalTradingDays = time_interval_data (beginDate, endDate, file)
    
    closingPriceVariation = closing_price_variation (intervalClosingPrices, totalTradingDays)
            
    countPositiveVariationDailyList, countPositiveVariation, validTradingDays = positive_days_counter_n_true_days (totalTradingDays, numberOfStocks, closingPriceVariation)
    
    frequencyOfPositiveFraction, positiveFraction = frequency_distribution_positive (numberOfStocks, countPositiveVariationDailyList, validTradingDays)
    
    experimentalFrequencyOfPositive = kernel_density_estimation (countPositiveVariation, positiveFraction, numberOfStocks)
    
    experimentalMean, experimentalStd = experimental_data_mean_std (positiveFraction, frequencyOfPositiveFraction, fixedMeanChoice = 'y')
    
    Uest, Dest = U_D_estimation (experimentalMean, experimentalStd, numberOfStocks)
    
    modelAnswer = binomial_model (Uest, Dest, positiveFraction, numberOfStocks)
    
    return Uest
    


#-----------------------------------------------------------------------------
#Function: Estimation of U according to day's interations.
#Description: For a day between beginDate and endDate, U is calculates using 
#informations from of previous 12 months. 
#Output: The U's values for all date analysed.
#-----------------------------------------------------------------------------

def dynamic_U_caller (beginDate, endDate, file, numberOfStocks):
    
    dynamicBeginDate = str(int(beginDate[0:4]) - 1) + beginDate[4:]
    dynamicEndDate = beginDate
    
    UList = []
    
    dayList = []
    
    while dynamicEndDate != endDate:
        
        tic = time.time()
        
        Uest = static_U_caller(dynamicBeginDate, dynamicEndDate, file, numberOfStocks)
        
        #(f'{beginDate}: {Uest[0]:.2f}')
        
        UList.append(Uest)

        today = datetime.datetime.strptime(beginDate, '%Y-%m-%d')
        
        dayList.append(today)
        
        tomorrow = today + datetime.timedelta(days = 1)
        
        beginDate = tomorrow.strftime('%Y-%m-%d')
        
        dynamicBeginDate = str(int(beginDate[0:4]) - 1) + beginDate[4:]
        dynamicEndDate = beginDate
        
        toc = time.time()
        
    print(f'Running time / day: {toc-tic}')

    return UList, dayList
    

###############################################################################
#USER LOG
###############################################################################

filename = 'Russell3000-99-2010.csv'

beginDate = '2000-01-01'

endDate = '2001-12-31'

file, numberOfStocks = file_reader (filename)

UList, dayList = dynamic_U_caller (beginDate, endDate, file, numberOfStocks)

limit = [1 for i in range (len(dayList))]

plt.plot(dayList, UList)
plt.plot(dayList, limit)
plt.ylabel('U')
plt.ylim(0,max(UList)+1)
plt.grid('--')
plt.show()

