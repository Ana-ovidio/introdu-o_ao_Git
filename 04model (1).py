import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.neighbors import KernelDensity
import math

print(math.gamma(4.5))

# Year to analyse
year = 2008
f = 'Russell3000-99-2010.csv'
# Read external file
file = read_csv(f)


# Select index from data column 
file = file.set_index('Unnamed: 0')

# Number of stocks
nStocks = len(file.columns)

print("nStocks = ",nStocks)

# Select data in a given year to be analysed
yfile = file.loc[str(year)+'-01-01':str(year)+'-12-31']

# Creates a list with information of dy=y_{i}-y_{i-1}
dif = []
for i in range (len(yfile)-1):
    dif.append(yfile.iloc[i+1].subtract(yfile.iloc[i]))

# Number of days
falseDays_=len(dif)

# Quantity of positive daily returns 
countDaily = []
for i in range(falseDays_): # filtra false days
    count = 0
    for j in range(nStocks):
        if dif[i][j] > 0.:      # Gol de mão pra incluir os zeros
            count += 1.
    if count!=0.:
        countDaily.append(int(count))
        
trueDays_=len(countDaily)
npCountDaily = np.zeros((trueDays_, 1))
for i in range(trueDays_):
    npCountDaily[i, 0] = countDaily[i] 
#In this case, there was 251 days with positive returns, and there was a quantity of these returns each day 

# Determines the distribution of frequencies for each percentual of ups
fkN = np.zeros((nStocks, 1)) #Frequency of positive returns   
idx = np.zeros((nStocks, 1))
teste = []
for i in range(nStocks):
    #countDaily.count(i) retorna a quantidade de vezes que o valor associado a i aparece na lista 
    #O valor de i será maior ou igual ao número de ações 
    fkN[i, 0] = countDaily.count(i)/trueDays_
    idx[i, 0] = i
# Graph distribution of frequencies
kN=idx/nStocks
fkN1 = fkN * 100
plt.plot(kN,fkN1,color='blue')#Plot of percentual of ups (+) verus the percentual of positive stonks 
plt.ylabel('Percentual of ups (%)')
plt.xlabel('k/N')
plt.grid()
plt.show()

print(countDaily)

print("sum fkN = ",sum(fkN))

# Kernel density estimates
bw=100 # Bandwidth
kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(npCountDaily)
log_dens = kde.score_samples(idx)
# Graph kernel estimate
dens=np.exp(log_dens)
plt.grid(linestyle='--')
plt.plot(idx/nStocks,nStocks*dens,'r',label='Kernel density')

#print(sum(dens))
plt.title("Kenel density")
plt.show()

#------------------------------------------------------------------------------
# Estimated f x kN

x=kN
y=fkN
#x=idx/nStocks
#y=dens

mean=0.
for i in range(len(kN)):
    mean+=x[i,0]*y[i]

#mean=0.5 # Media setada

x_=kN-mean
var=0.0
for i in range(len(x_)):
    var+=x_[i]*x_[i]*fkN[i]
std=var**0.5
print("Mean = %s, Std = %s"%(mean,std))
"""
gauss=[]
#calculate of gaussian distribuition based in the average and the standard deviation of k/N and fkN frequencies. 
for i in range(len(kN)):
    gauss.append(0.05*math.exp(-0.5*((kN[i]-mean)/std)**2)/(std*math.sqrt(2*math.pi)))
    

plt.plot(kN,gauss,label='Gaussian fitted solution')
#plt.legend()
plt.title('Gaussian fitted solution')
plt.grid('--')
plt.show()
"""
# Estimated U and D
m=mean
s=std

e=m
a=(m*(1-m)-s*s)/(s*s-m*(1-m)/nStocks)

U_est=a*e
D_est=a*(1-e)

print("U_est = %s, D_est = %s"%(U_est,D_est))
#--------------------------------------------------------------------------

    



