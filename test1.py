"""
2018-10-18
Mini project
"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import sklearn.linear_model as sl

DJIA = pd.read_csv('^DJI.csv')

# Not a professional plot
#plt.plot(DJIA.Close)
#plt.show()

# More profesional plot but can be even more professional
SP500 = pd.read_csv('^GSPC.csv')
#SP500.index = pd.to_datetime(SP500.Date)
#plt.plot(SP500.Close)
#plt.show()
n1=len(DJIA)-1
DJIA['Log_Returns']=np.log(DJIA['Adj Close'])-np.log(DJIA['Adj Close']).shift(periods=1)
Sample_Mean_DJIA=np.sum(DJIA['Log_Returns'])/n1
#print(Sample_Mean)
VarDJIA=np.var(DJIA['Log_Returns'])
A_Return_DJIA=252*Sample_Mean_DJIA*100
A_Volatility_DJIA=np.sqrt(252*VarDJIA)*100

n2=len(SP500)-1
SP500['Log_Returns']=np.log(SP500['Adj Close'])-np.log(SP500['Adj Close']).shift(periods=1)
Sample_Mean_SP500=np.sum(SP500['Log_Returns'])/n2
#print(Sample_Mean1)
#print(SP500)
VarSP500=np.var(SP500['Log_Returns'])
A_Return_SP500=252*Sample_Mean_SP500*100
A_Volatility_SP500=np.sqrt(252*VarSP500)*100

Std_DJIA=np.std(DJIA['Log_Returns'])
Skew_DJIA=(np.sum((DJIA['Log_Returns']-Sample_Mean_DJIA)**3))/((len(DJIA)-1)*(Std_DJIA)**3)
print(Skew_DJIA)

Std_SP500=np.std(SP500['Log_Returns'])
Skew_SP500=(np.sum((SP500['Log_Returns']-Sample_Mean_SP500)**3))/((len(SP500)-1)*(Std_SP500)**3)
print(Skew_SP500)

Kurt_DJIA=(np.sum((DJIA['Log_Returns']-Sample_Mean_DJIA)**4))/((len(DJIA)-1)*(Std_DJIA)**4)
print(Kurt_DJIA)

Kurt_SP500=(np.sum((SP500['Log_Returns']-Sample_Mean_SP500)**4))/((len(SP500)-1)*(Std_SP500)**4)
print(Kurt_SP500)
def JB(T,Skew,Kurt):
    return T*((Skew**2/6)+((Kurt-3)**2/24))

JB_DJIA=JB(n1,Skew_DJIA,Kurt_DJIA)
JB_SP500=JB(n2,Skew_SP500,Kurt_SP500)
print(JB_DJIA)
print(JB_SP500)



#correlation
Correlation = np.corrcoef(DJIA['Log_Returns'].dropna(),SP500['Log_Returns'].dropna())
print(Correlation)
v=(VarDJIA/n1+VarSP500/n2)**2/(((VarDJIA/n1)**2/(n1-1))+((VarSP500/n1)**2/(n2-1)))
T_Statistics = (Sample_Mean_DJIA-Sample_Mean_SP500)/np.sqrt((VarDJIA/n1)+(VarSP500/n2))
print(T_Statistics)
Critical_value = 1.960
def T_Test():
    if np.absolute(T_Statistics)>Critical_value:
        print("Reject null hypothesis that mean1=mean2")
    else:
        print("Do not reject null hypothesis that mean1=mean2")
T_Test()
print(v)

F_Statistics=VarDJIA/VarSP500
print(F_Statistics)
F_Critical_Value=1
def F_Test():
    if F_Statistics < F_Critical_Value or F_Statistics > F_Critical_Value:
        print('Reject null hypothersis that sigma1=sigma2')
    else:
        print('do not reject null hypothersis that sigma1=sigma2')

F_Test()

#Task 3
#DJIA['Log_Returns']=DJIA['Log_Returns'].dropna
y=DJIA.loc[1: ,'Log_Returns'].values.reshape(-1,1)
x=SP500.loc[1: ,'Log_Returns'].values.reshape(-1,1)
#print(y)
#print(x)


#SP500['Log_Returns']=SP500['Log_Returns'].dropna
#=SP500['Log_Returns'].dropna
#print(DJIA['Log_Returns'])
regr = sl.LinearRegression().fit(x,y)
print(regr.coef_)
print(regr.intercept_)