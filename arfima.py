# coding: utf-8

# In[4]:

from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import xlrd
from scipy.special import gamma
#计算hurst指数函数
from __future__ import division
from collections import Iterable

import numpy as np 
from pandas import Series
from scipy.special import gamma

datasheet=[]
data=xlrd.open_workbook(filename='19_data.xlsx')
datasheet=data.sheet_by_name('Sheet1')
data_1=[]
for r in range(datasheet.nrows):
    data_1.append(datasheet.cell(r,0).value)
c1=np.array(data_1,dtype=np.float)

data_2=[]
for r in range(datasheet.nrows):
    data_2.append(datasheet.cell(r,1).value)
c2=np.array(data_2,dtype=np.float)

data_3=[]
for r in range(datasheet.nrows):
    data_3.append(datasheet.cell(r,2).value)
c3=np.array(data_3,dtype=np.float)

data_4=[]
for r in range(datasheet.nrows):
    data_4.append(datasheet.cell(r,3).value)
c4=np.array(data_4,dtype=np.float)

data_5=[]
for r in range(datasheet.nrows):
    data_5.append(datasheet.cell(r,4).value)
c5=np.array(data_5,dtype=np.float)

data_6=[]
for r in range(datasheet.nrows):
    data_6.append(datasheet.cell(r,5).value)
c6=np.array(data_6,dtype=np.float)

data_7=[]
for r in range(datasheet.nrows):
    data_7.append(datasheet.cell(r,6).value)
c7=np.array(data_7,dtype=np.float)

data_8=[]
for r in range(datasheet.nrows):
    data_8.append(datasheet.cell(r,7).value)
c8=np.array(data_8,dtype=np.float)

data_9=[]
for r in range(datasheet.nrows):
    data_9.append(datasheet.cell(r,8).value)
c9=np.array(data_9,dtype=np.float)

data_10=[]
for r in range(datasheet.nrows):
    data_10.append(datasheet.cell(r,9).value)
c10=np.array(data_10,dtype=np.float)

data_11=[]
for r in range(datasheet.nrows):
    data_11.append(datasheet.cell(r,10).value)
c11=np.array(data_11,dtype=np.float)

data_12=[]
for r in range(datasheet.nrows):
    data_12.append(datasheet.cell(r,11).value)
c12=np.array(data_12,dtype=np.float)

data_13=[]
for r in range(datasheet.nrows):
    data_13.append(datasheet.cell(r,12).value)
c13=np.array(data_13,dtype=np.float)

data_14=[]
for r in range(datasheet.nrows):
    data_14.append(datasheet.cell(r,13).value)
c14=np.array(data_14,dtype=np.float)

data_15=[]
for r in range(datasheet.nrows):
    data_15.append(datasheet.cell(r,14).value)
c15=np.array(data_15,dtype=np.float)

data_16=[]
for r in range(datasheet.nrows):
    data_16.append(datasheet.cell(r,15).value)
c16=np.array(data_16,dtype=np.float)

data_17=[]
for r in range(datasheet.nrows):
    data_17.append(datasheet.cell(r,16).value)
c17=np.array(data_17,dtype=np.float)

data_18=[]
for r in range(datasheet.nrows):
    data_18.append(datasheet.cell(r,17).value)
c18=np.array(data_18,dtype=np.float)

data_19=[]
for r in range(datasheet.nrows):
    data_19.append(datasheet.cell(r,18).value)
c19=np.array(data_19,dtype=np.float)

#取前46个数据为训练样本
k=46 #datasize
c1_=c1[0:k]
c2_=c2[0:k]
c3_=c3[0:k]
c4_=c4[0:k]
c5_=c5[0:k]
c6_=c6[0:k]
c7_=c7[0:k]
c8_=c8[0:k]
c9_=c9[0:k]
c10_=c10[0:k]
c11_=c11[0:k]
c12_=c12[0:k]
c13_=c13[0:k]
c14_=c14[0:k]
c15_=c15[0:k]
c16_=c16[0:k]
c17_=c17[0:k]
c18_=c18[0:k]
c19_=c19[0:k]

class CalculateValues():
    def calcHurst2(self, ts):
       
        if not isinstance(ts, Iterable):
            print ('error')
            return

        n_min, n_max = 2, len(ts)//3
        RSlist = []
        for cut in range(n_min, n_max):
            children = len(ts) // cut
            children_list = [ts[i*children:(i+1)*children] for i in range(cut)]
            L = []
            for a_children in children_list:
                Ma = np.mean(a_children)
                Xta = Series(map(lambda x: x-Ma, a_children)).cumsum()
                Ra = max(Xta) - min(Xta)
                Sa = np.std(a_children)
                rs = Ra / Sa
                L.append(rs)
            RS = np.mean(L)
            RSlist.append(RS)
        return np.polyfit(np.log(range(2+len(RSlist),2,-1)), np.log(RSlist), 1)[0]

    def calculate(self):
        h1=self.calcHurst2(c1_)
        h2=self.calcHurst2(c2_)
        h3=self.calcHurst2(c3_)
        h4=self.calcHurst2(c4_)
        h7=self.calcHurst2(c7_)
        h8=self.calcHurst2(c8_)
        h12=self.calcHurst2(c12_)
        h13=self.calcHurst2(c13_)
        h14=self.calcHurst2(c14_)
        h16=self.calcHurst2(c16_)
        h2=1
        h3=1
        h7=1
        h8=1
        h12=1
        h16=1
        print(self.calcHurst2(c1_))
        print(self.calcHurst2(c2_))
        print(self.calcHurst2(c3_))
        print(self.calcHurst2(c4_))
        print(self.calcHurst2(c7_))
        print(self.calcHurst2(c8_))
        print(self.calcHurst2(c12_))
        print(self.calcHurst2(c13_))
        print(self.calcHurst2(c14_))
        print(self.calcHurst2(c16_))
        print("...")
        print(h1)
        print(h2)
        print(h3)
        print(h4)
        print(h7)
        print(h8)
        print(h12)
        print(h13)
        print(h14)
        print(h16)

        n=46
        g=[0]*n

        #选择数据对象!!
        h_value=h16
        #计算系数矩阵元素
        d=h_value-0.5
        for i in range(n):
            g[i]=(gamma(i-d))/(gamma(i+1)*gamma(-d))
        print(g)
        #建立系数矩阵
        coefficient_matrix=[[0 for i in range(n)] for i in range(n)]
        for  i in range(n):
            for j in range(n-i):
                coefficient_matrix[i][j+i]=g[i]
                
        coefficient_matrix


        # In[3]:

        type(coefficient_matrix)


        # In[6]:


        y=[0]*n

        for i in range(n):
            for j in range(n):
                y[i]=y[i]+c16_[j]*coefficient_matrix[j][i]
        print(y)

        import statsmodels.api as sm
        fig = plt.figure(figsize=(15,8))
        ax1=fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(y,lags=44,ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(y,lags=44,ax=ax2)
        plt.show()

        print(sm.tsa.adfuller(y))


        # In[7]:

        #拟合arfima模型
        #选取差分后的序列进行模型参数估计
        y_for_test=y
        series_for_test=pd.Series(y_for_test)
        series_for_test.index=pd.Index(sm.tsa.datetools.dates_from_range('2000','2045'))
        arma_mod_a= sm.tsa.ARMA(series_for_test,(1,0)).fit(disp=False)
        arma_mod_b= sm.tsa.ARMA(series_for_test,(0,1)).fit(disp=False)
        arma_mod_c= sm.tsa.ARMA(series_for_test,(0,2)).fit(disp=False)
        arma_mod_d= sm.tsa.ARMA(series_for_test,(1,0)).fit(disp=False)
        #print(arma_mod_c1.params)
        print(arma_mod_a.aic,arma_mod_a.bic)
        print(arma_mod_b.aic,arma_mod_b.bic)
        print(arma_mod_c.aic,arma_mod_c.bic)
        print(arma_mod_d.aic,arma_mod_d.bic)
        print(d)
        type(arma_mod_a)


        # In[16]:

        #根据aic和bic信息准侧，选择用于预测的模型参数，预测结果
        selected_arma=arma_mod_b#需要选择！！
        predict_outcome=selected_arma.predict()
        forecast_outcome=selected_arma.forecast(6)
        print(predict_outcome)
        print(forecast_outcome)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax = series_for_test.ix['2004':].plot(ax=ax)
        fig= selected_arma.plot_predict('2046', '2052', dynamic=True, ax=ax, plot_insample=False)
        plt.show()


        # In[90]:

        import statsmodels.tsa.stattools as ts
        ts.adfuller(y[4:])


        # In[92]:

        #计算真值

        from scipy.special import gamma
        n=52
        g=[0]*n

        #选择数据对象!!
        h_value=h1
        #计算系数矩阵元素
        d=h_value-0.5
        for i in range(n):
            g[i]=gamma(i-d)/(gamma(i+1)*gamma(-d))
        print(g)
        #建立系数矩阵
        coefficient_matrix=[[0 for i in range(n)] for i in range(n)]
        for  i in range(n):
            for j in range(n-i):
                coefficient_matrix[i][j+i]=g[i]
                
        y=[0]*n

        for i in range(n):
            for j in range(n):
                y[i]=y[i]+c1[j]*coefficient_matrix[j][i]#需要设置用于乘以系数矩阵的序列
        print(y)
        import statsmodels.api as sm
        fig = plt.figure(figsize=(15,8))
        ax1=fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(y[4:],lags=41,ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(y[4:],lags=41,ax=ax2)
        plt.show()

        for i in range(46,52):
            print(y[i])


        # In[18]:

        for i in range(0,52):
            print(y[i])


        # In[19]:

        y


        # In[ ]:

        import arfima#class
        arfima.fit()