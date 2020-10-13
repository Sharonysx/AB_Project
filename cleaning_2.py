# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:17:43 2020

@author: 38030
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from scipy import stats
from scipy.stats.mstats import winsorize
import gc
import os
os.chdir(r'C:\Users\38030\OneDrive\CORNELL\2020F\AB')

#%%
sp500_ab=pd.read_csv('sp500_ab.csv')
sp500_ab['yearmonth']=sp500_ab['date']//100
price_financial=pd.read_csv('price+financial.csv')
price_raw=pd.read_csv('sp500_cons_prc_new.csv')
error=['0051B','1974B','3CCIKO','4741B','4764B','5235B','6525B','6712B','9297B','1717B','2091B','5050B']
price_raw=price_raw[~price_raw.tic.isin(error)]
price_cons=pd.read_csv('sp500_cons_part_prc.csv')
error=['0051B','1974B','3CCIKO','4741B','4764B','5235B','6525B','6712B','9297B','1717B','2091B','5050B']
price_cons=price_cons[~price_cons.tic.isin(error)]

#%% cap return calculation
price_cons['yearmonth']=price_cons['datadate']//100
price_cons['cap']=price_cons['prccm']*price_cons['cshom']
price_cons.sort_values(by=['gvkey','datadate'],inplace=True)
price_cons['cap_t1']=price_cons.groupby(['gvkey'])['cap'].shift(1)
price_cons['total_cap_t1'] = price_cons.groupby(['yearmonth'])['cap_t1'].transform(sum)
price_cons['w']=price_cons['cap_t1']/price_cons['total_cap_t1']
price_cons['cap_ret']= price_cons['w']*(price_cons['trt1m']/100)
cap_ret=price_cons.groupby(['yearmonth'])['cap_ret'].sum().reset_index()

compare = pd.merge(cap_ret,sp500_ab)
compare.columns=['yearmonth','cap_ret','date','spx_ab']
compare=compare[1:]
compare['diff']=compare['cap_ret']-compare['spx_ab']
plt.plot(compare['diff'])
plt.savefig('spx_ret_diff.png',bbox_inches='tight') 

#%% technical features(monthly)
price_raw=price_raw[~price_raw.tic.isin(error)]
test=price_raw.groupby(by=['gvkey','iid']).size().reset_index()
test=test.sort_values(by=['gvkey',0],ascending=False).drop_duplicates(subset=['gvkey'],keep='first')
price_raw=pd.merge(price_raw,test[['gvkey','iid']])
price_raw.sort_values(by=['gvkey','tic'],ascending=False,inplace=True)
price_raw=price_raw.drop_duplicates(subset=['gvkey','datadate'])
price_raw['MA3']=price_raw.groupby(['gvkey'])['trt1m'].transform(lambda x: x.rolling(3).mean())
price_raw['MA6']=price_raw.groupby(['gvkey'])['trt1m'].transform(lambda x: x.rolling(6).mean())
price_raw['MA12']=price_raw.groupby(['gvkey'])['trt1m'].transform(lambda x: x.rolling(12).mean())
price_raw['EMA3']=price_raw.groupby(['gvkey'])['trt1m'].transform(lambda x: x.ewm(3).mean())
price_raw['EMA6']=price_raw.groupby(['gvkey'])['trt1m'].transform(lambda x: x.ewm(6).mean())
price_raw['EMA12']=price_raw.groupby(['gvkey'])['trt1m'].transform(lambda x: x.ewm(12).mean())
price_raw['mkt_cap']=price_raw['prccm']*price_raw['cshom']
dividend = pd.read_csv('dividend_monthly.csv')
dividend=dividend[['gvkey','datadate','dvpspm','dvpsxm']]
dividend.drop_duplicates(['gvkey','datadate'],inplace=True)
price_raw=pd.merge(price_raw,dividend,how='left')
price_raw['dvpspm'].fillna(0,inplace=True)
price_raw['dvpsxm'].fillna(0,inplace=True)
price_raw.to_csv('sp500_cons_prc.csv',index=False)

#%% valuation and quality features(quarterly)
data=pd.read_csv('financial_sp500.csv')
data['dvpq_3y']=data.groupby(['gvkey'])['dvpq'].transform(lambda x: x.rolling(12).mean())
data['dvpq_y']=data.groupby(['gvkey'])['dvpq'].transform(lambda x: x.rolling(4).mean())
data['saleq_y']=data.groupby(['gvkey'])['saleq'].transform(lambda x: x.rolling(4).mean())
data['oibdpq_y']=data.groupby(['gvkey'])['oibdpq'].transform(lambda x: x.rolling(4).mean())
data['atq_y']=data.groupby(['gvkey'])['atq'].transform(lambda x: x.rolling(4).mean())
data['revtq_y']=data.groupby(['gvkey'])['revtq'].transform(lambda x: x.rolling(4).mean())
data['niq_y']=data.groupby(['gvkey'])['niq'].transform(lambda x: x.rolling(4).mean())
data['icaptq_y']=data.groupby(['gvkey'])['icaptq'].transform(lambda x: x.rolling(4).mean())
data['xrdq_y']=data.groupby(['gvkey'])['xrdq'].transform(lambda x: x.rolling(4).mean())
data['NI_growth']= data.groupby(['gvkey'])['niq_y'].transform(lambda x: x.diff(1)/x.shift(1))
data['earnings'] = (data['niq']-data['dvpq'])*1e6
data['earningsVol'] = data.groupby(['gvkey'])['earnings'].transform(lambda x: x.rolling(12).std())
data['sales_growth']= data.groupby(['gvkey'])['saleq_y'].transform(lambda x: x.diff(1)/x.shift(1))


data.drop(columns=['datadate'],inplace=True)
data.drop_duplicates(['gvkey','rdq'],keep='last',inplace=True)
rdq_month=pd.DataFrame()
rdq=data[['rdq','gvkey']].drop_duplicates()
month_date=price_raw['datadate'].drop_duplicates()
for d in month_date:
    temp=rdq[rdq.rdq<=d]
    temp=temp.groupby(['gvkey'])['rdq'].max().reset_index()
    temp['datadate']=d
    rdq_month=rdq_month.append(temp)
price_financial=pd.merge(price_raw,rdq_month,how='left')
price_financial=pd.merge(price_financial,data)


data1=pd.read_csv('financial_sp500_annual.csv')
data1=data1[['gvkey','fdate','at', 'dv', 'dvt', 'ebit', 'ebitda', 'epspx', 'fincf', 'gla',
       'ivncf', 'lt', 'oancf', 'revt', 'costat']]
fdate=data1[['fdate','gvkey']].drop_duplicates()
month_date=price_financial['datadate'].drop_duplicates()
fdate_month=pd.DataFrame()
for d in month_date:
    temp=fdate[fdate.fdate<=d]
    temp=temp.groupby(['gvkey'])['fdate'].max().reset_index()
    temp['datadate']=d
    fdate_month=fdate_month.append(temp)
price_financial=pd.merge(price_financial,fdate_month,how='left')
data1.drop_duplicates(['gvkey','fdate'],keep='last',inplace=True)
price_financial=pd.merge(price_financial,data1)

price_financial['ROE'] = price_financial['niq_y']/(price_financial['cshom']*price_financial['prccm'])*1e6
price_financial['ROA'] = price_financial['niq_y']/price_financial['atq_y']
price_financial['leverage'] = (price_financial['dlttq']+price_financial['dlcq'])/price_financial['seqq']
price_financial['NetPM'] = price_financial['niq_y']/price_financial['revtq_y']
price_financial['ROI'] = price_financial['prccm']/price_financial['icaptq_y']/1e6
price_financial['Asset Turnover'] = price_financial['revtq_y']/price_financial['atq_y']
price_financial['PE']=price_financial['prccm']/price_financial['epspi12']
price_financial['PB']=price_financial['prccm']/(price_financial['atq']-price_financial['ltq'])*price_financial['cshom']/1e6
price_financial['PS']=price_financial['prccm']/price_financial['saleq_y']*price_financial['cshom']/1e6
price_financial['dvpsxm'].fillna(0,inplace=True)
price_financial['dividend_yield']=price_financial.groupby(['gvkey'])['dvpsxm'].transform(lambda x: x.rolling(12).sum())/price_financial['prccm']
price_financial['xrdq/sale']=price_financial['xrdq_y']/price_financial['saleq_y']
price_financial['dps']=price_financial.groupby(['gvkey'])['dvpsxm'].transform(lambda x: x.rolling(12).sum())
price_financial['EV']=price_financial['mkt_cap']+price_financial['ltq']*1e6
price_financial['ev_ebitda']=price_financial['EV']/price_financial['ebitda']/1e6
price_financial['net_cash_flow']=(price_financial['fincf']+price_financial['ivncf']+price_financial['oancf'])*1e6
price_financial['NCFP']=price_financial['net_cash_flow']/price_financial['mkt_cap']
price_financial['industry_id']=price_financial['gind']//10000
price_financial['Size'] = np.log(price_financial['mkt_cap'])
price_financial['PriceCap'] = price_financial['prccm']/price_financial['mkt_cap']
price_financial['cumulative_ret'] = price_financial.groupby(by=['gvkey','iid'])['trt1m'].transform(lambda x: (x/100+1).cumprod())
def temp(x):
    return(list(x)[0])
price_financial['adj_close'] = price_financial.groupby(['gvkey','iid'])['prccm'].transform(temp)
price_financial['adj_close'] = price_financial['cumulative_ret']*price_financial['adj_close']
price_financial['trading_volumn_size']=np.log(price_financial['cshtrm'])
price_financial.sort_values(by=['gvkey','datadate'],inplace=True)
price_financial['turnover']=price_financial['cshtrm']/price_financial['cshom']
price_financial['12m_turnover']=price_financial.groupby(by=['gvkey'])['turnover'].transform(lambda x: x.rolling(12).mean())
price_financial['abnormal_volume']=(price_financial['turnover']-price_financial['12m_turnover'])

price_financial.to_csv('price+financial.csv',index=False)
price_financial=pd.read_csv('price+financial.csv')

features = price_financial[['iid','gvkey','tic','datadate','industry_id','trt1m','adj_close',
                     'abnormal_volume','MA3','MA6','MA12','EMA3','EMA6','EMA12','NI_growth',
                     'Size','PriceCap','ROE','ROA','ROI','NetPM','epspi12','earnings', 'earningsVol','Asset Turnover',
                     'PE', 'PB', 'PS', 'dividend_yield','ev_ebitda','NCFP']]

sp500_cons_daily=pd.read_csv('sp500_cons_daily.csv')

sp500_cons_daily['adj_close']=sp500_cons_daily['prccd']*(sp500_cons_daily['trfd']/100+1)
sp500_cons_daily['adj_ret']=sp500_cons_daily.groupby(['gvkey','iid','tic'])['adj_close'].transform(lambda x: x.diff(1)/x.shift(1))
sp500_cons_daily['vol']=sp500_cons_daily.groupby(['gvkey','iid','tic'])['adj_ret'].transform(lambda x: x.rolling(252).std()*np.sqrt(252))
sp500_cons_daily['yearmonth'] = sp500_cons_daily['datadate']//100
test=sp500_cons_daily.groupby(by=['yearmonth','gvkey','iid','tic'])['datadate'].max().reset_index()
sp500_cons_daily1=pd.merge(sp500_cons_daily,test)


features['yearmonth']=features['datadate']//100
features=pd.merge(features,sp500_cons_daily1[['gvkey','iid','tic','yearmonth','vol']],how='left')
features=features[features.datadate>20100000]
features=features[features.datadate<20200000]
features.sort_values(by=['gvkey','iid','datadate'],inplace=True)
features.drop(columns=['yearmonth'],inplace=True)
features.to_csv('features.csv',index=False)

#%% Select SPX constituents stocks
cons_features = pd.merge(price_cons[['gvkey','datadate','conm','tic']],features)
cons_features = cons_features[['iid','gvkey','tic','datadate','industry_id','trt1m','adj_close',
                     'abnormal_volume','MA3','MA6','MA12','EMA3','EMA6','EMA12','NI_growth',
                     'Size','PriceCap','ROE','ROA','ROI','NetPM','epspi12','earnings', 'earningsVol','Asset Turnover',
                     'PE', 'PB', 'PS', 'dividend_yield','ev_ebitda','NCFP']]


cons_features.to_csv('sp500_cons_part_features.csv',index=False)

cons_features = pd.read_csv('sp500_cons_part_features.csv')

X=cons_features.copy()
features = list(cons_features.columns)[7:]
def z_score(x):
    return((x-x.mean())/x.std())
X[features]=X.groupby(by=['datadate','industry_id'])[features].transform(z_score)
def winsor(x):
    return(winsorize(x,limits=0.05))
X[features]=X.groupby(['datadate','industry_id'])[features].transform(winsor)
spx_norm=X.copy()
spx=pd.read_csv('sp500_monthly.csv')
spx_norm.sort_values(by=['datadate','gvkey','tic'],inplace=True)
spx_norm['yearmonth'] = spx_norm['datadate']//100
spx_norm=pd.merge(spx_norm,spx)
spx_norm['forward_return'] = spx_norm.groupby(['gvkey','tic'])['trt1m'].transform(lambda x: x.shift(-1))/100
spx_norm['excess_ret'] = spx_norm['trt1m']/100-spx_norm['sp500_ret']
spx_norm['forward_excess_ret'] = spx_norm.groupby(by=['gvkey','tic'])['excess_ret'].transform(lambda x: x.shift(-1))


### Labels
def binary_class(df):
    return(pd.qcut(df,2,labels=False))   


def multi_class(df):
    return(pd.qcut(df,10,labels=False))
    
    

spx_norm['binary_class'] = spx_norm.groupby(['datadate'])['forward_excess_ret'].transform(binary_class)
spx_norm.dropna(subset=['forward_excess_ret'],inplace=True)
spx_norm['multi_class'] = spx_norm.groupby(['datadate'])['forward_excess_ret'].transform(multi_class)

spx_norm.to_csv('spx_cons_norm.csv',index=False)

#%%
sp500_daily_part=pd.read_csv('sp500_cons_part_prc_daily.csv')

sp500_cons_daily=pd.read_csv('sp500_cons_daily.csv')
sp500_cons_daily['adj_close']=sp500_cons_daily['prccd']*(sp500_cons_daily['trfd']/100+1)
sp500_cons_daily.drop_duplicates(subset=['gvkey','datadate'],inplace=True)
sp500_cons_daily['adj_ret']=sp500_cons_daily.groupby(['gvkey','tic'])['adj_close'].transform(lambda x: x.diff(1)/x.shift(1))
sp500_cons_daily['vol']=sp500_cons_daily.groupby(['gvkey','tic'])['adj_ret'].transform(lambda x: x.rolling(252).std()*np.sqrt(252))
sp500_cons_daily.sort_values(by=['gvkey','datadate'],inplace=True)
sp500_cons_daily['turnover']=sp500_cons_daily['cshtrd']/sp500_cons_daily['cshoc']
sp500_cons_daily['annual_turnover']=sp500_cons_daily.groupby(by=['gvkey','iid'])['turnover'].transform(lambda x: x.rolling(12).mean())
sp500_cons_daily['abnormal_volume']=(sp500_cons_daily['turnover']-sp500_cons_daily['annual_turnover'])

# quarter+daily
data=pd.read_csv('financial_sp500.csv')
data['dvpq_3y']=data.groupby(['gvkey'])['dvpq'].transform(lambda x: x.rolling(12).mean())
data['dvpq_y']=data.groupby(['gvkey'])['dvpq'].transform(lambda x: x.rolling(4).mean())
data['saleq_y']=data.groupby(['gvkey'])['saleq'].transform(lambda x: x.rolling(4).mean())
data['oibdpq_y']=data.groupby(['gvkey'])['oibdpq'].transform(lambda x: x.rolling(4).mean())
data['atq_y']=data.groupby(['gvkey'])['atq'].transform(lambda x: x.rolling(4).mean())
data['revtq_y']=data.groupby(['gvkey'])['revtq'].transform(lambda x: x.rolling(4).mean())
data['niq_y']=data.groupby(['gvkey'])['niq'].transform(lambda x: x.rolling(4).mean())
data['icaptq_y']=data.groupby(['gvkey'])['icaptq'].transform(lambda x: x.rolling(4).mean())
data['xrdq_y']=data.groupby(['gvkey'])['xrdq'].transform(lambda x: x.rolling(4).mean())
data['NI_growth']= data.groupby(['gvkey'])['niq_y'].transform(lambda x: x.diff(1)/x.shift(1))
data['earnings'] = (data['niq']-data['dvpq'])*1e6
data['earningsVol'] = data.groupby(['gvkey'])['earnings'].transform(lambda x: x.rolling(12).std())
data['sales_growth']= data.groupby(['gvkey'])['saleq_y'].transform(lambda x: x.diff(1)/x.shift(1))

data.drop(columns=['datadate'],inplace=True)
data.drop_duplicates(['gvkey','rdq'],keep='last',inplace=True)
rdq_daily=pd.DataFrame()
rdq=data[['rdq','gvkey']].drop_duplicates()
daily_date=sp500_cons_daily['datadate'].drop_duplicates()
for d in daily_date:
    temp=rdq[rdq.rdq<=d]
    temp=temp.groupby(['gvkey'])['rdq'].max().reset_index()
    temp['datadate']=d
    rdq_daily=rdq_daily.append(temp)
price_financial=pd.merge(sp500_cons_daily,rdq_daily,how='left')
price_financial=pd.merge(price_financial,data)

# annual+daily
data1=pd.read_csv('financial_sp500_annual.csv')
data1=data1[['gvkey','fdate','at', 'dv', 'dvt', 'ebit', 'ebitda', 'epspx', 'fincf', 'gla',
       'ivncf', 'lt', 'oancf', 'revt', 'costat']]
fdate=data1[['fdate','gvkey']].drop_duplicates()
fdate_daily=pd.DataFrame()
for d in daily_date:
    temp=fdate[fdate.fdate<=d]
    temp=temp.groupby(['gvkey'])['fdate'].max().reset_index()
    temp['datadate']=d
    fdate_daily=fdate_daily.append(temp)
price_financial=pd.merge(price_financial,fdate_daily,how='left')
data1.drop_duplicates(['gvkey','fdate'],keep='last',inplace=True)
price_financial=pd.merge(price_financial,data1)


price_financial['ROE'] = price_financial['niq_y']/(price_financial['cshoc']*price_financial['prccd'])*1e6
price_financial['ROA'] = price_financial['niq_y']/price_financial['atq_y']
price_financial['leverage'] = (price_financial['dlttq']+price_financial['dlcq'])/price_financial['seqq']
price_financial['NetPM'] = price_financial['niq_y']/price_financial['revtq_y']
price_financial['ROI'] = price_financial['prccd']/price_financial['icaptq_y']/1e6
price_financial['Asset Turnover'] = price_financial['revtq_y']/price_financial['atq_y']
price_financial['PE']=price_financial['prccd']/price_financial['epspi12']
price_financial['PB']=price_financial['prccd']/(price_financial['atq']-price_financial['ltq'])*price_financial['cshoc']/1e6
price_financial['PS']=price_financial['prccd']/price_financial['saleq_y']*price_financial['cshoc']/1e6
price_financial['div'].fillna(0,inplace=True)
price_financial['dividend_yield']=price_financial.groupby(['gvkey','tic'])['div'].transform(lambda x: x.rolling(12).sum())/price_financial['prccd']
price_financial['xrdq/sale']=price_financial['xrdq_y']/price_financial['saleq_y']
price_financial['dps']=price_financial.groupby(['gvkey'])['div'].transform(lambda x: x.rolling(252).sum())
price_financial['EV']=price_financial['mkt_cap']+price_financial['ltq']*1e6
price_financial['ev_ebitda']=price_financial['EV']/price_financial['ebitda']/1e6
price_financial['net_cash_flow']=(price_financial['fincf']+price_financial['ivncf']+price_financial['oancf'])*1e6
price_financial['NCFP']=price_financial['net_cash_flow']/price_financial['mkt_cap']
price_financial['industry_id']=price_financial['gind']//10000
price_financial['mkt_cap'] = price_financial['prccd']*price_financial['cshoc']
price_financial['Size'] = np.log(price_financial['mkt_cap'])
price_financial['MA5']=price_financial.groupby(['gvkey','tic'])['adj_ret'].transform(lambda x: x.rolling(5).mean())
price_financial['MA10']=price_financial.groupby(['gvkey','tic'])['adj_ret'].transform(lambda x: x.rolling(10).mean())
price_financial['MA20']=price_financial.groupby(['gvkey','tic'])['adj_ret'].transform(lambda x: x.rolling(20).mean())
price_financial['EMA5']=price_financial.groupby(['gvkey','tic'])['adj_ret'].transform(lambda x: x.ewm(5).mean())
price_financial['EMA10']=price_financial.groupby(['gvkey','tic'])['adj_ret'].transform(lambda x: x.ewm(10).mean())
price_financial['EMA20']=price_financial.groupby(['gvkey','tic'])['adj_ret'].transform(lambda x: x.ewm(20).mean())

#price_financial.to_csv('price+financial_daily.csv',index=False)

features = price_financial[['iid','gvkey','tic','datadate','industry_id','adj_ret','adj_close',
                     'abnormal_volume','MA5','MA10','MA20','EMA5','EMA10','EMA20','NI_growth',
                     'Size','ROE','ROA','ROI','NetPM','epspi12','earnings', 'earningsVol','Asset Turnover',
                     'PE', 'PB', 'PS', 'dividend_yield','ev_ebitda','NCFP']]


features=features[features.datadate>20100000]
features=features[features.datadate<20200000]
features.sort_values(by=['gvkey','iid','datadate'],inplace=True)

cons_features = pd.merge(sp500_daily_part[['gvkey','datadate','tic']],price_financial)
cons_features = cons_features[['iid','gvkey','tic','datadate','industry_id','adj_ret','adj_close',
                     'abnormal_volume','MA5','MA10','MA20','EMA5','EMA10','EMA20','NI_growth',
                     'Size','ROE','ROA','ROI','NetPM','epspi12','earnings', 'earningsVol','Asset Turnover',
                     'PE', 'PB', 'PS', 'dividend_yield','ev_ebitda','NCFP']]

cons_features.to_csv('sp500_cons_part_features_daily.csv',index=False)
X=cons_features.iloc[:,7:]
X['year']=cons_features['year']
X['industry_id']=cons_features['industry_id']
def z_score(x):
    return((x-x.mean())/x.std())
X=X.groupby(by=['year','industry_id']).apply(z_score)
spx_norm=cons_features.copy()
spx_norm.iloc[:,7:]=X
spx_norm['forward_ret']=spx_norm.groupby(['gvkey','tic'])['adj_ret'].transform(lambda x: x.shift(-1))
spx_norm.to_csv('spx_cons_norm_daily.csv',index=False)
