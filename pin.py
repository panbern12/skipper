
#%%
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data1 = pd.read_csv(r'‪C:\Users\c69241a\Desktop\pin.csv'.strip('\u202a'), encoding = "ISO-8859-1")
pd.options.display.max_columns = None
pd.options.display.max_rows = None


#%%

## Data cleaning
#Categorical -  worst_riskclass_l3m, worst_accstatus_l12m, customer_marital_status, customer_employment_type
#Numerical - no_daysarr1_29_evr, capo_enqs_ever_cb, no_guarantors_mdi, no_daysarrs90_119_evr,no_daysarrs0_l3m
#worst_accstatus_l12m
#
#
#
#

## Missing 
# impute 0 (no_guarantors_mdi, no_daysarrs90_119_evr)
#
#

## Mode 
# impute 8 (customer_martial_status)
#
#

data1[['worst_riskclass_l3m','no_daysarrs1_29_evr','capo_enqs_ever_cb','no_guarantors_mdi','worst_accstatus_l12m','customer_marital_status','no_daysarrs90_119_evr','no_daysarrs0_l3m','customer_employment_type','curbal_ugx']] = data1[['worst_riskclass_l3m','no_daysarrs1_29_evr','capo_enqs_ever_cb','no_guarantors_mdi','worst_accstatus_l12m','customer_marital_status','no_daysarrs90_119_evr','no_daysarrs0_l3m','customer_employment_type','curbal_ugx']].apply(pd.to_numeric, errors = 'coerce')


#%%

feature_cols = ['worst_riskclass_l3m',
       'no_daysarrs1_29_evr', 'capo_enqs_ever_cb', 'no_guarantors_mdi',
       'worst_accstatus_l12m', 'customer_marital_status',
       'no_daysarrs90_119_evr', 'no_daysarrs0_l3m', 'customer_employment_type',
       'curbal_ugx']


#%%
data1['y'] = data1['y'].fillna(data1['y'].mean())
data1['y'] = data1['y'].replace([np.inf, -np.inf], np.nan).fillna(data1['y'].mean())

#%%

for field in feature_cols:
    data1[field] = data1[field].fillna(data1[field].mean())
    data1[field] = data1[field].replace([np.inf, -np.inf], np.nan).fillna(data1[field].mean())
data1['customer_marital_status'] = data1['customer_marital_status'].fillna(0)

x = data1[feature_cols]

y = data1['y']

# %%
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=9)

lin_reg_mod = LinearRegression()
lin_reg_mod.fit(X_train, y_train)
pred = lin_reg_mod.predict(X_test)

test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
test_set_r2 = r2_score(y_test, pred)
# %%
def WoE(dataframe, field, target='y', compute=True):



    pd.set_option('use_inf_as_na', True)



    xtab_1 = pd.crosstab(dataframe[field], dataframe[target], normalize=1, dropna=False)
    xtab_2 = pd.crosstab(dataframe[field], dataframe[target], dropna=False)
    xtab_1.columns = ['pct_Bads', 'pct_Goods'] 
    xtab_2.columns = ['Bads', 'Goods'] 
    xtab_3 = pd.merge(left=xtab_2, right=xtab_1, left_on=field, right_on=field)



    if compute==True:
        xtab_3['WoE'] = xtab_3.apply(lambda x: np.log(x['pct_Goods']/x['pct_Bads']), axis=1)
# Information Value = sum (% of non-events – % of events) * WoE
        xtab_3['IV'] = xtab_3.apply(lambda x: (x['pct_Goods'] - x['pct_Bads']) * x['WoE'], axis=1)
        xtab_3.replace([np.inf, -np.inf], np.nan)


    return xtab_3

def fineclass (dataframe, field, target="y", no_classes=10, min_volume=30, print_msg=False, low=0):
    str1 = "Running fineclass for ''" + field + "' aiming for " + str(no_classes) + " classes, each with a minimum of "
    str1 = str1 + str(min_volume) + " events and non-events"

    if print_msg:
        print("#", str1, "#", sep="n")
        woe_tab = WoE(dataframe, field, target=target, compute=False)

        assert isinstance(no_classes, int)
        assert no_classes > 0
        assert isinstance(min_volume, int)
        assert min_volume > 0

        threshold = 1/no_classes

        woe_tab2 = woe_tab.copy()
        woe_tab['cum_pct_Bads'] = woe_tab.pct_Bads.cumsum()
        woe_tab['cum_pct_Goods'] = woe_tab.pct_Goods.cumsum()
        woe_tab['cum_Bads'] = woe_tab.Bads.cumsum()
        woe_tab['cum_Goods'] = woe_tab.Goods.cumsum()
        woe_tab['diff_pct_Bads'] = np.NaN
        woe_tab['diff_pct_Goods'] = np.NaN
        woe_tab['diff_Bads'] = np.NaN
        woe_tab['diff_Goods'] = np.NaN

    last_Goods, last_Bads, last_pct_Goods, last_pct_Bads = 0, 0, 0, 0
    bins = []
    # data step

    for obs in woe_tab.index:
        woe_tab.at[obs, 'diff_Goods'] = woe_tab.at[obs, 'cum_Goods'] - last_Goods
        woe_tab.at[obs, 'diff_Bads'] = woe_tab.at[obs, 'cum_Bads'] - last_Bads
        woe_tab.at[obs, 'diff_pct_Goods'] = woe_tab.at[obs, 'cum_pct_Goods'] - last_pct_Goods
        woe_tab.at[obs, 'diff_pct_Bads'] = woe_tab.at[obs, 'cum_pct_Bads'] - last_pct_Bads
    # if we have enough goods, and enough bads, and a ibg enough % of population, then declare class completed.
        if (woe_tab.at[obs, 'diff_Goods'] >= min_volume and woe_tab.at[obs, 'diff_Bads'] >= min_volume and (woe_tab.at[obs, 'diff_pct_Goods'] + woe_tab.at[obs, 'diff_pct_Bads']) >= threshold*2):
            bins.append(obs)
            last_Goods = woe_tab.at[obs, 'cum_Goods'] 
            last_Bads = woe_tab.at[obs, 'cum_Bads'] 
            last_pct_Goods = woe_tab.at[obs, 'cum_pct_Goods'] 
            last_pct_Bads = woe_tab.at[obs, 'cum_pct_Bads'] 
            pass

# the last of our ‘bins’ stops short of the end of the dataset, meaning the range above this will be inquorate.
# remove this bin:
    del bins[-1] 
    
    bins.append(woe_tab.index.max())

    # add a starting bin
    bins.insert(0, woe_tab.index.min()-1)

    # use the bins to cut categories:
    woe_tab['fineclass'] = pd.cut(woe_tab.index, bins)

    woe_tab['bin_end'] = ser1

    # group by our fine classes
    woe_grp = woe_tab2.groupby(pd.cut(woe_tab.index, bins))
    woe_tab3 = woe_grp[list(woe_tab2.columns.values)].agg('sum')
    woe_tab3['WoE'] = woe_tab3.apply(lambda x: np.log(x['pct_Goods']/x['pct_Bads']), axis=1)
    
    # Information Value = sum (% of non-events – % of events) * WoE
    woe_tab3['IV'] = woe_tab3.apply(lambda x: (x['pct_Goods'] - x['pct_Bads']) * x['WoE'], axis=1)
    woe_tab3.index.name=field

    return woe_tab3, bins









worst_riskclass_l3m = WoE(data1,'worst_riskclass_l3m', compute=True)
no_daysarrs1_29_evr = WoE(data1, 'no_daysarrs1_29_evr', compute=True)
capo_enqs_ever_cb = WoE(data1,'capo_enqs_ever_cb', compute=True)
no_guarantors_mdi = WoE(data1, 'no_guarantors_mdi', compute=True)
worst_accstatus_l12m = WoE(data1, 'worst_accstatus_l12m', compute=True)
customer_marital_status = WoE(data1, 'customer_marital_status', compute=True)
no_daysarrs90_119_evr = WoE(data1, 'no_daysarrs90_119_evr', compute=True)
no_daysarrs0_l3m = WoE(data1, 'no_daysarrs0_l3m', compute=True)
customer_employment_type = WoE(data1,  'customer_employment_type', compute=True)
curbal_ugx = WoE(data1, 'curbal_ugx', compute=True)

worst_riskclass_l3mGrp, worst_riskclass_l3mBins = fineclass(data1, 'worst_riskclass_l3m', print_msg=True)
no_daysarrs1_29_evrGrp, no_daysarrs1_29_evrBins = fineclass(data1, 'no_daysarrs1_29_evr', print_msg=True)
capo_enqs_ever_cbGrp, capo_enqs_ever_cbBins = fineclass(data1, 'capo_enqs_ever_cb', print_msg=True)
no_guarantors_mdiGrp, no_guarantors_mdiBins = fineclass(data1, 'no_guarantors_mdi', print_msg=True)
worst_accstatus_l12mGrp, worst_accstatus_l12mBins = fineclass(data1, 'worst_accstatus_l12m', print_msg=True)
customer_marital_statusGrp, customer_marital_statusBins = fineclass(data1, 'customer_marital_status', print_msg=True)
no_daysarrs90_119_evrGrp, no_daysarrs90_119_evrBins = fineclass(data1, 'no_daysarrs90_119_evr', print_msg=True)
no_daysarrs0_l3mGrp, no_daysarrs0_l3mBins = fineclass(data1, 'no_daysarrs0_l3m', print_msg=True)
customer_employment_typeGrp, customer_employment_typeBins = fineclass(data1, 'customer_employment_type', print_msg=True)
curbal_ugxGrp, curbal_ugxBins = fineclass(data1, 'curbal_ugx', print_msg=True)

#%%
n = len(feature_cols)
alpha = lin_reg_mod.intercept_
beta_worst_riskclass_l3m  = lin_reg_mod.coef_[0]  
beta_no_daysarrs1_29_evr    = lin_reg_mod.coef_[1]    
beta_capo_enqs_ever_cb   = lin_reg_mod.coef_[2]    
beta_no_guarantors_mdi = lin_reg_mod.coef_[3] 
beta_worst_accstatus_l12m = lin_reg_mod.coef_[4]
beta_customer_marital_status = lin_reg_mod.coef_[5]
beta_no_daysarrs90_119_evr= lin_reg_mod.coef_[6]
beta_no_daysarrs0_l3m = lin_reg_mod.coef_[7]
beta_customer_employment_type = lin_reg_mod.coef_[8]
beta_curbal_ugx = lin_reg_mod.coef_[9]
#Scaling for a maximum total Scorecard Point of 600
factor      = 20/np.log(2)
offset      = 600-factor*np.log(20)

print("factor:{0}, offset:{1}".format(factor, offset))

#Scorecard Point calculation
woe_worst_riskclass_l3m = worst_riskclass_l3mGrp['WoE'].sum()
worst_riskclass_l3mGrp['variable_score'] = round((beta_worst_riskclass_l3m*woe_worst_riskclass_l3m+(alpha/n))*factor + (offset/n),1)
worst_riskclass_l3mGrp['bin_score'] = round((beta_worst_riskclass_l3m*worst_riskclass_l3mGrp['WoE']+(alpha/n))*factor + (offset/n), 1)

woe_no_daysarrs1_29_evr = no_daysarrs1_29_evrGrp['WoE'].sum()
no_daysarrs1_29_evrGrp['variable_score'] = round((beta_no_daysarrs1_29_evr*woe_no_daysarrs1_29_evr+(alpha/n))*factor + (offset/n),1)
no_daysarrs1_29_evrGrp['bin_score'] = round((beta_no_daysarrs1_29_evr*no_daysarrs1_29_evrGrp['WoE']+(alpha/n))*factor + (offset/n), 1)

woe_capo_enqs_ever_cb = capo_enqs_ever_cbGrp['WoE'].sum()
capo_enqs_ever_cbGrp['variable_score'] = round((beta_capo_enqs_ever_cb*woe_capo_enqs_ever_cb+(alpha/n))*factor + (offset/n),1)
capo_enqs_ever_cbGrp['bin_score'] = round((beta_capo_enqs_ever_cb*capo_enqs_ever_cbGrp['WoE']+(alpha/n))*factor + (offset/n), 1)

woe_no_guarantors_mdi = no_guarantors_mdiGrp['WoE'].sum()
no_guarantors_mdiGrp['variable_score'] = round((beta_no_guarantors_mdi*woe_no_guarantors_mdi+(alpha/n))*factor + (offset/n),1)
no_guarantors_mdiGrp['bin_score'] = round((beta_no_guarantors_mdi*no_guarantors_mdiGrp['WoE']+(alpha/n))*factor + (offset/n), 1)

woe_worst_accstatus_l12m = worst_accstatus_l12mGrp['WoE'].sum()
worst_accstatus_l12mGrp['variable_score'] = round((beta_worst_accstatus_l12m*woe_worst_accstatus_l12m+(alpha/n))*factor + (offset/n), 1)
worst_accstatus_l12mGrp['bin_score'] = round((beta_worst_accstatus_l12m*worst_accstatus_l12mGrp['WoE']+(alpha/n))*factor + (offset/n), 1)

woe_customer_marital_status = customer_marital_statusGrp['WoE'].sum()
customer_marital_statusGrp['variable_score'] = round((beta_customer_marital_status*woe_customer_marital_status+(alpha/n))*factor + (offset/n), 1)
customer_marital_statusGrp['bin_score'] = round((beta_customer_marital_status*customer_marital_statusGrp['WoE']+(alpha/n))*factor + (offset/n), 1)

woe_no_daysarrs90_119_evr = no_daysarrs90_119_evrGrp['WoE'].sum()
no_daysarrs90_119_evrGrp['variable_score'] = round((beta_no_daysarrs90_119_evr*woe_no_daysarrs90_119_evr+(alpha/n))*factor + (offset/n), 1)
no_daysarrs90_119_evrGrp['bin_score'] = round((beta_no_daysarrs90_119_evr*no_daysarrs90_119_evrGrp['WoE']+(alpha/n))*factor + (offset/n), 1)

woe_no_daysarrs0_l3m = no_daysarrs0_l3mGrp['WoE'].sum()
no_daysarrs0_l3mGrp['variable_score'] = round((beta_no_daysarrs0_l3m*woe_no_daysarrs0_l3m+(alpha/n))*factor + (offset/n), 1)
no_daysarrs0_l3mGrp['bin_score'] = round((beta_no_daysarrs0_l3m*no_daysarrs0_l3mGrp['WoE']+(alpha/n))*factor + (offset/n), 1)

woe_customer_employment_type = customer_employment_typeGrp['WoE'].sum()
customer_employment_typeGrp['variable_score'] = round((beta_customer_employment_type*woe_customer_employment_type+(alpha/n))*factor + (offset/n), 1)
customer_employment_typeGrp['bin_score'] = round((beta_customer_employment_type*customer_employment_typeGrp['WoE']+(alpha/n))*factor + (offset/n), 1)

woe_curbal_ugx = curbal_ugxGrp['WoE'].sum()
curbal_ugxGrp['variable_score'] = round((beta_curbal_ugx*woe_curbal_ugx+(alpha/n))*factor + (offset/n), 1)
curbal_ugxGrp['bin_score'] = round((beta_curbal_ugx*curbal_ugxGrp['WoE']+(alpha/n))*factor + (offset/n), 1)
# %%


