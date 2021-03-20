#%%
# packages required
from sklearn.base import BaseEstimator, TransformerMixin
import pandas, numpy
from sklearn.pipeline import Pipeline
from pandas_profiling import ProfileReport


# %%
class HandlingDataTypes(BaseEstimator, TransformerMixin): 

    def __init__(self):
        pass 
    def fit(self,X,y=None): 
        return self 
    def transform(self, X): 
        
        # convert to categorical columns
        X[['customer_marital_status','customer_employment_type', 'worst_riskclass_l3m', 'worst_accstatus_l12m','y']] =X[['customer_marital_status','customer_employment_type', 'worst_riskclass_l3m', 'worst_accstatus_l12m','y']].astype('category')

        # convert to numeric columns
        X[['no_daysarrs1_29_evr','capo_enqs_ever_cb','no_guarantors_mdi','no_daysarrs90_119_evr','no_daysarrs0_l3m','curbal_ugx']] = X[['no_daysarrs1_29_evr','capo_enqs_ever_cb','no_guarantors_mdi','no_daysarrs90_119_evr','no_daysarrs0_l3m','curbal_ugx']].apply(pandas.to_numeric, errors = 'coerce')

        return X

#%%

class HandlingMissingValues(BaseEstimator, TransformerMixin):

    def __init__(self): 
        pass 
    def fit(self,X, y=None): 
        return self 
    def transform(self, X):

        # replacing empty strings 
        X.replace(" ", numpy.nan, inplace=True)

        values = {'customer_marital_status': '?', 'customer_employment_type': '?', 'worst_riskclass_l3m': '?', 'worst_accstatus_l12m': '?','no_daysarrs90_119_evr': 0,'no_daysarrs1_29_evr': 0,'capo_enqs_ever_cb': 0,'no_guarantors_mdi': 0,'no_daysarrs0_l3m': 0, 'curbal_ugx':0}

        X.fillna(value=values, inplace = True)

        return X


# %%

### PipeLines 

cleaningPipeLine = Pipeline(steps=[ ('CleaningMissingValues',HandlingMissingValues()), ('ConvertingDataTypes', HandlingDataTypes())])

#%%
## execute PipeLine 

cleandata = cleaningPipeLine.fit_transform(pandas.read_csv(r'‪C:\Users\c69241a\Desktop\pin.csv'.strip('\u202a'), encoding = "ISO-8859-1"))

# %%
# pandas profiling report 
profile = ProfileReport(cleandata, title='Credit Score Data', explorative=True)

profile.to_file("cleandataexploration.html")
# %%
