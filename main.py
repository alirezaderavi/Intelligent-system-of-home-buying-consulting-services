import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error ,mean_squared_error
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering

df=pd.read_csv('desktop/data project/train_data.csv',na_values=np.NaN)
dftest=pd.read_csv('desktop/data project/test_data.csv',na_values=np.NaN)
columns_name=df.columns
df.columns


var = 'LotFrontage'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

df.sort_values(by = 'LotFrontage', ascending = False)[:2]
df = df.drop(df[df.index==1166].index)
df = df.drop(df[df.index==839].index)

var = 'MasVnrArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df.sort_values(by = 'LotFrontage', ascending = False)[:1]
df = df.drop(df[df.index==1015].index)

var = 'MSSubClass'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'GarageArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'GarageCars'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'Fireplaces'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'TotRmsAbvGrd'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'KitchenAbvGr'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'BedroomAbvGr'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'HalfBath'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'FullBath'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'BsmtHalfBath'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'BsmtFullBath'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'LowQualFinSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = '2ndFlrSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = '1stFlrSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'BsmtUnfSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'BsmtFinSF2'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df.sort_values(by = 'LotFrontage', ascending = False)[:1]
df = df.drop(df[df.index==178].index)

var = 'BsmtFinSF1'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'YearRemodAdd'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'YearBuilt'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'OverallCond'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'OverallQual'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'LotArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df.sort_values(by = 'LotFrontage', ascending = False)[:3]
df = df.drop(df[df.index==205].index)
df = df.drop(df[df.index==999].index)
df = df.drop(df[df.index==1062].index)

var = 'GarageYrBlt'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'MiscVal'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df.sort_values(by = 'LotFrontage', ascending = False)[:2]
df = df.drop(df[df.index==1203].index)
df = df.drop(df[df.index==1087].index)

var = 'PoolArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'ScreenPorch'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = '3SsnPorch'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'EnclosedPorch'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'OpenPorchSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'WoodDeckSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'GrLivArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df.sort_values(by = 'GrLivArea', ascending = False)[:1]
df = df.drop(df[df.index==466].index)

var = 'TotalBsmtSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


comp=np.arange(1,df.shape[1])               

for i in comp:
    print(df.columns[i],sum(df[df.columns[i]].isna()),':',(sum(df[df.columns[i]].isna())/df.shape[0])*100)

df = df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
dftest = dftest.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)

categorical_feature_mask = df.dtypes==object
categorical_cols = df.columns[categorical_feature_mask].tolist()
num_feature_mask= df.dtypes!=object
num_cols = df.columns[num_feature_mask].tolist()
df_cat=pd.get_dummies(df,drop_first=True,columns=categorical_cols)
df_cat.columns.tolist()
df_num=df[num_cols]
df_cat2 = pd.DataFrame()

categorical_feature_mask_test = dftest.dtypes==object
categorical_cols_test = dftest.columns[categorical_feature_mask_test].tolist()
num_feature_mask_test= dftest.dtypes!=object
num_cols_test = dftest.columns[num_feature_mask_test].tolist()
dftest_cat=pd.get_dummies(dftest,drop_first=True,columns=categorical_cols_test)
dftest_cat.columns.tolist()
dftest_num=df[num_cols_test]
dftest_cat2 = pd.DataFrame()


df_cat2['MSZoning']=df_cat['MSZoning_FV']*0+df_cat['MSZoning_RH']*1+df_cat['MSZoning_RL']*2+df_cat['MSZoning_RM']*3
df_cat2['Street']=df_cat['Street_Pave']*1
df_cat2['LotShape']=df_cat['LotShape_IR2']*0+df_cat['LotShape_IR3']*1+df_cat['LotShape_Reg']*2
df_cat2['LandContour']=df_cat['LandContour_HLS']*0+df_cat['LandContour_Low']*1+df_cat['LandContour_Lvl']*2
df_cat2['Utilities']=df_cat['Utilities_NoSeWa']*1
df_cat2['LotConfig']=df_cat['LotConfig_CulDSac']*0+df_cat['LotConfig_FR2']*1+df_cat['LotConfig_FR3']*2+df_cat['LotConfig_Inside']*3
df_cat2['LandSlope']=df_cat['LandSlope_Mod']*0+df_cat['LandSlope_Sev']*1
df_cat2['Neighborhood']=df_cat['Neighborhood_Blueste']*0+df_cat['Neighborhood_BrDale']*1+df_cat['Neighborhood_BrkSide']*2+df_cat['Neighborhood_ClearCr']*3+df_cat['Neighborhood_CollgCr']*4+df_cat['Neighborhood_Crawfor']*5+df_cat['Neighborhood_Edwards']*6+df_cat['Neighborhood_Gilbert']*7+df_cat['Neighborhood_IDOTRR']*8+df_cat['Neighborhood_MeadowV']*9+df_cat['Neighborhood_Mitchel']*10+df_cat['Neighborhood_NAmes']*11+df_cat['Neighborhood_NPkVill']*12+df_cat['Neighborhood_NWAmes']*13+df_cat['Neighborhood_NoRidge']*14+df_cat['Neighborhood_NridgHt']*15+df_cat['Neighborhood_OldTown']*16+df_cat['Neighborhood_SWISU']*17+df_cat['Neighborhood_Sawyer']*18+df_cat['Neighborhood_SawyerW']*19+df_cat['Neighborhood_Somerst']*20+df_cat['Neighborhood_StoneBr']*21+df_cat['Neighborhood_Timber']*22+df_cat['Neighborhood_Veenker']*23
df_cat2['Condition1']=df_cat['Condition1_Feedr']*0+df_cat['Condition1_Norm']*1+df_cat['Condition1_PosA']*2+df_cat['Condition1_PosN']*3+df_cat['Condition1_RRAe']*4+df_cat['Condition1_RRAn']*5+df_cat['Condition1_RRNe']*6+df_cat['Condition1_RRNn']*7
df_cat2['Condition2']=df_cat['Condition2_Feedr']*0+df_cat['Condition2_Norm']*1+df_cat['Condition2_PosA']*2+df_cat['Condition2_PosN']*3+df_cat['Condition2_RRAe']*4+df_cat['Condition2_RRAn']*5+df_cat['Condition2_RRNn']*6
df_cat2['BldgType']=df_cat['BldgType_2fmCon']*0+df_cat['BldgType_Duplex']*1+df_cat['BldgType_Twnhs']*2+df_cat['BldgType_TwnhsE']*3
df_cat2['HouseStyle']=df_cat['HouseStyle_1.5Unf']*0+df_cat['HouseStyle_1Story']*1+df_cat['HouseStyle_2.5Fin']*2+df_cat['HouseStyle_2.5Unf']*3+df_cat['HouseStyle_2Story']*4+df_cat['HouseStyle_SFoyer']*5+df_cat['HouseStyle_SLvl']*6
df_cat2['RoofStyle']=df_cat['RoofStyle_Gable']*0+df_cat['RoofStyle_Gambrel']*1+df_cat['RoofStyle_Hip']*2+df_cat['RoofStyle_Mansard']*3+df_cat['RoofStyle_Shed']*4
df_cat2['RoofMatl']=df_cat['RoofMatl_Membran']*0+df_cat['RoofMatl_Metal']*1+df_cat['RoofMatl_Roll']*2+df_cat['RoofMatl_Tar&Grv']*3+df_cat['RoofMatl_WdShake']*4+df_cat['RoofMatl_WdShngl']*4
df_cat2['Exterior1st']=df_cat['Exterior1st_AsphShn']*0+df_cat['Exterior1st_BrkComm']*1+df_cat['Exterior1st_BrkFace']*2+df_cat['Exterior1st_CBlock']*3+df_cat['Exterior1st_CemntBd']*4+df_cat['Exterior1st_HdBoard']*5+df_cat['Exterior1st_ImStucc']*6+df_cat['Exterior1st_MetalSd']*7+df_cat['Exterior1st_Plywood']*8+df_cat['Exterior1st_Stone']*9+df_cat['Exterior1st_Stucco']*10+df_cat['Exterior1st_VinylSd']*11+df_cat['Exterior1st_Wd Sdng']*12+df_cat['Exterior1st_WdShing']*13
df_cat2['Exterior2nd']=df_cat['Exterior2nd_AsphShn']*0+df_cat['Exterior2nd_Brk Cmn']*1+df_cat['Exterior2nd_BrkFace']*2+df_cat['Exterior2nd_CBlock']*3+df_cat['Exterior2nd_CmentBd']*4+df_cat['Exterior2nd_HdBoard']*5+df_cat['Exterior2nd_ImStucc']*6+df_cat['Exterior2nd_MetalSd']*7+df_cat['Exterior2nd_Other']*8+df_cat['Exterior2nd_Plywood']*9+df_cat['Exterior2nd_Stone']*10+df_cat['Exterior2nd_Stucco']*11+df_cat['Exterior2nd_VinylSd']*12+df_cat['Exterior2nd_Wd Sdng']*13+df_cat['Exterior2nd_Wd Shng']*14
df_cat2['MasVnrType']=df_cat['MasVnrType_BrkFace']*0+df_cat['MasVnrType_None']*1+df_cat['MasVnrType_Stone']*2
df_cat2['ExterCond']=df_cat['ExterQual_Fa']*0+df_cat['ExterQual_Gd']*1+df_cat['ExterQual_TA']*2+df_cat['ExterCond_Fa']*3+df_cat['ExterCond_Gd']*4+df_cat['ExterCond_Po']*5+df_cat['ExterCond_TA']*6
df_cat2['Foundation']=df_cat['Foundation_CBlock']*0+df_cat['Foundation_PConc']*1+df_cat['Foundation_Slab']*2+df_cat['Foundation_Stone']*3+df_cat['Foundation_Wood']*4
df_cat2['BsmtQual']=df_cat['BsmtQual_Fa']*0+df_cat['BsmtQual_Gd']*1+df_cat['BsmtQual_TA']*2
df_cat2['BsmtCond']=df_cat['BsmtCond_Gd']*0+df_cat['BsmtCond_Po']*1+df_cat['BsmtCond_TA']*2
df_cat2['BsmtExposure']=df_cat['BsmtExposure_Gd']*0+df_cat['BsmtExposure_Mn']*1+df_cat['BsmtExposure_No']*2
df_cat2['BsmtFinType1']=df_cat['BsmtFinType1_BLQ']*0+df_cat['BsmtFinType1_GLQ']*1+df_cat['BsmtFinType1_LwQ']*2+df_cat['BsmtFinType1_Rec']*3+df_cat['BsmtFinType1_Unf']*4
df_cat2['BsmtFinType2']=df_cat['BsmtFinType2_BLQ']*0+df_cat['BsmtFinType2_GLQ']*1+df_cat['BsmtFinType2_LwQ']*2+df_cat['BsmtFinType2_Rec']*3+df_cat['BsmtFinType2_Unf']*4
df_cat2['Heating']=df_cat['Heating_GasA']*0+df_cat['Heating_GasW']*1+df_cat['Heating_Grav']*2+df_cat['Heating_OthW']*3+df_cat['Heating_Wall']*4
df_cat2['HeatingQC']=df_cat['HeatingQC_Fa']*0+df_cat['HeatingQC_Gd']*1+df_cat['HeatingQC_Po']*2+df_cat['HeatingQC_TA']*3
df_cat2['CentralAir']=df_cat['CentralAir_Y']*1
df_cat2['Electrical']=df_cat['Electrical_FuseF']*0+df_cat['Electrical_FuseP']*1+df_cat['Electrical_Mix']*2+df_cat['Electrical_SBrkr']*3
df_cat2['KitchenQual']=df_cat['KitchenQual_Fa']*0+df_cat['KitchenQual_Gd']*1+df_cat['KitchenQual_TA']*2
df_cat2['Functional']=df_cat['Functional_Maj2']*0+df_cat['Functional_Min1']*1+df_cat['Functional_Min2']*2+df_cat['Functional_Mod']*3+df_cat['Functional_Sev']*4+df_cat['Functional_Typ']*5
df_cat2['FireplaceQu']=df_cat['FireplaceQu_Fa']*0+df_cat['FireplaceQu_Gd']*1+df_cat['FireplaceQu_Po']*2+df_cat['FireplaceQu_TA']*3
df_cat2['GarageType']=df_cat['GarageType_Attchd']*0+df_cat['GarageType_Basment']*1+df_cat['GarageType_BuiltIn']*2+df_cat['GarageType_CarPort']*3+df_cat['GarageType_Detchd']*4
df_cat2['GarageFinish']=df_cat['GarageFinish_RFn']*0+df_cat['GarageFinish_Unf']*1
df_cat2['GarageQual']=df_cat['GarageQual_Fa']*0+df_cat['GarageQual_Gd']*1+df_cat['GarageQual_Po']*2+df_cat['GarageQual_TA']*3
df_cat2['GarageCond']=df_cat['GarageCond_Fa']*0+df_cat['GarageCond_Gd']*1+df_cat['GarageCond_Po']*2+df_cat['GarageCond_TA']*3
df_cat2['PavedDrive']=df_cat['PavedDrive_P']*0+df_cat['PavedDrive_Y']*1
df_cat2['SaleType']=df_cat['SaleType_CWD']*0+df_cat['SaleType_Con']*1+df_cat['SaleType_ConLD']*2+df_cat['SaleType_ConLI']*3+df_cat['SaleType_ConLw']*4+df_cat['SaleType_New']*5+df_cat['SaleType_Oth']*6+df_cat['SaleType_WD']*7
df_cat2['SaleCondition']=df_cat['SaleCondition_AdjLand']*0+df_cat['SaleCondition_Alloca']*1+df_cat['SaleCondition_Family']*2+df_cat['SaleCondition_Normal']*3+df_cat['SaleCondition_Partial']*4

dftest_cat2['MSZoning']=df_cat['MSZoning_FV']*0+df_cat['MSZoning_RH']*1+df_cat['MSZoning_RL']*2+df_cat['MSZoning_RM']*3
dftest_cat2['Street']=df_cat['Street_Pave']*1
dftest_cat2['LotShape']=df_cat['LotShape_IR2']*0+df_cat['LotShape_IR3']*1+df_cat['LotShape_Reg']*2
dftest_cat2['LandContour']=df_cat['LandContour_HLS']*0+df_cat['LandContour_Low']*1+df_cat['LandContour_Lvl']*2
dftest_cat2['Utilities']=df_cat['Utilities_NoSeWa']*1
dftest_cat2['LotConfig']=df_cat['LotConfig_CulDSac']*0+df_cat['LotConfig_FR2']*1+df_cat['LotConfig_FR3']*2+df_cat['LotConfig_Inside']*3
dftest_cat2['LandSlope']=df_cat['LandSlope_Mod']*0+df_cat['LandSlope_Sev']*1
dftest_cat2['Neighborhood']=df_cat['Neighborhood_Blueste']*0+df_cat['Neighborhood_BrDale']*1+df_cat['Neighborhood_BrkSide']*2+df_cat['Neighborhood_ClearCr']*3+df_cat['Neighborhood_CollgCr']*4+df_cat['Neighborhood_Crawfor']*5+df_cat['Neighborhood_Edwards']*6+df_cat['Neighborhood_Gilbert']*7+df_cat['Neighborhood_IDOTRR']*8+df_cat['Neighborhood_MeadowV']*9+df_cat['Neighborhood_Mitchel']*10+df_cat['Neighborhood_NAmes']*11+df_cat['Neighborhood_NPkVill']*12+df_cat['Neighborhood_NWAmes']*13+df_cat['Neighborhood_NoRidge']*14+df_cat['Neighborhood_NridgHt']*15+df_cat['Neighborhood_OldTown']*16+df_cat['Neighborhood_SWISU']*17+df_cat['Neighborhood_Sawyer']*18+df_cat['Neighborhood_SawyerW']*19+df_cat['Neighborhood_Somerst']*20+df_cat['Neighborhood_StoneBr']*21+df_cat['Neighborhood_Timber']*22+df_cat['Neighborhood_Veenker']*23
dftest_cat2['Condition1']=df_cat['Condition1_Feedr']*0+df_cat['Condition1_Norm']*1+df_cat['Condition1_PosA']*2+df_cat['Condition1_PosN']*3+df_cat['Condition1_RRAe']*4+df_cat['Condition1_RRAn']*5+df_cat['Condition1_RRNe']*6+df_cat['Condition1_RRNn']*7
dftest_cat2['Condition2']=df_cat['Condition2_Feedr']*0+df_cat['Condition2_Norm']*1+df_cat['Condition2_PosA']*2+df_cat['Condition2_PosN']*3+df_cat['Condition2_RRAe']*4+df_cat['Condition2_RRAn']*5+df_cat['Condition2_RRNn']*6
dftest_cat2['BldgType']=df_cat['BldgType_2fmCon']*0+df_cat['BldgType_Duplex']*1+df_cat['BldgType_Twnhs']*2+df_cat['BldgType_TwnhsE']*3
dftest_cat2['HouseStyle']=df_cat['HouseStyle_1.5Unf']*0+df_cat['HouseStyle_1Story']*1+df_cat['HouseStyle_2.5Fin']*2+df_cat['HouseStyle_2.5Unf']*3+df_cat['HouseStyle_2Story']*4+df_cat['HouseStyle_SFoyer']*5+df_cat['HouseStyle_SLvl']*6
dftest_cat2['RoofStyle']=df_cat['RoofStyle_Gable']*0+df_cat['RoofStyle_Gambrel']*1+df_cat['RoofStyle_Hip']*2+df_cat['RoofStyle_Mansard']*3+df_cat['RoofStyle_Shed']*4
dftest_cat2['RoofMatl']=df_cat['RoofMatl_Membran']*0+df_cat['RoofMatl_Metal']*1+df_cat['RoofMatl_Roll']*2+df_cat['RoofMatl_Tar&Grv']*3+df_cat['RoofMatl_WdShake']*4+df_cat['RoofMatl_WdShngl']*4
dftest_cat2['Exterior1st']=df_cat['Exterior1st_AsphShn']*0+df_cat['Exterior1st_BrkComm']*1+df_cat['Exterior1st_BrkFace']*2+df_cat['Exterior1st_CBlock']*3+df_cat['Exterior1st_CemntBd']*4+df_cat['Exterior1st_HdBoard']*5+df_cat['Exterior1st_ImStucc']*6+df_cat['Exterior1st_MetalSd']*7+df_cat['Exterior1st_Plywood']*8+df_cat['Exterior1st_Stone']*9+df_cat['Exterior1st_Stucco']*10+df_cat['Exterior1st_VinylSd']*11+df_cat['Exterior1st_Wd Sdng']*12+df_cat['Exterior1st_WdShing']*13
dftest_cat2['Exterior2nd']=df_cat['Exterior2nd_AsphShn']*0+df_cat['Exterior2nd_Brk Cmn']*1+df_cat['Exterior2nd_BrkFace']*2+df_cat['Exterior2nd_CBlock']*3+df_cat['Exterior2nd_CmentBd']*4+df_cat['Exterior2nd_HdBoard']*5+df_cat['Exterior2nd_ImStucc']*6+df_cat['Exterior2nd_MetalSd']*7+df_cat['Exterior2nd_Other']*8+df_cat['Exterior2nd_Plywood']*9+df_cat['Exterior2nd_Stone']*10+df_cat['Exterior2nd_Stucco']*11+df_cat['Exterior2nd_VinylSd']*12+df_cat['Exterior2nd_Wd Sdng']*13+df_cat['Exterior2nd_Wd Shng']*14
dftest_cat2['MasVnrType']=df_cat['MasVnrType_BrkFace']*0+df_cat['MasVnrType_None']*1+df_cat['MasVnrType_Stone']*2
dftest_cat2['ExterCond']=df_cat['ExterQual_Fa']*0+df_cat['ExterQual_Gd']*1+df_cat['ExterQual_TA']*2+df_cat['ExterCond_Fa']*3+df_cat['ExterCond_Gd']*4+df_cat['ExterCond_Po']*5+df_cat['ExterCond_TA']*6
dftest_cat2['Foundation']=df_cat['Foundation_CBlock']*0+df_cat['Foundation_PConc']*1+df_cat['Foundation_Slab']*2+df_cat['Foundation_Stone']*3+df_cat['Foundation_Wood']*4
dftest_cat2['BsmtQual']=df_cat['BsmtQual_Fa']*0+df_cat['BsmtQual_Gd']*1+df_cat['BsmtQual_TA']*2
dftest_cat2['BsmtCond']=df_cat['BsmtCond_Gd']*0+df_cat['BsmtCond_Po']*1+df_cat['BsmtCond_TA']*2
dftest_cat2['BsmtExposure']=df_cat['BsmtExposure_Gd']*0+df_cat['BsmtExposure_Mn']*1+df_cat['BsmtExposure_No']*2
dftest_cat2['BsmtFinType1']=df_cat['BsmtFinType1_BLQ']*0+df_cat['BsmtFinType1_GLQ']*1+df_cat['BsmtFinType1_LwQ']*2+df_cat['BsmtFinType1_Rec']*3+df_cat['BsmtFinType1_Unf']*4
dftest_cat2['BsmtFinType2']=df_cat['BsmtFinType2_BLQ']*0+df_cat['BsmtFinType2_GLQ']*1+df_cat['BsmtFinType2_LwQ']*2+df_cat['BsmtFinType2_Rec']*3+df_cat['BsmtFinType2_Unf']*4
dftest_cat2['Heating']=df_cat['Heating_GasA']*0+df_cat['Heating_GasW']*1+df_cat['Heating_Grav']*2+df_cat['Heating_OthW']*3+df_cat['Heating_Wall']*4
dftest_cat2['HeatingQC']=df_cat['HeatingQC_Fa']*0+df_cat['HeatingQC_Gd']*1+df_cat['HeatingQC_Po']*2+df_cat['HeatingQC_TA']*3
dftest_cat2['CentralAir']=df_cat['CentralAir_Y']*1
dftest_cat2['Electrical']=df_cat['Electrical_FuseF']*0+df_cat['Electrical_FuseP']*1+df_cat['Electrical_Mix']*2+df_cat['Electrical_SBrkr']*3
dftest_cat2['KitchenQual']=df_cat['KitchenQual_Fa']*0+df_cat['KitchenQual_Gd']*1+df_cat['KitchenQual_TA']*2
dftest_cat2['Functional']=df_cat['Functional_Maj2']*0+df_cat['Functional_Min1']*1+df_cat['Functional_Min2']*2+df_cat['Functional_Mod']*3+df_cat['Functional_Sev']*4+df_cat['Functional_Typ']*5
dftest_cat2['FireplaceQu']=df_cat['FireplaceQu_Fa']*0+df_cat['FireplaceQu_Gd']*1+df_cat['FireplaceQu_Po']*2+df_cat['FireplaceQu_TA']*3
dftest_cat2['GarageType']=df_cat['GarageType_Attchd']*0+df_cat['GarageType_Basment']*1+df_cat['GarageType_BuiltIn']*2+df_cat['GarageType_CarPort']*3+df_cat['GarageType_Detchd']*4
dftest_cat2['GarageFinish']=df_cat['GarageFinish_RFn']*0+df_cat['GarageFinish_Unf']*1
dftest_cat2['GarageQual']=df_cat['GarageQual_Fa']*0+df_cat['GarageQual_Gd']*1+df_cat['GarageQual_Po']*2+df_cat['GarageQual_TA']*3
dftest_cat2['GarageCond']=df_cat['GarageCond_Fa']*0+df_cat['GarageCond_Gd']*1+df_cat['GarageCond_Po']*2+df_cat['GarageCond_TA']*3
dftest_cat2['PavedDrive']=df_cat['PavedDrive_P']*0+df_cat['PavedDrive_Y']*1
dftest_cat2['SaleType']=df_cat['SaleType_CWD']*0+df_cat['SaleType_Con']*1+df_cat['SaleType_ConLD']*2+df_cat['SaleType_ConLI']*3+df_cat['SaleType_ConLw']*4+df_cat['SaleType_New']*5+df_cat['SaleType_Oth']*6+df_cat['SaleType_WD']*7
dftest_cat2['SaleCondition']=df_cat['SaleCondition_AdjLand']*0+df_cat['SaleCondition_Alloca']*1+df_cat['SaleCondition_Family']*2+df_cat['SaleCondition_Normal']*3+df_cat['SaleCondition_Partial']*4




df = pd.concat([df_num, df_cat2], axis=1)
dftest = pd.concat([dftest_num, dftest_cat2], axis=1)


comp=np.arange(1,df.shape[1])               

for i in comp:
    print(df.columns[i],sum(df[df.columns[i]].isna()),':',(sum(df[df.columns[i]].isna())/df.shape[0])*100)


imp = SimpleImputer(missing_values=np.NaN, strategy='mean')

a=df.columns
x=imp.fit_transform(df)
y=df['SalePrice'].values

df = pd.DataFrame(x,columns=a)

a=dftest.columns
x=imp.fit_transform(dftest)

dftest = pd.DataFrame(x,columns=a)
df = df.drop(['SalePrice'],axis=1)


corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

comp=np.arange(1,df.shape[1])               

for i in comp:
    print(df.columns[i],sum(df[df.columns[i]].isna()),':',(sum(df[df.columns[i]].isna())/df.shape[0])*100)
          


X=df.values
#y gablan tarif shode

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train= y_train.reshape(-1, 1)
y_test= y_test.reshape(-1, 1)
y_train = sc_X.fit_transform(y_train)
y_test = sc_X.fit_transform(y_test)

SGD = SGDRegressor()
ElasticNet = ElasticNet(random_state=0)
Lasso = Lasso(alpha=0.1)
lm = LinearRegression()
svr = SVR(kernel = 'rbf')
rfr = RandomForestRegressor(n_estimators = 100, random_state = 0)


SGD.fit(X_train,y_train)
ElasticNet.fit(X_train,y_train)
Lasso.fit(X_train,y_train)
lm.fit(X_train,y_train)
svr.fit(X_train,y_train)
rfr.fit(X_train, y_train)



predictions = lm.predict(X_test)
predictions= predictions.reshape(-1,1)
lineStart = y_test.min() 
lineEnd = y_test.max()  
plt.figure(figsize=(10,8))
plt.scatter(y_test,predictions,color = 'k',alpha=0.5)
plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.xlim(lineStart, lineEnd)
plt.ylim(lineStart, lineEnd)
plt.show()
print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))


predictions = svr.predict(X_test)
predictions= predictions.reshape(-1,1)
lineStart = y_test.min() 
lineEnd = y_test.max()  
plt.figure(figsize=(10,8))
plt.scatter(y_test,predictions,color = 'k',alpha=0.5)
plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.xlim(lineStart, lineEnd)
plt.ylim(lineStart, lineEnd)
plt.show()
print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))


predictions = Lasso.predict(X_test)
predictions= predictions.reshape(-1,1)
lineStart = y_test.min() 
lineEnd = y_test.max()  
plt.figure(figsize=(10,8))
plt.scatter(y_test,predictions,color = 'k',alpha=0.5)
plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.xlim(lineStart, lineEnd)
plt.ylim(lineStart, lineEnd)
plt.show()
print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))

ytest=rfr.predict(dftest)
dftest['SalePrice']=ytest


comp=np.arange(2,15)               
average_silhouette=-1
for n in comp:
    from sklearn.cluster import KMeans
    KMeans = KMeans(n_clusters=n).fit(X)
    labels_kmeans=KMeans.labels_
    if average_silhouette<silhouette_score(df,labels_kmeans):
        average_silhouette=silhouette_score(df,labels_kmeans)
        best_n=n
print ('Kmeans score:',average_silhouette,'with n_clusters=',best_n)

m=2
from sklearn.cluster import Birch
Birch = Birch(n_clusters=m).fit(X)
labels_Birch=Birch.labels_
print ('Birch score:',silhouette_score(df,labels_Birch),'with n_clusters=',m)

comp=np.arange(2,100)
average_silhouette=-1
for bw in comp:               
    from sklearn.cluster import MeanShift
    MeanShift=MeanShift(bandwidth=bw).fit(X)
    labels_MeanShift=MeanShift.labels_
    if average_silhouette<silhouette_score(df,labels_MeanShift): 
        average_silhouette=silhouette_score(df,labels_MeanShift)
        bestbw=bw
print ('MeanShift score:',average_silhouette,'with bandwidth=',bestbw)


