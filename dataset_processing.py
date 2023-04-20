import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from datasets.FeatureBinarizer import FeatureBinarizer
from sklearn.model_selection import train_test_split

def load_and_transform_data(dataset, doFeatureBinarizer=True, doOrdinalEncode=False):
    """
    Retrieves the dataset from it's folder, splits in train and test set, 
        removes colums and lines with too many nans, inputs missing values, and perform ordinal or binary encoding before
        returning the train and test data and labels.
    """
    ds_folder = "./datasets/"
    
    thr_nan_ratio = 0.4
    nb_val_numerical = 9
    
    if dataset == "adult":
        ds = pd.read_csv(ds_folder+"Adult/adult.data", header=None, na_values='?', skipinitialspace=True)
        lb = ds[14]; ds = ds.loc[:,:13]
        tst = pd.read_csv(ds_folder+"Adult/adult.test", header=None, na_values='?', skipinitialspace=True)
        tst_lb = tst[14]; tst = tst.loc[:,:13]
    elif dataset == "magic": 
        ds = pd.read_csv(ds_folder+"Magic/magic04.data", header=None, skipinitialspace=True)
        lb = ds[10]; ds = ds.loc[:,:9]
    elif dataset == "house": 
        ds = pd.read_csv(ds_folder+"House/house_16H.csv", skipinitialspace=True)
        lb = ds.iloc[:,16]; ds = ds.iloc[:,:16]
    elif dataset == "heloc": 
        ds = pd.read_csv(ds_folder+"Heloc/heloc_dataset_v1.csv", na_values=['-7', '-8', '-9'], skipinitialspace=True)
        lb = ds['RiskPerformance']; ds = ds.iloc[:,1:]
    elif dataset == "mushroom": 
        ds = pd.read_csv(ds_folder+"Mushroom/agaricus-lepiota.data", header=None, na_values='?', skipinitialspace=True)
        lb = ds[0]; ds = ds.iloc[:,1:]
    elif dataset == "chess": 
        ds = pd.read_csv(ds_folder+"Chess/kr-vs-kp.data", header=None, skipinitialspace=True)
        lb = ds.iloc[:,36]; ds = ds.iloc[:,:36]
    elif dataset == "ads": 
        ds = pd.read_csv(ds_folder+"Ads/ad.data", header=None, na_values='?', skipinitialspace=True)
        lb = ds.iloc[:,1558]; ds = ds.iloc[:,:1558]
    elif dataset == "nursery": 
        ds = pd.read_csv(ds_folder+"Nursery/nursery.data", header=None, skipinitialspace=True)
        lb = ds.iloc[:,8]; ds = ds.iloc[:,:8]
    elif dataset == "car": 
        ds = pd.read_csv(ds_folder+"Car/car.data", header=None, skipinitialspace=True)
        lb = ds.iloc[:,6]; ds = ds.iloc[:,:6]
    elif dataset == "pageblocks": 
        ds = pd.read_csv(ds_folder+"Pageblocks/page-blocks.data", header=None, skipinitialspace=True, sep=' ')
        lb = ds.iloc[:,10]; ds = ds.iloc[:,:10]
    elif dataset == "pendigits": 
        ds = pd.read_csv(ds_folder+"Pendigits/pendigits.tra", header=None, skipinitialspace=True)
        lb = ds.iloc[:,16]; ds = ds.iloc[:,:16]
        tst = pd.read_csv(ds_folder+"Pendigits/pendigits.tes", header=None, skipinitialspace=True)
        tst_lb = tst.iloc[:,16]; tst = tst.iloc[:,:16]
    elif dataset == "contraceptivemc": 
        ds = pd.read_csv(ds_folder+"ContraceptiveMC/cmc.data", header=None, skipinitialspace=True)
        lb = ds.iloc[:,9]; ds = ds.iloc[:,:9]
    elif dataset == "drive": 
        ds = pd.read_csv(ds_folder+"Drive/Sensorless_drive_diagnosis.txt", header=None, skipinitialspace=True, sep=' ')
        lb = ds.iloc[:,48]; ds = ds.iloc[:,:48]
    elif dataset == "yeast": 
        ds = pd.read_csv(ds_folder+"Yeast/yeast-train.csv", header=0, skipinitialspace=True)
        lb = ds.iloc[:,-14:]; ds = ds.iloc[:,:-14]
        tst = pd.read_csv(ds_folder+"Yeast/yeast-test.csv", header=0, skipinitialspace=True)
        tst_lb = tst.iloc[:,-14:]; tst = tst.iloc[:,:-14]
    elif dataset == "scene": 
        ds = pd.read_csv(ds_folder+"Scene/scene-train.csv", header=None, skipinitialspace=True)
        lb = ds.iloc[:,-6:]; ds = ds.iloc[:,:-6]
        tst = pd.read_csv(ds_folder+"Scene/scene-test.csv", header=None, skipinitialspace=True)
        tst_lb = tst.iloc[:,-6:]; tst = tst.iloc[:,:-6]    
    
    if dataset not in ["adult", "pendigits", "shuttle", "yeast", "scene"]:
        ds, tst, lb, tst_lb = train_test_split(ds, lb, test_size=0.25, random_state=42, stratify=lb)
    
    #remove lines with a too high amount of nan (in term of proportion)
    perc_nan_line = ds.isna().mean(axis=1)
    ds = ds[perc_nan_line<=thr_nan_ratio]
    lb = lb[perc_nan_line<=thr_nan_ratio]
    # remove the columns with a too high amount of nan (in term of proportion)
    perc_nan_attr = ds.isna().mean()
    ds = ds.iloc[:,perc_nan_attr.values<=thr_nan_ratio]
    tst = tst.iloc[:,perc_nan_attr.values<=thr_nan_ratio]
    
    # find numerical and categorical columns
    objects = list(ds.columns[(ds.dtypes == np.dtype('O')).to_numpy().nonzero()[0]])
    if dataset == 'diabetes': objects.append('discharge_disposition_id'); objects.append('admission_source_id')
    if dataset == "yeast": numerical_cols = [col for col in ds.columns]
    else: numerical_cols = [col for col in ds.columns if col not in objects and np.unique(ds[col].dropna()).shape[0] > nb_val_numerical]
    categorical_cols = [col for col in ds.columns if col not in numerical_cols]

    # Input the missing values
    if len(categorical_cols) != 0:
        imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ds[categorical_cols] = imp_cat.fit_transform(ds[categorical_cols])
        tst[categorical_cols] = imp_cat.transform(tst[categorical_cols])
    if len(numerical_cols) != 0:
        imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
        ds[numerical_cols] = imp_num.fit_transform(ds[numerical_cols])
        tst[numerical_cols] = imp_num.transform(tst[numerical_cols])
    print('after removing nans', ds.shape)
    print (np.unique(lb, return_counts=True)[1]/ds.shape[0])
    
    X = ds.copy()
    X_tst = tst.copy()
    if doFeatureBinarizer:
        fb = FeatureBinarizer(colCateg=categorical_cols, negations=False)
        X = fb.fit_transform(X)
        X_tst = fb.transform(X_tst)
    
    elif doOrdinalEncode:
        enc = OneHotEncoder(sparse=False)
        X = pd.concat((X[numerical_cols],pd.DataFrame(enc.fit_transform(X[categorical_cols])).set_index(X.index)),axis=1)
        X_tst = pd.concat((X_tst[numerical_cols], pd.DataFrame(enc.transform(X_tst[categorical_cols])).set_index(X_tst.index)),axis=1)
            
    if dataset not in ["yeast", "scene"]:
        le = LabelEncoder()
        Y = le.fit_transform(lb).astype(int)
        Y_tst = le.transform(tst_lb).astype(int)
    else: 
        Y = lb
        Y_tst = tst_lb
    
    return X, Y, X_tst, Y_tst

if __name__ == "__main__":  
    dataset = "yeast" 
    # adult, magic, house, heloc, mushroom, chess, ads, nursery, car, pageblocks, pendigits, contraceptivemc, drive, yeast, scene
    X, Y, X_tst, Y_tst = load_and_transform_data(dataset, doFeatureBinarizer=True, doOrdinalEncode=False)
