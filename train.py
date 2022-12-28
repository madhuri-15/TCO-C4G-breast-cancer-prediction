# Importing libraries
import pickle                           # To save the model
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics

from sklearn.linear_model import LogisticRegression

def data_format(df, data_type='test'):
    df.rename(columns={col: col.lower() for col in df.columns.tolist()}, inplace=True)

    # column 1 - menopaus
    menopaus_values = {
        0 : 'premenopausal',
        1 : 'postmenopausal',
        9 : np.nan 
        }
        
    df['menopaus'] = df.menopaus.replace(menopaus_values)
    
    # column 2 - agegrp
    agegrp_values = {
        1 : '35-39',
        2 : '40-44',
        3 : '45-49',
        4 : '50-54',
        5 : '55-59',
        6 : '60-64',
        7 : '65-69',
        8 : '70-74',
        9 : '75-59',
        10: '80-84'
    }
    df['agegrp'] = df.agegrp.replace(agegrp_values)


    # column 3 - density
    density_values = {
        1 : 'almost entirely fat',
        2 : 'scattered fibroglandular',
        3 : 'heterogeneously',
        4 : 'extremely dense',
        9 : np.nan
    }
    df['density'] = df.density.replace(density_values)

    # column 4 - race
    race_values = {
        1 : 'white',
        2 : 'asian/pacific',
        3 : 'black',
        4 : 'native american',
        5 : 'other/mixed',
        9 : np.nan
    }

    df['race'] = df.race.replace(race_values)


    # column 5 - hispanic 
    hispanic_values = {
        0 : 'no',
        1 : 'yes',
        9 : np.nan
    }

    df['hispanic'] = df.hispanic.replace(hispanic_values)


    # column 6 - bmi - body mass index
    bmi_values = {
        1 : '10-24.99',
        2 : '25-29.99',
        3 : '30-34.99',
        4 : '35 or more',
        9 : np.nan
    }

    df['bmi'] = df.bmi.replace(bmi_values)


    # column 7 - agefirst - age at the first birth
    agefirst_values = {
        0 : 'age <30',
        1 : 'age 30 or greater',
        2 : 'Nulliparous',
        9 : np.nan
    }
    df['agefirst'] = df.agefirst.replace(agefirst_values)

    # column 8 - nrelbc - Number of first degree relatives with breast cancer - number of relative with breast cancer
    nrelbc_values = {
        0 : 'zero',
        1 : 'one',
        2 : '2 or more',
        9 : np.nan
    }

    df['nrelbc'] = df. nrelbc.replace(nrelbc_values)


    # column 9 - brstproc - previous breast procedure
    brstproc_values = {
        0 : 'no',
        1 : 'yes',
        9 : np.nan
    }
    df['brstproc'] = df.brstproc.replace(brstproc_values)



    # column 10 - lastmamm - result of last mammogram before the index mammogram
    lastmamm_values = {
        0 : 'negative',
        1 : 'false positive',
        9 : np.nan
    }
    df['lastmamm'] = df.lastmamm.replace(lastmamm_values)


    # column 11 - surgmeno - surgical menopause 
    surgmeno_values = {
        0 : 'natural',
        1 : 'surgical',
        9 : np.nan
    }
    df['surgmeno'] = df.surgmeno.replace(surgmeno_values)


    # column 12 - hrt - current hormone therapy
    hrt_values = {
        0 : 'no',
        1 : 'yes',
        9 : np.nan
    }
    df['hrt'] = df.hrt.replace(hrt_values)


    # column 13 - invasive - diagnosis of invasive brease cancer within one year of the index screening mammogram
    invasive_values = {
        0 : 'no',
        1 : 'yes'
    }
    df['invasive'] = df.invasive.replace(invasive_values)

    if data_type == 'train':
        # column 14 - cancer - diagnosis of invasive or ductual carcinoma in situ breast cancer within one year of the index screening mammogram
        cancer_values = {
            0 : 'no',
            1 : 'yes'
        }
        df['cancer'] = df.cancer.replace(cancer_values)

    return df


def get_data(PATH):
    data = pd.read_csv(PATH)
    data = data_format(data, data_type='train')
                       
    data = data.set_index(['id'])
    
    # split the data into X and y
    y = (data['cancer']=='yes').astype('int')
    X = data.drop(['cancer'], axis=1)

    # replace the missing values in the categorical data with mode value of that columns
    for col in X.columns[X.isnull().any().tolist()]:
        X[col].fillna(value=X[col].mode()[0], inplace=True)

    # split the data into training, validation and test dataset in ratio 60:30:10.
    x_train,  x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

    return (x_train, y_train), (x_test, y_test)


def train(x_train, y_train, C):

    dicts = x_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C,
                               class_weight='balanced',
                               multi_class='ovr',
                               max_iter=1000,
                               random_state=42,
                               n_jobs=-1
                               )

    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_preds = model.predict_proba(X)[:, 1]

    return y_preds


    
if __name__ == '__main__':
    
    # parameters
    C = 0.5
    output_file = "model_C=%s.bin" %C
    
    PATH = "../code/data/data-training.csv"
    
    df_train, df_test = get_data(PATH)
    x_train, y_train = df_train
    x_test, y_test = df_test


    # Training final model
    print("Training final model....")
    dv, model = train(x_train, y_train, C)
    y_preds = predict(x_test, dv, model)

    # Evaluation
    fpr, tpr, thr = metrics.roc_curve(y_test, y_preds)
    score = metrics.auc(fpr, tpr)
    print("ROC Score :: %.3f" %score)
    
    # Save the model and dictionary vectorizer
    with open(f"../code/model/{output_file}", 'wb') as f_out:
        pickle.dump((dv, model), f_out)

    print(f"Model saved as {output_file} file")


    
    

