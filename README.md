# Balancing data
Version Date: 2020-03-16
<hr>

## Table of Contents
[TOC]

## Prerequisites
The main script will need Numpy, Scikit Learn and Pandas to be installed, even though almost all code will work with only the first two: Pandas is only needed for getting the data and outputting results. 

#### Dependencies
* **Numpy:** _https://anaconda.org/anaconda/numpy_
    *    How to install **_Numpy_** via conda: `conda install numpy`

* **Scikit Learn:** _http://scikit-learn.org/stable/install.html_
    *    How to install **_Sklearn_** via conda: `conda install scikit-learn`

* **Pandas:** _https://pandas.pydata.org/pandas-docs/stable/install.html_
    *    How to install **_Pandas_** via conda: `conda install pandas`

## Module Overview

### Description
Classification algorithms are appropriate Machine Learning techniques for building patterns from categorical or nominal data. This data is collected in a categorical column with a relatively few values called _uniques_, and the aim of the underlying classification algorithm will be to classify the new data in one unique or in another one. 

Frequently, in reality it happens that the same number of uniques is not observed: it can happen that many cases of a unique are observed, and very few of another unique. The quintessential example of this phenomenon can be found in fraud detection or quality control, both boolean data, where the class with less uniques ("fraud" in the first case, "rejected piece" in the second one) is extremely small compared with the class with more uniques ("not fraud" in the first case, "accepted piece" in the second one). 

Classification algorithms have difficulties to learn properly from this kind of data, because they tend to be biased in favour of majoritary classes. Therefore, a module should be developed to deal with this _imbalancing_ cases, to compensate the bias of classifiers. This is precisely what this module does. 

### Flowchart
![](D:\AML_Repositories\f2_balance\flowcharts\balancing.jpg)

## List of scripts and their functions
* balance_strategy.py
    * size_predictor
    * is_balanceable
    * data_imb
    * generate_balanced_data

## balance_strategy.py - Code explanations
In classification, a dataset is called imbalanced when target column values' frequencies differ greatly from each other. In this situations statistical learning is very hard to perform because models often learn more information about majority classes (classes of the target column that appear most frequently). 

To avoid poor results in learning, balancing techniques can be developed. This script implements two different variants, the first of them focused on observations of the dataset, and the second one based on algorithms' weights. Both techniques are blended in _clfFit_ function, whose purpose is to imitate/generalize what a method _fit_ does. 

Moreover, not all imbalanced datasets are tratable by balancing techniques: it is impossible to learn from a very imbalanced dataset (ex. binary classification with 10000000 obs. from one class, 5 from the other). Thus, a tratability filter is necessary to decide which degree of imbalancing is going to be as the maximum accepted. 

### Prerequisites - Imports
* **Scipy**, **Pandas** and **Scikit-learn** packages:
  * Numpy: `import numpy as np`
  * Pandas: `import pandas as pd`
  * Scikit-learn K-means clustering algorithm: `from sklearn.cluster import KMeans as km`

### function _size_predictor_
```python
def size_predictor(uniq, quant):

    from sklearn.cluster import KMeans as km

    quantLen = len(quant)                                                       # ASSUMPTION: No missings in target column

    if quantLen >= 3:                                                           # Multiclass and imbalanced classification
        clustAlgorithm = km(n_clusters = 2, random_state = 0)
        uniqLabels = clustAlgorithm.fit(quant.reshape(-1, 1)).labels_

        if len(set(uniqLabels)) == 1:
            raise ValueError('Dataset is perfectly balanced!')

        maximums = max(quant[uniqLabels == 0]), max(quant[uniqLabels == 1])
        majorClass = int(np.argmax(maximums))
        minorClass = 1 - majorClass
        boolMaj = uniqLabels == majorClass
        boolMin = np.invert(boolMaj)
        majPercent = sum(boolMaj)/len(uniqLabels)
        minPercent = 1 - majPercent
        quantMaj, quantMin = quant[boolMaj], quant[boolMin]
        upper = min(quantMaj)
        lower = max(quantMin)

        if majPercent >= 0.75:
            middle = int((2*upper + lower)/3)
        elif majPercent <= 0.25:
            middle = int((upper + 2*lower)/3)
        else:
            middle = int((upper + lower)/2)
                    
        dictValues = quant.copy()
        limitRUS = np.round(quantMaj/2, 0).astype('int')
        limitSMOTE = np.round(3*quantMin, 0).astype('int')
        midArrayRUS = np.maximum(middle*np.ones(len(quantMaj)), limitRUS)
        midArraySMOTE = np.minimum(middle*np.ones(len(quantMin)), limitSMOTE)
        dictValues[boolMaj], dictValues[boolMin] = midArrayRUS, midArraySMOTE
        dictCounts = dict(zip(uniq, dictValues))

    elif quantLen == 2:                                                         # Binary and imbalanced classification
        boolMaj = int(np.argmax(quant))
        boolMin = 1 - boolMaj
        middle = int(np.mean(quant))

        limitRUS, limitSMOTE = int((quant[boolMaj])/2), int(3*quant[boolMin])
        dictCounts = dict(zip(uniq, quant))
        dictCounts[uniq[boolMaj]] = max(middle, limitRUS)
        dictCounts[uniq[boolMin]] = min(middle, limitSMOTE)

    return uniq, quant, dictCounts, boolMaj, boolMin, middle
```

#### Description
To balance a dataset (and, in general, to do everything in life) it is necessary to know how to do it: balancing models needs to decide how many observations will have each class. Moreover, to study viability of this balancing is important to compute this quantities before fitting any model. 

This function is a predictor which returns how classes are divided into majority and minority (for that purpose a clustering technique is used), and also returns how many observations will have each class in balanced dataset, for posterior evaluations. 

#### I/O
* Parameters:
    * _**uniq**_ (_array-like_): Array containing unique values of target column
    * _**quant**_ (_array-like_): Array containing frequencies of each value of target column

* Returns:
    * Array (_array-like_) containing unique values of target column
    * Array (_array-like_) containing frequencies of each value of target column
    * Dictionary (_dict_) to decide how many observations (value, _int_) must be in each class (key of dictionary)
    * Array (_array-like_) of booleans telling what classes are majority classes
    * Array (_array-like_) of booleans telling what classes are minority classes
    * Number of observations (_int_) for each class in final dataset 

### function _is_balanceable_
```python
def is_balanceable(df, target_col_name):
     
    y = df[target_col_name].values
    uniq, quant = np.unique(y, return_counts = True)                            # ASSUMPTION: No missings in target column

    maxQuant, minQuant = max(quant), min(quant)
    is_manageable = minQuant/maxQuant >= 0.01                              # Reject data if greatest class has 100 times more obs. than smallest class

    if not is_manageable:
        return False, False, tuple()

    resPredictor = size_predictor(uniq, quant)
    dictUniques = resPredictor[2]
    RUSSMOTEBoolean = sum(list(dictUniques.values())) <= 1.2 * len(y)           # Avoid resampling if it implies augmenting data size more than a 20%
    
    return is_manageable, RUSSMOTEBoolean, resPredictor
```

#### Description
When a dataset is imbalanced, balancing tools are run. However, sometimes the dataset is too balanced and this tools cannot do anything to improve algorithms' results. Therefore, a filter is needed to detect when it an imbalanced dataset cannot be balanced. 

#### I/O
* Parameters:
    * _**df**_ (_pandas.DataFrame_): data frame containing all the data
    * _**targetColumnName**_ (_str_): name of target column

* Returns:
    * Decision (_bool_) about if dataset is too imbalanced to try balancing
    * Decision (_bool_) about if dataset is going to be balanced (otherwise algorithm's weights will be tuned)
    * Results of function _sizePredictor_ (_array-like_)

#### Flowchart
![](D:\AML_Repositories\f2_balance\flowcharts\Balancing_strategy.jpg)

### function _data_imb_
```python
def data_imb(X, y, uniq, quant, boolMaj, boolMin, middle):
    '''Function performing sequential under- and over-sampling'''
    from imblearn.under_sampling import RandomUnderSampler as RUS
    from imblearn.over_sampling import SMOTE

    # Starting parameters
    dictCounts = dict(zip(uniq, quant))
    quantMaj, quantMin = quant[boolMaj], quant[boolMin]
    quantLen = len(quant)

    # Performing random under-sampling
    limRUS = np.round(quantMaj/2, 0).astype('int')
    if quantLen >= 3:
        dictValues = quant.copy()
        midArrayRUS = np.maximum(middle*np.ones(len(quantMaj)), limRUS)
        dictValues[boolMaj] = midArrayRUS
        dictCounts = dict(zip(uniq, dictValues))

    elif quantLen == 2:
        dictCounts[uniq[boolMaj]] = max(middle, limRUS)

    XRus, yRus = RUS(ratio = dictCounts, random_state = 0).fit_sample(X, y)
    
    # Performing SMOTE over-sampling
    limSMOTE = np.round(3*quantMin, 0).astype('int')
    if quantLen >= 3:                                                      
        dictValues = np.array(list(dictCounts.values()))
        midArraySMOTE = np.minimum(middle*np.ones(len(quantMin)), limSMOTE)
        dictValues[boolMin] = midArraySMOTE
        dictCountsRus = dict(zip(uniq, dictValues))

    elif quantLen == 2:
        dictCountsRus = dictCounts.copy()
        dictCountsRus[uniq[boolMin]] = min(middle, limSMOTE)

    XRusSmote, yRusSmote = SMOTE(ratio = dictCountsRus, 
                                 k_neighbors = 4, 
                                 random_state = 0).fit_sample(XRus, yRus)

    return XRusSmote, yRusSmote
```

#### Description
All previous functions decide how to balance a dataset, but they do not balance it. To successfully complete the balancing task, we only need one function that, taking all the information calculated up to this point, balances a dataset. This is the purpose of this function: it takes the arrays X and y containing the data and the information computed 
previously, and return balanced X and y. 

#### I/O
* Parameters:
    * _**X**_ (_array-like_): Numpy 2D-array containing observations of independent variables
    * _**y**_ (_array-like_): Numpy 1D-array with observations of target variable
    * _**uniq**_ (_array-like_): Array containing uniques of _y_
    * _**quant**_ (_array-like_): Array containing number of times each unique appears at _y_
    * _**boolMaj**_ (_array-like_): Boolean array with True at positions where the unique defines a majority class
    * _**boolMin**_ (_array-like_): Boolean array with True at positions where the unique defines a minority class  
    * _**middle**_ (_int_): the hypothetical middle point between number of observations of minority and majority classes
* Returns:
    * 2-D array (_array-like_) containing observations of the balanced dataset
    * Array (_array-like_) contanining the target variable of the balanced dataset

### function _generate_balanced_data_
```python
def generate_balanced_data(df, route_df_prop):

    import pickle as pkl
    from f8_core.f81_problem_type.f811_clf.balance_clf import data_imb
    
    with open(route_df_prop, 'rb') as file:
        dfProp = pkl.load(file)
        
    targetColumnName = dfProp.selectedTargetColumn
    target_index = df.columns.tolist().index(targetColumnName)
    df_no_target = df.drop(columns = targetColumnName)
    X, y = df_no_target.values, df[targetColumnName].values
    uniq, quant, _, boolMaj, boolMin, middle = dfProp.res_predictor
    
    XBalanced, yBalanced = data_imb(X, y, uniq, quant, boolMaj, boolMin, middle)
    
    df_balanced = pd.DataFrame(XBalanced)
    df_balanced.columns = df_no_target.columns.copy()
    if target_index == len(df_balanced.columns):
        df_balanced[targetColumnName] = yBalanced
    else:
        df_balanced.insert(loc = target_index, \
                           column = targetColumnName, \
                           value = yBalanced)
    
    return df_balanced
```

#### Description
The previous function _data_imb_ takes a lot of inputs contaning information about how to balance and balances a dataset of the form _X, y_ returning 
another pair _X, y_. 

However, in the real world of AlchemyML data comes in dataframes, information
about balancing is encapsulated in a pickle and balanced data needs to be 
returned in a dataframe. All this processing and transformations are 
made in this function. 

#### I/O
* Parameters:
    * _**df**_ (_pandas.DataFrame_): data frame containing all the data
    * _**route_df_prop**_ (_str_): location of the pickle

* Returns:
    * Balanced dataframe (_pandas.DataFrame_) 

#### Flowchart
![](D:\AML_Repositories\f2_balance\flowcharts\Balancing_strategy.jpg)

