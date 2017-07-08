# Random Forest Classification

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
def prepare_data(Train_filename,Test_filename ):
    trainset = pd.read_csv(Train_filename)
    temp = trainset.Survived
    trainset.drop('Survived',1,inplace=True)
    testset = pd.read_csv(Test_filename)
    dataset = trainset.append(testset)
    dataset.reset_index(inplace=True)
    dataset.drop('index',1,inplace=True)
    
    dataset['Title'] = dataset['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    dataset.drop('Name',1,inplace=True)
    
    
    Title_Dictionary = {
                            "Capt":       "Officer",
                            "Col":        "Officer",
                            "Major":      "Officer",
                            "Jonkheer":   "Royalty",
                            "Don":        "Royalty",
                            "Sir" :       "Royalty",
                            "Dr":         "Officer",
                            "Rev":        "Officer",
                            "the Countess":"Royalty",
                            "Dona":       "Royalty",
                            "Mme":        "Mrs",
                            "Mlle":       "Miss",
                            "Ms":         "Mrs",
                            "Mr" :        "Mr",
                            "Mrs" :       "Mrs",
                            "Miss" :      "Miss",
                            "Master" :    "Master",
                            "Lady" :      "Royalty"
    
                            }
        
    # we map each title
    dataset['Title'] = dataset.Title.map(Title_Dictionary)
    
    grouped = dataset.groupby(['Sex','Pclass','Title'])
    grouped.median()
    
    #calculate None age values
    def fillAges(row):
            if row['Sex']=='female' and row['Pclass'] == 1:
                if row['Title'] == 'Miss':
                    return 30
                elif row['Title'] == 'Mrs':
                    return 45
                elif row['Title'] == 'Officer':
                    return 49
                elif row['Title'] == 'Royalty':
                    return 39
    
            elif row['Sex']=='female' and row['Pclass'] == 2:
                if row['Title'] == 'Miss':
                    return 20
                elif row['Title'] == 'Mrs':
                    return 30
    
            elif row['Sex']=='female' and row['Pclass'] == 3:
                if row['Title'] == 'Miss':
                    return 18
                elif row['Title'] == 'Mrs':
                    return 31
    
            elif row['Sex']=='male' and row['Pclass'] == 1:
                if row['Title'] == 'Master':
                    return 6
                elif row['Title'] == 'Mr':
                    return 41.5
                elif row['Title'] == 'Officer':
                    return 52
                elif row['Title'] == 'Royalty':
                    return 40
    
            elif row['Sex']=='male' and row['Pclass'] == 2:
                if row['Title'] == 'Master':
                    return 2
                elif row['Title'] == 'Mr':
                    return 30
                elif row['Title'] == 'Officer':
                    return 41.5
    
            elif row['Sex']=='male' and row['Pclass'] == 3:
                if row['Title'] == 'Master':
                    return 6
                elif row['Title'] == 'Mr':
                    return 26
    
    dataset.Age = dataset.apply(lambda r: fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    dataset.Cabin = dataset.apply(lambda r: 'U' if pd.isnull(r['Cabin']) else r['Cabin'][0], axis = 1)
    
    X = dataset.iloc[:,[0,1,2,3,4,5,8,10]].values
    y = temp.values
    
    
    
    #Encoding cateorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_2 = LabelEncoder()
    X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
    labelencoder_X_6 = LabelEncoder()
    X[:,6] = labelencoder_X_6.fit_transform(X[:,6])
    labelencoder_X_7 = LabelEncoder()
    X[:,7] = labelencoder_X_7.fit_transform(X[:,7])
    onehotencoder = OneHotEncoder(categorical_features= [2])
    X = onehotencoder.fit_transform(X).toarray()
    
    onehotencoder = OneHotEncoder(categorical_features= [7])
    X = onehotencoder.fit_transform(X).toarray()
    
    onehotencoder = OneHotEncoder(categorical_features= [16])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:,1:]
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X[0:16,17:20] = sc.fit_transform(X[0:16,17:20])
    train = X[0:891]
    test = X[891:]
    return train, test, y
X_train,X_test, Y = prepare_data('train.csv','test.csv')

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_features='sqrt')
classifier = classifier.fit(X_train,Y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
parameters ={
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [200,210,240,250],
                 'criterion': ['gini','entropy']
                 }
cross_validation = StratifiedKFold(Y, n_folds=5)
grid_search = GridSearchCV(classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = cross_validation,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, Y)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Predict again
y_pred = grid_search.predict(X_test)

#Prepare test results to submit
result = pd.DataFrame()
result['PassengerId'] = pd.read_csv('test.csv')['PassengerId']
result['Survived'] = y_pred
result[['PassengerId','Survived']].to_csv('predictions.csv',index=False)

