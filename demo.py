# -*- coding: utf-8 -*-

# To evaluate time performance
import time as time
# Required Python Machine learning Packages
import pandas as pd
# For preprocessing the data
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To model the decision tree classifier
from sklearn.tree import DecisionTreeClassifier
# To model the NN classifier
from sklearn.neural_network import MLPClassifier
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
# To calculate the precision, recall and fscore of the model
from sklearn.metrics import precision_recall_fscore_support
# For NN stats, import report util
from sklearn.metrics import classification_report


def stats_output(test, predict, fitTime, predTime, name):
    print "****************************************"
    type = 0
    precisions, recalls, fscores, _ = precision_recall_fscore_support(test, predict)
    print "**Evaluate performance of algorithm %s**" % name
    print "accuracy score is %.4f" % accuracy_score(target_test, target_pred, normalize=True)
    for (precision, recall, fscore) in zip(precisions, recalls, fscores):
        print ""
        print "precision of class %d is %.4f" % (type, precision)
        print "recall of class %d is %.4f" % (type, recall)
        print "fscore of class %d is %.4f" % (type, fscore)
        type += 1
    print ""
    print "(1)Fit stage elapse time is %.5f" % fitTime
    print "(2)Prediction stage elapse time is %.5f" % predTime
    print "****************************************\n\n"

if __name__ == "__main__":
    # print "this is test"
    adult_df = pd.read_csv('adult.data',
                           header=None, delimiter=' *, *', engine='python')
    adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                        'marital_status', 'occupation', 'relationship',
                        'race', 'sex', 'capital_gain', 'capital_loss',
                        'hours_per_week', 'native_country', 'income']

    adult_df_rev = adult_df
    for value in ['workclass', 'education',
                  'marital_status', 'occupation',
                  'relationship', 'race', 'sex',
                  'native_country', 'income']:
        adult_df_rev[value].replace(['?'], [adult_df_rev.describe(include='all')[value][2]],
                                    inplace=True)

    le = preprocessing.LabelEncoder()
    workclass_cat = le.fit_transform(adult_df.workclass)
    education_cat = le.fit_transform(adult_df.education)
    marital_cat = le.fit_transform(adult_df.marital_status)
    occupation_cat = le.fit_transform(adult_df.occupation)
    relationship_cat = le.fit_transform(adult_df.relationship)
    race_cat = le.fit_transform(adult_df.race)
    sex_cat = le.fit_transform(adult_df.sex)
    native_country_cat = le.fit_transform(adult_df.native_country)

    # initialize the encoded categorical columns
    adult_df_rev['workclass_cat'] = workclass_cat
    adult_df_rev['education_cat'] = education_cat
    adult_df_rev['marital_cat'] = marital_cat
    adult_df_rev['occupation_cat'] = occupation_cat
    adult_df_rev['relationship_cat'] = relationship_cat
    adult_df_rev['race_cat'] = race_cat
    adult_df_rev['sex_cat'] = sex_cat
    adult_df_rev['native_country_cat'] = native_country_cat

    # drop the old categorical columns from dataframe
    dummy_fields = ['workclass', 'education', 'marital_status',
                    'occupation', 'relationship', 'race',
                    'sex', 'native_country']
    adult_df_rev = adult_df_rev.drop(dummy_fields, axis=1)

    adult_df_rev = adult_df_rev.reindex_axis(['age', 'workclass_cat', 'fnlwgt', 'education_cat',
                                              'education_num', 'marital_cat', 'occupation_cat',
                                              'relationship_cat', 'race_cat', 'sex_cat', 'capital_gain',
                                              'capital_loss', 'hours_per_week', 'native_country_cat',
                                              'income'], axis=1)
    features = adult_df_rev.values[:, :14]
    target = adult_df_rev.values[:, 14]
    features_train, features_test, target_train, target_test = train_test_split(features,
                                                                                target, test_size=0.33, random_state=10)

    # sklearn gaussian bayes start
    clf = GaussianNB()

    st = time.time()
    clf.fit(features_train, target_train)
    fitTime = time.time() - st

    st = time.time()
    target_pred = clf.predict(features_test)
    predTime = time.time() - st

    stats_output(target_test, target_pred, fitTime, predTime, "GAUSSIAN BAYES(SKLEARN)")
    # sklearn gaussian bayes end

    # sklearn decision tree start
    clf = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                                 max_depth=3, min_samples_leaf=5)

    st = time.time()
    clf.fit(features_train, target_train)
    fitTime = time.time() - st

    st = time.time()
    target_pred = clf.predict(features_test)
    predTime = time.time() - st

    stats_output(target_test, target_pred, fitTime, predTime, "DECISION TREE(SKLEARN)")
    # sklearn gaussian bayes end
    clf = MLPClassifier(hidden_layer_sizes=(1,))

    st = time.time()
    clf.fit(features_train, target_train)
    fitTime = time.time() - st

    st = time.time()
    target_pred = clf.predict(features_test)
    predTime = time.time() - st

    stats_output(target_test, target_pred, fitTime, predTime, "NEURAL NETWORK(SKLEARN)")
    exit()
