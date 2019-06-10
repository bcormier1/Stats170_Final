import matplotlib.pyplot as plt
import urllib.request
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import time
import requests
import json
import datetime
from collections import defaultdict
from dateutil.relativedelta import *
import re
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


'''
This function takes a dataframe that has each row as one quarter of data
The number of deatures is irrelevant as long as its consistent.
Returns a dataframe with "quarters" of data side by side labeled feature1- featuren
'''
def reformat(df, quarters):
    #gets each row to contain x quarters of data by combining x rows into 1
    toSplit = []
    for ticker in df.ticker.unique():
        toAdd = []
        added = 0
        subdf = df[ticker == df.ticker]
        i = 0
        while i < subdf.shape[0]:
            toAdd.extend(list(subdf.iloc[i]))
            added += 1
            if added == quarters:
                toSplit.append(toAdd)
                toAdd = []
                added = 0
            i += 1
    newFrame = pd.DataFrame(toSplit)

    #Rename all columns w/ quarter
    mylist = list(df.columns) * quarters
    counts = Counter(mylist) # so we have: {'name':3, 'state':1, 'city':1, 'zip':2}
    for s,num in counts.items():
        if num > 1: # ignore strings that only appear once
            for suffix in reversed(range(1, num + 1)): # suffix starts at 1 and increases by 1 each time
                mylist[mylist.index(s)] = str(s) + str(suffix) # replace each appearance of s
    newFrame.columns = mylist
    return newFrame

'''
If you already have a datafile with the stock data, this will just grab that
If not, it goes into web scraping to pull all the stocks it can from the list of 
SP500 on wikipedia
'''
def getStockData():
    #Note: Changes were made to make this run locally without database access. You'd need to set up an engine
    #to use with the commented code to make this work
    try:
        return pd.read_csv("mainStockFrame.csv")
    except:
        print("Downloading data! Please wait.")
    data = (pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'))
    data[0].Symbol
    bigdf = pd.DataFrame(columns = pd.read_csv("Fundamentals.csv").columns)
    tickset = data[0].Symbol
    print(len(tickset))
    for t in tickset:
        try:
            newData = pd.read_csv("http://www.stockpup.com/data/{}_quarterly_financial_data.csv".format(t)).replace("None", -1)
            newData["ticker"] = t
            #Instead of a concat, we would write this frame to a database called sp500fundamentals via 
            #df.to_sql('sp500fundamentals', con=engine, if_exists = "append")
            #You would also connect your engine obviously to whatever postgres db you have locally
            bigdf = pd.concat([bigdf, newData])
        except:
            continue
            print("Failed on: " + t)

    #Once finished with writing to database, we want to pull from it
    #pd.read_sql_query(SELECT * FROM sp500fundamentals, con = engine)
    return bigdf


'''
Adds the percent price change Q to Q as well as the target variable of last year price change
'''
def addPriceChange(data, quarters):
    for i in range(2,quarters + 1):
        # percent change
        print()
        data['PriceChange{}/{}'.format(i-1,i)] = data['Price{}'.format(i)].astype('float') / data['Price{}'.format(i-1)].astype('float') - 1

    data['PriceChange{}/{}'.format(quarters-4, quarters)] = data['Price{}'.format(quarters)].astype('float')/data['Price{}'.format(quarters-4)].astype('float') - 1
    return data

'''
Adds YOY earning growth NOT Q/Q earnings growth.
'''
def addEarningsGrowth(data, quarters):
    for i in range(5, quarters + 1):
        data['EarningsGrowth{}/{}'.format(i, i - 4)] = data['Earnings{}'.format(i)].astype('float') / data['Earnings{}'.format(i - 4)].astype('float') - 1
    return data

'''
Returns a dict where key = year and val = gdp
'''
def getGDPdict():
    dict = {}
    df = pd.read_csv("GDP.csv")
    for index, row in df.iterrows():
        dict[str(row['date'])[0:-2]] = row['change-current']
    #Data ended in 2015, we found these values from various resources to impute
    dict["2016"] = 1.567
    dict["2017"] = 2.217
    dict["2018"] = 2.9
    dict["2019"] = 3.2
    return dict

#Joins the bond rate file with df
def addBondRates(df):
    bondRates = pd.read_csv("FRB_H15.csv")
    return df.merge(bondRates, left_on='Quarter end', right_on='Series Description').replace("ND", np.nan)

#Uses gdpdict to add gdp's to stocks
def combine(stocks, gdp):
    GDPCol = []
    for index, row in stocks.iterrows():
        try:
            year = str(row["Quarter end"])[0:4]
            GDPCol.append(gdp[year])
        except:
            print("No GDP data for: " + year)
    stocks["GDP"] = GDPCol
    return stocks

'''
If you dont give a year, this randomly splits the data to train and test data
If you do give a year it considers all past data before year training and 
any data with q1 in year as testing

NOTE: The y lists are not lists they are dataframes (for easy merging later. Y["Actual"] is the actual target
'''

def train_test(df, seed, quarters, year = None):
    df = df.dropna(axis = 1)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep]
    columnList = []
    y = df[["ticker1", 'PriceChange{}/{}'.format(quarters-4, quarters)]]

    for i in df.select_dtypes(include=['int64', 'float64']):
        if i == 'PriceChange{}/{}'.format(quarters-4, quarters):
            y["Actual"] = pd.qcut(df[i],4,labels=[1,2,3,4])
        elif "Unnamed" not in i:
            toAdd = True
            for j in range(quarters - 4, quarters + 1):
                if str(j) in i:
                    toAdd = False
            if toAdd:
                columnList.append(i)
    X = df[columnList]

    if year == None:
        sss = ShuffleSplit(n_splits=1, test_size=0.33, random_state = seed)
        sss.get_n_splits(X, y)
        train_index, test_index = next(sss.split(X, y))

    else:
        i = 0
        test_index = []
        train_index = []
        for index, row in df.iterrows():
            if (int(row["Quarter end1"][0:4]) == year):
                test_index.append(i)
            elif (int(row["Quarter end1"][0:4]) < year):
                train_index.append(i)
            i += 1

    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, X_test, y_train, y_test, train_index, test_index, df


#Given a model this will test it and return our statistics and a frame with predicitons
def getResults(df, X_train, X_test, y_train, y_test, model, scorer, test_indices, quarters):
    #model.fit(X_train, y_train["Actual"])
    y_preds_proba = pd.DataFrame(model.predict_proba(X_test))
    y_preds_proba.columns =[1,2,3,4]
    subDf = df.iloc[test_indices, :]
    y_preds = model.predict(X_test)
    finaldf = pd.DataFrame()
    finaldf = X_test.reset_index(drop=True).merge(pd.DataFrame(model.predict_proba(X_test)).reset_index(drop=True), left_index=True, right_index=True)
    finaldf = finaldf.merge(y_test.reset_index(drop = True), left_index=True, right_index=True)
    finaldf["Pred"] = y_preds
    return (finaldf, scorer(y_test["Actual"], y_preds))


#Returns accuracy by class, but only works for nclasses<=4
def accuracy_by_class(results):
    trues = {1:0,2:0,3:0,4:0}
    falses = {1:0,2:0,3:0,4:0}

    for i,row in results.iterrows():
        if row['Pred'] == row['Actual']:
            trues[row['Actual']] += 1
        else:
            falses[row['Pred']] += 1

    for key in trues.keys():
        if trues[key] == 0 and falses[key] == 0:
            falses[key] = 1
    precision = {1:(trues[1]/(trues[1]+falses[1])),2:(trues[2]/(trues[2]+falses[2])),3:(trues[3]/(trues[3]+falses[3])),
                4:(trues[4]/(trues[4]+falses[4]))}
    return(precision)

#Returns mean of all targets and the mean of stocks chosen by the model according to the criteria described in report
def play(df, thresholdLow, thresholdHigh, quarters, weight = False):
    usedRows = []
    for index, row in df.iterrows():
        if row[3] > thresholdHigh and row[0] < thresholdLow:
            usedRows.append(row['PriceChange{}/{}'.format(quarters-4, quarters)])
    return (np.mean(usedRows), np.mean([row['PriceChange{}/{}'.format(quarters-4, quarters)] for index, row in df.iterrows()]))
