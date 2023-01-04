import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from importlib import reload

# DBSCAN import
from sklearn.cluster import DBSCAN
# Scaler import
from sklearn.preprocessing import MinMaxScaler

# import modules
import src.wrangle as wr

# define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

df = wr.get_logs()
ds = df[df.field == 'DS']
wd = df[df.field == 'WebDev']


####### QUESTION 1
def get_top_lessons(df: pd.DataFrame, name:str='total', i:int=5, viz:bool=False):
    '''
    Finds i most popular lessons in a dataframe
    
    Parameters:
        df: logs dataframe
        name: name of the dataframe
        i: number of rows to return
    Returns:
        data frame
    '''
    # series with groupby results
    s = df[~(df.full_lesson_name == 'index')].groupby('full_lesson_name').page.count().sort_values(ascending=False).head(i)
    if viz:
        popular = s.reset_index()
        popular['field'] = name
    else:
        popular= s.reset_index().rename({'lesson':name, 'page':name+'_page'}, axis=1)
    return popular

def top_lessons_df():
    '''
    displays most popular lessons as data frame
    '''
    popular_lessons = \
        pd.concat([get_top_lessons(df, 'total', 5), get_top_lessons(ds, 'DS', 5), get_top_lessons(wd, 'WD', 5)], axis=1)
    display(popular_lessons)

def viz_top_lessons():
    '''
    vizualizes top lessons
    '''
    popular_lessons2 = pd.concat([ #get_top_lessons(f, 'total', 5, viz=True), \
                              get_top_lessons(ds, 'DS', 5, viz=True),\
                              get_top_lessons(wd, 'WD', 5, viz=True)],\
                              axis=0, ignore_index=True)
    plt.figure(figsize=(20, 5))
    ax = sns.barplot(data=popular_lessons2, x='full_lesson_name', y='page', hue='field')
    plt.xticks(rotation=45)
    plt.show()

def viz_top_lessons_dev():
    popular_lessons2 = get_top_lessons(wd, 'WD', 5, viz=True)
                              
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=popular_lessons2, x='full_lesson_name', y='page')
    plt.xticks(rotation=45)
    plt.show()

def viz_top_lessons_ds():
    popular_lessons2 = get_top_lessons(ds, 'DS', 6, viz=True)[1:]
                              
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=popular_lessons2, x='full_lesson_name', y='page')
    plt.xticks(rotation=45)
    plt.show()

####### QUESTION 2
def cohort_ds_lesson():
    count_lessons_ds = ds[~(ds.full_lesson_name=='index')].groupby(['full_lesson_name','cohort_name']).\
        page.count().sort_values(ascending=False).reset_index()
    count_topics_ds = ds[~(ds.full_lesson_name=='index')].groupby(['topic','cohort_name']).\
        page.count().sort_values(ascending=False).reset_index()
    print('DATA SCIENCE COHORTS')
    print('--------------------')
    print('FULL LESSON NAME')
    display(count_lessons_ds.head(5))
    print()    
    
    print('CHAPTER NAME')
    display(count_topics_ds.head(5))
    print()

def cohort_wd_lesson():
    count_lessons_wd = wd[~(wd.full_lesson_name=='index')].groupby(['full_lesson_name','cohort_name']).\
        page.count().sort_values(ascending=False).reset_index()
    count_topics_wd = wd[~(wd.full_lesson_name=='index')].groupby(['topic','cohort_name']).\
        page.count().sort_values(ascending=False).reset_index()
    print('WEB DEV COHORTS')
    print('--------------------')
    print('FULL LESSON NAME')
    display(count_lessons_wd.head(5))
    print()    
    
    print('CHAPTER NAME')
    display(count_topics_wd.head(5))
    print()

####### QUESTION 4
def compute_pct_b(pages_per_user: pd.Series, span: int, weight: float, user: int) -> pd.DataFrame:
    '''
    This function adds the %b of a bollinger band range for the page views of a single user's log activity
    
    Parameters:
        pages_per_user: pandas Series
            - contains information how many pages the user visited per time perido
        span: int 
            - number time periods(days, weeks etc) to compute exponential moving average
        weight: float
            - number of standrard deviations to compute anomalities
        user: int
            - user id
    
    Returns:
        Data Frame with following columns:
        pagers_per_user, midband, stdev, upper bound, lower bound, %b and user info
    '''
    # compute midband
    midband = pages_per_user.ewm(span=span).mean()

    # compute exponential stdev
    stdev = pages_per_user.ewm(span=span).std()

    # compute upper and lower bands
    ub = midband + stdev*weight
    lb = midband - stdev*weight
    bb = pd.DataFrame({'pages':pages_per_user, 'midband':midband, 'stdev':stdev, 'ub':ub, 'lb':lb})
    # compute %b
    bb['pct_b'] = (bb['pages'] - bb['lb'])/(bb['ub'] - bb['lb'])
    bb['user'] = user
    return bb
def one_user_df_prep(df: pd.DataFrame, user: int) -> pd.Series:
    '''
    This function returns a Series consisting of data for only a single defined user
    
    Parameters:
        df: DataFrame with all logs and user id numbers
        user: user id number
    '''
    df = df[df.id == user].copy()
    df.date = pd.to_datetime(df.date)
    df = df.set_index(df.date)
    pages_one_user = df['page'].resample('d').count()
    return pages_one_user

def find_anomalies(df, user, span, weight, plot=False):
    '''
    This function returns the records where a user's daily activity exceeded the upper limit of a bollinger band range
    '''
    
    # Reduce dataframe to represent a single user
    pages_one_user = one_user_df_prep(df, user)
    
    # Add bollinger band data to dataframe
    my_df = compute_pct_b(pages_one_user, span, weight, user)
    
    # Plot data if requested (plot=True)
    if plot:
        plot_bands(my_df, user)
    
    # Return only records that sit outside of bollinger band upper limit
    return my_df[my_df.pct_b>1]

def suspicios_ids(df):
    ts = df.set_index(df.date)
    pages = ts['page'].resample('d').count()
    anomalies = pd.DataFrame()
    for u in list(ts.id.unique()):
        user_df = find_anomalies(ts, u, 30, 2)
        anomalies = pd.concat([anomalies, user_df], axis=0)
    anomalies = anomalies.sort_values(by='pages', ascending=False)
    users = anomalies.iloc[0:5].user
    for u in users:
        suspicious_user = one_user_df_prep(df, u)
        suspicious_user.plot(figsize=(12,5), label=u)
        plt.title('Top 10 anomalies')
        plt.legend()
    plt.show()


####### QUESTION 6
def topic_reference_after(df):
    df0 = df[(df.date > df.end_date) & ~(df.full_lesson_name == 'index')]
    dfw = df0.loc[df0['field'] == 'WebDev']
    dfd = df0.loc[df0['field'] == 'DS']
    dfs = df0.loc[df0['field'] == 'Staff']
    df0.groupby('topic').page.count().sort_values(ascending=False).head()
    dfw.groupby('topic').page.count().sort_values(ascending=False).head()
    dfd.groupby('topic').page.count().sort_values(ascending=False).head()
    dfs.groupby('topic').page.count().sort_values(ascending=False).head()
    
    print('Data Science')
    display(dfd.topic.head(1))
    print()
    print('Web Dev')
    display(dfw.topic.head(1))
    print()


####### QUESTION 7

def get_bottom_lessons(df: pd.DataFrame, name:str='total', i:int=10, viz:bool=False):
    '''
    Finds least popular lessons in a dataframe
    Parameters:
        df: logs dataframe
        name: name of the dataframe
        i: number of rows to return
    Returns:
        data frame
    '''
    # series with groupby results
    s = df.groupby('full_lesson_name').page.count().sort_values(ascending=True).head(i)
    if viz:
        popular = s.reset_index()
        popular['field'] = name
    else:
        popular= s.reset_index().rename({'lesson':name, 'page':name+'_page'}, axis=1)
    return popular