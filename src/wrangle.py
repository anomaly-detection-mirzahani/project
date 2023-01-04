import pandas as pd
import numpy as np
import json
import os

filename = 'data/clean_data.csv'


########## FUNCTION TO MODIFY DATAFRAME ROWS #######
def return_first_element(l: list) -> str:
    '''
    Checks if the parameter is a list, 
    extracts the first value and returns it
    
    Parameters:
        l: list of strings
    Returns:
        string
    '''
    if type(l) == list:
        return l[0]


####### Acquire and clean data ########

# read csv file
df = pd.read_table("data/anonymized-curriculum-access.txt", sep = '\s', header = None, engine='python',
                   names = ['date', 'time', 'page', 'id', 'cohort', 'ip'])

# read json file and save it to the dictionary
with open("data/cohorts.json", "r") as json_file:
    cohorts_dict = json.load(json_file)

# convert keys into numbers
cohorts_dict = {float(k):v for k,v in cohorts_dict.items()}

# create a column that holds the names of cohorts
df['cohort_name'] = df.cohort.map(cohorts_dict)

# set the name of the main page as index
df.page = np.where(df.page == '/', 'index', df.page)

# one null value in the column 'page' replace with 'index'
df.page = df.page.fillna('index')
# handle the null values in columns cohort and cohort_name
df.cohort = np.where(df.cohort.isnull(), 0, df.cohort)
df.cohort_name = np.where(df.cohort_name.isnull(), 'No Name', df.cohort_name)

# incorrect web address replaced
# df.loc[178330, 'page'] = 'cohorts/26/grades.csv'
# df[df.topic.str.contains('%20https:')] -> weird, user 580

# create a columns 'lesson' 
# splits 'page' into 2 parts. 1st - topic, 2nd - chapter of the lesson (here:lesson)
df['topic'] = df.page.str.split('/',1, expand=True)[0]
df['lesson'] = df.page.str.split('/',1, expand=True)[1]
# split lesson again
# as a result we get a list of 2 values
df.lesson = df.lesson.str.split('/', 1)
# apply the function that saves only the first value of the list as a string
df.lesson = df.lesson.apply(return_first_element)
#df.lesson = np.where(df.lesson.isnull(), df.topic, df.lesson)


# Yuvia's solution
#lessons = pd.read_csv('data/lessons.csv', index_col='Unnamed: 0')
#lessons.Topic = np.where(lessons.Topic.isnull(), 'index', lessons.Topic)
#lessons.Lesson = np.where(lessons.Lesson.isnull(), lessons.Topic, lessons.Lesson)

# add Yuvia's lessons
#df = pd.concat([df, lessons], axis=1)

df['full_lesson_name'] = np.where(df.lesson.isnull(), df.topic, df.topic + ' ' + df.lesson)
df.lesson = np.where(df.lesson.isnull(), df.topic, df.lesson)

# create a dictionary with data science cohorts, staff and no name cohorts
field = {'Bayes':'DS', 'Curie':'DS', 'Darden':'DS', 'Florence':'DS', 'Easley':'DS', \
         'Mirzakhani':'DS', 'Staff':'Staff', 'No Name':'No Name'}
# create a column 'field' with values: DS, Staff, No Name
df['field'] = df.cohort_name.map(field)
# fill null values with WD -> web developers
df.field = df.field.fillna('WebDev')

# create a column date_time that holds date and time together
df.insert(0,'date_time', df.date + ' ' + df.time)
# delete the column time
try:
    del df['time']
except KeyError:
    print('The column doesn\'t exist')

# read json file and save it to the dictionary
with open("data/end_dates.json", "r") as json_file1:
    dates_dict = json.load(json_file1)
df['end_date'] = df.cohort_name.map(dates_dict)


df.to_csv(filename)

#### FUNCTIONS ######

def change_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Changes floats to integers, date type to datetime format,
    objects with small number of unique values to categories

    Parameters:
        df: pd.DataFrame
            - cleaned dataframe with logs
    Returns:
        df with new data types
    '''
    # change data types
    # integer below 255
    df.cohort = df.cohort.astype('uint8')
    # dates
    #df.end_date = pd.to_datetime(df.end_date)
    for col in ['date_time', 'date', 'end_date']:
        df[col] = pd.to_datetime(df[col])
    # categories
    for col in ['cohort_name', 'field']:
        df[col] = pd.Categorical(df[col])
    
    return df

####### Return clean data
def get_logs1():
    '''
    returns dataframe
    '''
    return change_dtypes(df)

def get_logs():
    if os.path.isfile(filename):
        df1 = pd.read_csv(filename)
        df1 = change_dtypes(df)
    else:
        df1 = get_logs1()
        df1.to_csv(filename)
    return df1