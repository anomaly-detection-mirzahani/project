import numpy as np
import pandas as pd
import seaborn as sns
import os
import json

import env

url = env.get_db_url('curriculum_logs')

sql_1 = '''
SELECT l.date, l.time, l.user_id, c.id, c.name, c.program_id, \
    c.start_date, c.created_at, c.end_date, l.ip, l.path
FROM logs l
LEFT JOIN cohorts c ON (c.id = l.cohort_id)
'''

df1 = pd.read_sql(sql_1, url)

df1.to_csv('data/logs1.csv', index=False)

df1.rename({'path':'endpoint', 'user_id':'user', 'id':'cohort_id', 'name':'cohort'}, axis=1, inplace=True)

coh_names = df1[['cohort_id', 'cohort']]
ci = coh_names.cohort_id.unique()
cn = coh_names.cohort.unique()
coh_dict = dict(zip(ci, cn))

with open("data/cohorts.json", "w") as outfile:
    json.dump(coh_dict, outfile)

end_dates = {}
for c in df1.cohort.unique():
    print(c)
    unique_dates = df1[df1.cohort == c].end_date.unique()
    if c == None:
        continue
    else:
        d = df1[df1.cohort == c].end_date.unique()[0]
        print(d)
        end_dates[c] = d

with open("data/end_dates.json", "w") as outfile:
    json.dump(end_dates, outfile)
