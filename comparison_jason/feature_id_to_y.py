# %%
import mysql.connector
import pandas as pd
import numpy as np

# %%

id = pd.read_csv('/home/jerry/Documents/country_gnn/comparison_jason/features_id.csv', index_col=0, dtype=(long))
y = np.zeros(id.shape)

# %%
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="0000",
  database="pagenet"
)
mycursor = mydb.cursor()
count = 0
for i in range(y.shape[0]):
    page_id = id.iloc[i].values.tolist()[0]
    sql = 'select 51_label from us_lgc_pages where id={};'.format(page_id)
    mycursor.execute(sql)
    print(page_id, mycursor.fetchall())
    count += 1
    if count % 100000 == 0:
        print(count, "record queried.")
mydb.commit()
print(count, "record queried.")
# %%
