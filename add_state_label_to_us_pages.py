# %%
import mysql.connector
import pandas as pd


# %%
# Initialize mysql connector
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="0000",
  database="pagenet"
)
mycursor = mydb.cursor()

# %%
# Add label indice to table state_label.
query = "select state from state_label;"
mycursor.execute(query)
result = mycursor.fetchall()
count = 0
for t in result:
    # print(t) ('Alabama',)
    sql = "update state_label set label={} where state='{}';".format(count,t[0])
    mycursor.execute(sql)
    count += 1
mydb.commit()
print(count, "record updated.")

# %%
# Store {state:label} in state_dict
sql = "select * from state_label;"
mycursor.execute(sql)
result = mycursor.fetchall()
state_dict = {}
for t in result:
    state, label = t
    state_dict[state] = label
# print(state_dict)

# %%
# Add state label indice to table us_pages
sql = "select idx, state from us_pages where state is not Null;"
mycursor.execute(sql)
result = mycursor.fetchall()
count = 0
for t in result:
    idx, state = t
    label = state_dict[state]
    sql = "update us_pages set label={} where idx={};".format(label, idx)
    mycursor.execute(sql)
    count += 1
    if count % 100000 == 0:
        mydb.commit()
        print(count, "record updated.")
mydb.commit()
print(count, "record updated.")

# %%
