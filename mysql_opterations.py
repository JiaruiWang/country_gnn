# %%
# import packages
import mysql.connector
import csv
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
# build state_str:index dict
state_order = '/home/jerry/Documents/jason_code/python code/state_labeler/adv/state_classes_DC.csv'
s2i = {}
with open(state_order, 'r') as file:
    lines = csv.reader(file)
    index = 0
    for line in lines:
        s2i[line[0]] = index
        index += 1
print(s2i)
# %%
# read data from table us_lgc_pages
sql = 'select new_idx, state from us_lgc_pages'
mycursor.execute(sql)
rows = mycursor.fetchall()
print(rows[0])
# %%
# update column 51_label in table us_lgc_pages
count = 0
for row in rows:
    new_idx = row[0]
    state_str = row[1]
    if state_str is None:
        state_str = 'California'
    sql = 'update us_lgc_pages set 51_label={} where new_idx={};'.format(s2i[state_str], new_idx)
    mycursor.execute(sql)
    count += 1
    if count % 100000 == 0:
        mydb.commit()
        print(count, 'rows updated.')
mydb.commit()
print(count, 'rows updated.')
# %%
