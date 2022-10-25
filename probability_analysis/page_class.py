# %%
# import lib
from functools import total_ordering
from unittest import result

#%%
# define class Page and County
@total_ordering
class Page:
    def __init__(self, idx, id, city, mask, label, label_state, pred, pred_state, threshold, counts, states, probs) -> None:
        self.idx = idx
        self.id = id
        self.original_labeled = mask
        self.label = label
        self.label_state = label_state
        
        self.city = city
        self.city_population = 0
        self.city_density = 0
        self.county =  ""
        self.county_fips = 0
        
        self.prediction = pred
        self.predicted_state = pred_state
        self.jenks_break_threshold = threshold
        self.possible_counts = counts
        self.possible_states = []
        for i in range(len(states)):
            self.possible_states.append((states[i], probs[i]))
        self.possible_states.sort(key = lambda x: -x[1])
        
    def __eq__(self, other: object) -> bool:
        return self.possible_counts == other.possible_counts

    def __lt__(self, other: object) -> bool:
        return self.possible_counts < other.possible_counts

    # def possible_counts(self) -> int:
    #     return self.possible_counts

    def __str__(self) -> str:
        output = (f"Page id: {self.id}, idx: {self.id}, city: {self.city}, original_labeled: {self.original_labeled}, label: {self.label}, "
                  f"label_state: {self.label_state}, prediction: {self.prediction}, predicted_state: {self.predicted_state}, "
                  f"jenks_break_threshold: {self.jenks_break_threshold}, count of possible states: {self.possible_counts}, ") 
        states_list = "POSSIBLE STATES:"
        for s, p in self.possible_states:
            states_list = states_list + f' {s}: {p},'
        output = output + states_list[0:-1]
        return output

    def __repr__(self) -> str:
        output = (f"Page id: {self.id}, idx: {self.id}, original_labeled: {self.original_labeled}, label: {self.label}, "
                  f"label_state: {self.label_state}, prediction: {self.prediction}, predicted_state: {self.predicted_state}, "
                  f"jenks_break_threshold: {self.jenks_break_threshold}, count of possible states: {self.possible_counts}, ") 
        states_list = "POSSIBLE STATES:"
        for s, p in self.possible_states:
            states_list = states_list + f' {s}: {p},'
        output = "\n" + output + states_list[0:-1]
        return output

class County:

    def __init__(self, county_name, county_fips, state, on_state_boarder,
            population, white, black, ameri_es, asian, hawn_pi, hispanic, other, mult_race):
        self.name = county_name
        self.county_fips = county_fips


#%% 
# load county data in the value list, column name in the header map.
import csv
c = 0
is_header = True
county_headers = {}
counties_fips = {}
countyfilename = '../data/raw_dir/USA_Counties.csv'
with open(countyfilename, 'r',encoding='utf-8-sig') as countyfile:
    countyreader = csv.reader(countyfile)
    for row in countyreader:
        # load header
        if is_header:
            header_idx = 0
            
            for r in row:
                county_headers[r] = header_idx
                header_idx += 1
            is_header = False
            print("county headers", county_headers)
            continue

        # store key value pair
        key = row[county_headers['FIPS']]
        counties_fips[key] = row

print(list(counties_fips.items())[0])


#%% 
# load city data in the value list, column name in the header map.
import csv
is_header = True
city_headers = {}
cities = {}
cityfilename = '../data/raw_dir/uscities.csv'
with open(cityfilename, 'r') as cityfile:
    cityreader = csv.reader(cityfile)
    for row in cityreader:
        # load header
        if is_header:
            header_idx = 0
            for r in row:
                city_headers[r] = header_idx
                header_idx += 1
            is_header = False
            print("city headers", city_headers)
            continue
        # print(row)
        
        # store key value pair
        if row[city_headers['state_id']] == 'DC':
            row[city_headers['state_name']] = "Washington D.C."

        key = (row[city_headers['city']],row[city_headers['state_name']])
        if key in cities:
            if cities[key][city_headers['population']] < row[city_headers['population']]:
                cities[key] = row
                # print(row)
        else:
            cities[key] = row

print(list(cities.items())[1])

#%%
# Load page data from csv file, add city and county data into pages.

import mysql.connector
pages = []
pages_dict = {}

mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="0000",
  database="pagenet"
)
mycursor = mydb.cursor()

cities_notfound = []
cities_found = []
inputfile = '../model/saint_all_label/idx_id_mask_label_state_pred_state_threshold_counts_possible_states.csv'
with open(inputfile, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:

        # page data
        idx = int(row[0])
        id = int(row[1])
        mask = int(row[2])
        label = int(row[3])
        label_state = str(row[4])
        pred = int(row[5])
        pred_state = str(row[6])
        threshold = float(row[7])
        counts = int(row[8])
        states = []
        probs = []
        for i in range(counts):
            states.append(row[9+i*2])
            probs.append(float(row[9+i*2+1]))

        # get city ground truth from mysql database
        sql = f"select city from us_lgc_pages where id={row[1]};"
        mycursor.execute(sql)
        result = mycursor.fetchall()
        # print(result, result[0][0])
        city = result[0][0]

        key = (city, label_state)
        
        if key in cities:
            cities_found.append(key)
        else:
            cities_notfound.append(key)

        page = Page(idx, id, city, mask, label, label_state, pred, pred_state, threshold, counts, states, probs)
        pages.append(page)
        pages_dict[id] = page

print(len(cities_found), len(cities_notfound))

#%%
print(cities_notfound[0:100])
# %%
sorted_pages = sorted(pages, reverse=True)

#%%
# Get count for pages with different number of possible states.
# 1 5111697
# 2 499577
# 3 155095
# 4 64826
# 5 26434
# 6 9811
# 7 3502
# 8 1421
# 9 565
# 10 263
# 11 101
# 12 55
# 13 29
# 14 9
# 15 4
# 16 2
# 17 2
# 18 2
possible_counts_pages = {}
for i in range(18):
    possible_counts_pages[i+1] = []

for page in sorted_pages:
    possible_counts_pages[page.possible_counts].append(page)

# Inter state scale?? Globalness scale?
for k in range(18):
    print(k+1, len(possible_counts_pages[k+1]))

# %%
# Load city names for pages
file_name = "/home/jerry/Documents/backup_data/page_basic_backup"
with open(file_name, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        if row[0] in pages_dict:
            pages_dict[row[0]].city = row[3]
# %%

