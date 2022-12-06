# %%
# import lib
from functools import total_ordering
from unittest import result

#%%
# define class Page and County
@total_ordering
class Page:
    def __init__(self, idx, id, city, name, category, likespage,
                 fan, outward, inward, mask, label, label_state, 
                 pred, pred_state, threshold, counts, states, probs) -> None:
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

        self.name = name
        self.category = category
        self.likespage = likespage
        self.fan = fan
        self.outward = outward
        self.inward = inward
        
        self.prediction = pred
        self.predicted_state = pred_state
        self.jenks_break_threshold = threshold
        self.possible_counts = counts
        self.possible_states = []
        for i in range(len(states)):
            self.possible_states.append((states[i], probs[i]))
        self.possible_states.sort(key = lambda x: -x[1])
        self.possible_states_tuple_key = tuple(states)
        self.tuple_key_only_centers = tuple(s for s in states if (s in [
                                    "California", "New York", "Florida", "Texas", "Illinois", "Pennsylvania"]))
        self.tuple_key_no_centers = tuple(s for s in states if (s not in [
                                    "California", "New York", "Florida", "Texas", "Illinois", "Pennsylvania"]))
        
    def __eq__(self, other: object) -> bool:
        return (self.possible_counts == other.possible_counts and 
                self.possible_states_tuple_key == other.possible_states_tuple_key and
                self.possible_states_tuple_key == other.possible_states_tuple_key and
                self.tuple_key_no_centers == other.tuple_key_no_centers and
                self.inward == other.inward and
                self.fan == other.fan)

    def __lt__(self, other: object) -> bool:
        # return self.possible_counts < other.possible_counts
        if self.possible_counts < other.possible_counts:
            return True
        elif self.possible_counts > other.possible_counts:
            return False
            
        if self.possible_states_tuple_key < other.possible_states_tuple_key:
            return True
        elif self.possible_states_tuple_key > other.possible_states_tuple_key:
            return False
        
        if self.tuple_key_only_centers < other.tuple_key_only_centers:
            return True
        elif self.tuple_key_only_centers > other.tuple_key_only_centers:
            return False

        if self.tuple_key_no_centers < other.tuple_key_no_centers:
            return True
        elif self.tuple_key_no_centers > other.tuple_key_no_centers:
            return False

        if self.fan < other.fan:
            return True
        elif self.fan > other.fan:
            return False

        if self.inward < other.inward:
            return True
        elif self.inward > other.inward:
            return False


        
        return False


    # def possible_counts(self) -> int:
    #     return self.possible_counts

    def __str__(self) -> str:
        output = (f"Page_id: {self.id}, idx: {self.id}, city: {self.city}, name: {self.name}, "
                  f"category: {self.category}, fan: {self.fan}, outward: {self.outward}, inward: {self.inward}, "
                  f"original_labeled: {self.original_labeled}, label: {self.label}, "
                  f"label_state: {self.label_state}, prediction: {self.prediction}, predicted_state: {self.predicted_state}, "
                  f"jenks_break_threshold: {self.jenks_break_threshold}, count_of_possible_states: {self.possible_counts}, ") 
        states_tuple_key = "Possible_state_tuple_key: ("
        for s in self.possible_states_tuple_key:
            states_tuple_key = states_tuple_key + f' {s},'
        output = output + states_tuple_key +'), '
        states_list = "POSSIBLE_STATES:"
        for s, p in self.possible_states:
            states_list = states_list + f' {s}: {p},'
        output = output + states_list[0:-1]
        return output

    def __repr__(self) -> str:
        output = (f"Page_id: {self.id}, idx: {self.id}, city: {self.city}, name: {self.name}, "
                  f"category: {self.category}, fan: {self.fan}, outward: {self.outward}, inward: {self.inward}, "
                  f"original_labeled: {self.original_labeled}, label: {self.label}, "
                  f"label_state: {self.label_state}, prediction: {self.prediction}, predicted_state: {self.predicted_state}, "
                  f"jenks_break_threshold: {self.jenks_break_threshold}, count_of_possible_states: {self.possible_counts}, ") 
        states_tuple_key = "Possible_state_tuple_key: ("
        for s in self.possible_states_tuple_key:
            states_tuple_key = states_tuple_key + f' {s},'
        output = output + states_tuple_key +'), '
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
countyfilename = '../../data/raw_dir/USA_Counties.csv'
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
print(len(list(counties_fips.items())))


#%% 
# load city data in the value list, column name in the header map.
import csv
is_header = True
city_headers = {}
cities = {}
cityfilename = '../../new_city_data/uscities.csv'
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
            print(row)

        key = (row[city_headers['city']].lower(),row[city_headers['state_name']].lower())
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
count_true_label = 0
cities_notfound = []
cities_found = []
inputfile = './id_label_state_pred_state_threshold_counts_possible_states.csv'
full_info = '/home/jerry/Documents/country_gnn/model/saint_all_label/us_lgc_page_id_name_category_city_likespage_fan_outward_inward.csv'

with open(inputfile, 'r') as file, open(full_info, 'r') as fullinfo:
    csvreader = csv.reader(file)
    inforeader = csv.reader(fullinfo, delimiter='\t')
    for (row, info) in zip(csvreader, inforeader):

        # page data
        info_id = int(info[0])
        name = str(info[1])
        category = str(info[2])
        city = str(info[3]).lower()
        likespage = int(info[4])
        fan = int(info[5])
        outward = int(info[6])
        inward = int(info[7])

        idx = 0
        id = int(row[0])

        if (id != info_id):
            print(id)
            break
        mask = 0
        label = int(row[1])

        # for data A
        if label == -1:
            continue
        else:
            count_true_label += 1
        label_state = str(row[2]).lower()
        pred = int(row[3])
        pred_state = str(row[4]).lower()
        threshold = float(row[5])
        counts = int(row[6])
        states = []
        probs = []
        for i in range(counts):
            states.append(row[7+i*2])
            probs.append(float(row[7+i*2+1]))

        page = Page(idx, id, city, name, category, likespage,
                 fan, outward, inward, mask, label, label_state, pred, pred_state, threshold, counts, states, probs)
        # truelabel -1 labelstate somehow is Washington D.C.
        key = (city, label_state)
        # need to use pred_state 
        # key = (city, pred_state)
        
        if key in cities:
            cities_found.append(key)
            page.city = cities[key][city_headers['city']]
            page.city_population = cities[key][city_headers['population']]
            page.city_density = cities[key][city_headers['density']]
            page.county =  cities[key][city_headers['county_name']]
            page.county_fips = cities[key][city_headers['county_fips']]
        else:
            cities_notfound.append(key)

        pages.append(page)
        pages_dict[id] = page

print(len(cities_found), len(cities_notfound))
print("count_true_label ", count_true_label)
#%%
print(cities_notfound[0:100])
# print(pages[0:100])

#%%
outputfile = './id_city_cityPopulation_county_countyFips.csv'
with open(outputfile, 'w') as file:
    csvwriter = csv.writer(file, delimiter='\t')
    for i in range(len(pages)):
        row = [pages[i].id, pages[i].city, pages[i].city_population,
               pages[i].county, pages[i].county_fips]
        csvwriter.writerow(row)

# %%
sorted_pages = sorted(pages, reverse=True)
#%%
# print(cities_notfound[0:100])
print(len(sorted_pages))
print(sorted_pages[0:10])
#%%
# Get count for pages with different number of possible states.
# 1 1934364
# 2 120582
# 3 37357
# 4 18959
# 5 10758
# 6 7319
# 7 4652
# 8 3437
# 9 2426
# 10 1691
# 11 1198
# 12 1512
# 13 587
# 14 469
# 15 644
# 16 688
# 17 150
# 18 190
# 19 28
# 20 224
# 21 80
# 22 16
# 23 16
# 24 0
# 25 2
# 26 5
# 27 1
# 28 0
# 29 0
# 30 5
# 31 39
possible_counts_pages = {}
for i in range(31):
    possible_counts_pages[i+1] = []

for page in sorted_pages:
    possible_counts_pages[page.possible_counts].append(page)

# Inter state scale?? Globalness scale?
for k in range(31):
    print(k+1, len(possible_counts_pages[k+1]))

#%%
# build maps for states tuple key
states_tuple_key= {}
for i in range(2, 32):
    # tuple key for each key length
    states_tuple_key[i] = {}
    for j in range(len(possible_counts_pages[i])):
        if (possible_counts_pages[i][j].tuple_key_only_centers,
            possible_counts_pages[i][j].tuple_key_no_centers) not in states_tuple_key[i]:
            states_tuple_key[i][(possible_counts_pages[i][j].tuple_key_only_centers,
                                 possible_counts_pages[i][j].tuple_key_no_centers)] = [possible_counts_pages[i][j]]
        else:
            states_tuple_key[i][(possible_counts_pages[i][j].tuple_key_only_centers,
                                 possible_counts_pages[i][j].tuple_key_no_centers)].append(possible_counts_pages[i][j])
#%%
# print unique key numbers for each key length in states tuple key

# 2 1140
# 3 4489
# 4 5398
# 5 4559
# 6 3407
# 7 2605
# 8 1854
# 9 1377
# 10 897
# 11 647
# 12 540
# 13 320
# 14 256
# 15 195
# 16 118
# 17 68
# 18 52
# 19 28
# 20 16
# 21 15
# 22 11
# 23 8
# 24 0
# 25 2
# 26 2
# 27 1
# 28 0
# 29 0
# 30 1
# 31 1
for k in range(2, 32):
    print(k, len(states_tuple_key[k]))

#%% 
# inspect tuple key map
outputfile = './tuple_keys_contains_pages_population_labeling_fan_order-6_centers_A_2147399.csv'
with open(outputfile, 'w') as file:
    csvwriter = csv.writer(file, delimiter='\t')
    for i in range(2, 32):
        # row = [i]
        key_list = list(states_tuple_key[i].keys())
        key_list.sort()
        for j in range(len(key_list)):
            tuple_page_list = states_tuple_key[i][key_list[j]]
            row = [i, key_list[j], len(tuple_page_list),
            tuple_page_list[0].id, tuple_page_list[0].fan, tuple_page_list[0].inward, tuple_page_list[0].outward]
            if (len(tuple_page_list)>1):
                row.extend([tuple_page_list[1].id, tuple_page_list[1].fan, tuple_page_list[1].inward, tuple_page_list[1].outward])
            if (len(tuple_page_list)>2):
                row.extend([tuple_page_list[2].id, tuple_page_list[2].fan, tuple_page_list[2].inward, tuple_page_list[2].outward])
            csvwriter.writerow(row)
# %%
# # Load city names for pages
# file_name = "/home/jerry/Documents/backup_data/page_basic_backup"
# with open(file_name, 'r') as file:
#     csvreader = csv.reader(file)
#     for row in csvreader:
#         if row[0] in pages_dict:
#             pages_dict[row[0]].city = row[3]
# %%
# count interstate page and intrastate page, total page for all states
total_page = [0 for i in range(51)]
interstate_page = [0 for i in range(51)]
intrastate_page = [0 for i in range(51)]

for i in range(len(possible_counts_pages[1])):
    label = possible_counts_pages[1][i].label
    total_page[label] += 1
    intrastate_page[label] += 1

for i in range(2, 32):
    for j in range(len(possible_counts_pages[i])):
        label = possible_counts_pages[i][j].label
        total_page[label] += 1
        interstate_page[label] += 1
#%%
print(sum(total_page))
print(sum(interstate_page))
print(sum(intrastate_page))
# 2147399
# 213035
# 1934364
#%%
print(interstate_page)
print(intrastate_page)
print(total_page)
#%%
interstate_page_over_total_page = []
intrastate_page_over_total_page = []
interstate_percentage = []
for i in range(51):
    interstate_page_over_total_page.append(interstate_page[i] / total_page[i])
    intrastate_page_over_total_page.append(intrastate_page[i] / total_page[i])
#%%
pl_class_names = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT",\
                  "DE", "FL", "GA", "HI", "ID", "IL", "IN",\
                  "IA", "KS", "KY", "LA", "ME", "MD", "MA",\
                  "MI", "MN", "MS", "MO", "MT", "NE", "NV",\
                  "NH", "NJ", "NM", "NY", "NC", "ND", "OH",\
                  "OK", "OR", "PA", "RI", "SC", "SD", "TN",\
                  "TX", "UT", "VT", "VA", "WA", "WV", "WI",\
                  "WY", "DC"]
for i in range(51):
    interstate_percentage.append(interstate_page_over_total_page[i]*100)
    print(f'{pl_class_names[i]}  {intrastate_page_over_total_page[i]*100:.1f}& {interstate_page_over_total_page[i]*100:.1f}&')
# AL  90.2& 9.8&
# AK  92.7& 7.3&
# AZ  92.1& 7.9&
# AR  88.3& 11.7&
# CA  91.9& 8.1&
# CO  91.4& 8.6&
# CT  84.1& 15.9&
# DE  86.9& 13.1&
# FL  92.4& 7.6&
# GA  89.1& 10.9&
# HI  92.3& 7.7&
# ID  92.7& 7.3&
# IL  90.7& 9.3&
# IN  90.6& 9.4&
# IA  88.8& 11.2&
# KS  90.9& 9.1&
# KY  87.1& 12.9&
# LA  92.2& 7.8&
# ME  84.6& 15.4&
# MD  79.8& 20.2&
# MA  85.7& 14.3&
# MI  92.8& 7.2&
# MN  92.0& 8.0&
# MS  89.5& 10.5&
# MO  82.8& 17.2&
# MT  93.7& 6.3&
# NE  87.5& 12.5&
# NV  83.0& 17.0&
# NH  83.1& 16.9&
# NJ  86.6& 13.4&
# NM  92.3& 7.7&
# NY  88.1& 11.9&
# NC  89.1& 10.9&
# ND  89.3& 10.7&
# OH  89.4& 10.6&
# OK  90.6& 9.4&
# OR  90.3& 9.7&
# PA  89.3& 10.7&
# RI  85.0& 15.0&
# SC  89.5& 10.5&
# SD  90.5& 9.5&
# TN  88.8& 11.2&
# TX  90.6& 9.4&
# UT  89.7& 10.3&
# VT  83.6& 16.4&
# VA  87.7& 12.3&
# WA  93.0& 7.0&
# WV  80.6& 19.4&
# WI  92.8& 7.2&
# WY  91.2& 8.8&
# DC  72.1& 27.9&
# %%
import matplotlib.pyplot as plt
import geopandas
# %%
states = geopandas.read_file('../../geopandas-tutorial-master/data/usa-states-census-2014.shp')
print(type(states))
# %%
states.plot()
# %%
import plotly.express as px
import pandas as pd
# %%
data = {'state_abbreviation': pl_class_names,
        'interstate_percentage': interstate_percentage}
df = pd.DataFrame(data)
df.to_csv('./state_interstate_percentage.csv')
# %%
fig = px.choropleth(df,
                    locations='state_abbreviation', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='interstate_percentage',
                    color_continuous_scale="Viridis_r", 
                    
                    )
fig.show()
# %%
