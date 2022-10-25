#%% 
# load city data in the value list, column name in the header map.
# city-state pair key will keep the duplicate citys from different states.
import csv
is_header = True
city_headers = {}
cities = {}
cityfilename = './us-cities-30k-noDC-noPR.csv'
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
        # When city-state key has duplicate in same state, we keep the city with largest population
        key = (row[city_headers['city']],row[city_headers['state_name']])

        if key in cities:
            if int(cities[key][city_headers['population']]) < int(row[city_headers['population']]):
                cities[key] = row
                # print(row)
        else:
            cities[key] = row
print(len(cities))
print(list(cities.items())[1])

# %%
# load states name create state label dict
state_label = {}
label_state = {}
statenamefile = './states.csv'
with open(statenamefile, 'r') as statename:
    cityreader = csv.reader(statename)
    counter = 0
    for row in cityreader:
        state = row[0]
        state_label[state] = counter
        label_state[counter] = state
        counter += 1

print(state_label)
print(label_state)

# %%
# build {city_name: {state: lable} } dict
cities_dup_states = {}
for key in cities.keys():
    city, state = key
    if city not in cities_dup_states:
        cities_dup_states[city] = {state: state_label[state]}
    else:
        cities_dup_states[city][state] = state_label[state]
print(len(cities_dup_states))
print(len(list(cities_dup_states.values())[0]))
print(list(cities_dup_states.items())[0])

#%% 
# load city data in the value list, column name in the header map.
# Unlike above city-state pair key will keep the duplicate citys from different states.
# This code will use city as key, and only keep one city with the largest population out of 
# all duplicate cities from different states.
import csv
is_header = True
city_headers = {}
cities_by_population = {}
cityfilename = './us-cities-30k-noDC-noPR.csv'
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
        
        # store key as city
        # When city key has duplicate in different states, we keep the city with largest population
        key = row[city_headers['city']]

        if key in cities_by_population:
            if int(cities_by_population[key][city_headers['population']]) < int(row[city_headers['population']]):
                cities_by_population[key] = row
                # print(row)
        else:
            cities_by_population[key] = row
print(len(cities_by_population))
print(list(cities_by_population.items())[1])

#%%
