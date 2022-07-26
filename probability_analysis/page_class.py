#%%
import csv
from functools import total_ordering

#%%
@total_ordering
class Page:
    def __init__(self, idx, id, mask, label, label_state, pred, pred_state, threshold, counts, states, probs) -> None:
        self.idx = idx
        self.id = id
        self.original_labeled = mask
        self.label = label
        self.label_state = label_state
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
        output = (f"Page id: {self.id}, idx: {self.id}, original_labeled: {self.original_labeled}, label: {self.label}, "
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

#%%
pages = []
count = 0
inputfile = '../model/saint_all_label/idx_id_mask_label_state_pred_state_threshold_counts_possible_states.csv'
with open(inputfile, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
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
        
        page = Page(idx, id, mask, label, label_state, pred, pred_state, threshold, counts, states, probs)
        pages.append(page)


# %%
sorted_pages = sorted(pages, reverse=True)

#%%
possible_counts_pages = {}
for i in range(18):
    possible_counts_pages[i+1] = []

for page in sorted_pages:
    possible_counts_pages[page.possible_counts].append(page)

# Inter state scale?? Globalness scale?
for k in range(18):
    print(k+1, len(possible_counts_pages[k+1]))

# %%
