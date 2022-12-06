#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include "graph.h"
// #include "tqdm.h"
using namespace std;


void Graph::parse_page (ifstream& is) {
	cout << "-----------------start parsing pages-------------------" << endl;
	// below columns are for the input file: 
	// ../data/raw_dir/us_pages_lgc_idx_id_mask_label_state.csv
	// unsigned long pageID = 0;
	// int new_idx = 0, state_idx = 0, cnt = 0;
	// bool mask;
	// string state, mask_str, new_idx_str, pageID_str, state_idx_str, line;
	unsigned long pageID = 0;
	int new_idx = 0, state_idx = 0, cnt = 0, trueLabel = 0, dupStates = 0,
		mostPopulationLabel = 0, firstRoundDupstate0 = 0;
	int count_washington = 0, count_5200_washington = 0, count_true_label = 0;
	bool mask;
	string city, line;
	string pageID_str, new_idx_str, dupStates_str, trueLabel_str,
		   mostPopulationLabel_str, state_idx_str, firstRoundDupstate0_str;
	while (is.good()) {
		getline(is, line);
		stringstream ss(line);
		getline(ss, pageID_str, ',');
		getline(ss, new_idx_str, ',');
		if (ss.peek() == '"') {
			ss.ignore(); // skip first char "
			getline(ss, city, '"'); // use the second char " as delimiter
			ss.ignore(); // skip the ,
		} else {
			getline(ss, city, ',');
		}
		getline(ss, dupStates_str, ',');
		getline(ss, trueLabel_str, ',');
		getline(ss, mostPopulationLabel_str, ',');
		getline(ss, state_idx_str, ',');
		getline(ss, firstRoundDupstate0_str, ',');
		pageID = stoul(pageID_str);
		new_idx = stoi(new_idx_str);
		city;
		dupStates = stoi(dupStates_str);
		trueLabel = stoi(trueLabel_str);
		mostPopulationLabel = stoi(mostPopulationLabel_str);
		state_idx = stoi(state_idx_str);
		firstRoundDupstate0 = stoi(firstRoundDupstate0_str);
		
		// use true label as state index, not the 2nd round population labels.
		
		if(city == "washington") {
			if (trueLabel == -1) {
				count_5200_washington++;
			}
			trueLabel = 50;
			count_washington++;
		}
		state_idx = trueLabel;
		// getline(is, state, '\n'); // is >> state will only read "New" from "New York".
		
		// cout << pageID << " " << new_idx << " " << city << " " << dupStates << " " << trueLabel << " "
		//  	 << mostPopulationLabel << " " << state_idx << " " << firstRoundDupstate0 << " is.eof "
		// 	 << is.eof() << " is.good() " << is.good() << endl;
		// if (cnt == 1000000) break;
		// if (new_idx == 208387) break;

		cnt++;
		mask = (trueLabel != -1);
		if (mask == true) {
			count_true_label++;
		}
		Page * current = new Page(new_idx, pageID, mask, state_idx
								//   , state
								  );
		
		// in ../data/raw_dir/us_pages_lgc_idx_id_mask_label_state.csv
		/*  
		new_idx was the 0 to 5873394, not new_idx as the index from the world page index, that's why 
		change pageMap[new_idx] = current; 
		to     pageMap[cnt - 1] = current;
		Here new_idx and cnt - 1 are both from 0 to 5873394
			0	4846711747	1	4	California
			1	5246634919	1	4	California
			2	5281959998	1	31	New York
			3	5340039981	1	45	Virginia
			4	5352734382	1	31	New York
			5	5381334667	0	4	California
			6	5417674986	0	4	California
			7	5459524943	0	31	New York
			8	5461604986	0	4	California
			9	5466504237	1	4	California
		
		*/
		pageMap[cnt - 1] = current;
		if (cnt % 200000 == 0) {
			cout << cnt << " pageIDs processed. " << endl;
		}
		
	}
	
	cout << "Done. "<< cnt << " pageIDs processed. " << endl;
	cout << "pageMap size " << pageMap.size() << endl;
	cout << "count_5200_washington " << count_5200_washington << endl;
	cout << "count_washington " << count_washington << endl;
	cout << "count_true_label " << count_true_label << endl;
	cout << "-----------------------------------------------" << endl;
}

void Graph::parse_edge (ifstream& is) {
	cout << "-----------------start parsing edges-------------------" << endl;
	int liking = 0, liked = 0, cnt = 0, internal_edge_cnt = 0;
	string line;
	while (is.good()) {
		is >> liking >> liked;
		getline(is, line, '\n'); // skip line has been read, is >> won't skip '\n' and '\t'
		if (is.eof()) break;
		
		if (cnt % 10000000 == 0) {
			cout << cnt << " edges scanned. " << internal_edge_cnt << " of them are accepted." << endl;
		}
		cnt++;
		if (pageMap.find(liking) == pageMap.end() || pageMap.find(liked) == pageMap.end()) {
			continue; // not a internal edge.
		}
		pageMap[liking]->outward_neighbor.push_back(liked);
		pageMap[liked]->inward_neighbor.push_back(liking);
		internal_edge_cnt++;
	}
	
	cout << "Done. " << cnt << " edges scanned. " << internal_edge_cnt << " of them are accepted." << endl;
	cout << "----------------------------------------------" << endl;
}

void Graph::get_neighbor_distributions() {
	int cnt = 0;
	int n = pageMap.size();
	for (int i = 0; i < n; i++) {
		count_all_kinds_neighbors(i);
		normalize_distributions(i);
		cnt++;
		if (cnt % 1000 == 0) {
			cout << cnt << " pages neighbor distribution processed. " << endl;
		}
	}
	cout << "Done. " << cnt << " pages neighbor distribution processed. " << endl;
	cout << "----------------------------------------------" << endl;
}

void Graph::get_neighbor_distributions_per_thread_vector(vector<int> &v) {
	int cnt = 0;
	for (int i : v) {
		count_all_kinds_neighbors(i);
		normalize_distributions(i);
		cnt++;
		if (cnt % 10000 == 0) {
			cout << cnt << " pages neighbor distribution processed. " << endl;
		}
	}
	cout << "Done. " << cnt << " pages neighbor distribution processed. " << endl;
	cout << "----------------------------------------------" << endl;
}

void Graph::get_neighbor_distributions_per_thread(int start, int end) {
	int cnt = 0;
	for (int i = start; i < end; i++) {
		count_all_kinds_neighbors(i);
		normalize_distributions(i);
		cnt++;
		if (cnt % 10000 == 0) {
			cout << cnt << " pages neighbor distribution processed. " << endl;
		}
	}
	cout << "Done. " << cnt << " pages neighbor distribution processed. " << endl;
	cout << "----------------------------------------------" << endl;
}

void Graph::normalize_distributions (int root) {
	Page *current = pageMap[root];
	normalize(current->inward_neighbor_distribution);
	normalize(current->hop2_inward_neighbor_distribution);
	normalize(current->outward_neighbor_distribution);
	normalize(current->hop2_outward_neighbor_distribution);
	normalize(current->undirected_neighbor_distribution);
	normalize(current->hop2_undirected_neighbor_distribution);
}

void Graph::normalize (vector<float> &v) {
	int s = sum(v);
	if (s == 0) s++;
	divide(v, s);
}

int Graph::sum (vector<float> &v) {
	int sum = 0;
	for (auto i : v) {
		sum += i;
	}
	return sum;
}

void Graph::divide (vector<float> & v, int sum) {
	for (int i = 0; i <= NUM_OF_STATES; i++) {
		v[i] = v[i] / sum;
	}
}

void Graph::count_all_kinds_neighbors (int root) {
	count_neighbor(root, 1, "inward");
	count_neighbor(root, 2, "inward");
	count_neighbor(root, 1, "outward");
	count_neighbor(root, 2, "outward");
	count_neighbor(root, 1, "undirected");
	count_neighbor(root, 2, "undirected");
}

void Graph::count_neighbor (int root, int distance, string direction) { // make sure root != 0, column is column # each page is going to fill.
	queue<pair<int, int>> unvisited; // primary queue in BFS.
	unordered_set<int> visited; // visited hash set.
	Page *root_page = pageMap[root], *current;
	int dist = 0; // denotes current rank.
	unvisited.push(make_pair(root, 0));
	while (distance >= dist && !unvisited.empty()) {
		current = pageMap[unvisited.front().first]; // current node.
		int state_idx = current->state_idx;
		dist = unvisited.front().second; // distance from root.
		if (dist > distance) {
			break;
		}

		if (visited.find(current->pageID) != visited.end()) { // this node has been visited before.
			unvisited.pop();
			continue;
		}

		if (dist != 0 && state_idx != -1) {
			if (direction == "inward" && distance == 1) {
				root_page->inward_neighbor_distribution[state_idx]++;
			} else if (direction == "inward" && distance == 2) {
				root_page->hop2_inward_neighbor_distribution[state_idx]++;
			} else if (direction == "outward" && distance == 1) {
				root_page->outward_neighbor_distribution[state_idx]++;
			} else if (direction == "outward" && distance == 2) {
				root_page->hop2_outward_neighbor_distribution[state_idx]++;
			} else if (direction == "undirected" && distance == 1) {
				root_page->undirected_neighbor_distribution[state_idx]++;
			} else if (direction == "undirected" && distance == 2) {
				root_page->hop2_undirected_neighbor_distribution[state_idx]++;
			}
		}

		if (direction == "inward") {
			for (size_t i = 0; i < current->inward_neighbor.size(); i++) { 
				unvisited.push(make_pair(current->inward_neighbor[i], dist + 1)); // push all its neighbors
			}
		} else if (direction == "outward") {
			for (size_t i = 0; i < current->outward_neighbor.size(); i++) {
				unvisited.push(make_pair(current->outward_neighbor[i], dist + 1)); // push all its neighbors
			}
		} else if (direction == "undirected") {
			for (size_t i = 0; i < current->inward_neighbor.size(); i++) { 
				unvisited.push(make_pair(current->inward_neighbor[i], dist + 1)); // push all its neighbors
			}
			for (size_t i = 0; i < current->outward_neighbor.size(); i++) {
				unvisited.push(make_pair(current->outward_neighbor[i], dist + 1)); // push all its neighbors
			}
		}

		visited.insert(current->pageID);
		unvisited.pop();
	}
}


void Graph::write(string directory) {
	cout << "-----------------start writing-------------------" << endl;

	ofstream ofs;
	string file = directory + "new_cities_true_label_more_washington_51_by_6_neighbor_distribution.csv";


	ofs.open(file);
	int cnt = 0;
	for (size_t i = 0; i < pageMap.size(); i++) {
		Page * current = pageMap[i];

		for (size_t i = 0; i < (current->inward_neighbor_distribution).size(); i++) {
			ofs << current->inward_neighbor_distribution[i];
			ofs << ",";
		}
		for (size_t i = 0; i < (current->outward_neighbor_distribution).size(); i++) {
			ofs << current->outward_neighbor_distribution[i];
			ofs << ",";
		}
		for (size_t i = 0; i < (current->undirected_neighbor_distribution).size(); i++) {
			ofs << current->undirected_neighbor_distribution[i];
			ofs << ",";
		}
		for (size_t i = 0; i < (current->hop2_inward_neighbor_distribution).size(); i++) {
			ofs << current->hop2_inward_neighbor_distribution[i];
			ofs << ",";
		}
		for (size_t i = 0; i < (current->hop2_outward_neighbor_distribution).size(); i++) {
			ofs << current->hop2_outward_neighbor_distribution[i];
			ofs << ",";
		}
		for (size_t i = 0; i < (current->hop2_undirected_neighbor_distribution).size(); i++) {
			ofs << current->hop2_undirected_neighbor_distribution[i];
			if (i != (current->hop2_undirected_neighbor_distribution).size() - 1) {
				ofs << ",";
			}
		}
		ofs << endl;
		if (cnt % 200000 == 0) {
			cout << cnt << " pages neighbor distribution had been wrote. " << endl;
		}
		cnt++;
	}
	cout << "Done. " << cnt << " pages neighbor distribution had been wrote. " << endl;
	cout << "----------------------------------------------" << endl;
	ofs.close();
//	ofstream(dictionary, ofstream::app);

}
