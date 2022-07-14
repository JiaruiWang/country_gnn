#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include "graph.h"

using namespace std;

void Graph::parse_seed (ifstream& is) {
	unsigned long page_id;
	int cnt = 0;
	string label, drain;
	while (getline(is, label, '\t')) {
		is >> page_id;
		cout << "label: " << label << "\t page_id: " << page_id << endl;
		pageMap[page_id] = new Page(page_id, label, 0);
		this->rootList.push_back(page_id);
		cnt++;
		getline(is, drain);
	}
	cout << cnt << " seed pages parsed" << endl;
}

void Graph::parse_class (ifstream& is) {
	string state;
	short cnt = 0;
	while (getline(is, state)) {
		stateIdx[state] = cnt;
		++cnt;
	}
	cout << cnt << 	" states processed. "  << endl;
}

void Graph::parse_state (ifstream& is) {
	string city, state;
	int cnt = 0, stored = 0;
	while (getline(is, city, '\t')) {
		getline(is, state);
		stateMap[city].insert(state);
		cnt++;
		stateList.insert(state);
	}
	for (auto it = stateMap.begin(); it != stateMap.end(); it++) {
		stored += (it->second).size();
	}
	cout << cnt << " cities processed. " << stored << " cities stored." << endl;
}

void Graph::parse_page (ifstream& is) {
	unsigned long pageID = 0, cnt = 0, known_cnt = 0, us_cnt = 0;
	while (is.good() && is >> pageID) {
		string country, city;
		cnt++;
		getline (is, country, '\t');
		getline (is, country, '\t');
		if (country == "Unknown") {
			getline(is, city);
			continue;
		}
		getline(is, city);
		Page * current = NULL;
		if (country != "United States") {
			// non us page
			//current = new Page(pageID, "others", rootList.size());
			current = new Page(pageID, "others", 2 * rootList.size());
		}
		else if (this->stateMap[city].size() == 1) {
			// us confirmed page.
			//current = new Page(pageID, *(stateMap[city].begin()), rootList.size());
			current = new Page(pageID, *(stateMap[city].begin()), 2 * rootList.size());
			us_cnt++;
		}
		else { // add us pages whose city belongs to multiple states.
			// current = new Page(pageID, "dup_states", 2 * rootList.size());
			// us_cnt++;
			continue;
		}
		pageMap[pageID] = current;
		known_cnt++;
		if (cnt % 2000000 == 0) {
			cout << cnt << " pageIDs processed. " << cnt - known_cnt << " are unknown page." << us_cnt << " are U.S. pages" << endl;
		}
	}
	cout << "-----------------------------------------------" << endl;
	cout << "Done. "<< cnt << " pageIDs processed. " << cnt - known_cnt << " are unknown page." << us_cnt << " are U.S. pages"  << endl;
}

void Graph::parse_edge (ifstream& is) {
	unsigned long liking = 0, liked = 0, cnt = 0, internal_edge_cnt = 0;
	while (is.good() && is >> liking >> liked) {
		cnt++;
		if (cnt % 10000000 == 0) {
			cout << cnt << " edges scanned. " << internal_edge_cnt << " of them are accepted." << endl;
		}
		if (pageMap.find(liking) == pageMap.end() || pageMap.find(liked) == pageMap.end()) {
			continue; // not a internal edge.
		}
		pageMap[liking]->outdegree.push_back(liked);
		pageMap[liked]->indegree.push_back(liking);
		internal_edge_cnt++;
	}
	cout << "----------------------------------------------" << endl;
	cout << cnt << " edges scanned. " << internal_edge_cnt << " of them are accepted." << endl;
}

void Graph::bfs_trigger() {
	// resize all rootList's distance vectors first to prevent from illegal memory access to rootList[i]->distance
	for (int i = 0; i < rootList.size(); i++) {
		//(pageMap[rootList[i]]->distance).resize(rootList.size());
		(pageMap[rootList[i]]->distance).resize(2 * rootList.size());  // inward and outward edges
	}

	for (int i = 0; i < rootList.size(); i++) {
		if (rootList[i]) {
			cout << "seed: " << rootList[i] << endl;
			bfs (rootList[i], 2 * i, true);
			bfs (rootList[i], 2 * i + 1, false);
		}
	}
}
void Graph::bfs (unsigned long root,
	int column, bool inward) { // make sure root != 0, column is column # each page is going to fill.
	queue<pair<unsigned long, int>> unvisited; // primary queue in BFS.
	unordered_set<unsigned long> visited; // visited hash set.
	Page *current;
	long nodeCount = 0;
	int rank = 0; // denotes current rank.
	unvisited.push(make_pair(root, 0));
	while (!unvisited.empty()) {
		current = pageMap[unvisited.front().first]; // current node.
		int dist = unvisited.front().second; // distance from root.
		/*
		if (dist >= 5) {
			cout << nodeCount << "\t" << current->pageID << "\t" << dist << endl;
		}
		*/
		if (visited.find(current->pageID) != visited.end()) { // this node has been visited before.
			unvisited.pop();
			continue;
		}
/*
		if (dist >= 5) {
			cout << "before assign distance" << endl;
		}
*/
		assert(current->distance.size() > column);
		current->distance[column] = dist;
		nodeCount++;
		if (nodeCount % 1000000 == 0) {
			cout << nodeCount << " pages traversed in BFS, seed# = " << root << endl;
		}

		if (inward) {
			for (int i = 0; i < current->indegree.size(); i++) {
				unvisited.push(make_pair(current->indegree[i], dist + 1)); // push all its neighbors
			}
		} else {
			for (int i = 0; i < current->outdegree.size(); i++) {
				unvisited.push(make_pair(current->outdegree[i], dist + 1)); // push all its neighbors
			}
		}
/*
		if (dist >= 5) {
			cout << "before insert and pop" << endl;
		}
*/
		visited.insert(current->pageID);
		unvisited.pop();
	}
	cout << nodeCount << " pages traversed in BFS, seed# = " << root << endl;
}

void Graph::neighbor_info(Page* p) {
	(p->inward_prob).resize(stateIdx.size());
	fill(p->inward_prob.begin(), p->inward_prob.end(), 0);
	(p->outward_prob).resize(stateIdx.size());
	fill(p->outward_prob.begin(), p->outward_prob.end(), 0);

	unsigned long inward_cnt = 0, outward_cnt = 0;
	for (const auto &i : p->indegree){
		Page* neighbor = pageMap[i];
		string state = neighbor->state;
		if (stateIdx.find(state) == stateIdx.end()) {
			continue;
		}
		++p->inward_prob[stateIdx[state]];
		++inward_cnt;
	}
	for (const auto &i : p->outdegree){
		Page* neighbor = pageMap[i];
		string state = neighbor->state;
		if (stateIdx.find(state) == stateIdx.end()) {
			continue;
		}
		++p->outward_prob[stateIdx[state]];
		++outward_cnt;
	}
	for (int i = 0; i < stateIdx.size(); ++i) {
		if (p->inward_prob[i] != 0) {
			p->inward_prob[i] = p->inward_prob[i] / inward_cnt;
		}
		if (p->outward_prob[i] != 0) {
			p->outward_prob[i] = p->outward_prob[i] / outward_cnt;
		}
	}
}

void Graph::write(string directory) {
	ofstream ofs, ofs_id;
	string dictionary = "rank.txt";
	string ext = ".csv";
	dictionary = directory + dictionary;
	unordered_set<unsigned long> seeds; // for quickly look up.
	ofs.open(dictionary);
	for (int i = 0; i < rootList.size(); i++) {
		seeds.insert(rootList[i]);
		ofs << pageMap[rootList[i]]->state << endl;
	}
	ofs.close();
	for (auto itr = pageMap.begin(); itr != pageMap.end(); itr++) {
		unsigned long id = itr->first;
		Page * current = itr->second;
		string state = current->state;
		if (seeds.find(id) != seeds.end()) {
			continue;
		}

		neighbor_info(current);

		ofs.open(directory + state + ext, ofstream::app);
		for (int i = 0; i < (current->distance).size(); i++) {
			ofs << current->distance[i];
			ofs << ",";
			/*
			if (i != (current->distance).size() - 1) {
				ofs << ",";
			}
			*/
		}
		for (int i = 0; i < (current->inward_prob).size(); i++) {
			ofs << current->inward_prob[i];
			ofs << ",";
		}
		for (int i = 0; i < (current->outward_prob).size(); i++) {
			ofs << current->outward_prob[i];
			if (i != (current->outward_prob).size() - 1) {
				ofs << ",";
			}
		}
		ofs << endl;
		ofs.close();

		ofs_id.open(directory + "id_" + state + ext, ofstream::app);
		ofs_id << id << endl;
		ofs_id.close();
	}
//	ofstream(dictionary, ofstream::app);

}
