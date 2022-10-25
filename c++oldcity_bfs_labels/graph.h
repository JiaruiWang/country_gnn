#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <assert.h>

#define NUM_OF_STATES 51

using namespace std;

/*
 * use like a struct.
*/


class Page {
public:
	const int new_idx;
	const unsigned long pageID;
	const bool true_label_mask;
	const int state_idx;	
	const string state;
	vector<short> distance;
	vector<float> inward_neighbor_distribution; // each state count in neighbors / all neighbors
	vector<float> hop2_inward_neighbor_distribution;
	vector<float> outward_neighbor_distribution;
	vector<float> hop2_outward_neighbor_distribution;
	vector<float> undirected_neighbor_distribution;
	vector<float> hop2_undirected_neighbor_distribution;
	vector<int> inward_neighbor; // why is vector??? in edges page_id list
	vector<int> outward_neighbor; // out edges page_id list
	Page (int idx, unsigned long id, bool mask, int state_idx, string state_name) : 
		new_idx(idx), pageID(id), true_label_mask(mask), state_idx(state_idx), state(state_name),
		inward_neighbor_distribution(NUM_OF_STATES, 0), hop2_inward_neighbor_distribution(NUM_OF_STATES, 0),
		outward_neighbor_distribution(NUM_OF_STATES, 0), hop2_outward_neighbor_distribution(NUM_OF_STATES, 0),
		undirected_neighbor_distribution(NUM_OF_STATES, 0), hop2_undirected_neighbor_distribution(NUM_OF_STATES, 0){
	};
};

class Graph {
public:
	void parse_page (ifstream& is);
	void parse_edge (ifstream& is);
	void get_neighbor_distributions ();
	void get_neighbor_distributions_per_thread(int start, int end);
	void get_neighbor_distributions_per_thread_vector(vector<int> &v);
	void write(string directory);

	unordered_map<int, Page *> pageMap;

private:
	unordered_map<string, unordered_set<string>> stateMap; // key:city, value: set(dup states)
	
	void normalize_distributions (int root);
	void count_all_kinds_neighbors (int root);
	void count_neighbor (int root, int distance, string direction);
	void normalize (vector<float> &v);
	void divide (vector<float> & v, int sum);
	int sum (vector<float> &v);
};
