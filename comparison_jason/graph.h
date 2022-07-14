#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <assert.h>

using namespace std;

/*
 * use like a struct.
*/
class Page {
public:
	const unsigned long pageID;
	const string state;
	vector<short> distance;
	vector<float> inward_prob;
	vector<float> outward_prob;
	vector<unsigned long> indegree;
	vector<unsigned long> outdegree;
	Page (unsigned long x, string label, int num) : pageID(x), state(label) {
		distance.resize(num);
		fill(distance.begin(), distance.end(), 0);
	};
};

class Graph {
public:
	void parse_seed (ifstream& is);
	void parse_page (ifstream& is);
	void parse_state(ifstream& is);
	void parse_edge (ifstream& is);
	void parse_class(ifstream& is);
	void write(string directory);
	void bfs_trigger ();
	void neighbor_info(Page* p);
private:
	void bfs (unsigned long root, int column, bool inward);
	unordered_map<unsigned long, Page *> pageMap;
	unordered_set<string> stateList;
	unordered_map<string, short> stateIdx;
	unordered_map<string, unordered_set<string>> stateMap;

	vector<unsigned long> rootList; // storing all seeds.
};
