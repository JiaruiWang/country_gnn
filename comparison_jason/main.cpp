#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "graph.h"

#define ARGV 7

using namespace std;

int main (int argc, char** argv) {
	if (argc != ARGV) {
		cout << "Parameter list:\n1. node file\n"
		<< "2. edge file\n3. seed file\n"
		<< "4. state file\n5. output file\n6. state class file\n";
	}
	string sample_path = "sample.txt";
	sample_path = argv[5] + sample_path;

	ifstream ns (argv[1]);
	ifstream es (argv[2]);
	ifstream ss (argv[3]);
	ifstream cs (argv[6]);
	ifstream state_stream(argv[4]);
	ofstream os (sample_path);
	if (!ns.good()) {
		cout << "Abort: node file open fail." << endl;
		return 0;
	}
	if (!es.good()) {
		cout << "Abort: edge file open fail." << endl;
		return 0;
	}
	if (!ss.good()) {
		cout << "Abort: seed file open fail." << endl;
		return 0;
	}
	if (!cs.good()) {
		cout << "Abort: state class file open fail." << endl;
		return 0;
	}
	if (!state_stream.good()) {
		cout << "Abort: state file open fail." << endl;
		return 0;
	}

	// open a sample file to validate os directory.
	if (!os.good()) {
		cout << "Abort: output file open fail." << endl;
		return 0;
	}
	os.close();
	Graph g;
	g.parse_seed(ss);
	g.parse_state(state_stream);
	g.parse_page(ns);
	g.parse_edge(es);
	g.parse_class(cs);
	cout << "BFS_trigger" << endl;
	g.bfs_trigger();
	g.write(argv[5]);
}
