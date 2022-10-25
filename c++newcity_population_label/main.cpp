#include <iostream>
#include <thread>
#include <vector>
#include <fstream>
#include <string>
#include "graph.h"

#define ARGV 7

using namespace std;

void print(vector<float> *v) {
	for (auto i : *v) {
		cout << i << " ";
	}
	cout << endl;
}

void print_int(vector<int> *v) {
	int cnt = 0;
	for (auto i : *v) {
		if (cnt > 293660) {
			cout << i << " ";
		}
	}
	cout << endl;
}

int main (int argc, char** argv) {
	if (argc != ARGV) {
		cout << "Parameter list:\n1. node file\n"
		<< "2. edge file\n3. state name file\n"
		<< "4. output file\n";
	}
	string sample_path = "sample.txt";
	sample_path = argv[4] + sample_path;

	ifstream ns (argv[1]);
	ifstream es (argv[2]);
	ifstream cs (argv[3]);
	ofstream os (sample_path);
	if (!ns.good()) {
		cout << "Abort: node file open fail." << endl;
		return 0;
	}
	if (!es.good()) {
		cout << "Abort: edge file open fail." << endl;
		return 0;
	}
	if (!cs.good()) {
		cout << "Abort: state name file open fail." << endl;
		return 0;
	}


	// open a sample file to validate os directory.
	if (!os.good()) {
		cout << "Abort: output file open fail." << endl;
		return 0;
	}
	os.close();

	int step = 293670;
	vector<int> range, test_range;
	for (int i = 0; i < 20; i++) {
		range.push_back(i * step);
		test_range.push_back(i);
	}
	range.push_back(5873395);
	test_range.push_back(20);
	

	vector<vector<int>> ranges(20, vector<int>());
	for (int i = 0; i < 5873395; i++) {
		ranges[i % 20].push_back(i);
	} // modify Graph::get_neighbor_distributions_per_thread(vector<int> &vector_of_index)
	print_int(&ranges[19]);
	Graph g;

	g.parse_page(ns);
	g.parse_edge(es);
	// g.get_neighbor_distributions();


	std::vector<std::thread> threads;
	cout << "-----------------start multithreading-------------------" << endl;
	for (int i = 0; i < 20; i++) {
		// int start = range[i], end = range[i + 1];
		// std::thread t(&Graph::get_neighbor_distributions_per_thread, &g, start, end);

		std::thread t(&Graph::get_neighbor_distributions_per_thread_vector, &g, std::ref(ranges[i]));
		threads.push_back(std::move(t));
		/* line 82 get error
		error: use of deleted function ‘std::thread::thread(const std::thread&)’
		The (misleading) error you see is the STL trying to fall back on copying types that 
		can't be moved, and failing because std::thread is deliberately non-copyable.
		*/
	}

	for (int i = 0; i < 20; i++) {
		threads[i].join();
	}
	cout << "-----------------Done with multithreading-------------------" << endl;
	// print(&g.pageMap[0]->inward_neighbor_distribution);
	// print(&g.pageMap[0]->hop2_inward_neighbor_distribution);
	// print(&g.pageMap[0]->outward_neighbor_distribution);
	// print(&g.pageMap[0]->hop2_outward_neighbor_distribution);
	// print(&g.pageMap[0]->undirected_neighbor_distribution);
	// print(&g.pageMap[0]->hop2_undirected_neighbor_distribution);

	// print(&g.pageMap[5873394]->inward_neighbor_distribution);
	// print(&g.pageMap[5873394]->hop2_inward_neighbor_distribution);
	// print(&g.pageMap[5873394]->outward_neighbor_distribution);
	// print(&g.pageMap[5873394]->hop2_outward_neighbor_distribution);
	// print(&g.pageMap[5873394]->undirected_neighbor_distribution);
	// print(&g.pageMap[5873394]->hop2_undirected_neighbor_distribution);

	g.write(argv[4]);
}
