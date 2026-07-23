#ifndef SEMANTIC_GRAPH_H
#define SEMANTIC_GRAPH_H

#include <vector>
#include <string>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace std;

class Graph {
public:
    int V;
    int E;
    vector<vector<int>> adj;
    vector<set<int>> adjSet;
    vector<vector<double>> semanticVectors;
    int vectorDim;

    Graph(int vertices);
    Graph(const string &filename);

    void addEdge(int u, int v);
    int getEdgeCount() const;
    bool isAdjacent(int u, int v) const;

    bool loadVectorsFromFile(const string &vectorFile);
    void setSemanticVector(int v, const vector<double> &vec);
    const vector<double>& getSemanticVector(int v) const;

    bool loadVectorsFromBinary(const string &binaryFile);
    bool loadVectorsFromChunks(const string &indexFile);
    void saveVectorsToBinary(const string &binaryFile) const;
    void normalizeVectors();
};

double calculateSimilarity(const vector<double> &vec1, const vector<double> &vec2);

#endif
