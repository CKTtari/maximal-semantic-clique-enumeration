#define main alg3_bb_interactive_main
#include "../src/alg3-raw.cpp"
#undef main

int main() {
    if (!acmsc_runtime::enforce_memory_limit()) return 2;
    omp_set_num_threads(32);

    {
        constexpr int n = 256;
        constexpr int dimension = 255;
        Graph graph(n);
        graph.semanticVectors.resize(n);
        graph.vectorDim = dimension;
        for (int u = 0; u < n; ++u)
            for (int v = u + 1; v < n; ++v) graph.addEdge(u, v);

        for (int v = 0; v < n; ++v) {
            vector<double> semantic(dimension, 0.0);
            semantic[v <= 1 ? 0 : v - 1] = 1.0;
            graph.setSemanticVector(v, semantic);
        }
        graph.normalizeVectors();

        VectorDB database;
        database.build(graph);
        AVGMiner miner(database, graph, 180);
        const long long count = miner.mine(0.9);
        if (count != 1) {
            fprintf(stderr,
                    "Expected one ACMSC from the 256-vertex clique, got %lld.\n",
                    count);
            return 1;
        }
    }

    {
        constexpr int n = 600;
        Graph graph(n);
        graph.semanticVectors.resize(n);
        graph.vectorDim = 1;
        for (int u = 0; u < n; ++u)
            for (int v = u + 1; v < n; ++v) graph.addEdge(u, v);
        for (int v = 0; v < n; ++v)
            graph.setSemanticVector(v, vector<double>{1.0});
        graph.normalizeVectors();

        VectorDB database;
        database.build(graph);
        AVGMiner miner(database, graph, 180);
        const long long count = miner.mine(0.9);
        if (count != 1) {
            fprintf(stderr,
                    "Expected one size-600 ACMSC, got %lld results.\n", count);
            return 1;
        }
    }
    printf("PASS: dynamic masks handled 256 vertices and reporting handled a size-600 ACMSC.\n");
    return 0;
}
