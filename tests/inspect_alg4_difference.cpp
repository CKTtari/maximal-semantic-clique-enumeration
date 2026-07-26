#include "../src/semantic_graph.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

struct FloatVectors {
    int n = 0;
    int dim = 0;
    vector<float> data;

    void build(const Graph& graph) {
        n = graph.V;
        dim = static_cast<int>(graph.getSemanticVector(0).size());
        data.resize(static_cast<size_t>(n) * dim);
        for (int u = 0; u < n; ++u) {
            const auto& source = graph.getSemanticVector(u);
            float* target = data.data() + static_cast<size_t>(u) * dim;
            for (int j = 0; j < dim; ++j) target[j] = static_cast<float>(source[j]);
        }
    }

    float dot(int u, int v) const {
        const float* a = data.data() + static_cast<size_t>(u) * dim;
        const float* b = data.data() + static_cast<size_t>(v) * dim;
        __m256 sum = _mm256_setzero_ps();
        int j = 0;
        for (; j <= dim - 8; j += 8)
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(a + j),
                                                  _mm256_loadu_ps(b + j)));
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float result = _mm_cvtss_f32(lo);
        for (; j < dim; ++j) result += a[j] * b[j];
        return result;
    }
};

vector<int> parse_clique(const string& line) {
    istringstream input(line);
    vector<int> clique;
    int vertex = 0;
    while (input >> vertex) clique.push_back(vertex);
    return clique;
}

set<string> read_cliques(const string& path) {
    ifstream input(path);
    set<string> result;
    string line;
    while (getline(input, line))
        if (!line.empty()) result.insert(line);
    return result;
}

long double surplus(const FloatVectors& vectors,
                    const vector<int>& clique,
                    long double tau) {
    long double sum = 0.0L;
    for (size_t i = 0; i < clique.size(); ++i)
        for (size_t j = i + 1; j < clique.size(); ++j)
            sum += static_cast<long double>(vectors.dot(clique[i], clique[j]));
    const long double edges =
        static_cast<long double>(clique.size() * (clique.size() - 1) / 2);
    return sum - tau * edges;
}

long double best_extension_surplus(const Graph& graph,
                                   const FloatVectors& vectors,
                                   const vector<int>& clique,
                                   long double tau) {
    long double current_sum = 0.0L;
    for (size_t i = 0; i < clique.size(); ++i)
        for (size_t j = i + 1; j < clique.size(); ++j)
            current_sum +=
                static_cast<long double>(vectors.dot(clique[i], clique[j]));

    long double best = -numeric_limits<long double>::infinity();
    for (int candidate = 0; candidate < graph.V; ++candidate) {
        if (binary_search(clique.begin(), clique.end(), candidate)) continue;
        bool adjacent = true;
        long double delta = 0.0L;
        for (int member : clique) {
            if (!graph.isAdjacent(candidate, member)) {
                adjacent = false;
                break;
            }
            delta += static_cast<long double>(vectors.dot(candidate, member));
        }
        if (!adjacent) continue;
        const size_t next_size = clique.size() + 1;
        const long double next_edges =
            static_cast<long double>(next_size * (next_size - 1) / 2);
        best = max(best, current_sum + delta - tau * next_edges);
    }
    return best;
}

void inspect_difference(const string& label,
                        const set<string>& left,
                        const set<string>& right,
                        const Graph& graph,
                        const FloatVectors& vectors,
                        long double tau) {
    size_t count = 0;
    size_t nonpositive = 0;
    size_t extendable = 0;
    long double minimum = numeric_limits<long double>::infinity();
    long double maximum = -numeric_limits<long double>::infinity();
    cout << label << '\n';
    for (const string& line : left) {
        if (right.count(line)) continue;
        const vector<int> clique = parse_clique(line);
        const long double own_surplus = surplus(vectors, clique, tau);
        const long double extension =
            best_extension_surplus(graph, vectors, clique, tau);
        ++count;
        if (own_surplus <= 0.0L) ++nonpositive;
        if (extension >= 0.0L) ++extendable;
        minimum = min(minimum, own_surplus);
        maximum = max(maximum, own_surplus);
        if (count <= 12) {
            cout << "  size=" << clique.size()
                 << " surplus=" << setprecision(18) << own_surplus
                 << " best_extension=" << extension << " clique=" << line << '\n';
        }
    }
    cout << "  count=" << count << " nonpositive=" << nonpositive
         << " extendable=" << extendable << " surplus_range=["
         << setprecision(18) << minimum << ',' << maximum << "]\n";
}

int canonical_deletion(const FloatVectors& vectors, const vector<int>& clique) {
    int best = -1;
    long double best_incident = 0.0L;
    for (int v : clique) {
        long double incident = 0.0L;
        for (int u : clique)
            if (u != v) incident += static_cast<long double>(vectors.dot(u, v));
        if (best == -1 || incident < best_incident ||
            (incident == best_incident && v < best)) {
            best = v;
            best_incident = incident;
        }
    }
    return best;
}

void trace_canonical_chain(const string& line,
                           const FloatVectors& vectors,
                           float tau) {
    vector<int> current = parse_clique(line);
    vector<int> removed;
    while (current.size() > 2) {
        const int deleted = canonical_deletion(vectors, current);
        removed.push_back(deleted);
        current.erase(find(current.begin(), current.end(), deleted));
    }
    reverse(removed.begin(), removed.end());

    vector<int> path = current;
    float sim_sum = vectors.dot(path[0], path[1]);
    cout << "Canonical path for selected differing clique:\n";
    cout << "  root=" << path[0] << ',' << path[1]
         << " sim=" << setprecision(18) << sim_sum << '\n';
    for (int added : removed) {
        float delta = 0.0f;
        for (int member : path) delta += vectors.dot(member, added);
        const int old_size = static_cast<int>(path.size());
        const float old_edges = static_cast<float>(old_size * (old_size - 1) / 2);
        const float next_edges = static_cast<float>((old_size + 1) * old_size / 2);
        const float mono_limit =
            (sim_sum / old_edges) * next_edges - sim_sum;
        const float w_test = sim_sum - tau * old_edges + delta - tau * old_size;
        path.push_back(added);
        const long double exact = surplus(vectors, path, tau);
        cout << "  add=" << added << " size=" << path.size()
             << " float_child_surplus=" << w_test
             << " delta_minus_mono=" << (delta - mono_limit)
             << " exact_child_surplus=" << exact
             << " canonical_delete=" << canonical_deletion(vectors, path) << '\n';
        sim_sum += delta;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "usage: inspect_alg4_difference <tau> <layered.cliques> "
                "<canonical.cliques>\n";
        return 2;
    }
    const long double tau = stold(argv[1]);
    const string graph_file = "dataset/dataset/FB15K-237_edges.txt";
    const string vector_file = "dataset/vectors/FB15K-237_vectors.index";
    Graph graph(graph_file);
    if (!graph.loadVectorsFromChunks(vector_file)) return 1;
    graph.normalizeVectors();
    for (auto& row : graph.adj) sort(row.begin(), row.end());

    FloatVectors vectors;
    vectors.build(graph);
    const set<string> layered = read_cliques(argv[2]);
    const set<string> canonical = read_cliques(argv[3]);
    inspect_difference("Only in layered", layered, canonical, graph, vectors, tau);
    inspect_difference("Only in canonical", canonical, layered, graph, vectors, tau);
    for (const string& line : layered) {
        if (!canonical.count(line)) {
            trace_canonical_chain(line, vectors, static_cast<float>(tau));
            return 0;
        }
    }
    for (const string& line : canonical) {
        if (!layered.count(line)) {
            trace_canonical_chain(line, vectors, static_cast<float>(tau));
            break;
        }
    }
    return 0;
}
