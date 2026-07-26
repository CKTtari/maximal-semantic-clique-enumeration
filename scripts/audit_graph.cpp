#include "experiment_runtime.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

struct Audit {
    string path;
    int vertices = 0;
    long long header_edges = 0;
    long long input_rows = 0;
    long long self_loops = 0;
    long long duplicate_edges = 0;
    long long edges = 0;
    int max_degree = 0;
    int degeneracy = 0;
    int max_forward_degree = 0;
    uint64_t wedges = 0;
    uint64_t triangles = 0;
};

static string json_escape(const string& value) {
    string escaped;
    for (char ch : value) {
        if (ch == '\\' || ch == '"') escaped.push_back('\\');
        escaped.push_back(ch);
    }
    return escaped;
}

static pair<int, vector<int>> degeneracy_order(const vector<vector<int>>& adjacency) {
    const int n = static_cast<int>(adjacency.size());
    vector<int> degree(n);
    int maximum = 0;
    for (int v = 0; v < n; ++v) {
        degree[v] = static_cast<int>(adjacency[v].size());
        maximum = max(maximum, degree[v]);
    }

    vector<int> bins(maximum + 1, 0);
    for (int value : degree) ++bins[value];
    int start = 0;
    for (int d = 0; d <= maximum; ++d) {
        const int count = bins[d];
        bins[d] = start;
        start += count;
    }
    vector<int> position(n), vertices(n);
    for (int v = 0; v < n; ++v) {
        position[v] = bins[degree[v]];
        vertices[position[v]] = v;
        ++bins[degree[v]];
    }
    for (int d = maximum; d >= 1; --d) bins[d] = bins[d - 1];
    bins[0] = 0;

    int degeneracy = 0;
    for (int i = 0; i < n; ++i) {
        const int v = vertices[i];
        degeneracy = max(degeneracy, degree[v]);
        for (int u : adjacency[v]) {
            if (degree[u] <= degree[v]) continue;
            const int old_degree = degree[u];
            const int old_position = position[u];
            const int first_position = bins[old_degree];
            const int first_vertex = vertices[first_position];
            if (u != first_vertex) {
                swap(vertices[old_position], vertices[first_position]);
                position[u] = first_position;
                position[first_vertex] = old_position;
            }
            ++bins[old_degree];
            --degree[u];
        }
    }
    return {degeneracy, vertices};
}

static Audit audit_graph(const fs::path& path) {
    ifstream input(path);
    Audit result;
    result.path = path.generic_string();
    if (!(input >> result.vertices >> result.header_edges) || result.vertices <= 0 ||
        result.header_edges < 0)
        throw runtime_error("Invalid graph header: " + path.string());

    vector<pair<int, int>> edges;
    edges.reserve(static_cast<size_t>(result.header_edges));
    int u = 0;
    int v = 0;
    while (input >> u >> v) {
        ++result.input_rows;
        if (u < 0 || u >= result.vertices || v < 0 || v >= result.vertices)
            throw runtime_error("Out-of-range endpoint in " + path.string());
        if (u == v) {
            ++result.self_loops;
            continue;
        }
        if (u > v) swap(u, v);
        edges.push_back({u, v});
    }
    sort(edges.begin(), edges.end());
    const auto unique_end = unique(edges.begin(), edges.end());
    result.duplicate_edges = distance(unique_end, edges.end());
    edges.erase(unique_end, edges.end());
    result.edges = static_cast<long long>(edges.size());

    vector<vector<int>> adjacency(result.vertices);
    for (const auto& edge : edges) {
        adjacency[edge.first].push_back(edge.second);
        adjacency[edge.second].push_back(edge.first);
    }
    vector<int> degree(result.vertices);
    for (int vertex = 0; vertex < result.vertices; ++vertex) {
        degree[vertex] = static_cast<int>(adjacency[vertex].size());
        result.max_degree = max(result.max_degree, degree[vertex]);
        result.wedges += static_cast<uint64_t>(degree[vertex]) *
                         static_cast<uint64_t>(degree[vertex] - 1) / 2;
    }

    const auto [degeneracy, removal_order] = degeneracy_order(adjacency);
    result.degeneracy = degeneracy;
    vector<int> rank(result.vertices);
    for (int i = 0; i < result.vertices; ++i) rank[removal_order[i]] = i;
    vector<vector<int>> forward(result.vertices);
    for (const auto& edge : edges) {
        const int from = rank[edge.first] < rank[edge.second] ? edge.first : edge.second;
        const int to = from == edge.first ? edge.second : edge.first;
        forward[from].push_back(to);
    }
    for (auto& row : forward) {
        sort(row.begin(), row.end(), [&](int left, int right) {
            return rank[left] < rank[right];
        });
        result.max_forward_degree =
            max(result.max_forward_degree, static_cast<int>(row.size()));
    }

    vector<uint32_t> marks(result.vertices, 0);
    uint32_t generation = 1;
    for (int from = 0; from < result.vertices; ++from) {
        if (++generation == 0) {
            fill(marks.begin(), marks.end(), 0);
            generation = 1;
        }
        for (int target : forward[from]) marks[target] = generation;
        for (int middle : forward[from])
            for (int target : forward[middle])
                if (marks[target] == generation) ++result.triangles;
    }
    return result;
}

static void write_json(const vector<Audit>& audits, ostream& output) {
    output << "{\n  \"schema_version\": 1,\n  \"graphs\": [\n";
    for (size_t i = 0; i < audits.size(); ++i) {
        const Audit& value = audits[i];
        const double average_degree =
            value.vertices ? 2.0 * value.edges / value.vertices : 0.0;
        const double transitivity =
            value.wedges ? 3.0 * value.triangles / value.wedges : 0.0;
        output << "    {\n"
               << "      \"path\": \"" << json_escape(value.path) << "\",\n"
               << "      \"vertices\": " << value.vertices << ",\n"
               << "      \"header_edges\": " << value.header_edges << ",\n"
               << "      \"input_rows\": " << value.input_rows << ",\n"
               << "      \"self_loops\": " << value.self_loops << ",\n"
               << "      \"duplicate_edges\": " << value.duplicate_edges << ",\n"
               << "      \"edges\": " << value.edges << ",\n"
               << "      \"average_degree\": " << setprecision(10)
               << average_degree << ",\n"
               << "      \"max_degree\": " << value.max_degree << ",\n"
               << "      \"degeneracy\": " << value.degeneracy << ",\n"
               << "      \"max_forward_degree\": " << value.max_forward_degree
               << ",\n"
               << "      \"wedges\": " << value.wedges << ",\n"
               << "      \"triangles\": " << value.triangles << ",\n"
               << "      \"transitivity\": " << setprecision(10) << transitivity
               << "\n    }" << (i + 1 == audits.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

int main(int argc, char** argv) {
    if (!acmsc_runtime::enforce_memory_limit()) {
        cerr << "Failed to enforce the 16 GiB process memory limit.\n";
        return 2;
    }
    if (argc < 4 || string(argv[1]) != "--output") {
        cerr << "usage: audit_graph --output OUTPUT.json GRAPH...\n";
        return 2;
    }
    const fs::path output_path = argv[2];
    vector<Audit> audits;
    for (int i = 3; i < argc; ++i) audits.push_back(audit_graph(argv[i]));
    ofstream output(output_path);
    if (!output) throw runtime_error("Cannot open output: " + output_path.string());
    write_json(audits, output);
    cout << "Audited " << audits.size() << " graph(s): " << output_path << "\n";
}
