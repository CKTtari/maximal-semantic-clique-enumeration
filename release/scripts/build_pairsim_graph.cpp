#include "../src/experiment_runtime.h"
#include "../src/semantic_graph.h"

#ifdef _WIN32
#include <malloc.h>
#endif

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define ALIGNED_ALLOC(size, align) _aligned_malloc(size, align)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(size, align) aligned_alloc(align, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

using namespace std;
namespace fs = std::filesystem;

namespace {

class Float32VectorStore {
public:
    ~Float32VectorStore() {
        if (values_) ALIGNED_FREE(values_);
    }

    void build(const Graph& graph) {
        vertices_ = graph.V;
        dimension_ = static_cast<int>(graph.getSemanticVector(0).size());
        const size_t bytes =
            (static_cast<size_t>(vertices_) * dimension_ * sizeof(float) + 63) &
            ~static_cast<size_t>(63);
        values_ = static_cast<float*>(ALIGNED_ALLOC(bytes, 64));
        if (!values_) throw bad_alloc();
#pragma omp parallel for schedule(static)
        for (int vertex = 0; vertex < vertices_; ++vertex) {
            const vector<double>& source = graph.getSemanticVector(vertex);
            float* target = values_ + static_cast<size_t>(vertex) * dimension_;
            for (int index = 0; index < dimension_; ++index)
                target[index] = static_cast<float>(source[index]);
        }
    }

    float dot(int left, int right) const {
        const float* a = values_ + static_cast<size_t>(left) * dimension_;
        const float* b = values_ + static_cast<size_t>(right) * dimension_;
        float result = 0.0f;
        int index = 0;
#ifdef __AVX2__
        __m256 sum = _mm256_setzero_ps();
        for (; index <= dimension_ - 8; index += 8)
            sum = _mm256_add_ps(sum, _mm256_mul_ps(
                                             _mm256_loadu_ps(a + index),
                                             _mm256_loadu_ps(b + index)));
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        result = _mm_cvtss_f32(lo);
#endif
        for (; index < dimension_; ++index) result += a[index] * b[index];
        return result;
    }

private:
    float* values_ = nullptr;
    int vertices_ = 0;
    int dimension_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        cerr << "Usage: build_pairsim_graph <graph> <vectors> <tau> <threads> <output>\n";
        return 2;
    }
    if (!acmsc_runtime::enforce_memory_limit()) {
        cerr << "Failed to enforce the 16 GiB memory limit.\n";
        return 2;
    }

    const string graph_path = argv[1];
    const string vector_path = argv[2];
    const float tau = static_cast<float>(stod(argv[3]));
    const int threads = max(1, min(32, stoi(argv[4])));
    const fs::path output_path = argv[5];
    if (fs::exists(output_path)) {
        cerr << "Refusing to overwrite existing output: " << output_path << '\n';
        return 2;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    Graph graph(graph_path);
    if (graph.V == 0) {
        cerr << "Failed to load graph.\n";
        return 1;
    }
    vector<set<int>>().swap(graph.adjSet);
    const bool loaded = vector_path.find(".index") != string::npos
                            ? graph.loadVectorsFromChunks(vector_path)
                            : graph.loadVectorsFromBinary(vector_path);
    if (!loaded) {
        cerr << "Failed to load vectors.\n";
        return 1;
    }
    graph.normalizeVectors();
    Float32VectorStore vectors;
    vectors.build(graph);
    vector<vector<double>>().swap(graph.semanticVectors);

    vector<pair<int, int>> source_edges;
    for (int source = 0; source < graph.V; ++source)
        for (int target : graph.adj[source])
            if (source < target) source_edges.push_back({source, target});

    vector<vector<pair<int, int>>> local_edges(threads);
#pragma omp parallel for schedule(dynamic, 4096)
    for (long long index = 0;
         index < static_cast<long long>(source_edges.size()); ++index) {
        const auto edge = source_edges[static_cast<size_t>(index)];
        if (vectors.dot(edge.first, edge.second) >= tau)
            local_edges[omp_get_thread_num()].push_back(edge);
    }

    vector<pair<int, int>> retained;
    size_t retained_count = 0;
    for (const auto& local : local_edges) retained_count += local.size();
    retained.reserve(retained_count);
    for (auto& local : local_edges)
        retained.insert(retained.end(), local.begin(), local.end());
    sort(retained.begin(), retained.end());

    fs::create_directories(output_path.parent_path());
    const fs::path temporary = output_path.string() + ".tmp";
    ofstream output(temporary, ios::binary);
    output << graph.V << ' ' << retained.size() << '\n';
    for (const auto& edge : retained)
        output << edge.first << ' ' << edge.second << '\n';
    output.close();
    fs::rename(temporary, output_path);

    cout << "vertices=" << graph.V << " source_edges=" << source_edges.size()
         << " retained_edges=" << retained.size() << " tau=" << tau << '\n';
    cout << "peak_memory_gib="
         << acmsc_runtime::peak_memory_bytes() / 1073741824.0 << '\n';
    return 0;
}
