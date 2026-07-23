#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <cstdlib>
#include <omp.h>
#include <cstdarg>
#include <immintrin.h>
#include <filesystem>

#include "semantic_graph.h"

using namespace std;
namespace fs = std::filesystem;

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(size, align) _aligned_malloc(size, align)
#define ALIGNED_FREE(ptr)          _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(size, align) aligned_alloc(align, size)
#define ALIGNED_FREE(ptr)          free(ptr)
#endif

// SIMD Float32VectorStore (copied from alg4-raw.cpp)
class Float32VectorStore {
    float* vecs_ = nullptr; int n_ = 0, d_ = 0;
public:
    ~Float32VectorStore() { if (vecs_) ALIGNED_FREE(vecs_); }
    int dim() const { return d_; }
    void build(const Graph& g) {
        n_ = g.V; d_ = (int)g.getSemanticVector(0).size();
        size_t vb = ((size_t)n_*d_*sizeof(float)+63) & ~63ULL;
        vecs_ = (float*)ALIGNED_ALLOC(vb, 64);
#pragma omp parallel for schedule(static)
        for (int ni = 0; ni < n_; ni++) {
            const vector<double>& dv = g.getSemanticVector(ni);
            float* xf = vecs_ + (size_t)ni*d_;
            for (int i = 0; i < d_; i++) xf[i] = (float)dv[i];
        }
    }
    inline float dot(int nu, int nv) const {
        const float* xu = vecs_ + (size_t)nu*d_;
        const float* xv = vecs_ + (size_t)nv*d_;
        float p = 0.f; int i = 0;
#ifdef __AVX2__
        __m256 acc = _mm256_setzero_ps();
        for (; i <= d_-8; i += 8)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(
                _mm256_loadu_ps(xu+i), _mm256_loadu_ps(xv+i)));
        { __m128 hi = _mm256_extractf128_ps(acc,1), lo = _mm256_castps256_ps128(acc);
          lo = _mm_add_ps(lo,hi); lo = _mm_hadd_ps(lo,lo); lo = _mm_hadd_ps(lo,lo);
          p = _mm_cvtss_f32(lo); }
#endif
        for (; i < d_; i++) p += xu[i]*xv[i];
        return p;
    }
};

struct DatasetConfig {
    int id;
    string name;
    string graph_file;
    string vec_index;
};

vector<DatasetConfig> DATASETS = {
    {1, "sc-ldoor", "dataset/dataset/sc-ldoor.txt", "dataset/vectors-256/sc-ldoor_vectors.bin"},
    {2, "sc-nasasrb", "dataset/dataset/sc-nasasrb.txt", "dataset/vectors-256/sc-nasasrb_vectors.bin"},
    {3, "sc-pkustk11", "dataset/dataset/sc-pkustk11.txt", "dataset/vectors-256/sc-pkustk11_vectors.bin"},
    {4, "soc-buzznet", "dataset/dataset/soc-buzznet.txt", "dataset/vectors-256/soc-buzznet_vectors.bin"},
    {5, "soc-digg", "dataset/dataset/soc-digg.txt", "dataset/vectors-256/soc-digg_vectors.bin"},
    {6, "tech-as-skitter", "dataset/dataset/tech-as-skitter.txt", "dataset/vectors-256/tech-as-skitter_vectors.bin"},
    {7, "FB15K-237", "dataset/dataset/FB15K-237_edges.txt", "dataset/vectors/FB15K-237_vectors.index"},
    {8, "WN18RR", "dataset/dataset/WN18RR_edges.txt", "dataset/vectors/WN18RR_vectors.index"},
    {9, "DBLP", "dataset/dataset/dblp_coauthor.txt", "dataset/vectors/dblp_coauthor_vectors.index"}
};

int main() {
    vector<int> PERCENTILES = {10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 93, 95, 97, 99};

    cout << "{" << endl;
    bool first_ds = true;

    for (const auto& cfg : DATASETS) {
        if (!first_ds) cout << "," << endl;
        first_ds = false;

        cout << "  \"" << cfg.id << "\": {" << endl;
        cout << "    \"name\": \"" << cfg.name << "\"," << endl;

        // Temporarily redirect cout to cerr so Graph loader messages don't pollute JSON
        auto old_cout_buf = cout.rdbuf(cerr.rdbuf());

        // Load graph (exactly as alg4-raw.cpp)
        Graph g(cfg.graph_file);
        if (g.V == 0) {
            cerr << "Failed to load graph: " << cfg.graph_file << endl;
            cout.rdbuf(old_cout_buf);
            continue;
        }

        // Load vectors (exactly as alg4-raw.cpp)
        if (fs::exists(cfg.vec_index)) {
            bool ok = (cfg.vec_index.find(".index") != string::npos)
                      ? g.loadVectorsFromChunks(cfg.vec_index) : g.loadVectorsFromBinary(cfg.vec_index);
            if (!ok) { cerr << "Failed to load vectors: " << cfg.vec_index << endl; cout.rdbuf(old_cout_buf); continue; }
        } else { cerr << "Vector file not found: " << cfg.vec_index << endl; cout.rdbuf(old_cout_buf); continue; }

        g.normalizeVectors();

        // Restore cout
        cout.rdbuf(old_cout_buf);

        // Build Float32VectorStore for fast SIMD dot product
        Float32VectorStore vecs;
        vecs.build(g);

        long long ec = 0;
        for (int i = 0; i < g.V; i++) ec += (long long)g.adj[i].size();
        ec /= 2;
        cout << "    \"vertices\": " << g.V << "," << endl;
        cout << "    \"edges\": " << ec << "," << endl;

        // Collect edges
        vector<pair<int,int>> edges;
        edges.reserve(ec);
        for (int u = 0; u < g.V; ++u) {
            for (int v : g.adj[u]) {
                if (u < v) edges.emplace_back(u, v);
            }
        }

        // Compute similarities in parallel using SIMD dot product
        size_t E = edges.size();
        vector<double> sims(E);

        #pragma omp parallel for schedule(dynamic, 4096)
        for (size_t i = 0; i < E; ++i) {
            sims[i] = vecs.dot(edges[i].first, edges[i].second);
        }

        // Sort
        sort(sims.begin(), sims.end());

        cout << "    \"min\": " << sims.front() << "," << endl;
        cout << "    \"max\": " << sims.back() << "," << endl;

        for (size_t i = 0; i < PERCENTILES.size(); ++i) {
            int p = PERCENTILES[i];
            size_t idx = static_cast<size_t>((p / 100.0) * (E - 1));
            if (idx >= E) idx = E - 1;
            cout << "    \"tau_" << p << "\": " << sims[idx];
            if (i + 1 < PERCENTILES.size()) cout << ",";
            cout << endl;
        }

        cout << "  }";
    }

    cout << endl << "}" << endl;
    return 0;
}
