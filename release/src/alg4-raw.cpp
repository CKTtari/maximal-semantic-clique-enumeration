// Canonical-parent DFS for average-constrained maximal semantic cliques.
//
// Feasibility follows the paper definition uniformly at every size:
//   * an edge/root is feasible iff sim >= tau;
//   * a larger state is feasible iff its semantic surplus is >= 0;
//   * terminal candidates may have zero surplus;
//   * inclusion-wise maximality is enforced by the final subset filter.
//
// Unlike the layered implementation, each feasible clique has one parent:
// delete the vertex with minimum incident-similarity sum (vertex id breaks
// ties). Deleting that vertex cannot lower the average similarity. A child is
// accepted only when the newly inserted vertex is this canonical deletion.
// Therefore no visited table or materialized size layer is needed.

#ifndef ACMSC_ENABLE_STATS
#define ACMSC_ENABLE_STATS 0
#endif

// Experimental generation-strategy switch. The standard build remains the
// canonical DFS (0); modes 1 and 2 are exact comparison variants used only by
// the ablation harness.
#ifndef ACMSC_ALG4_GENERATION
#define ACMSC_ALG4_GENERATION 0
#endif
#ifndef ACMSC_ENGINE_REORDER
#define ACMSC_ENGINE_REORDER 1
#endif
#ifndef ACMSC_ENGINE_BITMAP
#define ACMSC_ENGINE_BITMAP 1
#endif
#ifndef ACMSC_ENGINE_INCREMENTAL_ACC
#define ACMSC_ENGINE_INCREMENTAL_ACC 1
#endif

static_assert(ACMSC_ALG4_GENERATION >= 0 && ACMSC_ALG4_GENERATION <= 2,
              "ACMSC_ALG4_GENERATION must be 0, 1, or 2");

#include "experiment_runtime.h"
#include "semantic_graph.h"

#ifdef _WIN32
#include <malloc.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <immintrin.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

struct Logger {
    FILE* fp = nullptr;

    void open(const string& path) {
        if (fp) fclose(fp);
        fp = path.empty() ? nullptr : fopen(path.c_str(), "w");
        if (!path.empty() && !fp)
            fprintf(stderr, "[Logger] Cannot open log: %s\n", path.c_str());
    }

    void print(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        vprintf(fmt, ap);
        va_end(ap);
        if (fp) {
            va_start(ap, fmt);
            vfprintf(fp, fmt, ap);
            va_end(ap);
            fflush(fp);
        }
    }

    ~Logger() {
        if (fp) fclose(fp);
    }
} logger;

class BlockedSparseBitmap {
public:
    struct Block {
        uint32_t idx;
        uint64_t bits;
    };

    struct Row {
        vector<Block> blocks;

        void build(const vector<int>& neighbors) {
            blocks.clear();
            if (neighbors.empty()) return;
            unordered_map<uint32_t, uint64_t> tmp;
            tmp.reserve(neighbors.size());
            for (int v : neighbors)
                tmp[static_cast<uint32_t>(v >> 6)] |= 1ULL << (v & 63);
            blocks.reserve(tmp.size());
            for (const auto& kv : tmp) blocks.push_back({kv.first, kv.second});
            sort(blocks.begin(), blocks.end(),
                 [](const Block& a, const Block& b) { return a.idx < b.idx; });
        }

        bool contains(int v) const {
            const uint32_t idx = static_cast<uint32_t>(v >> 6);
            const uint64_t bit = 1ULL << (v & 63);
            int lo = 0;
            int hi = static_cast<int>(blocks.size()) - 1;
            while (lo <= hi) {
                const int mid = (lo + hi) >> 1;
                if (blocks[mid].idx == idx) return (blocks[mid].bits & bit) != 0;
                if (blocks[mid].idx < idx)
                    lo = mid + 1;
                else
                    hi = mid - 1;
            }
            return false;
        }

        void intersect_to(const Row& other, vector<int>& out) const {
            out.clear();
            size_t i = 0;
            size_t j = 0;
            while (i < blocks.size() && j < other.blocks.size()) {
                if (blocks[i].idx == other.blocks[j].idx) {
                    uint64_t common = blocks[i].bits & other.blocks[j].bits;
                    while (common) {
                        out.push_back(static_cast<int>(blocks[i].idx * 64 +
                                                       __builtin_ctzll(common)));
                        common &= common - 1;
                    }
                    ++i;
                    ++j;
                } else if (blocks[i].idx < other.blocks[j].idx) {
                    ++i;
                } else {
                    ++j;
                }
            }
        }
    };

    explicit BlockedSparseBitmap(int n) : rows_(n) {}
    void build(int u, const vector<int>& neighbors) { rows_[u].build(neighbors); }
    const Row& row(int u) const { return rows_[u]; }

private:
    vector<Row> rows_;
};

class Float32VectorStore {
public:
    ~Float32VectorStore() {
        if (vecs_) ALIGNED_FREE(vecs_);
    }

    void build(const Graph& graph, const vector<int>& new_to_old) {
        n_ = graph.V;
        d_ = static_cast<int>(graph.getSemanticVector(0).size());
        const size_t bytes = (static_cast<size_t>(n_) * d_ * sizeof(float) + 63) &
                             ~static_cast<size_t>(63);
        vecs_ = static_cast<float*>(ALIGNED_ALLOC(bytes, 64));
        if (!vecs_) throw bad_alloc();

#pragma omp parallel for schedule(static)
        for (int ni = 0; ni < n_; ++ni) {
            const vector<double>& source = graph.getSemanticVector(new_to_old[ni]);
            float* target = vecs_ + static_cast<size_t>(ni) * d_;
            for (int j = 0; j < d_; ++j) target[j] = static_cast<float>(source[j]);
        }
        logger.print("  Float32VectorStore: %dx%d\n", n_, d_);
    }

    inline float dot(int u, int v) const {
        const float* a = vecs_ + static_cast<size_t>(u) * d_;
        const float* b = vecs_ + static_cast<size_t>(v) * d_;
        float result = 0.0f;
        int j = 0;
#ifdef __AVX2__
        __m256 sum = _mm256_setzero_ps();
        for (; j <= d_ - 8; j += 8)
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(a + j),
                                                  _mm256_loadu_ps(b + j)));
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        result = _mm_cvtss_f32(lo);
#endif
        for (; j < d_; ++j) result += a[j] * b[j];
        return result;
    }

private:
    float* vecs_ = nullptr;
    int n_ = 0;
    int d_ = 0;
};

struct GraphReorder {
    vector<int> old_to_new;
    vector<int> new_to_old;

    void build(const Graph& graph) {
        const int n = graph.V;
        old_to_new.assign(n, -1);
        new_to_old.assign(n, -1);
#if !ACMSC_ENGINE_REORDER
        iota(old_to_new.begin(), old_to_new.end(), 0);
        iota(new_to_old.begin(), new_to_old.end(), 0);
        return;
#endif
        vector<int> by_degree(n);
        iota(by_degree.begin(), by_degree.end(), 0);
        sort(by_degree.begin(), by_degree.end(), [&](int a, int b) {
            return graph.adj[a].size() > graph.adj[b].size();
        });

        int next = 0;
        queue<int> pending;
        for (int start : by_degree) {
            if (old_to_new[start] != -1) continue;
            old_to_new[start] = next;
            new_to_old[next++] = start;
            pending.push(start);
            while (!pending.empty()) {
                const int u = pending.front();
                pending.pop();
                for (int v : graph.adj[u]) {
                    if (old_to_new[v] != -1) continue;
                    old_to_new[v] = next;
                    new_to_old[next++] = v;
                    pending.push(v);
                }
            }
        }
    }
};

struct VectorDB {
    Float32VectorStore vectors;
    GraphReorder reorder;

    void build(const Graph& graph) {
        logger.print("\n=== Building VectorDB ===\n");
        reorder.build(graph);
        vectors.build(graph, reorder.new_to_old);
        logger.print("=== DB ready ===\n\n");
    }
};

struct AccStore {
    vector<double> values;
    vector<uint32_t> generations;
    uint32_t generation = 1;

    void init(int n) {
#if ACMSC_ENGINE_INCREMENTAL_ACC
        values.assign(n, 0.0);
        generations.assign(n, 0);
#else
        (void)n;
#endif
    }

    void next_generation() {
#if ACMSC_ENGINE_INCREMENTAL_ACC
        if (++generation == 0) {
            fill(generations.begin(), generations.end(), 0);
            generation = 1;
        }
#endif
    }

    double get(int v) const {
        return generations[v] == generation ? values[v] : 0.0;
    }

    void add(int v, float value) {
        if (generations[v] != generation) {
            generations[v] = generation;
            values[v] = 0.0;
        }
        values[v] += static_cast<double>(value);
    }

    void set(int v, double value) {
        generations[v] = generation;
        values[v] = value;
    }
};

struct SearchScratch {
    deque<vector<int>> candidates;
    deque<vector<pair<int, double>>> undo;
    deque<vector<float>> member_deltas;
    deque<vector<long double>> member_old_values;

    vector<int>& candidate_buffer(int depth) {
        while (static_cast<int>(candidates.size()) <= depth) candidates.emplace_back();
        candidates[depth].clear();
        return candidates[depth];
    }

    vector<pair<int, double>>& undo_buffer(int depth) {
        while (static_cast<int>(undo.size()) <= depth) undo.emplace_back();
        undo[depth].clear();
        return undo[depth];
    }

    vector<float>& delta_buffer(int depth) {
        while (static_cast<int>(member_deltas.size()) <= depth)
            member_deltas.emplace_back();
        member_deltas[depth].clear();
        return member_deltas[depth];
    }


    vector<long double>& member_old_buffer(int depth) {
        while (static_cast<int>(member_old_values.size()) <= depth)
            member_old_values.emplace_back();
        member_old_values[depth].clear();
        return member_old_values[depth];
    }
};

struct LocalStats {
#if ACMSC_ENABLE_STATS
    uint64_t states = 0;
    uint64_t candidate_tests = 0;
    uint64_t canonical_rejects = 0;
    uint64_t w_prunes = 0;
    uint64_t duplicate_states = 0;
    vector<uint64_t> states_by_size;

    void record_state(int size) {
        ++states;
        if (static_cast<int>(states_by_size.size()) <= size)
            states_by_size.resize(size + 1, 0);
        ++states_by_size[size];
    }
#else
    void record_state(int) {}
    void record_candidate() {}
    void record_canonical_reject() {}
    void record_w_prune() {}
    void record_duplicate() {}
#endif

#if ACMSC_ENABLE_STATS
    void record_candidate() { ++candidate_tests; }
    void record_canonical_reject() { ++canonical_rejects; }
    void record_w_prune() { ++w_prunes; }
    void record_duplicate() { ++duplicate_states; }
#endif
};

struct VectorHash {
    size_t operator()(const vector<int>& values) const noexcept {
        size_t result = 1469598103934665603ULL;
        for (int value : values) {
            result ^= static_cast<size_t>(static_cast<uint32_t>(value));
            result *= 1099511628211ULL;
        }
        return result;
    }
};

struct LayerState {
    vector<int> clique;
    double sim_sum = 0.0;
};

struct RootEdge {
    int u;
    int v;
    float similarity;
};

class CanonicalAvgSimMiner {
public:
    CanonicalAvgSimMiner(const VectorDB& db, Graph& graph, int timeout_seconds)
        : db_(db), graph_(graph), timeout_seconds_(timeout_seconds) {}

    ~CanonicalAvgSimMiner() {
        if (sink_) fclose(sink_);
    }

    void set_sink(const string& path) {
        if (sink_) fclose(sink_);
        sink_ = path.empty() ? nullptr : fopen(path.c_str(), "w");
        if (!path.empty() && !sink_)
            logger.print("[Sink] Cannot open: %s\n", path.c_str());
    }

    long long mine(double tau) {
        tau_ = static_cast<float>(tau);
        timed_out_.store(false);
        collected_.clear();
        bitmap_.reset();

        const auto all_start = chrono::steady_clock::now();
        if (timeout_seconds_ > 0)
            deadline_ = all_start + chrono::seconds(timeout_seconds_);
        logger.print("\n=== %s AvgSim MCE tau=%.3f ===\n",
                     generation_label(), tau_);

        build_similarity_graph();
        if (deadline_reached()) return report_timeout(all_start);

        bitmap_.reset();
#if ACMSC_ENGINE_BITMAP
        bitmap_ = make_unique<BlockedSparseBitmap>(graph_.V);
#pragma omp parallel for schedule(static)
        for (int u = 0; u < graph_.V; ++u) {
            vector<int> ids;
            ids.reserve(neighbors_[u].size());
            for (const auto& edge : neighbors_[u]) ids.push_back(edge.first);
            bitmap_->build(u, ids);
        }
#endif

#if ACMSC_ALG4_GENERATION == 0
        run_canonical_search();
#elif ACMSC_ALG4_GENERATION == 1
        run_visited_search();
#else
        run_layered_search();
#endif
        if (timed_out_.load()) return report_timeout(all_start);

        const long long final_count = post_validate();
        const long long total_ms = chrono::duration_cast<chrono::milliseconds>(
                                       chrono::steady_clock::now() - all_start)
                                       .count();
        logger.print("\n=== TOTAL: %lld ms,  %lld maximal cliques ===\n",
                     total_ms, final_count);
        logger.print("Peak process memory: %.2f GiB (limit 16.00 GiB)\n",
                     acmsc_runtime::peak_memory_bytes() / 1073741824.0);
        if (sink_) fflush(sink_);
        return final_count;
    }

private:
    const VectorDB& db_;
    Graph& graph_;
    float tau_ = 0.0f;
    int timeout_seconds_ = acmsc_runtime::kDefaultTimeoutSeconds;
    vector<vector<pair<int, float>>> neighbors_;
    unique_ptr<BlockedSparseBitmap> bitmap_;
    vector<vector<int>> collected_;
    FILE* sink_ = nullptr;
    atomic<bool> timed_out_{false};
    chrono::steady_clock::time_point deadline_;

    static const char* generation_label() {
#if ACMSC_ALG4_GENERATION == 0
        return "Canonical DFS";
#elif ACMSC_ALG4_GENERATION == 1
        return "Visited DFS";
#else
        return "Layered";
#endif
    }

    bool deadline_reached() {
        if (timeout_seconds_ <= 0) return false;
        if (chrono::steady_clock::now() < deadline_) return false;
        timed_out_.store(true);
        return true;
    }

    long long report_timeout(chrono::steady_clock::time_point start) {
        const long long elapsed = chrono::duration_cast<chrono::milliseconds>(
                                      chrono::steady_clock::now() - start)
                                      .count();
        collected_.clear();
        logger.print("[TIMEOUT] incomplete run stopped after %lld ms\n", elapsed);
        logger.print("Peak process memory: %.2f GiB (limit 16.00 GiB)\n",
                     acmsc_runtime::peak_memory_bytes() / 1073741824.0);
        return -1;
    }

    float lookup_similarity(int u, int v) const {
        const auto& row = neighbors_[u];
        int lo = 0;
        int hi = static_cast<int>(row.size()) - 1;
        while (lo <= hi) {
            const int mid = (lo + hi) >> 1;
            if (row[mid].first == v) return row[mid].second;
            if (row[mid].first < v)
                lo = mid + 1;
            else
                hi = mid - 1;
        }
        return 0.0f;
    }

    bool adjacent(int u, int v) const {
#if ACMSC_ENGINE_BITMAP
        return bitmap_->row(u).contains(v);
#else
        const auto& row = neighbors_[u];
        const auto it = lower_bound(row.begin(), row.end(), v,
                                    [](const pair<int, float>& edge, int target) {
                                        return edge.first < target;
                                    });
        return it != row.end() && it->first == v;
#endif
    }

    void common_neighbors(int u, int v, vector<int>& output) const {
#if ACMSC_ENGINE_BITMAP
        bitmap_->row(u).intersect_to(bitmap_->row(v), output);
#else
        output.clear();
        const auto& left = neighbors_[u];
        const auto& right = neighbors_[v];
        size_t i = 0;
        size_t j = 0;
        while (i < left.size() && j < right.size()) {
            if (left[i].first == right[j].first) {
                output.push_back(left[i].first);
                ++i;
                ++j;
            } else if (left[i].first < right[j].first) {
                ++i;
            } else {
                ++j;
            }
        }
#endif
    }

    double candidate_delta(const vector<int>& clique, int candidate,
                           const AccStore& accumulator) const {
#if ACMSC_ENGINE_INCREMENTAL_ACC
        (void)clique;
        return accumulator.get(candidate);
#else
        (void)accumulator;
        double total = 0.0;
        for (int member : clique)
            total += static_cast<double>(lookup_similarity(member, candidate));
        return total;
#endif
    }

    void build_similarity_graph() {
        logger.print("Stage1: computing and caching all edge similarities...\n");
        const auto start = chrono::steady_clock::now();
        const int n = graph_.V;
        vector<vector<pair<int, float>>> half(n);
        long long total = 0;
        long long above = 0;

#pragma omp parallel for schedule(dynamic, 100) reduction(+ : total, above)
        for (int u = 0; u < n; ++u) {
            const int reordered_u = db_.reorder.old_to_new[u];
            for (int v : graph_.adj[u]) {
                if (v <= u) continue;
                const float sim = db_.vectors.dot(reordered_u,
                                                  db_.reorder.old_to_new[v]);
                half[u].push_back({v, sim});
                ++total;
                if (sim >= tau_) ++above;
            }
        }

        neighbors_.assign(n, {});
        for (int u = 0; u < n; ++u) {
            for (const auto& edge : half[u]) {
                neighbors_[u].push_back(edge);
                neighbors_[edge.first].push_back({u, edge.second});
            }
        }
        vector<vector<pair<int, float>>>().swap(half);

#pragma omp parallel for schedule(dynamic, 200)
        for (int u = 0; u < n; ++u)
            sort(neighbors_[u].begin(), neighbors_[u].end());

        const long long elapsed = chrono::duration_cast<chrono::milliseconds>(
                                      chrono::steady_clock::now() - start)
                                      .count();
        logger.print("  [Stage1] edges: %lld, sim>=%.3f: %lld\n", total, tau_, above);
        logger.print("  [Stage1 time]: %lld ms\n", elapsed);
    }

    bool is_canonical_child(const vector<int>& clique,
                            const vector<long double>& incident,
                            int added,
                            vector<float>& deltas) const {
        const int k = static_cast<int>(clique.size());
        deltas.resize(k);
        long double added_incident = 0.0L;
        for (int i = 0; i < k; ++i) {
            const float sim = lookup_similarity(clique[i], added);
            deltas[i] = sim;
            added_incident += static_cast<long double>(sim);
        }

        int minimum_vertex = added;
        long double minimum_incident = added_incident;
        for (int i = 0; i < k; ++i) {
            const long double child_incident =
                incident[i] + static_cast<long double>(deltas[i]);
            if (child_incident < minimum_incident ||
                (child_incident == minimum_incident && clique[i] < minimum_vertex)) {
                minimum_incident = child_incident;
                minimum_vertex = clique[i];
            }
        }
        return minimum_vertex == added;
    }

    void emit_candidate(const vector<int>& clique, vector<vector<int>>& output) {
        vector<int> sorted = clique;
        sort(sorted.begin(), sorted.end());
        output.push_back(move(sorted));
    }

    void search(vector<int>& clique,
                vector<long double>& member_incident,
                double sim_sum,
                vector<int>& candidates,
                AccStore& accumulator,
                SearchScratch& scratch,
                vector<vector<int>>& output,
                LocalStats& stats,
                int depth,
                uint32_t& deadline_tick) {
        const int k = static_cast<int>(clique.size());
        stats.record_state(k);
        if (timeout_seconds_ > 0 && ((++deadline_tick & 4095U) == 0) &&
            deadline_reached())
            return;

        const double edge_count = static_cast<double>(
            (static_cast<long long>(k) * (k - 1)) / 2);
        const double surplus =
            sim_sum - static_cast<double>(tau_) * edge_count;
        const double tau_k = static_cast<double>(tau_) * k;
        bool has_valid_extension = false;

        for (int v : candidates) {
            if (timed_out_.load()) return;
            stats.record_candidate();
            if (timeout_seconds_ > 0 && ((++deadline_tick & 16383U) == 0) &&
                deadline_reached())
                return;

            const double delta = candidate_delta(clique, v, accumulator);
            if (surplus + delta - tau_k < 0.0) {
                stats.record_w_prune();
                continue;
            }
            has_valid_extension = true;

            vector<float>& member_deltas = scratch.delta_buffer(depth);
            if (!is_canonical_child(clique, member_incident, v, member_deltas)) {
                stats.record_canonical_reject();
                continue;
            }

            vector<int>& child_candidates = scratch.candidate_buffer(depth);
            child_candidates.reserve(candidates.size());
            for (int w : candidates)
                if (w != v && adjacent(v, w)) child_candidates.push_back(w);

#if ACMSC_ENGINE_INCREMENTAL_ACC
            vector<pair<int, double>>& undo = scratch.undo_buffer(depth);
            undo.reserve(child_candidates.size());
            for (int w : child_candidates) {
                const float sim = lookup_similarity(w, v);
                const double old_value = accumulator.get(w);
                undo.push_back({w, old_value});
                accumulator.set(w, old_value + sim);
            }
#endif

            vector<long double>& old_incident = scratch.member_old_buffer(depth);
            old_incident.assign(member_incident.begin(), member_incident.end());
            for (int i = 0; i < k; ++i)
                member_incident[i] =
                    old_incident[i] + static_cast<long double>(member_deltas[i]);
            long double added_incident = 0.0L;
            for (float sim : member_deltas)
                added_incident += static_cast<long double>(sim);
            member_incident.push_back(added_incident);
            clique.push_back(v);

            search(clique, member_incident, sim_sum + delta, child_candidates,
                   accumulator, scratch, output, stats, depth + 1, deadline_tick);

            clique.pop_back();
            member_incident.pop_back();
            for (int i = 0; i < k; ++i)
                member_incident[i] = old_incident[i];
#if ACMSC_ENGINE_INCREMENTAL_ACC
            for (const auto& entry : undo) accumulator.set(entry.first, entry.second);
#endif
        }

        if (!timed_out_.load() && !has_valid_extension && k >= 2 && surplus >= 0.0)
            emit_candidate(clique, output);
    }

    void search_visited(vector<int>& clique,
                        double sim_sum,
                        vector<int>& candidates,
                        AccStore& accumulator,
                        SearchScratch& scratch,
                        unordered_set<vector<int>, VectorHash>& visited,
                        vector<vector<int>>& output,
                        LocalStats& stats,
                        int depth,
                        uint32_t& deadline_tick) {
        const int k = static_cast<int>(clique.size());
        stats.record_state(k);
        if (timeout_seconds_ > 0 && ((++deadline_tick & 4095U) == 0) &&
            deadline_reached())
            return;

        const double edge_count = static_cast<double>(
            (static_cast<long long>(k) * (k - 1)) / 2);
        const double surplus =
            sim_sum - static_cast<double>(tau_) * edge_count;
        const double tau_k = static_cast<double>(tau_) * k;
        bool has_valid_extension = false;

        for (int v : candidates) {
            if (timed_out_.load()) return;
            stats.record_candidate();
            if (timeout_seconds_ > 0 && ((++deadline_tick & 16383U) == 0) &&
                deadline_reached())
                return;

            const double delta = candidate_delta(clique, v, accumulator);
            if (surplus + delta - tau_k < 0.0) {
                stats.record_w_prune();
                continue;
            }
            has_valid_extension = true;

            vector<int> child_key = clique;
            child_key.push_back(v);
            sort(child_key.begin(), child_key.end());
            if (!visited.insert(child_key).second) {
                stats.record_duplicate();
                continue;
            }

            vector<int>& child_candidates = scratch.candidate_buffer(depth);
            child_candidates.reserve(candidates.size());
            for (int w : candidates)
                if (w != v && adjacent(v, w)) child_candidates.push_back(w);

#if ACMSC_ENGINE_INCREMENTAL_ACC
            vector<pair<int, double>>& undo = scratch.undo_buffer(depth);
            undo.reserve(child_candidates.size());
            for (int w : child_candidates) {
                const double old_value = accumulator.get(w);
                undo.push_back({w, old_value});
                accumulator.set(w, old_value + lookup_similarity(w, v));
            }
#endif

            clique.push_back(v);
            search_visited(clique, sim_sum + delta, child_candidates,
                           accumulator, scratch, visited, output, stats,
                           depth + 1, deadline_tick);
            clique.pop_back();
#if ACMSC_ENGINE_INCREMENTAL_ACC
            for (const auto& entry : undo) accumulator.set(entry.first, entry.second);
#endif
        }

        if (!timed_out_.load() && !has_valid_extension && k >= 2 && surplus >= 0.0)
            emit_candidate(clique, output);
    }

    vector<RootEdge> build_roots() const {
        vector<RootEdge> roots;
        for (int u = 0; u < graph_.V; ++u)
            for (const auto& edge : neighbors_[u])
                if (edge.first > u && edge.second >= tau_)
                    roots.push_back({u, edge.first, edge.second});
        return roots;
    }

    void run_canonical_search() {
        logger.print("Stage2: building canonical edge roots...\n");
        vector<RootEdge> roots = build_roots();
        logger.print("  roots: %zu (sim >= %.3f)\n", roots.size(), tau_);

        const int threads = omp_get_max_threads();
        vector<AccStore> accumulators(threads);
        vector<SearchScratch> scratch(threads);
        vector<vector<vector<int>>> output(threads);
        vector<LocalStats> stats(threads);
        vector<uint32_t> deadline_ticks(threads, 0);
        for (auto& accumulator : accumulators) accumulator.init(graph_.V);

        const auto search_start = chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long ri = 0; ri < static_cast<long long>(roots.size()); ++ri) {
            if (timed_out_.load()) continue;
            const int tid = omp_get_thread_num();
            const RootEdge root = roots[ri];
            AccStore& accumulator = accumulators[tid];
            accumulator.next_generation();

            vector<int> root_candidates;
            common_neighbors(root.u, root.v, root_candidates);
#if ACMSC_ENGINE_INCREMENTAL_ACC
            for (int w : root_candidates) {
                accumulator.add(w, lookup_similarity(w, root.u));
                accumulator.add(w, lookup_similarity(w, root.v));
            }
#endif

            vector<int> clique{root.u, root.v};
            vector<long double> incident{static_cast<long double>(root.similarity),
                                         static_cast<long double>(root.similarity)};
            search(clique, incident, root.similarity, root_candidates,
                   accumulator, scratch[tid], output[tid], stats[tid], 0,
                   deadline_ticks[tid]);
        }

        size_t output_count = 0;
        for (int t = 0; t < threads; ++t) {
            output_count += output[t].size();
#if ACMSC_ENABLE_STATS
        }

        uint64_t states = 0;
        uint64_t tests = 0;
        uint64_t rejects = 0;
        uint64_t w_prunes = 0;
        uint64_t duplicate_states = 0;
        vector<uint64_t> state_histogram;
        for (int t = 0; t < threads; ++t) {
            states += stats[t].states;
            tests += stats[t].candidate_tests;
            rejects += stats[t].canonical_rejects;
            w_prunes += stats[t].w_prunes;
            duplicate_states += stats[t].duplicate_states;
            if (state_histogram.size() < stats[t].states_by_size.size())
                state_histogram.resize(stats[t].states_by_size.size(), 0);
            for (size_t i = 0; i < stats[t].states_by_size.size(); ++i)
                state_histogram[i] += stats[t].states_by_size[i];
#endif
        }

        collected_.reserve(output_count);
        for (auto& local : output)
            for (auto& clique : local) collected_.push_back(move(clique));

        const long long elapsed = chrono::duration_cast<chrono::milliseconds>(
                                      chrono::steady_clock::now() - search_start)
                                      .count();
        logger.print("Stage3: canonical DFS: %lld ms\n", elapsed);
#if ACMSC_ENABLE_STATS
        logger.print("STAT states=%llu candidate_tests=%llu canonical_rejects=%llu "
                     "w_prunes=%llu duplicate_states=%llu raw_candidates=%zu\n",
                     static_cast<unsigned long long>(states),
                     static_cast<unsigned long long>(tests),
                     static_cast<unsigned long long>(rejects),
                     static_cast<unsigned long long>(w_prunes),
                     static_cast<unsigned long long>(duplicate_states),
                     collected_.size());
        logger.print("  Canonical state count by clique size:\n");
        for (size_t size = 2; size < state_histogram.size(); ++size)
            if (state_histogram[size])
                logger.print("    size %3zu: %llu\n", size,
                             static_cast<unsigned long long>(state_histogram[size]));
#endif
    }

    void run_visited_search() {
        logger.print("Stage2: building visited-DFS edge roots...\n");
        vector<RootEdge> roots = build_roots();
        logger.print("  roots: %zu (sim >= %.3f)\n", roots.size(), tau_);

        const int threads = omp_get_max_threads();
        vector<AccStore> accumulators(threads);
        vector<SearchScratch> scratch(threads);
        vector<vector<vector<int>>> output(threads);
        vector<LocalStats> stats(threads);
        vector<uint32_t> deadline_ticks(threads, 0);
        for (auto& accumulator : accumulators) accumulator.init(graph_.V);

        const auto search_start = chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long ri = 0; ri < static_cast<long long>(roots.size()); ++ri) {
            if (timed_out_.load()) continue;
            const int tid = omp_get_thread_num();
            const RootEdge root = roots[ri];
            AccStore& accumulator = accumulators[tid];
            accumulator.next_generation();

            vector<int> root_candidates;
            common_neighbors(root.u, root.v, root_candidates);
#if ACMSC_ENGINE_INCREMENTAL_ACC
            for (int w : root_candidates) {
                accumulator.add(w, lookup_similarity(w, root.u));
                accumulator.add(w, lookup_similarity(w, root.v));
            }
#endif

            vector<int> clique{root.u, root.v};
            unordered_set<vector<int>, VectorHash> visited;
            visited.reserve(1024);
            visited.insert(clique);
            search_visited(clique, root.similarity, root_candidates,
                           accumulator, scratch[tid], visited, output[tid],
                           stats[tid], 0, deadline_ticks[tid]);
        }

        size_t output_count = 0;
        for (const auto& local : output) output_count += local.size();
        collected_.reserve(output_count);
        for (auto& local : output)
            for (auto& clique : local) collected_.push_back(move(clique));

        const long long elapsed = chrono::duration_cast<chrono::milliseconds>(
                                      chrono::steady_clock::now() - search_start)
                                      .count();
        logger.print("Stage3: visited DFS: %lld ms\n", elapsed);
#if ACMSC_ENABLE_STATS
        uint64_t states = 0;
        uint64_t tests = 0;
        uint64_t duplicates = 0;
        uint64_t w_prunes = 0;
        for (const auto& local : stats) {
            states += local.states;
            tests += local.candidate_tests;
            duplicates += local.duplicate_states;
            w_prunes += local.w_prunes;
        }
        logger.print("STAT states=%llu candidate_tests=%llu canonical_rejects=0 "
                     "w_prunes=%llu duplicate_states=%llu raw_candidates=%zu\n",
                     static_cast<unsigned long long>(states),
                     static_cast<unsigned long long>(tests),
                     static_cast<unsigned long long>(w_prunes),
                     static_cast<unsigned long long>(duplicates),
                     collected_.size());
#endif
    }

    void run_layered_search() {
        logger.print("Stage2: building initial feasible edge layer...\n");
        vector<RootEdge> roots = build_roots();
        vector<LayerState> layer;
        layer.reserve(roots.size());
        for (const RootEdge& root : roots)
            layer.push_back({{root.u, root.v}, root.similarity});
        logger.print("  roots: %zu (sim >= %.3f)\n", roots.size(), tau_);

        const int threads = omp_get_max_threads();
        vector<vector<LayerState>> next_local(threads);
        vector<vector<vector<int>>> terminal_local(threads);
        LocalStats stats;
        uint64_t duplicate_states = 0;
        size_t max_layer_states = layer.size();
        const auto search_start = chrono::steady_clock::now();

        while (!layer.empty() && !timed_out_.load()) {
            for (auto& local : next_local) local.clear();
            for (auto& local : terminal_local) local.clear();
            const int k = static_cast<int>(layer.front().clique.size());

#pragma omp parallel for schedule(dynamic, 16)
            for (long long index = 0; index < static_cast<long long>(layer.size());
                 ++index) {
                if (timed_out_.load()) continue;
                const int tid = omp_get_thread_num();
                const LayerState& state = layer[static_cast<size_t>(index)];
                bool has_valid_extension = false;

                if (timeout_seconds_ > 0 && ((index & 4095LL) == 0) &&
                    deadline_reached())
                    continue;

                vector<int> candidates;
                const int first = state.clique.front();
                candidates.reserve(neighbors_[first].size());
                for (const auto& edge : neighbors_[first]) {
                    const int v = edge.first;
                    bool common = true;
                    for (size_t ci = 1; ci < state.clique.size(); ++ci) {
                    if (!adjacent(state.clique[ci], v)) {
                            common = false;
                            break;
                        }
                    }
                    if (common) candidates.push_back(v);
                }

                const double edge_count = static_cast<double>(
                    (static_cast<long long>(k) * (k - 1)) / 2);
                const double surplus =
                    state.sim_sum - static_cast<double>(tau_) * edge_count;
                const double tau_k = static_cast<double>(tau_) * k;
                for (int v : candidates) {
#if ACMSC_ENABLE_STATS
#pragma omp atomic update
                    stats.candidate_tests++;
#endif
                    double delta = 0.0;
                    for (int u : state.clique)
                        delta += static_cast<double>(lookup_similarity(u, v));
                    if (surplus + delta - tau_k < 0.0) {
#if ACMSC_ENABLE_STATS
#pragma omp atomic update
                        stats.w_prunes++;
#endif
                        continue;
                    }
                    has_valid_extension = true;
                    LayerState child;
                    child.clique = state.clique;
                    child.clique.push_back(v);
                    sort(child.clique.begin(), child.clique.end());
                    child.sim_sum = state.sim_sum + delta;
                    next_local[tid].push_back(move(child));
                }

                if (!has_valid_extension && surplus >= 0.0)
                    terminal_local[tid].push_back(state.clique);
            }

#if ACMSC_ENABLE_STATS
            stats.states += static_cast<uint64_t>(layer.size());
            if (stats.states_by_size.size() <= static_cast<size_t>(k))
                stats.states_by_size.resize(static_cast<size_t>(k) + 1, 0);
            stats.states_by_size[static_cast<size_t>(k)] += layer.size();
#endif
            for (auto& local : terminal_local)
                for (auto& clique : local) collected_.push_back(move(clique));

            vector<LayerState> next;
            size_t generated = 0;
            for (const auto& local : next_local) generated += local.size();
            next.reserve(generated);
            for (auto& local : next_local)
                for (auto& state : local) next.push_back(move(state));
            sort(next.begin(), next.end(), [](const LayerState& a, const LayerState& b) {
                return a.clique < b.clique;
            });
            const auto unique_end = unique(
                next.begin(), next.end(),
                [](const LayerState& a, const LayerState& b) {
                    return a.clique == b.clique;
                });
            duplicate_states += static_cast<uint64_t>(next.end() - unique_end);
            next.erase(unique_end, next.end());
            max_layer_states = max(max_layer_states, next.size());
            layer.swap(next);
        }

        const long long elapsed = chrono::duration_cast<chrono::milliseconds>(
                                      chrono::steady_clock::now() - search_start)
                                      .count();
        logger.print("Stage3: layered generation: %lld ms\n", elapsed);
#if ACMSC_ENABLE_STATS
        logger.print("STAT states=%llu candidate_tests=%llu canonical_rejects=0 "
                     "w_prunes=%llu duplicate_states=%llu max_layer_states=%zu "
                     "raw_candidates=%zu\n",
                     static_cast<unsigned long long>(stats.states),
                     static_cast<unsigned long long>(stats.candidate_tests),
                     static_cast<unsigned long long>(stats.w_prunes),
                     static_cast<unsigned long long>(duplicate_states),
                     max_layer_states, collected_.size());
#endif
    }

    long long post_validate() {
        logger.print("Stage4: inclusion-maximal filtering...\n");
        const auto start = chrono::steady_clock::now();
        const size_t raw = collected_.size();
        sort(collected_.begin(), collected_.end(),
             [](const vector<int>& a, const vector<int>& b) {
                 if (a.size() != b.size()) return a.size() > b.size();
                 return a < b;
             });
        const auto duplicate_end = unique(collected_.begin(), collected_.end());
        const size_t generated_duplicates =
            static_cast<size_t>(collected_.end() - duplicate_end);
        collected_.erase(duplicate_end, collected_.end());

        // The index contains retained strict supersets only. Equal-size
        // cliques query a frozen index before the group is inserted.
        vector<vector<size_t>> inverted(graph_.V);
        vector<unsigned char> valid(collected_.size(), 1);
        size_t subset_pruned = 0;
        for (size_t begin = 0; begin < collected_.size();) {
            size_t end = begin + 1;
            while (end < collected_.size() &&
                   collected_[end].size() == collected_[begin].size())
                ++end;

            const long long group_size = static_cast<long long>(end - begin);
#pragma omp parallel for if(group_size >= 4096) schedule(dynamic, 64)
            for (long long offset = 0; offset < group_size; ++offset) {
                const size_t ci = begin + static_cast<size_t>(offset);
                const vector<int>& clique = collected_[ci];
                int rarest = clique.front();
                size_t shortest = inverted[rarest].size();
                for (int v : clique) {
                    const size_t length = inverted[v].size();
                    if (length < shortest) {
                        shortest = length;
                        rarest = v;
                    }
                }

                for (size_t larger_index : inverted[rarest]) {
                    const vector<int>& larger = collected_[larger_index];
                    if (includes(larger.begin(), larger.end(),
                                 clique.begin(), clique.end())) {
                        valid[ci] = 0;
                        break;
                    }
                }
            }

            for (size_t ci = begin; ci < end; ++ci) {
                if (!valid[ci]) {
                    ++subset_pruned;
                    continue;
                }
                for (int v : collected_[ci]) inverted[v].push_back(ci);
            }
            begin = end;
        }

        map<int, long long> size_distribution;
        long long sum_sizes = 0;
        for (size_t i = 0; i < collected_.size(); ++i) {
            if (!valid[i]) continue;
            ++size_distribution[static_cast<int>(collected_[i].size())];
            sum_sizes += static_cast<long long>(collected_[i].size());
            if (sink_) {
                for (size_t j = 0; j < collected_[i].size(); ++j) {
                    if (j) fputc(' ', sink_);
                    fprintf(sink_, "%d", collected_[i][j]);
                }
                fputc('\n', sink_);
            }
        }

        const size_t final_count = collected_.size() - subset_pruned;
        const long long elapsed = chrono::duration_cast<chrono::milliseconds>(
                                      chrono::steady_clock::now() - start)
                                      .count();
        logger.print("  raw=%zu generated_duplicates=%zu subset_pruned=%zu\n",
                     raw, generated_duplicates, subset_pruned);
        logger.print("  [Stage4 time]: %lld ms\n", elapsed);
        logger.print("\n--- Clique size distribution ---\n");
        for (const auto& item : size_distribution)
            logger.print("  size %3d: %lld (%.2f%%)\n", item.first, item.second,
                         final_count ? 100.0 * item.second / final_count : 0.0);
        if (final_count)
            logger.print("  Average clique size: %.2f\n",
                         static_cast<double>(sum_sizes) / final_count);
        logger.print("--- Total cliques: %zu ---\n", final_count);
        return static_cast<long long>(final_count);
    }
};

}  // namespace

int main(int argc, char* argv[]) {
    if (!acmsc_runtime::enforce_memory_limit()) {
#ifdef _WIN32
        fprintf(stderr, "Failed to enforce 16 GiB process memory limit (error %lu).\n",
                static_cast<unsigned long>(GetLastError()));
#else
        fprintf(stderr, "Failed to enforce 16 GiB process memory limit.\n");
#endif
        return 2;
    }

    int dataset_id = 2;
    int requested_threads = acmsc_runtime::kDefaultThreads;
    int timeout_seconds = acmsc_runtime::kDefaultTimeoutSeconds;
    string log_path;
    string graph_override;
    string vector_override;
    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg == "dataset" && i + 1 < argc)
            dataset_id = atoi(argv[++i]);
        else if (arg == "threads" && i + 1 < argc)
            requested_threads = atoi(argv[++i]);
        else if (arg == "timeout" && i + 1 < argc)
            timeout_seconds = atoi(argv[++i]);
        else if (arg == "log" && i + 1 < argc)
            log_path = argv[++i];
        else if (arg == "graph" && i + 1 < argc)
            graph_override = argv[++i];
        else if (arg == "vectors" && i + 1 < argc)
            vector_override = argv[++i];
    }

    const int threads = max(1, min(requested_threads, omp_get_num_procs()));
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    if (!log_path.empty()) logger.open(log_path);

    string graph_file;
    string vector_file;
    acmsc_runtime::select_dataset(dataset_id, graph_file, vector_file);
    if (!graph_override.empty()) graph_file = graph_override;
    if (!vector_override.empty()) vector_file = vector_override;
    logger.print("============================================\n");
    logger.print("  Alg4 Canonical-Parent DFS\n");
    logger.print("  mine <tau> | sink <file> | log <file> | quit\n");
    logger.print("============================================\n");
    logger.print("Dataset: %s (ID=%d)\n", graph_file.c_str(), dataset_id);
    if (timeout_seconds > 0)
        logger.print("Threads: %d, timeout: %ds, memory limit: 16 GiB\n",
                     threads, timeout_seconds);
    else
        logger.print("Threads: %d, timeout: external/unlimited, memory limit: 16 GiB\n",
                     threads);

    const auto load_start = chrono::steady_clock::now();
    Graph graph(graph_file);
    if (graph.V == 0) {
        fprintf(stderr, "Failed to load graph.\n");
        return 1;
    }

    vector<set<int>>().swap(graph.adjSet);
    bool vectors_loaded = false;
    if (fs::exists(vector_file)) {
        vectors_loaded = vector_file.find(".index") != string::npos
                             ? graph.loadVectorsFromChunks(vector_file)
                             : graph.loadVectorsFromBinary(vector_file);
    }
    if (!vectors_loaded) {
        fprintf(stderr, "Failed to load vectors: %s\n", vector_file.c_str());
        return 1;
    }
    graph.normalizeVectors();

    long long edge_count = 0;
    for (const auto& row : graph.adj) edge_count += static_cast<long long>(row.size());
    edge_count /= 2;
    logger.print("Graph: V=%d E=%lld\n", graph.V, edge_count);
    logger.print("Load: %lld ms\n",
                 static_cast<long long>(chrono::duration_cast<chrono::milliseconds>(
                                            chrono::steady_clock::now() - load_start)
                                            .count()));

    VectorDB db;
    db.build(graph);
    vector<vector<double>>().swap(graph.semanticVectors);

    CanonicalAvgSimMiner miner(db, graph, timeout_seconds);
    logger.print("\n>> ");
    string line;
    while (getline(cin, line)) {
        if (line.empty()) {
            logger.print(">> ");
            continue;
        }
        istringstream input(line);
        string command;
        input >> command;
        if (command == "mine") {
            double tau = 0.6;
            input >> tau;
            try {
                miner.mine(tau);
            } catch (const bad_alloc&) {
                fprintf(stderr,
                        "bad_alloc: the 16 GiB process memory limit was reached.\n");
                return 3;
            }
        } else if (command == "sink") {
            string path;
            input >> path;
            miner.set_sink(path);
            logger.print("Sink: %s\n", path.c_str());
        } else if (command == "log") {
            string path;
            input >> path;
            logger.open(path);
            logger.print("Log: %s\n", path.c_str());
        } else if (command == "quit" || command == "exit") {
            break;
        } else {
            logger.print("Commands: mine <tau> | sink <file> | log <file> | quit\n");
        }
        logger.print(">> ");
    }
    logger.print("Bye.\n");
    return 0;
}
