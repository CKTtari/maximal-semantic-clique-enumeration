// ============================================================

#ifndef ACMSC_ENABLE_STATS
#define ACMSC_ENABLE_STATS 0
#endif
#ifndef ACMSC_ENGINE_REORDER
#define ACMSC_ENGINE_REORDER 1
#endif
#ifndef ACMSC_ENGINE_BITMAP
#define ACMSC_ENGINE_BITMAP 0
#endif
#ifndef ACMSC_ENGINE_INCREMENTAL_ACC
#define ACMSC_ENGINE_INCREMENTAL_ACC 0
#endif
// ALG2 standard: pivot-free semantic BK baseline.
//
// 特性：
//   - 输出信息保存到日志（Logger 同时写 stdout + 文件）
//   - 分阶段计时：build_graph / degeneracy / search / 总计
//   - 输出团总数 + size 分布
//   - 无任何统计计数器，时间效率纯洁准确
// ============================================================

#include "experiment_runtime.h"
#include "semantic_graph.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <immintrin.h>
#include <string>
#include <sstream>
#include <queue>
#include <cassert>
#include <cstdarg>
#include <memory>

#ifdef _WIN32
    #include <malloc.h>
    #define ALIGNED_ALLOC(size, align) _aligned_malloc(size, align)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #define ALIGNED_ALLOC(size, align) aligned_alloc(align, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

using namespace std;
namespace fs = std::filesystem;

// ============================================================
// Logger：同时写 stdout + 日志文件
// ============================================================
struct Logger {
    FILE* fp = nullptr;
    void open(const string& path) {
        if (fp) { fclose(fp); fp = nullptr; }
        if (!path.empty()) {
            fp = fopen(path.c_str(), "w");
            if (!fp) fprintf(stderr, "[Logger] Cannot open: %s\n", path.c_str());
        }
    }
    ~Logger() { if (fp) fclose(fp); }
    void print(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
        if (fp) {
            va_start(ap, fmt); vfprintf(fp, fmt, ap); va_end(ap);
            fflush(fp);
        }
    }
} logger;

// ============================================================
// BlockedSparseBitmap
// 邻接关系的 bitmap 表示，支持 O(|row|/64) 交集。
// ============================================================
class BlockedSparseBitmap {
public:
    struct Block { uint32_t idx; uint64_t bits; };
    struct Row {
        vector<Block> blocks;
        void build(const vector<int>& neighbors) {
            blocks.clear();
            if (neighbors.empty()) return;
            unordered_map<uint32_t,uint64_t> tmp;
            tmp.reserve(neighbors.size());
            for (int v : neighbors) tmp[(uint32_t)(v>>6)] |= 1ULL<<(v&63);
            blocks.reserve(tmp.size());
            for (auto& kv : tmp) blocks.push_back({kv.first, kv.second});
            sort(blocks.begin(), blocks.end(),
                 [](const Block& a, const Block& b){ return a.idx < b.idx; });
        }
        bool contains(int v) const {
            uint32_t bidx=(uint32_t)(v>>6); uint64_t bbit=1ULL<<(v&63);
            int lo=0, hi=(int)blocks.size()-1;
            while (lo<=hi) {
                int mid=(lo+hi)>>1;
                if (blocks[mid].idx==bidx) return (blocks[mid].bits&bbit)!=0;
                blocks[mid].idx<bidx ? lo=mid+1 : hi=mid-1;
            }
            return false;
        }
        void intersect_to(const Row& other, vector<int>& out) const {
            out.clear();
            if (blocks.empty()||other.blocks.empty()) return;
            size_t i=0,j=0;
            while (i<blocks.size()&&j<other.blocks.size()) {
                if (blocks[i].idx==other.blocks[j].idx) {
                    uint64_t common=blocks[i].bits&other.blocks[j].bits;
                    while (common) {
                        out.push_back((int)(blocks[i].idx*64+__builtin_ctzll(common)));
                        common&=common-1;
                    }
                    i++; j++;
                } else if (blocks[i].idx<other.blocks[j].idx) i++; else j++;
            }
        }
        int intersect_count(const Row& other) const {
            if (blocks.empty()||other.blocks.empty()) return 0;
            int cnt=0; size_t i=0,j=0;
            while (i<blocks.size()&&j<other.blocks.size()) {
                if (blocks[i].idx==other.blocks[j].idx) {
                    cnt+=__builtin_popcountll(blocks[i].bits&other.blocks[j].bits); i++; j++;
                } else if (blocks[i].idx<other.blocks[j].idx) i++; else j++;
            }
            return cnt;
        }
    };
private:
    vector<Row> rows_; int n_;
public:
    BlockedSparseBitmap() : n_(0) {}
    explicit BlockedSparseBitmap(int n) : n_(n), rows_(n) {}
    void build(int u, const vector<int>& nb) { rows_[u].build(nb); }
    bool contains(int u, int v) const { return rows_[u].contains(v); }
    const Row& get_row(int u) const { return rows_[u]; }
    int size() const { return n_; }
};

// ============================================================
// Float32VectorStore
// 节点向量的 float32 紧凑存储，支持 AVX2 加速内积。
// ============================================================
class Float32VectorStore {
    float* vecs_ = nullptr;
    int n_ = 0, d_ = 0;
public:
    ~Float32VectorStore() { if (vecs_) ALIGNED_FREE(vecs_); }
    int dim() const { return d_; }

    void build(const Graph& g, const vector<int>& order = {}) {
        n_ = g.V;
        d_ = (int)g.getSemanticVector(0).size();
        size_t vb = ((size_t)n_ * d_ * sizeof(float) + 63) & ~63ULL;
        vecs_ = (float*)ALIGNED_ALLOC(vb, 64);
        bool has_order = !order.empty();
        #pragma omp parallel for schedule(static)
        for (int ni = 0; ni < n_; ni++) {
            int oi = has_order ? order[ni] : ni;
            const vector<double>& dv = g.getSemanticVector(oi);
            float* xf = vecs_ + (size_t)ni * d_;
            for (int i = 0; i < d_; i++) xf[i] = (float)dv[i];
        }
        logger.print("  Float32VectorStore: %dx%d\n", n_, d_);
    }

    inline float dot(int nu, int nv) const {
        const float* xu = vecs_ + (size_t)nu * d_;
        const float* xv = vecs_ + (size_t)nv * d_;
        float p = 0.f; int i = 0;
#ifdef __AVX2__
        __m256 acc = _mm256_setzero_ps();
        for (; i <= d_-8; i+=8)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(
                _mm256_loadu_ps(xu+i), _mm256_loadu_ps(xv+i)));
        { __m128 hi = _mm256_extractf128_ps(acc, 1),
                 lo = _mm256_castps256_ps128(acc);
          lo = _mm_add_ps(lo, hi);
          lo = _mm_hadd_ps(lo, lo);
          lo = _mm_hadd_ps(lo, lo);
          p  = _mm_cvtss_f32(lo); }
        for (; i < d_; i++) p += xu[i] * xv[i];
#else
        for (; i < d_; i++) p += xu[i] * xv[i];
#endif
        return p;
    }
};

// ============================================================
// GraphReorder
// 按度数降序做 BFS 重排，改善向量访问局部性。
// ============================================================
struct GraphReorder {
    vector<int> old_to_new, new_to_old;
    void build(const Graph& g) {
        int n = g.V;
        old_to_new.assign(n, -1);
        new_to_old.assign(n, -1);
#if !ACMSC_ENGINE_REORDER
        iota(old_to_new.begin(), old_to_new.end(), 0);
        iota(new_to_old.begin(), new_to_old.end(), 0);
        return;
#endif
        vector<int> order(n);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(),
             [&](int a, int b){ return g.adj[a].size() > g.adj[b].size(); });
        int nid = 0;
        queue<int> q;
        for (int s : order) {
            if (old_to_new[s] != -1) continue;
            old_to_new[s] = nid; new_to_old[nid] = s; nid++; q.push(s);
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (int v : g.adj[u]) if (old_to_new[v] == -1) {
                    old_to_new[v] = nid; new_to_old[nid] = v; nid++; q.push(v);
                }
            }
        }
    }
};

// ============================================================
// VectorDB
// 持有重排序映射和向量存储，供相似度计算使用。
// ============================================================
struct VectorDB {
    Float32VectorStore vecs;
    GraphReorder       reorder;
    const Graph*       graph = nullptr;

    void build(const Graph& g) {
        graph = &g;
        logger.print("\n=== Building VectorDB ===\n");
        reorder.build(g);
        vecs.build(g, reorder.new_to_old);
        logger.print("=== DB ready ===\n\n");
    }
};

// ============================================================
// AvgSimMCEMiner
// ============================================================
class AvgSimMCEMiner {
private:
    const VectorDB& db_;
    const Graph&    g_;
    float           tau_;

    vector<vector<pair<int,float>>> nb_;
    vector<unordered_map<int,float>> sim_map_;

    BlockedSparseBitmap* bitmap_ = nullptr;
    vector<int>          ordering_;

    std::atomic<long long> clique_count_{0};
#if ACMSC_ENABLE_STATS
    std::atomic<unsigned long long> search_states_{0};
    std::atomic<unsigned long long> candidate_expansions_{0};
    std::atomic<unsigned long long> check_states_{0};
    std::atomic<unsigned long long> check_calls_{0};
    std::atomic<unsigned long long> feasible_terminals_{0};
#endif
    std::unique_ptr<std::atomic<long long>[]> size_hist_;
    int                    size_hist_size_ = 0;
    FILE*                  sink_ = nullptr;
    std::mutex             sink_mutex_;
    int                    timeout_seconds_ = 0;
    std::atomic<bool>      timed_out_{false};
    chrono::steady_clock::time_point deadline_;

    bool deadline_reached() {
        if (timeout_seconds_ <= 0 || chrono::steady_clock::now() < deadline_)
            return false;
        timed_out_.store(true, std::memory_order_relaxed);
        return true;
    }

    inline void emit(const vector<int>& C) {
        int sz = (int)C.size();
        if (sz <= 1) return;
        clique_count_.fetch_add(1, std::memory_order_relaxed);
        if (sz < size_hist_size_)
            size_hist_[sz].fetch_add(1, std::memory_order_relaxed);
        if (sink_) {
            std::lock_guard<std::mutex> lk(sink_mutex_);
            for (int i = 0; i < sz; i++) {
                if (i) fputc(' ', sink_);
                fprintf(sink_, "%d", C[i]);
            }
            fputc('\n', sink_);
        }
    }

    inline float lookup_sim(int u, int v) const {
        auto it = sim_map_[u].find(v);
        return it != sim_map_[u].end() ? it->second : 0.0f;
    }

    inline bool adjacent(int u, int v) const {
#if ACMSC_ENGINE_BITMAP
        return bitmap_->contains(u, v);
#else
        return sim_map_[u].find(v) != sim_map_[u].end();
#endif
    }

    void intersect_neighbors(int v, const vector<int>& vertices,
                             vector<int>& output) const {
        output.clear();
        output.reserve(vertices.size());
#if ACMSC_ENGINE_BITMAP
        const auto& row = bitmap_->get_row(v);
        if (vertices.size() >= 64) {
            BlockedSparseBitmap::Row vertex_row;
            vertex_row.build(vertices);
            row.intersect_to(vertex_row, output);
            return;
        }
#endif
        for (int u : vertices)
            if (adjacent(v, u)) output.push_back(u);
    }

    struct AccStore {
        vector<double>   val;
        vector<uint32_t> gen;
        uint32_t         cur = 0;
        void init(int n) {
#if ACMSC_ENGINE_INCREMENTAL_ACC
            val.assign(n, 0.0);
            gen.assign(n, 0);
#else
            (void)n;
#endif
            cur = 1;
        }
        inline double get(int w) const { return gen[w]==cur ? val[w] : 0.0; }
        inline void  add(int w, float s) {
#if ACMSC_ENGINE_INCREMENTAL_ACC
            if (gen[w] != cur) { val[w] = 0.0; gen[w] = cur; }
            val[w] += static_cast<double>(s);
#else
            (void)w; (void)s;
#endif
        }
        inline void  sub(int w, float s) {
#if ACMSC_ENGINE_INCREMENTAL_ACC
            if (gen[w] == cur) val[w] -= static_cast<double>(s);
#else
            (void)w; (void)s;
#endif
        }
    };

    bool search(
        vector<int>& C,
        double       sim_sum,
        vector<int>& cands,
        vector<int>& excl,
        AccStore&    acc,
        bool         is_check,
        int          min_size = 0
    ) {
        if (deadline_reached()) return false;
#if ACMSC_ENABLE_STATS
        search_states_.fetch_add(1, std::memory_order_relaxed);
        if (is_check) check_states_.fetch_add(1, std::memory_order_relaxed);
#endif
        bool found_any_longer = false;
        int  k = (int)C.size();

        vector<int> to_enumerate(cands.begin(), cands.end());

        for (int v : to_enumerate) {
#if ACMSC_ENABLE_STATS
            candidate_expansions_.fetch_add(1, std::memory_order_relaxed);
#endif
            cands.erase(find(cands.begin(), cands.end(), v));

            vector<int> cands_new;
            cands_new.reserve(cands.size());
            for (int u : cands)
                if (adjacent(v, u)) cands_new.push_back(u);

            vector<int> excl_new;
            if (!excl.empty()) intersect_neighbors(v, excl, excl_new);

#if ACMSC_ENGINE_INCREMENTAL_ACC
            vector<pair<int,float>> undo;
            undo.reserve(cands_new.size() + excl_new.size());
            for (int w : cands_new) {
                float s = lookup_sim(w, v);
                acc.add(w, s);
                undo.push_back({w, s});
            }
            for (int w : excl_new) {
                float s = lookup_sim(w, v);
                acc.add(w, s);
                undo.push_back({w, s});
            }
#endif

            double sim_sum_new = sim_sum;
            for (int u : C) {
                sim_sum_new += static_cast<double>(lookup_sim(u, v));
            }

            C.push_back(v);
            bool sub = search(C, sim_sum_new, cands_new, excl_new,
                              acc, is_check, min_size);
            C.pop_back();

#if ACMSC_ENGINE_INCREMENTAL_ACC
            for (auto& [w, s] : undo) acc.sub(w, s);
#endif

            if (timed_out_.load(std::memory_order_relaxed)) return false;

            if (sub) {
                found_any_longer = true;
                if (is_check) return true;
            }

            excl.push_back(v);
        }

        if (!found_any_longer) {
            long long edges = (long long)k * (k-1) / 2;
            bool avg_ok = (k <= 1) ||
                          (edges > 0 && sim_sum >=
                              static_cast<double>(tau_) * edges);

            if (avg_ok) {
#if ACMSC_ENABLE_STATS
                feasible_terminals_.fetch_add(1, std::memory_order_relaxed);
#endif
                if (is_check) {
                    return (k > min_size);
                }

                bool subsumed = false;
                if (k >= 2 && !excl.empty()) {
#if ACMSC_ENABLE_STATS
                    check_calls_.fetch_add(1, std::memory_order_relaxed);
#endif
                    vector<int> excl_as_cands = excl;
                    vector<int> empty_excl;
                    subsumed = search(C, sim_sum, excl_as_cands, empty_excl,
                                      acc, true, k);
                }

                if (!subsumed) {
                    emit(C);
                }

                return true;
            }
        }

        return found_any_longer;
    }

    void build_graph() {
        int n = g_.V;
        logger.print("[Stage1] Computing and caching all edge similarities...\n");
        auto t0 = chrono::high_resolution_clock::now();

        vector<vector<pair<int,float>>> half_adj(n);
        long long total_edges = 0;
        #pragma omp parallel for schedule(dynamic, 100) reduction(+:total_edges)
        for (int u = 0; u < n; u++) {
            int nu = db_.reorder.old_to_new[u];
            for (int v : g_.adj[u]) {
                if (v <= u) continue;
                int nv = db_.reorder.old_to_new[v];
                float s = db_.vecs.dot(nu, nv);
                half_adj[u].push_back({v, s});
                total_edges++;
            }
        }
        logger.print("  edges cached: %lld\n", (long long)total_edges);

        nb_.assign(n, {});
        for (int u = 0; u < n; u++)
            for (auto& [v, s] : half_adj[u]) {
                nb_[u].push_back({v, s});
                nb_[v].push_back({u, s});
            }
        half_adj.clear();

        sim_map_.resize(n);
        for (int u = 0; u < n; u++) {
            sim_map_[u].reserve(nb_[u].size());
            for (auto& [v, s] : nb_[u]) sim_map_[u][v] = s;
        }

#if ACMSC_ENGINE_BITMAP
        bitmap_ = new BlockedSparseBitmap(n);
        #pragma omp parallel for schedule(static)
        for (int u = 0; u < n; u++) {
            vector<int> ids;
            ids.reserve(nb_[u].size());
            for (auto& [v, s] : nb_[u]) ids.push_back(v);
            bitmap_->build(u, ids);
        }
#else
        bitmap_ = nullptr;
#endif

        logger.print("  [Stage1] Done: %lld ms\n",
                     (long long)chrono::duration_cast<chrono::milliseconds>(
                         chrono::high_resolution_clock::now()-t0).count());
    }

    void build_degeneracy_ordering() {
        logger.print("[Stage2] Degeneracy ordering...\n");
        auto t0 = chrono::high_resolution_clock::now();
        int n = g_.V;
        ordering_.clear();
        ordering_.reserve(n);

        vector<int> deg(n);
        for (int v = 0; v < n; v++) deg[v] = (int)nb_[v].size();
        int max_deg = deg.empty() ? 1 : *max_element(deg.begin(), deg.end());
        if (max_deg == 0) max_deg = 1;

        vector<vector<int>> bucket(max_deg+1);
        vector<int> pos(n, -1), cur(deg);
        for (int v = 0; v < n; v++) {
            pos[v] = (int)bucket[cur[v]].size();
            bucket[cur[v]].push_back(v);
        }
        vector<bool> removed(n, false);
        int cur_min = 0;
        for (int i = 0; i < n; i++) {
            while (cur_min <= max_deg && bucket[cur_min].empty()) cur_min++;
            if (cur_min > max_deg) break;
            int v = bucket[cur_min].back();
            bucket[cur_min].pop_back();
            pos[v] = -1;
            removed[v] = true;
            ordering_.push_back(v);
            for (auto& [u, s] : nb_[v]) {
                if (removed[u]) continue;
                int od = cur[u], op = pos[u];
                if (!bucket[od].empty() && op < (int)bucket[od].size()) {
                    int last = bucket[od].back();
                    bucket[od][op] = last;
                    pos[last] = op;
                    bucket[od].pop_back();
                }
                cur[u]--;
                pos[u] = (int)bucket[cur[u]].size();
                bucket[cur[u]].push_back(u);
                if (cur[u] < cur_min) cur_min = cur[u];
            }
        }
        logger.print("  [Stage2] Done: %lld ms\n",
                     (long long)chrono::duration_cast<chrono::milliseconds>(
                         chrono::high_resolution_clock::now()-t0).count());
    }

    bool run_search(long long& search_ms) {
        logger.print("[Stage3] Semantic BK search...\n");
        if (timeout_seconds_ > 0)
            logger.print("  tau=%.3f  threads=%d  timeout=%ds\n", tau_,
                         omp_get_max_threads(), timeout_seconds_);
        else
            logger.print("  tau=%.3f  threads=%d  timeout=external/unlimited\n",
                         tau_, omp_get_max_threads());
        auto t0 = chrono::high_resolution_clock::now();
        if (timeout_seconds_ > 0)
            deadline_ = chrono::steady_clock::now() + chrono::seconds(timeout_seconds_);
        int n = g_.V;
        int total = (int)ordering_.size();

        vector<int> order_pos(n, 0);
        for (int i = 0; i < total; i++)
            order_pos[ordering_[i]] = i;

        int nthreads = omp_get_max_threads();
        vector<AccStore> stores(nthreads);
        for (auto& s : stores) s.init(n);

        auto last_print = chrono::high_resolution_clock::now();
        std::atomic<int> progress_i{0};
        std::mutex print_mutex;

        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < total; i++) {
            if (timed_out_.load(std::memory_order_relaxed)) continue;
            auto now = chrono::high_resolution_clock::now();

            int v   = ordering_[i];
            int tid = omp_get_thread_num();
            AccStore& acc = stores[tid];

#if ACMSC_ENGINE_INCREMENTAL_ACC
            acc.cur++;
            if (acc.cur == 0) {
                fill(acc.gen.begin(), acc.gen.end(), 0);
                acc.cur = 1;
            }
#endif

            vector<int> cands, excl;
            cands.reserve(nb_[v].size());
            excl.reserve(nb_[v].size());
            for (auto& [u, s] : nb_[v]) {
                if (order_pos[u] > i)      cands.push_back(u);
                else if (order_pos[u] < i) excl.push_back(u);
            }
#if ACMSC_ENGINE_INCREMENTAL_ACC
            for (auto& [u, s] : nb_[v]) acc.add(u, s);
#endif

            vector<int> C = {v};
            search(C, 0.0f, cands, excl, acc, false);

            int cur_i = progress_i.fetch_add(1, std::memory_order_relaxed) + 1;
            if (chrono::duration_cast<chrono::seconds>(now - last_print).count() >= 5) {
                std::lock_guard<std::mutex> lk(print_mutex);
                if (chrono::duration_cast<chrono::seconds>(
                        chrono::high_resolution_clock::now() - last_print).count() >= 5) {
                    long long cand_so_far = clique_count_.load();
                    logger.print("  progress %d/%d (%.1f%%)  cliques=%lld  elapsed=%llds\n",
                               cur_i, total, 100.0 * cur_i / total,
                               cand_so_far,
                               (long long)chrono::duration_cast<chrono::seconds>(
                                   now - t0).count());
                    last_print = now;
                }
            }
        }

        search_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        if (timed_out_.load(std::memory_order_relaxed)) {
            logger.print("  [TIMEOUT] incomplete search stopped after %lld ms\n",
                         search_ms);
            return false;
        }
        logger.print("  [Stage3] Done: %lld ms  cliques=%lld\n",
                     search_ms, (long long)clique_count_.load());
        return true;
    }

public:
    AvgSimMCEMiner(const VectorDB& db, const Graph& g, int timeout_seconds)
        : db_(db), g_(g), tau_(0.f),
          timeout_seconds_(max(0, timeout_seconds)) {}
    ~AvgSimMCEMiner() {
        delete bitmap_;
        if (sink_) fclose(sink_);
    }

    void set_sink(const char* fn) {
        if (sink_) { fclose(sink_); sink_ = nullptr; }
        if (fn && fn[0]) sink_ = fopen(fn, "w");
    }

    long long mine(double tau) {
        tau_ = (float)tau;
        clique_count_.store(0);
        timed_out_.store(false);
#if ACMSC_ENABLE_STATS
        search_states_.store(0);
        candidate_expansions_.store(0);
        check_states_.store(0);
        check_calls_.store(0);
        feasible_terminals_.store(0);
#endif
        delete bitmap_; bitmap_ = nullptr;
        nb_.clear(); sim_map_.clear(); ordering_.clear();

        logger.print("\n=== ALG2 SemBK  tau=%.3f ===\n", tau);
        auto t0 = chrono::high_resolution_clock::now();

        build_graph();
        int max_degree = 0;
        for (const auto& row : nb_)
            max_degree = max(max_degree, static_cast<int>(row.size()));
        size_hist_size_ = max(2, max_degree + 2);
        size_hist_ = make_unique<atomic<long long>[]>(size_hist_size_);
        for (int i = 0; i < size_hist_size_; ++i) size_hist_[i].store(0);
        auto t_after_graph = chrono::high_resolution_clock::now();
        long long graph_ms = chrono::duration_cast<chrono::milliseconds>(t_after_graph - t0).count();

        build_degeneracy_ordering();
        auto t_after_degen = chrono::high_resolution_clock::now();
        long long degen_ms = chrono::duration_cast<chrono::milliseconds>(t_after_degen - t_after_graph).count();

        long long search_ms = 0;
        if (!run_search(search_ms)) {
            logger.print("[TIMEOUT] incomplete run; no result count is reported.\n");
            logger.print("Peak process memory: %.2f GiB (limit 16.00 GiB)\n",
                         acmsc_runtime::peak_memory_bytes() / 1073741824.0);
            return -1;
        }
        auto t_after_search = chrono::high_resolution_clock::now();

        long long total_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        long long final_count = clique_count_.load();

        logger.print("\n=== TOTAL: %lld ms,  %lld maximal cliques ===\n",
                     total_ms, final_count);
        logger.print("Peak process memory: %.2f GiB (limit 16.00 GiB)\n",
                     acmsc_runtime::peak_memory_bytes() / 1073741824.0);
#if ACMSC_ENABLE_STATS
        logger.print("STAT search_states=%llu candidate_expansions=%llu "
                     "check_calls=%llu check_states=%llu feasible_terminals=%llu\n",
                     search_states_.load(), candidate_expansions_.load(),
                     check_calls_.load(), check_states_.load(),
                     feasible_terminals_.load());
#endif

        logger.print("\n--- Phase timing summary ---\n");
        logger.print("  Graph build        : %6lld ms  (%5.1f%%)\n",
                     graph_ms, total_ms>0?100.0*graph_ms/total_ms:0.0);
        logger.print("  Degeneracy ordering: %6lld ms  (%5.1f%%)\n",
                     degen_ms, total_ms>0?100.0*degen_ms/total_ms:0.0);
        logger.print("  BK search          : %6lld ms  (%5.1f%%)\n",
                     search_ms, total_ms>0?100.0*search_ms/total_ms:0.0);
        logger.print("----------------------------\n");

        logger.print("\n--- Clique size distribution ---\n");
        long long total = clique_count_.load();
        long long sum_sizes = 0;
        for (int i = 2; i < size_hist_size_; i++) {
            long long c = size_hist_[i].load();
            if (c > 0) {
                logger.print("  size %3d: %lld (%.2f%%)\n",
                           i, c, total>0?100.0*c/total:0.0);
                sum_sizes += (long long)i*c;
            }
        }
        if (total > 0)
            logger.print("  Average clique size: %.2f\n", (double)sum_sizes/total);
        logger.print("--------------------------------\n");

        return clique_count_.load();
    }
};

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    if (!acmsc_runtime::enforce_memory_limit()) {
        fprintf(stderr, "Failed to enforce the 16 GiB process memory limit.\n");
        return 2;
    }

    string graph_file, vec_index, log_path, graph_override, vector_override;
    int dataset_id = 2;
    int requested_threads = acmsc_runtime::kDefaultThreads;
    int timeout_seconds = acmsc_runtime::kDefaultTimeoutSeconds;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "dataset" && i + 1 < argc) {
            dataset_id = atoi(argv[++i]);
        } else if (arg == "log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if (arg == "threads" && i + 1 < argc) {
            requested_threads = atoi(argv[++i]);
        } else if (arg == "graph" && i + 1 < argc) {
            graph_override = argv[++i];
        } else if (arg == "vectors" && i + 1 < argc) {
            vector_override = argv[++i];
        } else if (arg == "timeout" && i + 1 < argc) {
            timeout_seconds = atoi(argv[++i]);
        }
    }
    const int thread_count = max(1, min(requested_threads, omp_get_num_procs()));
    omp_set_dynamic(0);
    omp_set_num_threads(thread_count);
    acmsc_runtime::select_dataset(dataset_id, graph_file, vec_index);
    if (!graph_override.empty()) graph_file = graph_override;
    if (!vector_override.empty()) vec_index = vector_override;

    if (!log_path.empty()) logger.open(log_path);

    logger.print("============================================\n");
    logger.print("  ALG2 Standard (SemBK)\n");
    logger.print("  mine <tau> | sink <file> | log <file> | quit\n");
    logger.print("============================================\n");
    logger.print("Dataset: %s (ID=%d)\n", graph_file.c_str(), dataset_id);

    auto t_load = chrono::high_resolution_clock::now();
    Graph g(graph_file);
    if (g.V == 0) { fprintf(stderr, "Failed to load graph\n"); return 1; }
    vector<set<int>>().swap(g.adjSet);

    bool vec_loaded = false;
    if (fs::exists(vec_index)) {
        if (vec_index.find(".index") != string::npos) {
            vec_loaded = g.loadVectorsFromChunks(vec_index);
        } else {
            vec_loaded = g.loadVectorsFromBinary(vec_index);
        }
        if (!vec_loaded) {
            fprintf(stderr, "Failed to load vectors from: %s\n", vec_index.c_str());
            return 1;
        }
    } else {
        fprintf(stderr, "Vector file not found: %s\n", vec_index.c_str());
        return 1;
    }
    g.normalizeVectors();

    long long ec = 0;
    for (int i = 0; i < g.V; i++) ec += (long long)g.adj[i].size();
    ec /= 2;
    logger.print("Graph: V=%d  E=%lld\n", g.V, ec);
    if (timeout_seconds > 0)
        logger.print("Threads: %d, timeout: %ds, memory limit: 16 GiB\n",
                     thread_count, timeout_seconds);
    else
        logger.print("Threads: %d, timeout: external, memory limit: 16 GiB\n",
                     thread_count);
    logger.print("Load: %lld ms\n\n",
               chrono::duration_cast<chrono::milliseconds>(
                   chrono::high_resolution_clock::now()-t_load).count());

    VectorDB db;
    db.build(g);
    vector<vector<double>>().swap(g.semanticVectors);
    AvgSimMCEMiner miner(db, g, timeout_seconds);

    logger.print("\n>> ");
    string line;
    while (getline(cin, line)) {
        if (line.empty()) { logger.print(">> "); continue; }
        istringstream iss(line);
        string cmd; iss >> cmd;
        if (cmd == "mine") {
            double tau = 0.2;
            iss >> tau;
            miner.mine(tau);
        } else if (cmd == "sink") {
            string fn; iss >> fn;
            miner.set_sink(fn.c_str());
            logger.print("Sink: %s\n", fn.c_str());
        } else if (cmd == "log") {
            string fn; iss >> fn;
            logger.open(fn);
            logger.print("Log: %s\n", fn.c_str());
        } else if (cmd == "quit" || cmd == "exit") {
            break;
        } else {
            logger.print("Commands: mine <tau> | sink <file> | log <file> | quit\n");
        }
        logger.print(">> ");
    }
    logger.print("Bye.\n");
    return 0;
}
