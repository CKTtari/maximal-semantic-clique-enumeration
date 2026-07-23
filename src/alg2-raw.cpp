// ============================================================
// ALG4_standard.cpp  语义 BK 极大团挖掘（标准版）
//
// 特性：
//   - 输出信息保存到日志（Logger 同时写 stdout + 文件）
//   - 分阶段计时：build_graph / degeneracy / search / 总计
//   - 输出团总数 + size 分布
//   - 无任何统计计数器，时间效率纯洁准确
// ============================================================

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
    static constexpr int   MAX_CLIQUE_SZ = 512;
    std::atomic<long long> size_hist_[MAX_CLIQUE_SZ]{};
    FILE*                  sink_ = nullptr;
    std::mutex             sink_mutex_;

    inline void emit(const vector<int>& C) {
        int sz = (int)C.size();
        if (sz <= 1) return;
        clique_count_.fetch_add(1, std::memory_order_relaxed);
        if (sz < MAX_CLIQUE_SZ)
            size_hist_[sz-1].fetch_add(1, std::memory_order_relaxed);
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

    struct AccStore {
        vector<float>    val;
        vector<uint32_t> gen;
        uint32_t         cur = 0;
        void init(int n) { val.assign(n, 0.f); gen.assign(n, 0); cur = 1; }
        inline float get(int w) const { return gen[w]==cur ? val[w] : 0.f; }
        inline void  add(int w, float s) {
            if (gen[w] != cur) { val[w] = 0.f; gen[w] = cur; }
            val[w] += s;
        }
        inline void  sub(int w, float s) {
            if (gen[w] == cur) val[w] -= s;
        }
    };

    bool search(
        vector<int>& C,
        float        sim_sum,
        vector<int>& cands,
        vector<int>& excl,
        AccStore&    acc,
        bool         is_check,
        int          min_size = 0
    ) {
        bool found_any_longer = false;
        int  k = (int)C.size();

        vector<int> to_enumerate(cands.begin(), cands.end());

        for (int v : to_enumerate) {
            cands.erase(find(cands.begin(), cands.end(), v));

            const auto& v_row = bitmap_->get_row(v);

            vector<int> cands_new;
            cands_new.reserve(cands.size());
            for (int u : cands)
                if (v_row.contains(u)) cands_new.push_back(u);

            vector<int> excl_new;
            if (!excl.empty()) {
                if ((int)excl.size() < 64) {
                    excl_new.reserve(excl.size());
                    for (int u : excl)
                        if (v_row.contains(u)) excl_new.push_back(u);
                } else {
                    BlockedSparseBitmap::Row excl_row;
                    excl_row.build(excl);
                    v_row.intersect_to(excl_row, excl_new);
                }
            }

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

            float sim_sum_new = sim_sum;
            for (int u : C) {
                sim_sum_new += lookup_sim(u, v);
            }

            C.push_back(v);
            bool sub = search(C, sim_sum_new, cands_new, excl_new,
                              acc, is_check, min_size);
            C.pop_back();

            for (auto& [w, s] : undo) acc.sub(w, s);

            if (sub) {
                found_any_longer = true;
                if (is_check) return true;
            }

            excl.push_back(v);
        }

        if (!found_any_longer) {
            long long edges = (long long)k * (k-1) / 2;
            bool avg_ok = (k <= 1) ||
                          (edges > 0 && sim_sum / (float)edges >= tau_);

            if (avg_ok) {
                if (is_check) {
                    return (k > min_size);
                }

                bool subsumed = false;
                if (k >= 2 && !excl.empty()) {
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

        bitmap_ = new BlockedSparseBitmap(n);
        #pragma omp parallel for schedule(static)
        for (int u = 0; u < n; u++) {
            vector<int> ids;
            ids.reserve(nb_[u].size());
            for (auto& [v, s] : nb_[u]) ids.push_back(v);
            bitmap_->build(u, ids);
        }

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

    void run_search(long long& search_ms) {
        logger.print("[Stage3] Semantic BK search...\n");
        logger.print("  tau=%.3f  threads=%d  timeout=600s\n", tau_, omp_get_max_threads());
        auto t0 = chrono::high_resolution_clock::now();
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
        std::atomic<bool> timed_out{false};
        std::mutex print_mutex;

        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < total; i++) {
            if (timed_out.load()) continue;

            auto now = chrono::high_resolution_clock::now();
            if (chrono::duration_cast<chrono::seconds>(now - t0).count() >= 600) {
                timed_out.store(true);
                continue;
            }

            int v   = ordering_[i];
            int tid = omp_get_thread_num();
            AccStore& acc = stores[tid];

            acc.cur++;
            if (acc.cur == 0) {
                fill(acc.gen.begin(), acc.gen.end(), 0);
                acc.cur = 1;
            }

            vector<int> cands, excl;
            cands.reserve(nb_[v].size());
            excl.reserve(nb_[v].size());
            for (auto& [u, s] : nb_[v]) {
                if (order_pos[u] > i)      cands.push_back(u);
                else if (order_pos[u] < i) excl.push_back(u);
            }
            for (auto& [u, s] : nb_[v]) acc.add(u, s);

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

        if (timed_out.load()) {
            logger.print("  [TIMEOUT] reached 600 seconds, stopping search\n");
        }

        search_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        logger.print("  [Stage3] Done: %lld ms  cliques=%lld\n",
                     search_ms, (long long)clique_count_.load());
    }

public:
    AvgSimMCEMiner(const VectorDB& db, const Graph& g)
        : db_(db), g_(g), tau_(0.f) {
        for (auto& h : size_hist_) h.store(0);
    }
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
        for (auto& h : size_hist_) h.store(0);
        delete bitmap_; bitmap_ = nullptr;
        nb_.clear(); sim_map_.clear(); ordering_.clear();

        logger.print("\n=== ALG4 MCE  tau=%.3f ===\n", tau);
        auto t0 = chrono::high_resolution_clock::now();

        build_graph();
        auto t_after_graph = chrono::high_resolution_clock::now();
        long long graph_ms = chrono::duration_cast<chrono::milliseconds>(t_after_graph - t0).count();

        build_degeneracy_ordering();
        auto t_after_degen = chrono::high_resolution_clock::now();
        long long degen_ms = chrono::duration_cast<chrono::milliseconds>(t_after_degen - t_after_graph).count();

        long long search_ms = 0;
        run_search(search_ms);
        auto t_after_search = chrono::high_resolution_clock::now();

        long long total_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        long long final_count = clique_count_.load();

        logger.print("\n=== TOTAL: %lld ms,  %lld maximal cliques ===\n",
                     total_ms, final_count);

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
        for (int i = 0; i < MAX_CLIQUE_SZ; i++) {
            long long c = size_hist_[i].load();
            if (c > 0) {
                logger.print("  size %3d: %lld (%.2f%%)\n",
                           i+1, c, total>0?100.0*c/total:0.0);
                sum_sizes += (long long)(i+1)*c;
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
    omp_set_num_threads(omp_get_max_threads());

    string graph_file, vec_index, log_path;
    int dataset_id = 8;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "dataset" || arg == "dataset") && i + 1 < argc) {
            dataset_id = atoi(argv[i + 1]);
        } else if (arg == "log" && i + 1 < argc) {
            log_path = argv[i + 1];
        }
    }

    switch (dataset_id) {
        case 1: graph_file = "dataset/dataset/sc-ldoor.txt"; vec_index = "dataset/vectors-256/sc-ldoor_vectors.bin"; break;
        case 2: graph_file = "dataset/dataset/sc-nasasrb.txt"; vec_index = "dataset/vectors-256/sc-nasasrb_vectors.bin"; break;
        case 3: graph_file = "dataset/dataset/sc-pkustk11.txt"; vec_index = "dataset/vectors-256/sc-pkustk11_vectors.bin"; break;
        case 4: graph_file = "dataset/dataset/soc-buzznet.txt"; vec_index = "dataset/vectors-256/soc-buzznet_vectors.bin"; break;
        case 5: graph_file = "dataset/dataset/soc-digg.txt"; vec_index = "dataset/vectors-256/soc-digg_vectors.bin"; break;
        case 6: graph_file = "dataset/dataset/tech-as-skitter.txt"; vec_index = "dataset/vectors-256/tech-as-skitter_vectors.bin"; break;
        case 7: graph_file = "dataset/dataset/FB15K-237_edges.txt"; vec_index = "dataset/vectors/FB15K-237_vectors.index"; break;
        case 8: graph_file = "dataset/dataset/WN18RR_edges.txt";    vec_index = "dataset/vectors/WN18RR_vectors.index";       break;
        case 9: graph_file = "dataset/dataset/dblp_coauthor.txt";    vec_index = "dataset/vectors/dblp_coauthor_vectors.index"; break;
        case 10: graph_file = "dataset/dataset/products.txt"; vec_index = "dataset/vectors/products_vectors.index"; break;
        default: graph_file = "dataset/dataset/sc-nasasrb.txt"; vec_index = "dataset/vectors-256/sc-nasasrb_vectors.bin"; break;
    }

    if (!log_path.empty()) logger.open(log_path);

    logger.print("============================================\n");
    logger.print("  ALG4 Standard (Semantic BK)\n");
    logger.print("  mine <tau> | sink <file> | log <file> | quit\n");
    logger.print("============================================\n");
    logger.print("Dataset: %s (ID=%d)\n", graph_file.c_str(), dataset_id);

    auto t_load = chrono::high_resolution_clock::now();
    Graph g(graph_file);
    if (g.V == 0) { fprintf(stderr, "Failed to load graph\n"); return 1; }

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
    logger.print("Load: %lld ms\n\n",
               chrono::duration_cast<chrono::milliseconds>(
                   chrono::high_resolution_clock::now()-t_load).count());

    VectorDB db;
    db.build(g);
    AvgSimMCEMiner miner(db, g);

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
