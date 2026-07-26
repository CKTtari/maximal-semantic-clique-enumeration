// ============================================================

#ifndef ACMSC_ENABLE_STATS
#define ACMSC_ENABLE_STATS 0
#endif
// ALG1_standard.cpp  结构极大团枚举 — Tomita Pivot BK（标准版）
//
// 特性：
//   - 输出信息保存到日志（Logger 同时写 stdout + 文件）
//   - 分阶段计时：build_adj / degeneracy / search / 总计
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
#include <numeric>
#include <omp.h>
#include <immintrin.h>
#include <string>
#include <sstream>
#include <queue>
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
// MCEMiner — Tomita Pivot BK，SOTA baseline
// ============================================================
class MCEMiner {
private:
    const Graph& g_;

    vector<vector<int>> adj_;
    BlockedSparseBitmap* bitmap_ = nullptr;
    vector<int>          ordering_;

    std::atomic<long long> clique_count_{0};
#if ACMSC_ENABLE_STATS
    std::atomic<unsigned long long> bk_states_{0};
    std::atomic<unsigned long long> pivot_tests_{0};
    std::atomic<unsigned long long> branch_expansions_{0};
#endif
    std::unique_ptr<std::atomic<long long>[]> size_hist_;
    int                    size_hist_size_ = 0;
    FILE*                  sink_     = nullptr;
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
        clique_count_.fetch_add(1, std::memory_order_relaxed);
        if (sz >= 0 && sz < size_hist_size_)
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

    void tomita_bk(vector<int>& C,
                   vector<int>& P,
                   vector<int>& X)
    {
        if (deadline_reached()) return;
#if ACMSC_ENABLE_STATS
        bk_states_.fetch_add(1, std::memory_order_relaxed);
#endif
        if (P.empty()) {
            if (X.empty()) {
                emit(C);
            }
            return;
        }

        int pivot = -1, best = -1;
        BlockedSparseBitmap::Row P_row;
        P_row.build(P);

        for (int u : P) {
#if ACMSC_ENABLE_STATS
            pivot_tests_.fetch_add(1, std::memory_order_relaxed);
#endif
            int cnt = bitmap_->get_row(u).intersect_count(P_row);
            if (cnt > best) { best = cnt; pivot = u; }
        }
        for (int u : X) {
#if ACMSC_ENABLE_STATS
            pivot_tests_.fetch_add(1, std::memory_order_relaxed);
#endif
            int cnt = bitmap_->get_row(u).intersect_count(P_row);
            if (cnt > best) { best = cnt; pivot = u; }
        }

        vector<int> S;
        S.reserve(P.size());
        if (pivot == -1) {
            S = P;
        } else {
            const auto& prow = bitmap_->get_row(pivot);
            for (int v : P)
                if (!prow.contains(v)) S.push_back(v);
        }

        for (int v : S) {
#if ACMSC_ENABLE_STATS
            branch_expansions_.fetch_add(1, std::memory_order_relaxed);
#endif
            const auto& v_row = bitmap_->get_row(v);

            vector<int> P_new;
            v_row.intersect_to(P_row, P_new);

            vector<int> X_new;
            if (!X.empty()) {
                BlockedSparseBitmap::Row X_row;
                if ((int)X.size() < 64) {
                    X_new.reserve(X.size());
                    for (int u : X)
                        if (v_row.contains(u)) X_new.push_back(u);
                } else {
                    X_row.build(X);
                    v_row.intersect_to(X_row, X_new);
                }
            }

            C.push_back(v);
            tomita_bk(C, P_new, X_new);
            C.pop_back();
            if (timed_out_.load(std::memory_order_relaxed)) return;

            P.erase(find(P.begin(), P.end(), v));
            X.push_back(v);
            P_row.build(P);
        }
    }

    void build_adj(long long& adj_ms) {
        int n = g_.V;
        logger.print("[Stage1] Building adjacency...\n");
        auto t0 = chrono::high_resolution_clock::now();

        adj_.assign(n, {});
        for (int u = 0; u < n; u++) {
            adj_[u].assign(g_.adj[u].begin(), g_.adj[u].end());
            sort(adj_[u].begin(), adj_[u].end());
        }

        bitmap_ = new BlockedSparseBitmap(n);
        #pragma omp parallel for schedule(static)
        for (int u = 0; u < n; u++)
            bitmap_->build(u, adj_[u]);

        long long ec = 0;
        for (int u = 0; u < n; u++) ec += (long long)adj_[u].size();
        ec /= 2;
        adj_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        logger.print("  [Stage1] Done: %lld ms  (V=%d, E=%lld)\n", adj_ms, n, ec);
    }

    void build_degeneracy_ordering(long long& degen_ms) {
        logger.print("[Stage2] Degeneracy ordering...\n");
        auto t0 = chrono::high_resolution_clock::now();
        int n = g_.V;
        ordering_.clear();
        ordering_.reserve(n);

        vector<int> deg(n);
        for (int v = 0; v < n; v++) deg[v] = (int)adj_[v].size();
        int max_deg = deg.empty() ? 1 : *max_element(deg.begin(), deg.end());
        if (max_deg == 0) max_deg = 1;

        vector<vector<int>> bucket(max_deg+1);
        vector<int> pos(n,-1), cur(deg);
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
            for (int u : adj_[v]) {
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
        degen_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        logger.print("  [Stage2] Done: %lld ms\n", degen_ms);
    }

    bool run_search(long long& search_ms) {
        logger.print("[Stage3] Tomita Pivot BK search...\n");
        if (timeout_seconds_ > 0)
            logger.print("  threads=%d  timeout=%ds\n", omp_get_max_threads(),
                         timeout_seconds_);
        else
            logger.print("  threads=%d  timeout=external/unlimited\n",
                         omp_get_max_threads());
        auto t0 = chrono::high_resolution_clock::now();
        if (timeout_seconds_ > 0)
            deadline_ = chrono::steady_clock::now() + chrono::seconds(timeout_seconds_);
        int n = g_.V;
        int total = (int)ordering_.size();

        vector<int> order_pos(n, 0);
        for (int i = 0; i < total; i++)
            order_pos[ordering_[i]] = i;

        auto last_print = chrono::high_resolution_clock::now();
        std::atomic<int> progress_i{0};
        std::mutex print_mutex;

        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < total; i++) {
            if (timed_out_.load(std::memory_order_relaxed)) continue;
            auto now = chrono::high_resolution_clock::now();

            int v = ordering_[i];

            vector<int> P, X;
            P.reserve(adj_[v].size());
            X.reserve(adj_[v].size());
            for (int u : adj_[v]) {
                if      (order_pos[u] > i) P.push_back(u);
                else if (order_pos[u] < i) X.push_back(u);
            }

            vector<int> C = {v};
            tomita_bk(C, P, X);

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
    explicit MCEMiner(const Graph& g, int timeout_seconds)
        : g_(g), timeout_seconds_(max(0, timeout_seconds)) {}
    ~MCEMiner() {
        delete bitmap_;
        if (sink_) fclose(sink_);
    }

    void set_sink(const char* fn) {
        if (sink_) { fclose(sink_); sink_ = nullptr; }
        if (fn && fn[0]) sink_ = fopen(fn, "w");
    }

    long long mine() {
        clique_count_.store(0);
        timed_out_.store(false);
#if ACMSC_ENABLE_STATS
        bk_states_.store(0);
        pivot_tests_.store(0);
        branch_expansions_.store(0);
#endif
        delete bitmap_; bitmap_ = nullptr;
        adj_.clear(); ordering_.clear();

        logger.print("\n=== ALG1 Standard MCE ===\n");
        auto t0 = chrono::high_resolution_clock::now();

        long long adj_ms = 0, degen_ms = 0, search_ms = 0;
        build_adj(adj_ms);
        int max_degree = 0;
        for (const auto& row : adj_)
            max_degree = max(max_degree, static_cast<int>(row.size()));
        size_hist_size_ = max(2, max_degree + 2);
        size_hist_ = make_unique<atomic<long long>[]>(size_hist_size_);
        for (int i = 0; i < size_hist_size_; ++i) size_hist_[i].store(0);
        build_degeneracy_ordering(degen_ms);
        if (!run_search(search_ms)) {
            logger.print("[TIMEOUT] incomplete run; no result count is reported.\n");
            logger.print("Peak process memory: %.2f GiB (limit 16.00 GiB)\n",
                         acmsc_runtime::peak_memory_bytes() / 1073741824.0);
            return -1;
        }

        long long total_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        long long final_count = clique_count_.load();

        logger.print("\n=== TOTAL: %lld ms,  %lld maximal cliques ===\n",
                     total_ms, final_count);
        logger.print("Peak process memory: %.2f GiB (limit 16.00 GiB)\n",
                     acmsc_runtime::peak_memory_bytes() / 1073741824.0);
#if ACMSC_ENABLE_STATS
        logger.print("STAT bk_states=%llu pivot_tests=%llu branch_expansions=%llu\n",
                     bk_states_.load(), pivot_tests_.load(),
                     branch_expansions_.load());
#endif

        logger.print("\n--- Phase timing summary ---\n");
        logger.print("  Adj build          : %6lld ms  (%5.1f%%)\n",
                     adj_ms, total_ms>0?100.0*adj_ms/total_ms:0.0);
        logger.print("  Degeneracy ordering: %6lld ms  (%5.1f%%)\n",
                     degen_ms, total_ms>0?100.0*degen_ms/total_ms:0.0);
        logger.print("  BK search          : %6lld ms  (%5.1f%%)\n",
                     search_ms, total_ms>0?100.0*search_ms/total_ms:0.0);
        logger.print("----------------------------\n");

        logger.print("\n--- Clique size distribution ---\n");
        long long total = clique_count_.load();
        long long sum_sizes = 0;
        for (int i = 1; i < size_hist_size_; i++) {
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

    string graph_file, vec_index, log_path, graph_override;
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
        } else if (arg == "timeout" && i + 1 < argc) {
            timeout_seconds = atoi(argv[++i]);
        }
    }
    const int thread_count = max(1, min(requested_threads, omp_get_num_procs()));
    omp_set_dynamic(0);
    omp_set_num_threads(thread_count);
    acmsc_runtime::select_dataset(dataset_id, graph_file, vec_index);
    if (!graph_override.empty()) graph_file = graph_override;

    if (!log_path.empty()) logger.open(log_path);

    logger.print("============================================\n");
    logger.print("  ALG1 Standard (Structural MCE)\n");
    logger.print("  mine | sink <file> | log <file> | quit\n");
    logger.print("============================================\n");
    logger.print("Dataset: %s (ID=%d)\n", graph_file.c_str(), dataset_id);

    auto t_load = chrono::high_resolution_clock::now();
    Graph g(graph_file);
    if (g.V == 0) { fprintf(stderr, "Failed to load graph\n"); return 1; }
    vector<set<int>>().swap(g.adjSet);

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

    MCEMiner miner(g, timeout_seconds);

    logger.print("\n>> ");
    string line;
    while (getline(cin, line)) {
        if (line.empty()) { logger.print(">> "); continue; }
        istringstream iss(line);
        string cmd; iss >> cmd;
        if (cmd == "mine") {
            miner.mine();
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
            logger.print("Commands: mine | sink <file> | log <file> | quit\n");
        }
        logger.print(">> ");
    }
    logger.print("Bye.\n");
    return 0;
}
