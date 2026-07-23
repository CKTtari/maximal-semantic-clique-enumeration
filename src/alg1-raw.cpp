// ============================================================
// ALG1_standard.cpp  结构极大团枚举 — Tomita Pivot BK（标准版）
//
// 特性：
//   - 输出信息保存到日志（Logger 同时写 stdout + 文件）
//   - 分阶段计时：build_adj / degeneracy / search / 总计
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
#include <numeric>
#include <omp.h>
#include <immintrin.h>
#include <string>
#include <sstream>
#include <queue>
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
    static constexpr int   MAX_CLIQUE_SZ = 512;
    std::atomic<long long> size_hist_[MAX_CLIQUE_SZ]{};
    FILE*                  sink_     = nullptr;
    std::mutex             sink_mutex_;

    inline void emit(const vector<int>& C) {
        int sz = (int)C.size();
        clique_count_.fetch_add(1, std::memory_order_relaxed);
        if (sz < MAX_CLIQUE_SZ)
            size_hist_[sz > 0 ? sz-1 : 0].fetch_add(1, std::memory_order_relaxed);
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
            int cnt = bitmap_->get_row(u).intersect_count(P_row);
            if (cnt > best) { best = cnt; pivot = u; }
        }
        for (int u : X) {
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

    void run_search(long long& search_ms) {
        logger.print("[Stage3] Tomita Pivot BK search...\n");
        logger.print("  threads=%d  timeout=600s\n", omp_get_max_threads());
        auto t0 = chrono::high_resolution_clock::now();
        int n = g_.V;
        int total = (int)ordering_.size();

        vector<int> order_pos(n, 0);
        for (int i = 0; i < total; i++)
            order_pos[ordering_[i]] = i;

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

        if (timed_out.load()) {
            logger.print("  [TIMEOUT] reached 600 seconds, stopping search\n");
        }

        search_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        logger.print("  [Stage3] Done: %lld ms  cliques=%lld\n",
                     search_ms, (long long)clique_count_.load());
    }

public:
    explicit MCEMiner(const Graph& g) : g_(g) {
        for (auto& h : size_hist_) h.store(0);
    }
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
        for (auto& h : size_hist_) h.store(0);
        delete bitmap_; bitmap_ = nullptr;
        adj_.clear(); ordering_.clear();

        logger.print("\n=== ALG1 Standard MCE ===\n");
        auto t0 = chrono::high_resolution_clock::now();

        long long adj_ms = 0, degen_ms = 0, search_ms = 0;
        build_adj(adj_ms);
        build_degeneracy_ordering(degen_ms);
        run_search(search_ms);

        long long total_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        long long final_count = clique_count_.load();

        logger.print("\n=== TOTAL: %lld ms,  %lld maximal cliques ===\n",
                     total_ms, final_count);

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
        case 1:  graph_file = "dataset/dataset/sc-ldoor.txt";         vec_index = "dataset/vectors-256/sc-ldoor_vectors.bin";       break;
        case 2:  graph_file = "dataset/dataset/sc-nasasrb.txt";       vec_index = "dataset/vectors-256/sc-nasasrb_vectors.bin";     break;
        case 3:  graph_file = "dataset/dataset/sc-pkustk11.txt";      vec_index = "dataset/vectors-256/sc-pkustk11_vectors.bin";    break;
        case 4:  graph_file = "dataset/dataset/soc-buzznet.txt";      vec_index = "dataset/vectors-256/soc-buzznet_vectors.bin";    break;
        case 5:  graph_file = "dataset/dataset/soc-digg.txt";         vec_index = "dataset/vectors-256/soc-digg_vectors.bin";       break;
        case 6:  graph_file = "dataset/dataset/tech-as-skitter.txt";  vec_index = "dataset/vectors-256/tech-as-skitter_vectors.bin"; break;
        case 7:  graph_file = "dataset/dataset/FB15K-237_edges.txt";  vec_index = "dataset/vectors/FB15K-237_vectors.index";        break;
        case 8:  graph_file = "dataset/dataset/WN18RR_edges.txt";     vec_index = "dataset/vectors/WN18RR_vectors.index";           break;
        case 9:  graph_file = "dataset/dataset/dblp_coauthor.txt";    vec_index = "dataset/vectors/dblp_coauthor_vectors.index";    break;
        case 10: graph_file = "dataset/dataset/products.txt";   vec_index = "dataset/vectors/products_vectors.index";     break;
        default: graph_file = "dataset/dataset/sc-nasasrb.txt";       vec_index = "dataset/vectors-256/sc-nasasrb_vectors.bin";     break;
    }

    if (!log_path.empty()) logger.open(log_path);

    logger.print("============================================\n");
    logger.print("  ALG1 Standard (Structural MCE)\n");
    logger.print("  mine | sink <file> | log <file> | quit\n");
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

    MCEMiner miner(g);

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
