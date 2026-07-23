// ============================================================
// ALG6_v2_standard.cpp  平均相似度约束极大团挖掘（标准版）
//
// 特性：
//   - 输出信息保存到日志（log_file）
//   - 分阶段计时 + 最终总时间
//   - 输出团总数 + size 分布
//   - 无任何统计计数器，时间效率纯洁准确
//
// ALG6 消融开关（脚本开头定义，默认全部启用）：
//   ENABLE_W_PRUNE    : W 剪枝
//   ENABLE_MONO_PRUNE : 单调剪枝
//   ENABLE_LAYER_DEDUP: 每层去重
// ============================================================

// ============================================================
// ===   消融开关（默认全部启用，可在此处改为 0 以关闭）    ===
// ============================================================
#define ENABLE_W_PRUNE     1   // W 剪枝
#define ENABLE_MONO_PRUNE  1   // 单调剪枝
#define ENABLE_LAYER_DEDUP 1   // 每层去重
// ============================================================

#include "semantic_graph.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <unordered_map>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <cstdarg>
#include <immintrin.h>
#include <string>
#include <sstream>
#include <queue>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(size, align) _aligned_malloc(size, align)
#define ALIGNED_FREE(ptr)          _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(size, align) aligned_alloc(align, size)
#define ALIGNED_FREE(ptr)          free(ptr)
#endif

using namespace std;
namespace fs = std::filesystem;

// ============================================================
// Logger：同时输出到 stdout 和日志文件
// ============================================================
struct Logger {
    FILE* fp = nullptr;
    void open(const string& path) {
        if (fp) { fclose(fp); fp = nullptr; }
        if (!path.empty()) {
            fp = fopen(path.c_str(), "w");
            if (!fp) fprintf(stderr, "[Logger] Cannot open log: %s\n", path.c_str());
        }
    }
    ~Logger() { if (fp) fclose(fp); }
    void print(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt); vprintf(fmt, ap);   va_end(ap);
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
            unordered_map<uint32_t,uint64_t> tmp; tmp.reserve(neighbors.size());
            for (int v : neighbors) tmp[(uint32_t)(v>>6)] |= 1ULL<<(v&63);
            blocks.reserve(tmp.size());
            for (auto& kv : tmp) blocks.push_back({kv.first, kv.second});
            sort(blocks.begin(), blocks.end(),
                 [](const Block& a, const Block& b){ return a.idx < b.idx; });
        }
        bool contains(int v) const {
            uint32_t bidx = (uint32_t)(v>>6); uint64_t bbit = 1ULL<<(v&63);
            int lo = 0, hi = (int)blocks.size()-1;
            while (lo <= hi) {
                int mid = (lo+hi)>>1;
                if (blocks[mid].idx == bidx) return (blocks[mid].bits & bbit) != 0;
                blocks[mid].idx < bidx ? lo = mid+1 : hi = mid-1;
            }
            return false;
        }
        void intersect_to(const Row& other, vector<int>& out) const {
            out.clear();
            if (blocks.empty() || other.blocks.empty()) return;
            size_t i = 0, j = 0;
            while (i < blocks.size() && j < other.blocks.size()) {
                if (blocks[i].idx == other.blocks[j].idx) {
                    uint64_t c = blocks[i].bits & other.blocks[j].bits;
                    while (c) { out.push_back((int)(blocks[i].idx*64+__builtin_ctzll(c))); c &= c-1; }
                    i++; j++;
                } else if (blocks[i].idx < other.blocks[j].idx) i++; else j++;
            }
        }
    };
private:
    vector<Row> rows_; int n_;
public:
    BlockedSparseBitmap() : n_(0) {}
    explicit BlockedSparseBitmap(int n) : n_(n), rows_(n) {}
    void       build   (int u, const vector<int>& nb) { rows_[u].build(nb); }
    bool       contains(int u, int v)           const { return rows_[u].contains(v); }
    const Row& get_row (int u)                  const { return rows_[u]; }
    int        size()                           const { return n_; }
};

// ============================================================
// Float32VectorStore
// ============================================================
class Float32VectorStore {
    float* vecs_ = nullptr; int n_ = 0, d_ = 0;
public:
    ~Float32VectorStore() { if (vecs_) ALIGNED_FREE(vecs_); }
    int dim() const { return d_; }
    void build(const Graph& g, const vector<int>& order = {}) {
        n_ = g.V; d_ = (int)g.getSemanticVector(0).size();
        size_t vb = ((size_t)n_*d_*sizeof(float)+63) & ~63ULL;
        vecs_ = (float*)ALIGNED_ALLOC(vb, 64);
        bool ho = !order.empty();
#pragma omp parallel for schedule(static)
        for (int ni = 0; ni < n_; ni++) {
            int oi = ho ? order[ni] : ni;
            const vector<double>& dv = g.getSemanticVector(oi);
            float* xf = vecs_ + (size_t)ni*d_;
            for (int i = 0; i < d_; i++) xf[i] = (float)dv[i];
        }
        logger.print("  Float32VectorStore: %dx%d\n", n_, d_);
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

// ============================================================
// GraphReorder
// ============================================================
struct GraphReorder {
    vector<int> old_to_new, new_to_old;
    void build(const Graph& g) {
        int n = g.V; old_to_new.assign(n,-1); new_to_old.assign(n,-1);
        vector<int> order(n); iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(),
             [&](int a, int b){ return g.adj[a].size() > g.adj[b].size(); });
        int nid = 0; queue<int> q;
        for (int s : order) {
            if (old_to_new[s] != -1) continue;
            old_to_new[s] = nid; new_to_old[nid] = s; nid++; q.push(s);
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (int v : g.adj[u]) if (old_to_new[v] == -1)
                    { old_to_new[v] = nid; new_to_old[nid] = v; nid++; q.push(v); }
            }
        }
    }
};

// ============================================================
// VectorDB
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
// LayerClique
// ============================================================
struct LayerClique {
    vector<int>   nodes;
    vector<int>   cands;
    vector<float> acc;
    float         sim_sum;
    bool          is_maximal = false;
};

// ============================================================
// AvgSimCliqueMiner（标准版，无统计计数器）
// ============================================================
class AvgSimCliqueMiner {
private:
    const VectorDB& db_;
    const Graph&    g_;
    float           tau_;

    vector<vector<pair<int,float>>> nb_;
    BlockedSparseBitmap* bitmap_ = nullptr;
    FILE* sink_ = nullptr;

    vector<vector<int>> collected_;
    mutex               collected_mtx_;

    // --------------------------------------------------------
    inline float lookup_sim(int u, int v) const {
        const auto& nb = nb_[u];
        int lo = 0, hi = (int)nb.size()-1;
        while (lo <= hi) {
            int mid = (lo+hi)>>1;
            if (nb[mid].first == v) return nb[mid].second;
            nb[mid].first < v ? lo = mid+1 : hi = mid-1;
        }
        return 0.f;
    }

    // --------------------------------------------------------
    // Stage 1: 构建边相似度图
    // --------------------------------------------------------
    void build_graph() {
        int n = g_.V;
        logger.print("Stage1: computing and caching all edge similarities...\n");
        auto t0 = chrono::high_resolution_clock::now();
        vector<vector<pair<int,float>>> half(n);
        long long total = 0, above = 0;
#pragma omp parallel for schedule(dynamic,100) reduction(+:total,above)
        for (int u = 0; u < n; u++) {
            int nu = db_.reorder.old_to_new[u];
            for (int v : g_.adj[u]) {
                if (v <= u) continue;
                float s = db_.vecs.dot(nu, db_.reorder.old_to_new[v]);
                half[u].push_back({v, s});
                total++;
                if (s > tau_) above++;
            }
        }
        logger.print("  [Stage1] edges: %lld, sim>%.3f: %lld\n", total, (double)tau_, above);
        nb_.assign(n, {});
        for (int u = 0; u < n; u++)
            for (auto& [v,s] : half[u]) { nb_[u].push_back({v,s}); nb_[v].push_back({u,s}); }
        half.clear();
#pragma omp parallel for schedule(dynamic,200)
        for (int u = 0; u < n; u++) sort(nb_[u].begin(), nb_[u].end());
        long long ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        logger.print("  [Stage1 time]: %lld ms\n", ms);
    }

    // --------------------------------------------------------
    // process_one：对一个 k-团做单步扩展
    // --------------------------------------------------------
    void process_one(const LayerClique& lc, vector<LayerClique>& next_out) {
        const int   k       = (int)lc.nodes.size();
        const float sim_sum = lc.sim_sum;
        const int   nc      = (int)lc.cands.size();

        float W       = sim_sum - tau_ * (float)((long long)k*(k-1)/2);
        float tau_k   = tau_ * (float)k;
        long long ek  = (long long)k*(k-1)/2;
        long long ek1 = (long long)(k+1)*k/2;
        float mono_lim = (k >= 2)
            ? (sim_sum / (float)ek) * (float)ek1 - sim_sum
            : 1e30f;

        bool any_extended = false;
        bool has_valid    = false;

        for (int ci = 0; ci < nc; ci++) {
            int   v  = lc.cands[ci];
            float dv = lc.acc[ci];

#if ENABLE_W_PRUNE
            if (W + dv - tau_k < 0.f) continue;
#endif
            has_valid = true;

#if ENABLE_MONO_PRUNE
            if (k >= 2 && dv > mono_lim) break;
#endif

            LayerClique nlc;
            nlc.sim_sum = sim_sum + dv;
            nlc.nodes   = lc.nodes;
            nlc.nodes.insert(lower_bound(nlc.nodes.begin(), nlc.nodes.end(), v), v);

            const auto& vrow = bitmap_->get_row(v);
            nlc.cands.reserve(nc);
            nlc.acc.reserve(nc);
            for (int di = 0; di < nc; di++) {
                int w = lc.cands[di];
                if (w == v) continue;
                if (!vrow.contains(w)) continue;
                float sv = lookup_sim(w, v);
                nlc.cands.push_back(w);
                nlc.acc.push_back(lc.acc[di] + sv);
            }

            // 按新 acc 升序排序
            {
                int nc2 = (int)nlc.cands.size();
                vector<int> idx(nc2); iota(idx.begin(), idx.end(), 0);
                sort(idx.begin(), idx.end(),
                     [&](int a, int b){ return nlc.acc[a] < nlc.acc[b]; });
                vector<int> sc(nc2); vector<float> sa(nc2);
                for (int i = 0; i < nc2; i++) { sc[i] = nlc.cands[idx[i]]; sa[i] = nlc.acc[idx[i]]; }
                nlc.cands = move(sc); nlc.acc = move(sa);
            }

            next_out.push_back(move(nlc));
            any_extended = true;
        }

        if (!any_extended && !has_valid && k >= 2 && W > 0.f) {
            LayerClique mc;
            mc.nodes      = lc.nodes;
            mc.sim_sum    = sim_sum;
            mc.is_maximal = true;
            next_out.push_back(move(mc));
        }
    }

    // --------------------------------------------------------
    // Stage 2+3: 分层并行扩展
    // --------------------------------------------------------
    void run_layered_expand() {
        int n   = g_.V;
        int nth = omp_get_max_threads();
        logger.print("Stage2: building layer-0 (sim > %.3f edges)...\n", (double)tau_);
        auto t_all = chrono::high_resolution_clock::now();

        // Layer 0
        vector<vector<LayerClique>> loc0(nth);
#pragma omp parallel for schedule(dynamic, 100)
        for (int u = 0; u < n; u++) {
            int tid = omp_get_thread_num();
            const auto& urow = bitmap_->get_row(u);
            for (auto& [v, s] : nb_[u]) {
                if (v <= u || s <= tau_) continue;
                LayerClique lc;
                lc.nodes   = {u, v};
                lc.sim_sum = s;
                urow.intersect_to(bitmap_->get_row(v), lc.cands);
                lc.acc.reserve(lc.cands.size());
                for (int w : lc.cands)
                    lc.acc.push_back(lookup_sim(w, u) + lookup_sim(w, v));
                int nc = (int)lc.cands.size();
                vector<int> idx(nc); iota(idx.begin(), idx.end(), 0);
                sort(idx.begin(), idx.end(), [&](int a, int b){ return lc.acc[a] < lc.acc[b]; });
                vector<int> sc(nc); vector<float> sa(nc);
                for (int i = 0; i < nc; i++) { sc[i] = lc.cands[idx[i]]; sa[i] = lc.acc[idx[i]]; }
                lc.cands = move(sc); lc.acc = move(sa);
                loc0[tid].push_back(move(lc));
            }
        }
        vector<LayerClique> current;
        for (auto& lv : loc0) for (auto& c : lv) current.push_back(move(c));
        loc0.clear();

        sort(current.begin(), current.end(),
             [](const LayerClique& a, const LayerClique& b){ return a.nodes < b.nodes; });
        current.erase(unique(current.begin(), current.end(),
             [](const LayerClique& a, const LayerClique& b){ return a.nodes == b.nodes; }),
             current.end());
        logger.print("  Layer-0: %zu 2-cliques (after dedup)\n", current.size());
        logger.print("Stage3: clique expansion by size layers...\n");

        int layer = 0;
        while (!current.empty()) {
            int  N     = (int)current.size();
            auto t_lyr = chrono::high_resolution_clock::now();

            vector<vector<LayerClique>> next_loc(nth);
#pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < N; i++) {
                int tid = omp_get_thread_num();
                process_one(current[i], next_loc[tid]);
            }
            { vector<LayerClique> tmp; tmp.swap(current); }

            {
                size_t tot = 0;
                for (auto& lv : next_loc) tot += lv.size();
                current.reserve(tot);
                for (auto& lv : next_loc) {
                    for (auto& c : lv) current.push_back(move(c));
                    vector<LayerClique>().swap(lv);
                }
            }
            next_loc.clear();

            size_t before = current.size();
            long long dedup_removed = 0;
#if ENABLE_LAYER_DEDUP
            sort(current.begin(), current.end(),
                 [](const LayerClique& a, const LayerClique& b){ return a.nodes < b.nodes; });
            current.erase(
                unique(current.begin(), current.end(),
                       [](const LayerClique& a, const LayerClique& b){
                           return a.nodes == b.nodes; }),
                current.end());
            dedup_removed = (long long)(before - current.size());
#endif

            vector<LayerClique> next_layer;
            next_layer.reserve(current.size());
            size_t maximal_this_layer = 0;
            for (auto& c : current) {
                if (c.is_maximal) {
                    lock_guard<mutex> lk(collected_mtx_);
                    collected_.push_back(c.nodes);
                    maximal_this_layer++;
                } else {
                    next_layer.push_back(move(c));
                }
            }
            { vector<LayerClique> tmp; tmp.swap(current); }
            current = move(next_layer);

            long long elapsed = chrono::duration_cast<chrono::milliseconds>(
                chrono::high_resolution_clock::now()-t_lyr).count();
            logger.print("  Layer %2d: in=%-8d  out_next=%-8zu  dedup_removed=%-8lld  "
                         "maximal_collected=%-6zu  t=%lldms\n",
                         layer, N, current.size(), dedup_removed,
                         maximal_this_layer, elapsed);
            layer++;
        }

        long long total_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t_all).count();
        logger.print("\n  [Stage2+3 time]: %d layers, %lld ms\n", layer, total_ms);
    }

    // --------------------------------------------------------
    // Stage 4: 后验精确去重 + 输出
    // --------------------------------------------------------
    long long post_validate() {
        logger.print("\nStage4: post-validation (exact dedup)...\n");
        auto t0 = chrono::high_resolution_clock::now();

        size_t raw = collected_.size();
        sort(collected_.begin(), collected_.end());
        collected_.erase(unique(collected_.begin(), collected_.end()), collected_.end());
        size_t after_exact = collected_.size();
        logger.print("  raw collected: %zu  after exact dedup: %zu  (removed %zu)\n",
                     raw, after_exact, raw - after_exact);

        // 按大小降序，大团优先
        sort(collected_.begin(), collected_.end(),
             [](const vector<int>& a, const vector<int>& b){ return a.size() > b.size(); });

        // 子集检查（倒排索引）
        unordered_map<int, vector<size_t>> inverted;
        vector<bool> valid(collected_.size(), true);
        size_t subset_pruned = 0;

        for (size_t ci = 0; ci < collected_.size(); ci++) {
            const vector<int>& C = collected_[ci];
            int csz = (int)C.size();

            int    rarest  = C[0];
            size_t min_len = inverted.count(C[0]) ? inverted[C[0]].size() : 0;
            for (int v : C) {
                auto it = inverted.find(v);
                size_t len = it != inverted.end() ? it->second.size() : 0;
                if (len < min_len) { min_len = len; rarest = v; }
            }

            auto it = inverted.find(rarest);
            if (it != inverted.end()) {
                for (size_t ti : it->second) {
                    const vector<int>& T = collected_[ti];
                    if ((int)T.size() <= csz) continue;
                    bool subset = true;
                    size_t ci2 = 0, ti2 = 0;
                    while (ci2 < C.size() && ti2 < T.size()) {
                        if      (C[ci2] == T[ti2]) { ci2++; ti2++; }
                        else if (C[ci2] >  T[ti2]) { ti2++; }
                        else { subset = false; break; }
                    }
                    if (ci2 < C.size()) subset = false;
                    if (subset) { valid[ci] = false; subset_pruned++; break; }
                }
            }
            if (valid[ci])
                for (int v : C) inverted[v].push_back(ci);
        }

        long long elapsed = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        size_t final_count = collected_.size() - subset_pruned;
        logger.print("  subset_pruned: %zu  final: %zu\n", subset_pruned, final_count);
        logger.print("  [Stage4 time]: %lld ms\n", elapsed);

        // 输出 & size 分布
        unordered_map<int,long long> sz_dist;
        long long sum_sizes = 0;
        for (size_t ci = 0; ci < collected_.size(); ci++) {
            if (!valid[ci]) continue;
            const vector<int>& C = collected_[ci];
            sz_dist[(int)C.size()]++;
            sum_sizes += (long long)C.size();
            if (sink_) {
                for (size_t i = 0; i < C.size(); i++) {
                    if (i) fputc(' ', sink_);
                    fprintf(sink_, "%d", C[i]);
                }
                fputc('\n', sink_);
            }
        }

        logger.print("\n--- Clique size distribution ---\n");
        vector<pair<int,long long>> sv(sz_dist.begin(), sz_dist.end());
        sort(sv.begin(), sv.end());
        for (auto& [sz, cnt] : sv)
            logger.print("  size %3d: %lld (%.2f%%)\n", sz, cnt,
                         final_count > 0 ? 100.0*cnt/final_count : 0.0);
        if (final_count > 0)
            logger.print("  Average clique size: %.2f\n", (double)sum_sizes/final_count);
        logger.print("--- Total cliques: %zu ---\n", final_count);

        return (long long)final_count;
    }

public:
    AvgSimCliqueMiner(const VectorDB& db, const Graph& g) : db_(db), g_(g), tau_(0.f) {}
    ~AvgSimCliqueMiner() { delete bitmap_; if (sink_) fclose(sink_); }

    void set_sink(const char* fn) {
        if (sink_) { fclose(sink_); sink_ = nullptr; }
        if (fn && fn[0]) sink_ = fopen(fn, "w");
    }

    long long mine(double tau) {
        tau_ = (float)tau;
        collected_.clear();
        delete bitmap_; bitmap_ = nullptr;

        logger.print("\n=== AvgSim Clique MCE  tau=%.3f ===\n", tau_);
        logger.print("Ablation: W_PRUNE=%d  MONO_PRUNE=%d  LAYER_DEDUP=%d\n",
                     ENABLE_W_PRUNE, ENABLE_MONO_PRUNE, ENABLE_LAYER_DEDUP);
        auto t0 = chrono::high_resolution_clock::now();

        build_graph();

        int n = g_.V;
        bitmap_ = new BlockedSparseBitmap(n);
#pragma omp parallel for schedule(static)
        for (int u = 0; u < n; u++) {
            vector<int> ids; ids.reserve(nb_[u].size());
            for (auto& [v,s] : nb_[u]) ids.push_back(v);
            bitmap_->build(u, ids);
        }

        run_layered_expand();

        long long final_count = post_validate();

        long long total_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        logger.print("\n=== TOTAL: %lld ms,  %lld maximal cliques ===\n", total_ms, final_count);
        if (sink_) fflush(sink_);
        return final_count;
    }
};

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    omp_set_num_threads(omp_get_max_threads());
    string graph_file, vec_index, log_path;
    int dataset_id = 2;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "dataset" && i+1 < argc) dataset_id = atoi(argv[++i]);
        else if (arg == "log"     && i+1 < argc) log_path = argv[++i];
    }
    switch (dataset_id) {
        case 1:  graph_file="dataset/dataset/sc-ldoor.txt";          vec_index="dataset/vectors-256/sc-ldoor_vectors.bin";        break;
        case 2:  graph_file="dataset/dataset/sc-nasasrb.txt";        vec_index="dataset/vectors-256/sc-nasasrb_vectors.bin";      break;
        case 3:  graph_file="dataset/dataset/sc-pkustk11.txt";       vec_index="dataset/vectors-256/sc-pkustk11_vectors.bin";     break;
        case 4:  graph_file="dataset/dataset/soc-buzznet.txt";       vec_index="dataset/vectors-256/soc-buzznet_vectors.bin";     break;
        case 5:  graph_file="dataset/dataset/soc-digg.txt";          vec_index="dataset/vectors-256/soc-digg_vectors.bin";        break;
        case 6:  graph_file="dataset/dataset/tech-as-skitter.txt";   vec_index="dataset/vectors-256/tech-as-skitter_vectors.bin"; break;
        case 7:  graph_file="dataset/dataset/FB15K-237_edges.txt";   vec_index="dataset/vectors/FB15K-237_vectors.index";         break;
        case 8:  graph_file="dataset/dataset/WN18RR_edges.txt";      vec_index="dataset/vectors/WN18RR_vectors.index";            break;
        case 9:  graph_file="dataset/dataset/dblp_coauthor.txt";     vec_index="dataset/vectors/dblp_coauthor_vectors.index";     break;
        case 10: graph_file="dataset/dataset/products.txt";          vec_index="dataset/vectors/products_vectors.index";          break;
        default: graph_file="dataset/dataset/sc-nasasrb.txt";        vec_index="dataset/vectors-256/sc-nasasrb_vectors.bin";      break;
    }

    if (!log_path.empty()) logger.open(log_path);

    logger.print("============================================\n");
    logger.print("  alg6_v2_standard (Clique Size-Layered MCE)\n");
    logger.print("  mine <tau> | sink <file> | log <file> | quit\n");
    logger.print("============================================\n");
    logger.print("Dataset: %s (ID=%d)\n", graph_file.c_str(), dataset_id);

    auto t_load = chrono::high_resolution_clock::now();
    Graph g(graph_file);
    if (g.V == 0) { fprintf(stderr, "Failed to load graph\n"); return 1; }
    if (fs::exists(vec_index)) {
        bool ok = (vec_index.find(".index") != string::npos)
                  ? g.loadVectorsFromChunks(vec_index) : g.loadVectorsFromBinary(vec_index);
        if (!ok) { fprintf(stderr, "Failed to load vectors: %s\n", vec_index.c_str()); return 1; }
    } else { fprintf(stderr, "Vector file not found: %s\n", vec_index.c_str()); return 1; }
    g.normalizeVectors();

    long long ec = 0;
    for (int i = 0; i < g.V; i++) ec += (long long)g.adj[i].size();
    ec /= 2;
    long long load_ms = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now()-t_load).count();
    logger.print("Graph: V=%d  E=%lld\n", g.V, ec);
    logger.print("Load: %lld ms\n\n", load_ms);

    VectorDB db; db.build(g);
    AvgSimCliqueMiner miner(db, g);

    logger.print("\n>> ");
    string line;
    while (getline(cin, line)) {
        if (line.empty()) { logger.print(">> "); continue; }
        istringstream iss(line); string cmd; iss >> cmd;
        if (cmd == "mine") {
            double tau = 0.2; iss >> tau;
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