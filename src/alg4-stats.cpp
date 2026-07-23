// ============================================================
// ALG7_v2_stats.cpp  平均相似度约束极大团挖掘（统计信息扩展版）
//
// 统计信息（全部精确，不影响原有逻辑）：
//
// 通用统计（每次 mine 重置）：
//   #recursive_calls       递归调用次数
//   #search_nodes          搜索树节点数（含叶子）
//   #candidate_expansions  候选扩展尝试次数（cands 遍历总次数）
//   #similarity_computations 相似度计算缓存访问次数
//
// ALG6 专项统计：
//   #mono_prune_triggers   单调剪枝触发次数（break 前的那次）
//   #w_prune_triggers      W 剪枝触发次数（continue 的次数）
//   dedup_ratio            最后的去重比例（去重删除数 / 总产出数）
//   avg_effective_rate     每次递归的有效扩展数/总扩展数（across all calls）
//
// 注：
//   - "递归调用" 对应每次 process_one 调用
//   - "搜索树节点" = 每次 process_one 处理的团（含 layer-0 种子）
//   - "候选扩展尝试" = process_one 中 for 循环的每次迭代（含被剪枝的）
//   - "有效扩展" = 实际产出了子状态的候选（未被 W/mono 剪枝、且进入了 next_out）
// ============================================================

#include "semantic_graph.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <cstdarg>
#include <numeric>
#include <omp.h>
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
// NodePool：per-thread 父链节点存储（无锁、无线程安全bug）
// ============================================================
struct NodePool {
    struct Entry {
        int node;
        int parent;
    };

    static constexpr int IDX_BITS  = 22;
    static constexpr int TID_SHIFT = IDX_BITS;
    static constexpr int MAX_LOCAL = 1 << IDX_BITS;
    static constexpr int MAX_TID   = 512;

    vector<vector<Entry>> tp;

    void init(int num_threads) {
        tp.resize(num_threads);
    }

    static int encode(int tid, int local_idx) {
        return (tid << TID_SHIFT) | local_idx;
    }
    static int decode_tid(int encoded) {
        return encoded >> TID_SHIFT;
    }
    static int decode_local(int encoded) {
        return encoded & (MAX_LOCAL - 1);
    }

    int alloc(int node, int parent) {
        int tid = omp_get_thread_num();
        auto& p = tp[tid];
        int local_idx = (int)p.size();
        p.push_back({node, parent});
        return encode(tid, local_idx);
    }

    void rebuild(int idx, vector<int>& out) const {
        out.clear();
        while (idx != -1) {
            int tid   = decode_tid(idx);
            int local = decode_local(idx);
            const Entry& e = tp[tid][local];
            out.push_back(e.node);
            idx = e.parent;
        }
        sort(out.begin(), out.end());
    }

    void clear() { for (auto& p : tp) p.clear(); }
    int  size()  const {
        int total = 0;
        for (auto& p : tp) total += (int)p.size();
        return total;
    }
};

// ============================================================
// CandPool：per-thread 扁平 cands/acc 存储
// ============================================================
struct CandPool {
    static constexpr int IDX_BITS  = 24;
    static constexpr int TID_SHIFT = IDX_BITS;
    static constexpr int MAX_LOCAL = 1 << IDX_BITS;

    struct ThreadData {
        vector<int>   cands;
        vector<float> acc;
    };
    vector<ThreadData> tp;

    void init(int nth) { tp.resize(nth); }

    static int encode(int tid, int offset) { return (tid << TID_SHIFT) | offset; }
    static int decode_tid(int e) { return e >> TID_SHIFT; }
    static int decode_local(int e) { return e & (MAX_LOCAL - 1); }

    int alloc(const int* c, const float* a, int sz) {
        if (sz == 0) return -1;
        int tid = omp_get_thread_num();
        auto& td = tp[tid];
        int offset = (int)td.cands.size();
        td.cands.insert(td.cands.end(), c, c + sz);
        td.acc.insert(td.acc.end(), a, a + sz);
        return encode(tid, offset);
    }

    const int*   cands_ptr(int encoded) const {
        return tp[decode_tid(encoded)].cands.data() + decode_local(encoded);
    }
    const float* acc_ptr(int encoded) const {
        return tp[decode_tid(encoded)].acc.data() + decode_local(encoded);
    }

    void clear() { for (auto& td : tp) { td.cands.clear(); td.acc.clear(); } }
    size_t total_entries() const {
        size_t s = 0;
        for (auto& td : tp) s += td.cands.size();
        return s;
    }
};

// ============================================================
// LayerClique（内存优化版 v2：nodes + cands/acc 全部池化）
// ============================================================
struct LayerClique {
    int   pool_idx   = -1;
    int   clique_sz  = 0;
    int   cand_idx   = -1;
    int   cand_sz    = 0;
    float sim_sum    = 0.f;
    bool  is_maximal = false;

    void rebuild_nodes(const NodePool& pool, vector<int>& out) const {
        pool.rebuild(pool_idx, out);
    }
};

// ============================================================
// MiningStats：所有统计计数器，mine() 开始时重置
// ============================================================
struct MiningStats {
    // 通用统计
    atomic<uint64_t> recursive_calls{0};        // process_one 被调用次数
    atomic<uint64_t> search_nodes{0};           // 搜索树节点数（含 layer-0 种子 + process_one 处理的每个团）
    atomic<uint64_t> candidate_expansions{0};   // 候选遍历总次数（for ci 的每次迭代）
    atomic<uint64_t> similarity_computations{0};// lookup_sim 调用次数

    // ALG6 专项
    atomic<uint64_t> w_prune_triggers{0};       // W 剪枝触发次数（continue 被执行）
    atomic<uint64_t> mono_prune_triggers{0};    // 单调剪枝触发次数（break 被执行）
    atomic<uint64_t> effective_expansions{0};   // 实际产出子状态的扩展次数
    // 去重统计在 run_layered_expand 中直接统计，非原子（单线程排序后统计）
    long long total_dedup_removed = 0;          // 所有层去重删除的总数
    long long total_produced      = 0;          // 所有层产出的总数（去重前）

    void reset() {
        recursive_calls        = 0;
        search_nodes           = 0;
        candidate_expansions   = 0;
        similarity_computations= 0;
        w_prune_triggers       = 0;
        mono_prune_triggers    = 0;
        effective_expansions   = 0;
        total_dedup_removed    = 0;
        total_produced         = 0;
    }

    void print(float tau, long long final_cliques) const {
        uint64_t rc  = recursive_calls.load();
        uint64_t sn  = search_nodes.load();
        uint64_t ce  = candidate_expansions.load();
        uint64_t sc  = similarity_computations.load();
        uint64_t wp  = w_prune_triggers.load();
        uint64_t mp  = mono_prune_triggers.load();
        uint64_t ee  = effective_expansions.load();

        double dedup_ratio   = (total_produced > 0)
            ? 100.0 * total_dedup_removed / total_produced : 0.0;
        double eff_rate      = (rc > 0)
            ? 100.0 * ee / (ce > 0 ? ce : 1) : 0.0;

        logger.print("\n");
        logger.print("+=========================================================+\n");
        logger.print("|   Extended Mining Statistics   tau=%.3f                 |\n", (double)tau);
        logger.print("+---------------------------------------------------------+\n");
        logger.print("|  [General]                                              |\n");
        logger.print("|  #recursive_calls          : %16llu          |\n", (unsigned long long)rc);
        logger.print("|  #search_nodes             : %16llu          |\n", (unsigned long long)sn);
        logger.print("|  #candidate_expansions     : %16llu          |\n", (unsigned long long)ce);
        logger.print("|  #similarity_computations  : %16llu          |\n", (unsigned long long)sc);
        logger.print("+---------------------------------------------------------+\n");
        logger.print("|  [ALG6 Pruning]                                         |\n");
        logger.print("|  #W_prune_triggers         : %16llu          |\n", (unsigned long long)wp);
        logger.print("|  #mono_prune_triggers      : %16llu          |\n", (unsigned long long)mp);
        logger.print("|  #effective_expansions     : %16llu          |\n", (unsigned long long)ee);
        logger.print("|  Effective expansion rate  : %15.2f%%          |\n", eff_rate);
        logger.print("+---------------------------------------------------------+\n");
        logger.print("|  [Dedup]                                                |\n");
        logger.print("|  Total produced (before dedup): %13lld          |\n", total_produced);
        logger.print("|  Total dedup removed          : %13lld          |\n", total_dedup_removed);
        logger.print("|  Dedup ratio                  : %13.2f%%          |\n", dedup_ratio);
        logger.print("+---------------------------------------------------------+\n");
        logger.print("|  Final maximal cliques     : %16lld          |\n", final_cliques);
        logger.print("+=========================================================+\n");
    }
};

// ============================================================
// AvgSimCliqueMiner（统计信息扩展版）
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

    mutable MiningStats stats_;

    // --------------------------------------------------------
    inline float lookup_sim(int u, int v) const {
        stats_.similarity_computations.fetch_add(1, memory_order_relaxed);
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
    // process_one：带完整统计插桩
    // --------------------------------------------------------
    void process_one(const LayerClique& lc, vector<LayerClique>& next_out,
                     NodePool& pool, const CandPool& cpool_r, CandPool& cpool_w) {
        stats_.recursive_calls.fetch_add(1, memory_order_relaxed);
        stats_.search_nodes.fetch_add(1, memory_order_relaxed);

        const int   k       = lc.clique_sz;
        const float sim_sum = lc.sim_sum;
        const int   nc      = lc.cand_sz;

        static thread_local vector<int> in_cands;
        static thread_local vector<float> in_acc;
        in_cands.clear(); in_acc.clear();
        if (nc > 0) {
            const int*   raw_c = cpool_r.cands_ptr(lc.cand_idx);
            const float* raw_a = cpool_r.acc_ptr(lc.cand_idx);
            in_cands.assign(raw_c, raw_c + nc);
            in_acc.assign(raw_a, raw_a + nc);
        }

        float W       = sim_sum - tau_ * (float)((long long)k*(k-1)/2);
        float tau_k   = tau_ * (float)k;
        long long ek  = (long long)k*(k-1)/2;
        long long ek1 = (long long)(k+1)*k/2;
        float mono_lim = (k >= 2)
            ? (sim_sum / (float)ek) * (float)ek1 - sim_sum
            : 1e30f;

        bool any_extended = false;
        bool has_valid    = false;

        static thread_local vector<int> tmp_cands;
        static thread_local vector<float> tmp_acc;

        for (int ci = 0; ci < nc; ci++) {
            stats_.candidate_expansions.fetch_add(1, memory_order_relaxed);

            int   v  = in_cands[ci];
            float dv = in_acc[ci];

            if (W + dv - tau_k < 0.f) {
                stats_.w_prune_triggers.fetch_add(1, memory_order_relaxed);
                continue;
            }
            has_valid = true;

            if (k >= 2 && dv > mono_lim) {
                stats_.mono_prune_triggers.fetch_add(1, memory_order_relaxed);
                break;
            }

            tmp_cands.clear();
            tmp_acc.clear();

            const auto& vrow = bitmap_->get_row(v);
            for (int di = 0; di < nc; di++) {
                int w = in_cands[di];
                if (w == v) continue;
                if (!vrow.contains(w)) continue;
                float sv = lookup_sim(w, v);
                tmp_cands.push_back(w);
                tmp_acc.push_back(in_acc[di] + sv);
            }

            int nc2 = (int)tmp_cands.size();
            if (nc2 > 1) {
                vector<int> idx(nc2); iota(idx.begin(), idx.end(), 0);
                sort(idx.begin(), idx.end(),
                     [&](int a, int b){ return tmp_acc[a] < tmp_acc[b]; });
                vector<int> sc(nc2); vector<float> sa(nc2);
                for (int i = 0; i < nc2; i++) { sc[i] = tmp_cands[idx[i]]; sa[i] = tmp_acc[idx[i]]; }
                tmp_cands = move(sc); tmp_acc = move(sa);
            }

            LayerClique nlc;
            nlc.sim_sum   = sim_sum + dv;
            nlc.clique_sz = k + 1;
            nlc.pool_idx  = pool.alloc(v, lc.pool_idx);
            nlc.cand_sz   = nc2;
            nlc.cand_idx  = cpool_w.alloc(tmp_cands.data(), tmp_acc.data(), nc2);

            next_out.push_back(move(nlc));
            any_extended = true;
            stats_.effective_expansions.fetch_add(1, memory_order_relaxed);
        }

        if (!any_extended && !has_valid && k >= 2 && W > 0.f) {
            LayerClique mc;
            mc.pool_idx   = lc.pool_idx;
            mc.clique_sz  = k;
            mc.sim_sum    = sim_sum;
            mc.is_maximal = true;
            next_out.push_back(move(mc));
        }
    }

    // --------------------------------------------------------
    // run_layered_expand：分层并行扩展，含去重统计
    // --------------------------------------------------------
    void run_layered_expand() {
        int n   = g_.V;
        int nth = omp_get_max_threads();
        logger.print("Stage2: building layer-0 (sim > %.3f edges)...\n", (double)tau_);
        auto t_all = chrono::high_resolution_clock::now();

        NodePool pool;
        pool.init(nth);

        CandPool cpool[2];
        cpool[0].init(nth);
        cpool[1].init(nth);
        int cbuf = 0;

        // Layer 0：seed 的每个 2-clique 也算一个搜索树节点
        vector<vector<LayerClique>> loc0(nth);
#pragma omp parallel for schedule(dynamic, 100)
        for (int u = 0; u < n; u++) {
            int tid = omp_get_thread_num();
            const auto& urow = bitmap_->get_row(u);
            for (auto& [v, s] : nb_[u]) {
                if (v <= u || s <= tau_) continue;
                LayerClique lc;
                lc.sim_sum   = s;
                lc.clique_sz = 2;
                int idx_v = pool.alloc(v, -1);
                lc.pool_idx  = pool.alloc(u, idx_v);

                static thread_local vector<int> l0_cands;
                static thread_local vector<float> l0_acc;
                l0_cands.clear();
                urow.intersect_to(bitmap_->get_row(v), l0_cands);
                l0_acc.resize(l0_cands.size());
                for (int i = 0; i < (int)l0_cands.size(); i++)
                    l0_acc[i] = lookup_sim(l0_cands[i], u) + lookup_sim(l0_cands[i], v);
                int nc = (int)l0_cands.size();
                if (nc > 1) {
                    vector<int> idx(nc); iota(idx.begin(), idx.end(), 0);
                    sort(idx.begin(), idx.end(), [&](int a, int b){ return l0_acc[a] < l0_acc[b]; });
                    vector<int> sc(nc); vector<float> sa(nc);
                    for (int i = 0; i < nc; i++) { sc[i] = l0_cands[idx[i]]; sa[i] = l0_acc[idx[i]]; }
                    l0_cands = move(sc); l0_acc = move(sa);
                }
                lc.cand_sz  = nc;
                lc.cand_idx = cpool[0].alloc(l0_cands.data(), l0_acc.data(), nc);
                loc0[tid].push_back(move(lc));
                stats_.search_nodes.fetch_add(1, memory_order_relaxed);
            }
        }
        vector<LayerClique> current;
        for (auto& lv : loc0) for (auto& c : lv) current.push_back(move(c));
        loc0.clear();

        // layer-0 去重
        size_t before0 = current.size();
        {
            vector<vector<int>> tmp_nodes(current.size());
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < current.size(); i++)
                current[i].rebuild_nodes(pool, tmp_nodes[i]);

            vector<int> order(current.size());
            iota(order.begin(), order.end(), 0);
            sort(order.begin(), order.end(),
                 [&](int a, int b){ return tmp_nodes[a] < tmp_nodes[b]; });
            vector<LayerClique> sorted_lc; sorted_lc.reserve(current.size());
            for (int i : order) sorted_lc.push_back(move(current[i]));
            sorted_lc.erase(
                unique(sorted_lc.begin(), sorted_lc.end(),
                       [&](const LayerClique& a, const LayerClique& b){
                           return tmp_nodes[&a - &sorted_lc[0]] == tmp_nodes[&b - &sorted_lc[0]];
                       }),
                sorted_lc.end());
            current = move(sorted_lc);
        }
        {
            long long rem0 = (long long)(before0 - current.size());
            stats_.total_produced     += (long long)before0;
            stats_.total_dedup_removed += rem0;
        }
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
                process_one(current[i], next_loc[tid], pool, cpool[cbuf], cpool[1-cbuf]);
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

            cpool[1-cbuf].clear();
            cbuf = 1 - cbuf;

            size_t before = current.size();
            stats_.total_produced += (long long)before;

            {
                vector<vector<int>> tmp_nodes(current.size());
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < current.size(); i++)
                    current[i].rebuild_nodes(pool, tmp_nodes[i]);

                vector<int> order(current.size());
                iota(order.begin(), order.end(), 0);
                sort(order.begin(), order.end(),
                     [&](int a, int b){ return tmp_nodes[a] < tmp_nodes[b]; });

                vector<LayerClique> sorted_lc; sorted_lc.reserve(current.size());
                for (int i : order) sorted_lc.push_back(move(current[i]));

                vector<LayerClique> deduped; deduped.reserve(current.size());
                for (size_t i = 0; i < sorted_lc.size(); i++) {
                    if (i > 0 && tmp_nodes[order[i]] == tmp_nodes[order[i-1]]) continue;
                    deduped.push_back(move(sorted_lc[i]));
                }
                current = move(deduped);
            }
            size_t after = current.size();
            long long dedup_removed = (long long)(before - after);
            stats_.total_dedup_removed += dedup_removed;

            vector<LayerClique> next_layer;
            next_layer.reserve(after);
            size_t maximal_this_layer = 0;
            {
                static thread_local vector<int> rbuf;
                for (auto& c : current) {
                    if (c.is_maximal) {
                        c.rebuild_nodes(pool, rbuf);
                        {
                            lock_guard<mutex> lk(collected_mtx_);
                            collected_.push_back(rbuf);
                        }
                        maximal_this_layer++;
                    } else {
                        next_layer.push_back(move(c));
                    }
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
    // post_validate：后验精确去重 + 子集检查
    // --------------------------------------------------------
    long long post_validate() {
        logger.print("\nStage4: post-validation (exact dedup + subset check)...\n");
        auto t0 = chrono::high_resolution_clock::now();

        size_t raw = collected_.size();
        sort(collected_.begin(), collected_.end());
        collected_.erase(unique(collected_.begin(), collected_.end()), collected_.end());
        size_t after_exact = collected_.size();
        logger.print("  raw collected: %zu  after exact dedup: %zu  (removed %zu exact duplicates)\n",
                     raw, after_exact, raw - after_exact);

        sort(collected_.begin(), collected_.end(),
             [](const vector<int>& a, const vector<int>& b){ return a.size() > b.size(); });

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
        logger.print("  subset check pruned: %zu  (should be 0 if algorithm is correct)\n", subset_pruned);
        logger.print("  final count: %zu\n", final_count);
        logger.print("  [Stage4 time]: %lld ms\n", elapsed);

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

        logger.print("\n--- Clique size distribution (after post-validation) ---\n");
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
        stats_.reset();
        collected_.clear();
        delete bitmap_; bitmap_ = nullptr;

        logger.print("\n=== AvgSim Clique MCE  tau=%.3f ===\n", tau_);
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

        stats_.print(tau_, final_count);

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
    logger.print("  ALG7_v2_stats (Clique Size-Layered MCE, Extended Stats)\n");
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