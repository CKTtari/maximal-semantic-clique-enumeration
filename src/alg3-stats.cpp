// ============================================================
// ALG5_stats.cpp  StrSub 结构极大团 + 语义子集枚举（统计信息扩展版）
//
// 统计信息（全部精确，不影响原有逻辑）：
//
// 通用统计：
//   #recursive_calls       BK递归调用次数 (bk_search)
//   #search_nodes          搜索树节点数（每次bk_search入口）
//   #candidate_expansions  候选扩展尝试次数（to_enum 遍历总次数）
//   #similarity_computations  lookup_sim 调用次数
//
// ALG3/StrSub 专项统计：
//   #phase1_calls          BK搜索调用次数（结构极大团枚举）
//   #phase2_calls          子集枚举调用次数（process_structural_clique）
//   #subsets_produced      Phase2产出的语义合法子集总数（去重前）
//   #topedge_prunes        Top-Edge Layer Pruning 跳过的 k 值数
//   #localmax_prunes       Local Maximality Filter 删除的子集数
//   phase1_ratio           BK在Phase1+2中的时间占比(%)
//   phase2_ratio           子集枚举在Phase1+2中的时间占比(%)
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
    };
private:
    vector<Row> rows_; int n_;
public:
    BlockedSparseBitmap() : n_(0) {}
    explicit BlockedSparseBitmap(int n) : n_(n), rows_(n) {}
    void build(int u, const vector<int>& nb) { rows_[u].build(nb); }
    bool contains(int u, int v) const { return rows_[u].contains(v); }
    int size() const { return n_; }
};

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
// MiningStats：ALG3/StrSub 统计信息
// ============================================================
struct MiningStats {
    atomic<uint64_t> recursive_calls{0};
    atomic<uint64_t> search_nodes{0};
    atomic<uint64_t> candidate_expansions{0};
    atomic<uint64_t> similarity_computations{0};

    // ALG3 专项
    atomic<uint64_t> phase1_calls{0};         // bk_search 调用次数
    atomic<uint64_t> phase2_calls{0};         // process_structural_clique 调用次数
    atomic<long long> subsets_produced{0};     // Phase2 产出子集数（去重前）
    atomic<long long> topedge_prunes{0};       // Top-Edge Pruning 跳过数
    atomic<long long> localmax_prunes{0};      // Local Max Filter 删除数
    long long final_count = 0;                 // 最终极大团数
    double phase1_ratio = 0.0;                 // BK 时间占比
    double phase2_ratio = 0.0;                 // 子集枚举时间占比

    void reset() {
        recursive_calls = 0;
        search_nodes = 0;
        candidate_expansions = 0;
        similarity_computations = 0;
        phase1_calls = 0;
        phase2_calls = 0;
        subsets_produced = 0;
        topedge_prunes = 0;
        localmax_prunes = 0;
        final_count = 0;
        phase1_ratio = 0.0;
        phase2_ratio = 0.0;
    }

    void print(float tau, long long final_cliques,
               long long graph_ms, long long degen_ms,
               long long phase12_ms, long long phase3_ms,
               long long total_ms) const {
        uint64_t rc  = recursive_calls.load();
        uint64_t sn  = search_nodes.load();
        uint64_t ce  = candidate_expansions.load();
        uint64_t sc  = similarity_computations.load();
        uint64_t p1c = phase1_calls.load();
        uint64_t p2c = phase2_calls.load();
        uint64_t sp  = (uint64_t)subsets_produced.load();
        uint64_t tep = (uint64_t)topedge_prunes.load();
        uint64_t lmp = (uint64_t)localmax_prunes.load();

        double dedup_rate = (sp > 0) ? 100.0 * lmp / sp : 0.0;

        logger.print("\n");
        logger.print("+============================================================+\n");
        logger.print("|   Extended Mining Statistics (ALG3/StrSub)  tau=%.3f      |\n", (double)tau);
        logger.print("+------------------------------------------------------------+\n");
        logger.print("|  [General]                                                 |\n");
        logger.print("|  #recursive_calls (BK)      : %14llu            |\n", (unsigned long long)rc);
        logger.print("|  #search_nodes              : %14llu            |\n", (unsigned long long)sn);
        logger.print("|  #candidate_expansions      : %14llu            |\n", (unsigned long long)ce);
        logger.print("|  #similarity_computations   : %14llu            |\n", (unsigned long long)sc);
        logger.print("+------------------------------------------------------------+\n");
        logger.print("|  [ALG3 Two-Phase]                                           |\n");
        logger.print("|  #phase1_calls (BK enum)    : %14llu            |\n", (unsigned long long)p1c);
        logger.print("|  #phase2_calls (subset enum): %14llu            |\n", (unsigned long long)p2c);
        logger.print("|  #subsets_produced (pre-dedup): %12lld            |\n", subsets_produced.load());
        logger.print("|  #topedge_prunes           : %14llu            |\n", (unsigned long long)tep);
        logger.print("|  #localmax_prunes          : %14llu            |\n", (unsigned long long)lmp);
        logger.print("|  LocalMax filter rate      : %13.2f %%           |\n", dedup_rate);
        logger.print("|  BK time ratio             : %13.2f %%           |\n", phase1_ratio);
        logger.print("|  Subset time ratio         : %13.2f %%           |\n", phase2_ratio);
        logger.print("+------------------------------------------------------------+\n");
        logger.print("|  [Phase Timing]                                            |\n");
        logger.print("|  Graph build        : %8lld ms  (%5.1f%%)               |\n",
                     graph_ms, total_ms>0?100.0*graph_ms/total_ms:0.0);
        logger.print("|  Degeneracy ordering: %8lld ms  (%5.1f%%)               |\n",
                     degen_ms, total_ms>0?100.0*degen_ms/total_ms:0.0);
        logger.print("|  BK + Subset enum    : %8lld ms  (%5.1f%%)               |\n",
                     phase12_ms, total_ms>0?100.0*phase12_ms/total_ms:0.0);
        logger.print("|  Global dedup (P3)   : %8lld ms  (%5.1f%%)               |\n",
                     phase3_ms, total_ms>0?100.0*phase3_ms/total_ms:0.0);
        logger.print("+============================================================+\n");
        logger.print("\n");
    }
};

class AVGMiner {
private:
    const VectorDB& db_;
    const Graph&    g_;
    float           tau_;

    vector<vector<pair<int,float>>> nb_;
    vector<unordered_map<int,float>> sim_map_;

    BlockedSparseBitmap* bitmap_ = nullptr;
    vector<int>          ordering_;
    vector<vector<int>>  candidates_;
    double               phase1_ratio_ = 0.0;
    double               phase2_ratio_ = 0.0;

    atomic<long long>    clique_count_{0};
    static constexpr int MAX_CLIQUE_SZ = 512;
    atomic<long long>    size_hist_[MAX_CLIQUE_SZ]{};
    FILE*                sink_ = nullptr;
    mutex                sink_mutex_;

    mutable MiningStats stats_;

    // ============================================================
    // Mask128：用两个 uint64_t 拼成 128-bit 掩码
    //   lo = bit 0..63，hi = bit 64..127
    //   上限从 63 升至 127，算法逻辑与原版完全一致
    // ============================================================
    struct Mask128 {
        uint64_t lo, hi;

        Mask128() : lo(0), hi(0) {}
        Mask128(uint64_t l, uint64_t h) : lo(l), hi(h) {}

        static Mask128 bit(int i) {
            if (i < 64) return Mask128(1ULL << i, 0);
            else        return Mask128(0, 1ULL << (i - 64));
        }

        static Mask128 low_k(int k) {
            if (k <= 0)  return Mask128(0, 0);
            if (k < 64)  return Mask128((1ULL << k) - 1, 0);
            if (k == 64) return Mask128(~0ULL, 0);
            if (k < 128) return Mask128(~0ULL, (1ULL << (k - 64)) - 1);
            return Mask128(~0ULL, ~0ULL);
        }

        static Mask128 end(int m) {
            if (m < 64) return Mask128(1ULL << m, 0);
            else        return Mask128(0, 1ULL << (m - 64));
        }

        bool operator==(const Mask128& o) const { return lo == o.lo && hi == o.hi; }
        bool operator!=(const Mask128& o) const { return !(*this == o); }
        bool operator<(const Mask128& o) const {
            return hi != o.hi ? hi < o.hi : lo < o.lo;
        }

        Mask128 operator&(const Mask128& o) const { return {lo & o.lo, hi & o.hi}; }
        Mask128 operator|(const Mask128& o) const { return {lo | o.lo, hi | o.hi}; }
        Mask128 operator^(const Mask128& o) const { return {lo ^ o.lo, hi ^ o.hi}; }
        Mask128 operator~() const { return {~lo, ~hi}; }

        bool zero() const { return lo == 0 && hi == 0; }

        Mask128 add128(const Mask128& o) const {
            uint64_t new_lo = lo + o.lo;
            uint64_t carry  = (new_lo < lo) ? 1ULL : 0ULL;
            uint64_t new_hi = hi + o.hi + carry;
            return {new_lo, new_hi};
        }

        int ctz() const {
            if (lo != 0) return __builtin_ctzll(lo);
            return 64 + __builtin_ctzll(hi);
        }

        Mask128 clear_lowest() const {
            if (lo != 0) return {lo & (lo - 1), hi};
            return {0, hi & (hi - 1)};
        }

        Mask128 neg() const {
            return (~*this).add128(Mask128(1, 0));
        }

        Mask128 shr(int n) const {
            if (n == 0)   return *this;
            if (n >= 128) return Mask128(0, 0);
            if (n >= 64)  return Mask128(hi >> (n - 64), 0);
            return Mask128((lo >> n) | (hi << (64 - n)), hi >> n);
        }

        Mask128 div_pow2(const Mask128& c) const {
            return shr(c.ctz());
        }

        Mask128 gosper_next() const {
            Mask128 c   = *this & neg();
            Mask128 r   = add128(c);
            Mask128 xr  = r ^ *this;
            Mask128 xr2 = xr.shr(2);
            Mask128 div = xr2.div_pow2(c);
            return div | r;
        }
    };

    inline void emit(const vector<int>& C) {
        clique_count_.fetch_add(1, memory_order_relaxed);
        int sz = (int)C.size();
        if (sz >= 1 && sz < MAX_CLIQUE_SZ)
            size_hist_[sz-1].fetch_add(1, memory_order_relaxed);
        if (sink_) {
            lock_guard<mutex> lk(sink_mutex_);
            for (int i = 0; i < sz; i++) {
                if (i) fputc(' ', sink_);
                fprintf(sink_, "%d", C[i]);
            }
            fputc('\n', sink_);
        }
    }

    inline float lookup_sim(int u, int v) const {
        stats_.similarity_computations.fetch_add(1, memory_order_relaxed);
        auto it = sim_map_[u].find(v);
        return it != sim_map_[u].end() ? it->second : 0.0f;
    }

    void process_structural_clique(
        const vector<int>& M,
        vector<vector<int>>& local_cands,
        long long& phase2_ns_local
    ) {
        stats_.phase2_calls.fetch_add(1, memory_order_relaxed);
        auto t0 = chrono::high_resolution_clock::now();
        int m = (int)M.size();
        if (m < 2) {
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
            return;
        }
        assert(m <= 127 && "clique too large for 128-bit bitmask");

        vector<int> sorted_M = M;
        sort(sorted_M.begin(), sorted_M.end());

        vector<vector<float>> sim_local(m, vector<float>(m, 0.0f));
        vector<float> all_edges;
        vector<float> node_contrib(m, 0.0f);
        float full_total = 0.0f;
        for (int i = 0; i < m; i++)
            for (int j = i+1; j < m; j++) {
                float s = lookup_sim(sorted_M[i], sorted_M[j]);
                sim_local[i][j] = s;
                sim_local[j][i] = s;
                node_contrib[i] += s;
                node_contrib[j] += s;
                all_edges.push_back(s);
                full_total += s;
            }

        int full_pairs = m * (m - 1) / 2;
        float full_avg = full_pairs > 0 ? full_total / full_pairs : 0.0f;
        if (full_avg >= tau_) {
            vector<int> subset = M;
            sort(subset.begin(), subset.end());
            local_cands.push_back(move(subset));
            stats_.subsets_produced.fetch_add(1, memory_order_relaxed);
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
            return;
        }

        sort(all_edges.rbegin(), all_edges.rend());

        if (all_edges[0] < tau_) {
            stats_.topedge_prunes.fetch_add(1, memory_order_relaxed);
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
            return;
        }

        int start_k = m;
        vector<float> prefix_sum(all_edges.size() + 1, 0.0f);
        for (int i = 0; i < (int)all_edges.size(); i++)
            prefix_sum[i+1] = prefix_sum[i] + all_edges[i];

        for (int k_candidate = m-1; k_candidate >= 2; k_candidate--) {
            int required_pairs = k_candidate * (k_candidate - 1) / 2;
            if (required_pairs > (int)all_edges.size()) continue;
            float top_avg = prefix_sum[required_pairs] / required_pairs;
            if (top_avg >= tau_) {
                start_k = k_candidate;
                break;
            } else {
                stats_.topedge_prunes.fetch_add(1, memory_order_relaxed);
            }
        }

        vector<int> node_order(m);
        iota(node_order.begin(), node_order.end(), 0);
        sort(node_order.begin(), node_order.end(),
             [&](int a, int b) { return node_contrib[a] < node_contrib[b]; });

        vector<Mask128> local_max;
        Mask128 union_mask;

        for (int k = start_k; k >= 2; k--) {
            Mask128 mask     = Mask128::low_k(k);
            Mask128 end_mask = Mask128::end(m);

            while (mask < end_mask) {
                bool covered = false;
                if (((mask & union_mask) ^ mask).zero()) {
                    for (const Mask128& lm : local_max) {
                        if (((mask & lm) ^ mask).zero()) { covered = true; break; }
                    }
                }

                if (!covered) {
                    float total = 0.0f;
                    Mask128 bits_i = mask;
                    while (!bits_i.zero()) {
                        int i = bits_i.ctz();
                        bits_i = bits_i.clear_lowest();
                        Mask128 bits_j = bits_i;
                        while (!bits_j.zero()) {
                            int j = bits_j.ctz();
                            bits_j = bits_j.clear_lowest();
                            total += sim_local[i][j];
                        }
                    }
                    int cnt = k * (k - 1) / 2;
                    float avg = cnt > 0 ? total / cnt : 0.0f;

                    if (avg >= tau_) {
                        local_max.push_back(mask);
                        union_mask = union_mask | mask;

                        vector<int> subset;
                        subset.reserve(k);
                        for (int i = 0; i < m; i++)
                            if ((mask & Mask128::bit(i)) != Mask128(0, 0)) subset.push_back(sorted_M[i]);
                        local_cands.push_back(move(subset));
                        stats_.subsets_produced.fetch_add(1, memory_order_relaxed);
                    }
                } else {
                    stats_.localmax_prunes.fetch_add(1, memory_order_relaxed);
                }

                mask = mask.gosper_next();
            }
        }
        phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
            chrono::high_resolution_clock::now() - t0).count();
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

    void bk_search(
        vector<int>&         C,
        vector<int>&         cands,
        vector<int>&         excl,
        AccStore&            acc,
        vector<vector<int>>& local_cands,
        atomic<bool>&        timed_out,
        long long&           phase2_ns_local
    ) {
        stats_.recursive_calls.fetch_add(1, memory_order_relaxed);
        stats_.search_nodes.fetch_add(1, memory_order_relaxed);

        if (timed_out.load()) return;

        if (cands.empty() && excl.empty()) {
            process_structural_clique(C, local_cands, phase2_ns_local);
            return;
        }

        int pivot = -1, pivot_cover = -1;
        for (int p : excl) {
            int cover = 0;
            for (int c : cands) if (bitmap_->contains(p, c)) cover++;
            if (cover > pivot_cover) { pivot_cover = cover; pivot = p; }
        }
        for (int p : cands) {
            int cover = 0;
            for (int c : cands) if (c != p && bitmap_->contains(p, c)) cover++;
            if (cover > pivot_cover) { pivot_cover = cover; pivot = p; }
        }

        vector<int> to_enum;
        to_enum.reserve(cands.size());
        for (int v : cands)
            if (pivot == -1 || !bitmap_->contains(pivot, v))
                to_enum.push_back(v);

        for (int v : to_enum) {
            stats_.candidate_expansions.fetch_add(1, memory_order_relaxed);
            if (timed_out.load()) return;

            cands.erase(find(cands.begin(), cands.end(), v));

            vector<int> cands_new, excl_new;
            cands_new.reserve(cands.size());
            excl_new.reserve(excl.size());
            for (int u : cands) if (bitmap_->contains(v, u)) cands_new.push_back(u);
            for (int u : excl)  if (bitmap_->contains(v, u)) excl_new.push_back(u);

            vector<pair<int,float>> undo;
            undo.reserve(cands_new.size() + excl_new.size());
            for (int w : cands_new) { float s=lookup_sim(w,v); acc.add(w,s); undo.push_back({w,s}); }
            for (int w : excl_new)  { float s=lookup_sim(w,v); acc.add(w,s); undo.push_back({w,s}); }

            C.push_back(v);
            bk_search(C, cands_new, excl_new, acc, local_cands, timed_out, phase2_ns_local);
            C.pop_back();

            for (auto& [w, s] : undo) acc.sub(w, s);
            excl.push_back(v);
        }
    }

    void build_graph() {
        int n = g_.V;
        logger.print("Stage1: computing and caching all edge similarities...\n");
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

        logger.print("  [Stage1 build_graph]: %lld ms\n",
            (long long)chrono::duration_cast<chrono::milliseconds>(
                chrono::high_resolution_clock::now()-t0).count());
    }

    void build_degeneracy_ordering() {
        logger.print("Stage1b: degeneracy ordering...\n");
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
        logger.print("  [Stage1b degeneracy]: %lld ms\n",
            (long long)chrono::duration_cast<chrono::milliseconds>(
                chrono::high_resolution_clock::now()-t0).count());
    }

    void run_phase1_2(long long& phase1_ms, long long& phase2_ms) {
        logger.print("Stage2+3: BK + intra-clique subset enumeration...\n");
        logger.print("  tau=%.3f  threads=%d\n", (double)tau_, omp_get_max_threads());

        int n = g_.V;
        int nthreads = omp_get_max_threads();
        atomic<bool> timed_out(false);
        atomic<int> progress_i(0);
        auto now = chrono::high_resolution_clock::now();
        auto last_print = now;
        mutex print_mutex;

        vector<int> order_pos(n, -1);
        for (int i = 0; i < (int)ordering_.size(); i++) order_pos[ordering_[i]] = i;

        int total = 0;
        for (int i = 0; i < n; i++) {
            int v = ordering_[i];
            for (auto& [u, s] : nb_[v])
                if (order_pos[u] > i) total++;
        }

        vector<vector<vector<int>>> per_thread_cands(nthreads);
        vector<long long> per_thread_phase1_ns(nthreads, 0);
        vector<long long> per_thread_phase2_ns(nthreads, 0);

        auto t_all = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic, 50)
        for (int i = 0; i < n; i++) {
            int tid = omp_get_thread_num();
            long long phase2_ns_local = 0;
            long long phase1_ns_local = 0;

            int v = ordering_[i];
            vector<int> cands, excl;
            cands.reserve(nb_[v].size());
            excl.reserve(nb_[v].size());
            for (auto& [u, s] : nb_[v]) {
                if (order_pos[u] > i)      cands.push_back(u);
                else if (order_pos[u] < i) excl.push_back(u);
            }
            AccStore acc;
            acc.init(n);
            for (auto& [u, s] : nb_[v]) acc.add(u, s);

            vector<int> C = {v};
            long long phase2_before = phase2_ns_local;
            auto t_bk = chrono::high_resolution_clock::now();
            bk_search(C, cands, excl, acc, per_thread_cands[tid], timed_out, phase2_ns_local);
            long long root_total_ns = chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t_bk).count();
            long long root_phase2_ns = phase2_ns_local - phase2_before;
            long long root_phase1_ns = root_total_ns - root_phase2_ns;
            if (root_phase1_ns > 0) phase1_ns_local += root_phase1_ns;

            int cur_i = progress_i.fetch_add(1, memory_order_relaxed) + 1;
            if (chrono::duration_cast<chrono::seconds>(now - last_print).count() >= 5) {
                lock_guard<mutex> lk(print_mutex);
                if (chrono::duration_cast<chrono::seconds>(
                        chrono::high_resolution_clock::now() - last_print).count() >= 5) {
                    logger.print("  progress %d/%d (%.1f%%)  candidates=%zu  elapsed=%llds\n",
                           cur_i, total, 100.0*cur_i/total,
                           candidates_.size(),
                           (long long)chrono::duration_cast<chrono::seconds>(
                               now - t_all).count());
                    last_print = now;
                }
            }

            per_thread_phase1_ns[tid] = phase1_ns_local;
            per_thread_phase2_ns[tid] = phase2_ns_local;
        }

        if (timed_out.load())
            logger.print("  [TIMEOUT] reached 6000 seconds\n");

        long long phase1_thread_ns = 0;
        long long phase2_thread_ns = 0;
        for (int t = 0; t < nthreads; t++) {
            phase1_thread_ns += per_thread_phase1_ns[t];
            phase2_thread_ns += per_thread_phase2_ns[t];
        }
        phase1_ms = phase1_thread_ns / 1000000;
        phase2_ms = phase2_thread_ns / 1000000;
        long long phase12_thread_ns = phase1_thread_ns + phase2_thread_ns;
        phase1_ratio_ = phase12_thread_ns > 0 ? 100.0 * phase1_thread_ns / phase12_thread_ns : 0.0;
        phase2_ratio_ = phase12_thread_ns > 0 ? 100.0 * phase2_thread_ns / phase12_thread_ns : 0.0;

        stats_.phase1_ratio = phase1_ratio_;
        stats_.phase2_ratio = phase2_ratio_;

        logger.print("  merging thread-local candidates...\n");
        size_t total_cands = 0;
        for (auto& tc : per_thread_cands) total_cands += tc.size();
        candidates_.reserve(total_cands);
        for (auto& tc : per_thread_cands)
            for (auto& c : tc) candidates_.push_back(move(c));

        long long wall_all = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - t_all).count();

        logger.print("  [Stage2+3 total]: %lld ms  candidates=%zu\n",
                     wall_all, candidates_.size());
        logger.print("    BK ratio in Phase1+2: %.2f%%\n", phase1_ratio_);
        logger.print("    Subset ratio in Phase1+2: %.2f%%\n", phase2_ratio_);
    }

    void run_phase3(long long& phase3_ms) {
        logger.print("Stage4: global dedup (subset pruning)...\n");
        auto t0 = chrono::high_resolution_clock::now();

        sort(candidates_.begin(), candidates_.end());
        candidates_.erase(
            unique(candidates_.begin(), candidates_.end()),
            candidates_.end());
        size_t after_exact = candidates_.size();
        logger.print("  after exact dedup: %zu\n", after_exact);

        sort(candidates_.begin(), candidates_.end(),
             [](const vector<int>& a, const vector<int>& b){
                 return a.size() > b.size(); });

        unordered_map<int, vector<size_t>> inverted;
        vector<bool> valid(candidates_.size(), true);
        size_t pruned = 0;

        for (size_t ci = 0; ci < candidates_.size(); ci++) {
            const vector<int>& C = candidates_[ci];
            int csz = (int)C.size();

            int    rarest = C[0];
            size_t min_len = inverted.count(C[0]) ? inverted[C[0]].size() : 0;
            for (int v : C) {
                auto it = inverted.find(v);
                size_t len = it != inverted.end() ? it->second.size() : 0;
                if (len < min_len) { min_len = len; rarest = v; }
            }

            auto it = inverted.find(rarest);
            if (it != inverted.end()) {
                for (size_t ti : it->second) {
                    const vector<int>& T = candidates_[ti];
                    if ((int)T.size() <= csz) continue;
                    bool subset = true;
                    size_t ci2 = 0, ti2 = 0;
                    while (ci2 < C.size() && ti2 < T.size()) {
                        if      (C[ci2] == T[ti2]) { ci2++; ti2++; }
                        else if (C[ci2] >  T[ti2]) { ti2++; }
                        else { subset = false; break; }
                    }
                    if (ci2 < C.size()) subset = false;
                    if (subset) { valid[ci] = false; pruned++; break; }
                }
            }
            if (valid[ci])
                for (int v : C) inverted[v].push_back(ci);
        }

        stats_.localmax_prunes.store(pruned, memory_order_relaxed);

        phase3_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        logger.print("  [Stage4 dedup]: %lld ms  pruned=%zu  final=%zu\n",
                     phase3_ms, pruned, candidates_.size() - pruned);

        for (size_t ci = 0; ci < candidates_.size(); ci++)
            if (valid[ci]) emit(candidates_[ci]);
    }

public:
    AVGMiner(const VectorDB& db, const Graph& g)
        : db_(db), g_(g), tau_(0.f) {
        for (auto& h : size_hist_) h.store(0);
    }
    ~AVGMiner() { delete bitmap_; if (sink_) fclose(sink_); }

    void set_sink(const char* fn) {
        if (sink_) { fclose(sink_); sink_ = nullptr; }
        if (fn && fn[0]) sink_ = fopen(fn, "w");
    }

    long long mine(double tau) {
        tau_ = (float)tau;
        clique_count_.store(0);
        for (auto& h : size_hist_) h.store(0);
        candidates_.clear();
        delete bitmap_; bitmap_ = nullptr;
        nb_.clear(); sim_map_.clear(); ordering_.clear();

        stats_.reset();

        logger.print("\n=== ALG5 MCE (stats)  tau=%.3f ===\n", tau);
        auto t0 = chrono::high_resolution_clock::now();

        auto t_graph = chrono::high_resolution_clock::now();
        build_graph();
        long long graph_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t_graph).count();

        auto t_degen = chrono::high_resolution_clock::now();
        build_degeneracy_ordering();
        long long degen_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t_degen).count();

        long long phase1_ms = 0, phase2_ms = 0, phase3_ms = 0;
        run_phase1_2(phase1_ms, phase2_ms);
        run_phase3(phase3_ms);

        long long total_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        long long final_count = clique_count_.load();

        stats_.final_count = final_count;

        logger.print("\n=== TOTAL: %lld ms,  %lld maximal cliques ===\n",
                     total_ms, final_count);

        stats_.print(tau_, final_count, graph_ms, degen_ms,
                     phase1_ms+phase2_ms, phase3_ms, total_ms);

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
        logger.print("--- Total: %lld ---\n", total);

        if (sink_) fflush(sink_);
        return final_count;
    }
};

int main(int argc, char* argv[]) {
    omp_set_num_threads(omp_get_max_threads());
    cout << "============================================" << endl;
    cout << "  ALG5 Standard (StrSub) - Stats Version" << endl;
    cout << "  mine <tau> | sink <file> | log <file> | dataset <id> | quit" << endl;
    cout << "============================================" << endl;

    string graph_file, vec_index, log_path;
    int dataset_id = 2;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "dataset" && i+1 < argc) dataset_id = atoi(argv[++i]);
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

    VectorDB db;
    Graph g(graph_file);
    if (g.V == 0) { fprintf(stderr, "Failed to load graph\n"); return 1; }
    if (fs::exists(vec_index)) {
        bool ok = (vec_index.find(".index") != string::npos)
                  ? g.loadVectorsFromChunks(vec_index) : g.loadVectorsFromBinary(vec_index);
        if (!ok) { fprintf(stderr, "Failed to load vectors: %s\n", vec_index.c_str()); return 1; }
    } else { fprintf(stderr, "Vector file not found: %s\n", vec_index.c_str()); return 1; }
    g.normalizeVectors();
    db.build(g);

    AVGMiner* miner = new AVGMiner(db, g);
    logger.print("Dataset loaded: ID=%d, V=%d\n", dataset_id, g.V);
    string line;
    while (true) {
        cout << ">> ";
        fflush(stdout);
        getline(cin, line);
        istringstream iss(line);
        string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "q") break;
        else if (cmd == "mine") {
            double tau;
            if (!(iss >> tau)) { cerr << "Usage: mine <tau>" << endl; continue; }
            logger.open("");
            auto t0 = chrono::high_resolution_clock::now();
            long long count = miner->mine(tau);
            long long ms = chrono::duration_cast<chrono::milliseconds>(
                chrono::high_resolution_clock::now()-t0).count();
            printf("time_avg %.3f  time_min %.3f  time_max %.3f  count %lld\n",
                   ms/1000.0, ms/1000.0, ms/1000.0, count);
        }
        else if (cmd == "sink") {
            string fn;
            iss >> fn;
            miner->set_sink(fn.c_str());
            cout << "Sink set to: " << fn << endl;
        }
        else if (cmd == "log") {
            string fn;
            iss >> fn;
            logger.open(fn);
            cout << "Log file: " << fn << endl;
        }
        else if (cmd == "dataset") {
            string ds_str;
            int new_ds_id;
            iss >> ds_str;
            new_ds_id = atoi(ds_str.c_str());
            switch (new_ds_id) {
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
            g = Graph(graph_file);
            if (g.V == 0) { fprintf(stderr, "Failed to load graph\n"); continue; }
            if (fs::exists(vec_index)) {
                bool ok = (vec_index.find(".index") != string::npos)
                          ? g.loadVectorsFromChunks(vec_index) : g.loadVectorsFromBinary(vec_index);
                if (!ok) { fprintf(stderr, "Failed to load vectors\n"); continue; }
            } else { fprintf(stderr, "Vector file not found\n"); continue; }
            g.normalizeVectors();
            db.build(g);
            delete miner;
            miner = new AVGMiner(db, g);
            dataset_id = new_ds_id;
            cout << "Dataset loaded: ID=" << new_ds_id << ", V=" << g.V << endl;
        }
        else {
            cout << "Unknown command: " << cmd << endl;
        }
    }
    delete miner;
    cout << "Bye." << endl;
    return 0;
}
