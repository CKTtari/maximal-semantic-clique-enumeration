// ============================================================
// ALG3 standard: structural maximal cliques + exact semantic subsets.
//
// Phase-2 improvements:
//   - native uint64_t subset enumeration for structural cliques up to size 63;
//   - the existing 128-bit path for sizes 64..127;
//   - a dynamic-bitset fallback above 127 (no mask-width size limit);
//   - a conservative vertex-aware upper bound for large size layers.
// ============================================================

#include "experiment_runtime.h"

#ifdef _WIN32
    #include <malloc.h>
    #define ALIGNED_ALLOC(size, align) _aligned_malloc(size, align)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #define ALIGNED_ALLOC(size, align) aligned_alloc(align, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

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
#include <limits>

#ifndef ALG3_VERTEX_BOUND_MIN_M
#define ALG3_VERTEX_BOUND_MIN_M 12
#endif

#ifndef ACMSC_ENABLE_STATS
#define ACMSC_ENABLE_STATS 0
#endif
#ifndef ACMSC_STRSUB_TOP_EDGE
#define ACMSC_STRSUB_TOP_EDGE 1
#endif
#ifndef ACMSC_STRSUB_VERTEX_BOUND
#define ACMSC_STRSUB_VERTEX_BOUND 1
#endif
#ifndef ACMSC_STRSUB_LOCAL_MAX
#define ACMSC_STRSUB_LOCAL_MAX 1
#endif
#ifndef ACMSC_ENGINE_REORDER
#define ACMSC_ENGINE_REORDER 1
#endif
#ifndef ACMSC_ENGINE_BITMAP
#define ACMSC_ENGINE_BITMAP 1
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

// ============================================================
// Float32VectorStore
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
#endif
        for (; i < d_; i++) p += xu[i] * xv[i];
        return p;
    }
};

// ============================================================
// GraphReorder: 按度排序后 BFS 展开的重排序（非 degeneracy order）
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
// AVGMiner（标准版，无统计计数器）
// ============================================================
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
    int                  timeout_seconds_ = acmsc_runtime::kDefaultTimeoutSeconds;
    atomic<uint64_t>      vertex_bound_skipped_layers_{0};
#if ACMSC_ENABLE_STATS
    atomic<uint64_t>      structural_cliques_{0};
    atomic<uint64_t>      subset_tests_{0};
    atomic<uint64_t>      covered_subsets_{0};
    atomic<uint64_t>      feasible_local_{0};
    uint64_t              exact_duplicates_ = 0;
    uint64_t              global_subsets_pruned_ = 0;
#endif

    atomic<long long>    clique_count_{0};
    vector<long long>    size_hist_;
    FILE*                sink_ = nullptr;
    mutex                sink_mutex_;

    static inline long double add_upper(long double a, long double b) {
        return nextafter(a + b, numeric_limits<long double>::infinity());
    }

    static inline long double divide_upper(long double value,
                                           long long divisor) {
        return nextafter(value / static_cast<long double>(divisor),
                         numeric_limits<long double>::infinity());
    }

    static bool combination_count_at_least(int n, int k,
                                           long double threshold) {
        if (k < 0 || k > n) return false;
        k = min(k, n - k);
        long double count = 1.0L;
        for (int i = 1; i <= k; ++i) {
            count *= static_cast<long double>(n - k + i) / i;
            if (count >= threshold) return true;
        }
        return count >= threshold;
    }

    // --------------------------------------------------------
    inline void emit(const vector<int>& C) {
        clique_count_.fetch_add(1, memory_order_relaxed);
        int sz = (int)C.size();
        if (sz >= 1) {
            if (size_hist_.size() <= static_cast<size_t>(sz))
                size_hist_.resize(static_cast<size_t>(sz) + 1, 0);
            ++size_hist_[sz];
        }
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

    // ============================================================
    // Mask128：用两个 uint64_t 拼成 128-bit 掩码
    //   lo = bit 0..63，hi = bit 64..127
    //
    // Used only by the middle-size fallback; larger cliques use dynamic masks.
    // process_structural_clique 中所有 uint64_t 掩码操作均逐词扩展，
    // 算法语义与原版完全一致。
    // ============================================================
    struct Mask128 {
        uint64_t lo, hi;

        Mask128() : lo(0), hi(0) {}
        Mask128(uint64_t l, uint64_t h) : lo(l), hi(h) {}

        // 第 i 位的单位掩码（0-indexed，0 <= i <= 127）
        static Mask128 bit(int i) {
            if (i < 64) return Mask128(1ULL << i, 0);
            else        return Mask128(0, 1ULL << (i - 64));
        }

        // 低 k 位全 1，等价于原来的 (1ULL<<k)-1，支持 k = 1..127
        static Mask128 low_k(int k) {
            if (k <= 0)  return Mask128(0, 0);
            if (k < 64)  return Mask128((1ULL << k) - 1, 0);
            if (k == 64) return Mask128(~0ULL, 0);
            if (k < 128) return Mask128(~0ULL, (1ULL << (k - 64)) - 1);
            return Mask128(~0ULL, ~0ULL);
        }

        // 等价于原来的 1ULL << m（end_mask），支持 m = 0..127
        static Mask128 end(int m) {
            if (m < 64) return Mask128(1ULL << m, 0);
            else        return Mask128(0, 1ULL << (m - 64));
        }

        bool operator==(const Mask128& o) const { return lo == o.lo && hi == o.hi; }
        bool operator!=(const Mask128& o) const { return !(*this == o); }
        // 无符号字典序比较（hi 优先），用于循环终止条件 mask < end_mask
        bool operator<(const Mask128& o) const {
            return hi != o.hi ? hi < o.hi : lo < o.lo;
        }

        Mask128 operator&(const Mask128& o) const { return {lo & o.lo, hi & o.hi}; }
        Mask128 operator|(const Mask128& o) const { return {lo | o.lo, hi | o.hi}; }
        Mask128 operator^(const Mask128& o) const { return {lo ^ o.lo, hi ^ o.hi}; }
        Mask128 operator~() const { return {~lo, ~hi}; }

        bool zero() const { return lo == 0 && hi == 0; }

        // 最低置位的位下标，等价于 __builtin_ctzll 扩展到 128-bit
        int ctz() const {
            if (lo != 0) return __builtin_ctzll(lo);
            return 64 + __builtin_ctzll(hi);
        }

        // 清除最低置位，等价于 x & (x-1) 扩展到 128-bit
        Mask128 clear_lowest() const {
            if (lo != 0) return {lo & (lo - 1), hi};
            return {0, hi & (hi - 1)};
        }

        // 128-bit 无符号加法（含进位）
        Mask128 add128(const Mask128& o) const {
            uint64_t new_lo = lo + o.lo;
            uint64_t carry  = (new_lo < lo) ? 1ULL : 0ULL;
            uint64_t new_hi = hi + o.hi + carry;
            return {new_lo, new_hi};
        }

        // 128-bit 取负：-x = ~x + 1
        Mask128 neg() const {
            return (~*this).add128(Mask128(1, 0));
        }

        // 逻辑右移 n 位（0 <= n < 128）
        Mask128 shr(int n) const {
            if (n == 0)   return *this;
            if (n >= 128) return Mask128(0, 0);
            if (n >= 64)  return Mask128(hi >> (n - 64), 0);
            return Mask128((lo >> n) | (hi << (64 - n)), hi >> n);
        }

        // 除以 2 的幂 c（c 必须是 2 的幂），等价于右移 ctz(c) 位
        Mask128 div_pow2(const Mask128& c) const {
            return shr(c.ctz());
        }

        // Gosper's hack：给定有 k 个置位的掩码，返回下一个同样有 k 个置位的掩码
        //   原 64-bit: c=x&-x; r=x+c; return (((r^x)>>2)/c)|r;
        //   128-bit 版逐步扩展，语义完全一致
        Mask128 gosper_next() const {
            Mask128 c   = *this & neg();      // c = x & (-x)，最低置位
            Mask128 r   = add128(c);          // r = x + c
            Mask128 xr  = r ^ *this;          // r ^ x
            Mask128 xr2 = xr.shr(2);         // >> 2
            Mask128 div = xr2.div_pow2(c);    // / c（c 是 2 的幂）
            return div | r;                   // | r
        }
    };
    // ============================================================

    // --------------------------------------------------------
    // Phase 2：在结构极大团内枚举语义合法子集
    //
    // Every representation enumerates the same subsets in descending size.
    // --------------------------------------------------------
    void process_structural_clique(
        const vector<int>& M,
        vector<vector<int>>& local_cands,
        long long& phase2_ns_local
    ) {
#if ACMSC_ENABLE_STATS
        structural_cliques_.fetch_add(1, memory_order_relaxed);
#endif
#if ACMSC_ENABLE_STATS
        auto t0 = chrono::high_resolution_clock::now();
#endif
        int m = (int)M.size();
        if (m < 2) {
#if ACMSC_ENABLE_STATS
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
#endif
            return;
        }
        vector<int> sorted_M = M;
        sort(sorted_M.begin(), sorted_M.end());

        vector<vector<float>> sim_local(m, vector<float>(m, 0.0f));
        vector<float> all_edges;
        double full_total = 0.0;
        for (int i = 0; i < m; i++)
            for (int j = i+1; j < m; j++) {
                float s = lookup_sim(sorted_M[i], sorted_M[j]);
                sim_local[i][j] = s;
                sim_local[j][i] = s;
                all_edges.push_back(s);
                full_total += static_cast<double>(s);
            }

        const long long full_pairs =
            static_cast<long long>(m) * (m - 1) / 2;
        if (full_pairs > 0 &&
            full_total >= static_cast<double>(tau_) * full_pairs) {
#if ACMSC_ENABLE_STATS
            feasible_local_.fetch_add(1, memory_order_relaxed);
#endif
            vector<int> subset = M;
            sort(subset.begin(), subset.end());
            local_cands.push_back(move(subset));
#if ACMSC_ENABLE_STATS
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
#endif
            return;
        }

        sort(all_edges.rbegin(), all_edges.rend());

        // Top1 边都不满足 tau，任何子团都不可能满足，直接放弃
#if ACMSC_STRSUB_TOP_EDGE
        if (all_edges[0] < tau_) {
#if ACMSC_ENABLE_STATS
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
#endif
            return;
        }
#endif

        int start_k = 2;
        vector<long double> prefix_sum(all_edges.size() + 1, 0.0L);
        for (int i = 0; i < (int)all_edges.size(); i++) {
            prefix_sum[i+1] =
                add_upper(prefix_sum[i], static_cast<long double>(all_edges[i]));
        }

        vector<char> layer_possible(m + 1, 0);
#if ACMSC_STRSUB_TOP_EDGE
        for (int k_candidate = m-1; k_candidate >= 2; k_candidate--) {
            const long long required_pairs =
                static_cast<long long>(k_candidate) * (k_candidate - 1) / 2;
            if (required_pairs > static_cast<long long>(all_edges.size())) continue;
            const long double top_avg =
                divide_upper(prefix_sum[static_cast<size_t>(required_pairs)],
                             required_pairs);
            if (top_avg >= static_cast<long double>(tau_))
                layer_possible[k_candidate] = 1;
        }
#else
        fill(layer_possible.begin() + 2, layer_possible.end(), 1);
#endif

        // For every k-subset S and v in S, the incident sum of v inside S
        // is at most the sum of v's top (k-1) incident similarities in M.
        // Summing the k largest such per-vertex bounds and dividing by
        // k(k-1) is therefore an admissible upper bound on avgSim(S).
#if ACMSC_STRSUB_VERTEX_BOUND
        if (m >= ALG3_VERTEX_BOUND_MIN_M) {
            vector<vector<long double>> incident_prefix(
                m, vector<long double>(m, 0.0L));
            for (int v = 0; v < m; ++v) {
                vector<float> incident;
                incident.reserve(m - 1);
                for (int u = 0; u < m; ++u)
                    if (u != v) incident.push_back(sim_local[v][u]);
                sort(incident.rbegin(), incident.rend());
                for (int r = 1; r < m; ++r)
                    incident_prefix[v][r] = add_upper(
                        incident_prefix[v][r - 1],
                        static_cast<long double>(incident[r - 1]));
            }

            vector<long double> vertex_caps(m);
            for (int k = 2; k < m; ++k) {
                if (!layer_possible[k]) continue;
                for (int v = 0; v < m; ++v)
                    vertex_caps[v] = incident_prefix[v][k - 1];
                nth_element(vertex_caps.begin(), vertex_caps.begin() + k,
                            vertex_caps.end(), greater<long double>());
                long double cap_sum = 0.0L;
                for (int i = 0; i < k; ++i)
                    cap_sum = add_upper(cap_sum, vertex_caps[i]);
                const long double vertex_avg_bound =
                    divide_upper(cap_sum,
                                 static_cast<long long>(k) * (k - 1));
                if (vertex_avg_bound < static_cast<long double>(tau_)) {
                    layer_possible[k] = 0;
#if ACMSC_ENABLE_STATS
                    vertex_bound_skipped_layers_.fetch_add(
                        1, memory_order_relaxed);
#endif
                }
            }
        }
#endif

        for (int k = m - 1; k >= 2; --k) {
            if (layer_possible[k]) {
                start_k = k;
                break;
            }
        }

        if (m <= 63) {
            vector<uint64_t> local_max;
            uint64_t union_mask = 0;

            struct LayerResult {
                vector<uint64_t> feasible_masks;
                uint64_t tested = 0;
                uint64_t covered = 0;
            };

            for (int k = start_k; k >= 2; --k) {
                if (!layer_possible[k]) continue;

#if !ACMSC_ENABLE_STATS
                const bool parallel_layer =
                    omp_in_parallel() && omp_get_num_threads() > 1 &&
                    combination_count_at_least(m, k, 250000.0L);
#else
                const bool parallel_layer = false;
#endif
                if (parallel_layer) {
                    const int prefix_count = m - k + 1;
                    vector<LayerResult> layer_results(prefix_count);

                    // Same-size subsets cannot strictly contain one another.
                    // Tasks read the frozen larger layers and merge before k-1.
                    #pragma omp taskgroup
                    {
                        for (int first = 0; first < prefix_count; ++first) {
                            #pragma omp task firstprivate(first, k) \
                                shared(layer_results, local_max, union_mask, sim_local)
                            {
                                LayerResult& result = layer_results[first];
                                const int remaining = k - 1;
                                vector<int> combination(remaining);
                                for (int i = 0; i < remaining; ++i)
                                    combination[i] = first + 1 + i;

                                while (true) {
                                    uint64_t candidate_mask = 1ULL << first;
                                    for (int index : combination)
                                        candidate_mask |= 1ULL << index;

                                    bool covered = false;
#if ACMSC_STRSUB_LOCAL_MAX
                                    if ((candidate_mask & union_mask) == candidate_mask) {
                                        for (uint64_t larger : local_max) {
                                            if ((candidate_mask & larger) == candidate_mask) {
                                                covered = true;
                                                break;
                                            }
                                        }
                                    }
#endif

                                    if (!covered) {
                                        ++result.tested;
                                        double total = 0.0;
                                        uint64_t bits_i = candidate_mask;
                                        while (bits_i != 0) {
                                            const int i = __builtin_ctzll(bits_i);
                                            bits_i &= bits_i - 1;
                                            uint64_t bits_j = bits_i;
                                            while (bits_j != 0) {
                                                const int j = __builtin_ctzll(bits_j);
                                                bits_j &= bits_j - 1;
                                                total += static_cast<double>(sim_local[i][j]);
                                            }
                                        }
                                        const int cnt = k * (k - 1) / 2;
                                        if (total >= static_cast<double>(tau_) * cnt)
                                            result.feasible_masks.push_back(candidate_mask);
                                    } else {
                                        ++result.covered;
                                    }

                                    int position = remaining - 1;
                                    while (position >= 0 &&
                                           combination[position] ==
                                               m - remaining + position)
                                        --position;
                                    if (position < 0) break;
                                    ++combination[position];
                                    for (int i = position + 1; i < remaining; ++i)
                                        combination[i] = combination[i - 1] + 1;
                                }
                            }
                        }
                    }

                    for (LayerResult& result : layer_results) {
#if ACMSC_ENABLE_STATS
                        subset_tests_.fetch_add(result.tested, memory_order_relaxed);
                        covered_subsets_.fetch_add(result.covered, memory_order_relaxed);
#endif
                        for (uint64_t feasible_mask : result.feasible_masks) {
#if ACMSC_ENABLE_STATS
                            feasible_local_.fetch_add(1, memory_order_relaxed);
#endif
#if ACMSC_STRSUB_LOCAL_MAX
                            local_max.push_back(feasible_mask);
                            union_mask |= feasible_mask;
#endif
                            vector<int> subset;
                            subset.reserve(k);
                            uint64_t bits = feasible_mask;
                            while (bits != 0) {
                                const int i = __builtin_ctzll(bits);
                                bits &= bits - 1;
                                subset.push_back(sorted_M[i]);
                            }
                            local_cands.push_back(move(subset));
                        }
                    }
                    continue;
                }

                uint64_t mask = (1ULL << k) - 1;
                const uint64_t end_mask = 1ULL << m;

                while (mask < end_mask) {
                    bool covered = false;
#if ACMSC_STRSUB_LOCAL_MAX
                    if ((mask & union_mask) == mask) {
                        for (uint64_t larger : local_max) {
                            if ((mask & larger) == mask) {
                                covered = true;
                                break;
                            }
                        }
                    }
#endif

                    if (!covered) {
#if ACMSC_ENABLE_STATS
                        subset_tests_.fetch_add(1, memory_order_relaxed);
#endif
                        double total = 0.0;
                        uint64_t bits_i = mask;
                        while (bits_i != 0) {
                            const int i = __builtin_ctzll(bits_i);
                            bits_i &= bits_i - 1;
                            uint64_t bits_j = bits_i;
                            while (bits_j != 0) {
                                const int j = __builtin_ctzll(bits_j);
                                bits_j &= bits_j - 1;
                                total += static_cast<double>(sim_local[i][j]);
                            }
                        }

                        const int cnt = k * (k - 1) / 2;
                        if (total >= static_cast<double>(tau_) * cnt) {
#if ACMSC_ENABLE_STATS
                            feasible_local_.fetch_add(1, memory_order_relaxed);
#endif
#if ACMSC_STRSUB_LOCAL_MAX
                            local_max.push_back(mask);
                            union_mask |= mask;
#endif
                            vector<int> subset;
                            subset.reserve(k);
                            uint64_t bits = mask;
                            while (bits != 0) {
                                const int i = __builtin_ctzll(bits);
                                bits &= bits - 1;
                                subset.push_back(sorted_M[i]);
                            }
                            local_cands.push_back(move(subset));
                        }
#if ACMSC_ENABLE_STATS
                    } else {
                        covered_subsets_.fetch_add(1, memory_order_relaxed);
#endif
                    }

                    const uint64_t c = mask & (~mask + 1);
                    const uint64_t r = mask + c;
                    mask = (((r ^ mask) >> 2) / c) | r;
                }
            }
#if ACMSC_ENABLE_STATS
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
#endif
            return;
        }

        if (m <= 127) {
            vector<Mask128> local_max;
            Mask128 union_mask;

            for (int k = start_k; k >= 2; k--) {
                if (!layer_possible[k]) continue;
                Mask128 mask = Mask128::low_k(k);
                const Mask128 end_mask = Mask128::end(m);

                while (mask < end_mask) {
                    bool covered = false;
#if ACMSC_STRSUB_LOCAL_MAX
                    if (((mask & union_mask) ^ mask).zero()) {
                        for (const Mask128& larger : local_max) {
                            if (((mask & larger) ^ mask).zero()) {
                                covered = true;
                                break;
                            }
                        }
                    }
#endif

                    if (!covered) {
#if ACMSC_ENABLE_STATS
                        subset_tests_.fetch_add(1, memory_order_relaxed);
#endif
                        double total = 0.0;
                        Mask128 bits_i = mask;
                        while (!bits_i.zero()) {
                            const int i = bits_i.ctz();
                            bits_i = bits_i.clear_lowest();
                            Mask128 bits_j = bits_i;
                            while (!bits_j.zero()) {
                                const int j = bits_j.ctz();
                                bits_j = bits_j.clear_lowest();
                                total += static_cast<double>(sim_local[i][j]);
                            }
                        }
                        const int cnt = k * (k - 1) / 2;
                        if (total >= static_cast<double>(tau_) * cnt) {
#if ACMSC_ENABLE_STATS
                            feasible_local_.fetch_add(1, memory_order_relaxed);
#endif
#if ACMSC_STRSUB_LOCAL_MAX
                            local_max.push_back(mask);
                            union_mask = union_mask | mask;
#endif
                            vector<int> subset;
                            subset.reserve(k);
                            for (int i = 0; i < m; ++i)
                                if (!(mask & Mask128::bit(i)).zero())
                                    subset.push_back(sorted_M[i]);
                            local_cands.push_back(move(subset));
                        }
#if ACMSC_ENABLE_STATS
                    } else {
                        covered_subsets_.fetch_add(1, memory_order_relaxed);
#endif
                    }
                    mask = mask.gosper_next();
                }
            }
#if ACMSC_ENABLE_STATS
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
#endif
            return;
        }

        // Dynamic fallback: no clique-size limit is imposed by the mask type.
        const int word_count = (m + 63) / 64;
        vector<vector<uint64_t>> local_max;
        vector<uint64_t> union_mask(word_count, 0);
        vector<uint64_t> mask(word_count, 0);

        for (int k = start_k; k >= 2; --k) {
            if (!layer_possible[k]) continue;
            vector<int> combination(k);
            iota(combination.begin(), combination.end(), 0);

            while (true) {
                fill(mask.begin(), mask.end(), 0);
                for (int index : combination)
                    mask[index >> 6] |= 1ULL << (index & 63);

                bool may_be_covered = true;
#if ACMSC_STRSUB_LOCAL_MAX
                for (int w = 0; w < word_count; ++w) {
                    if ((mask[w] & union_mask[w]) != mask[w]) {
                        may_be_covered = false;
                        break;
                    }
                }
#else
                may_be_covered = false;
#endif
                bool covered = false;
#if ACMSC_STRSUB_LOCAL_MAX
                if (may_be_covered) {
                    for (const auto& larger : local_max) {
                        bool subset = true;
                        for (int w = 0; w < word_count; ++w) {
                            if ((mask[w] & larger[w]) != mask[w]) {
                                subset = false;
                                break;
                            }
                        }
                        if (subset) {
                            covered = true;
                            break;
                        }
                    }
                }
#endif

                if (!covered) {
#if ACMSC_ENABLE_STATS
                    subset_tests_.fetch_add(1, memory_order_relaxed);
#endif
                    double total = 0.0;
                    for (int i = 0; i < k; ++i)
                        for (int j = i + 1; j < k; ++j)
                            total += static_cast<double>(
                                sim_local[combination[i]][combination[j]]);
                    const long long cnt =
                        static_cast<long long>(k) * (k - 1) / 2;
                    if (total >= static_cast<double>(tau_) * cnt) {
#if ACMSC_ENABLE_STATS
                        feasible_local_.fetch_add(1, memory_order_relaxed);
#endif
#if ACMSC_STRSUB_LOCAL_MAX
                        local_max.push_back(mask);
                        for (int w = 0; w < word_count; ++w)
                            union_mask[w] |= mask[w];
#endif
                        vector<int> subset;
                        subset.reserve(k);
                        for (int index : combination)
                            subset.push_back(sorted_M[index]);
                        local_cands.push_back(move(subset));
                    }
#if ACMSC_ENABLE_STATS
                } else {
                    covered_subsets_.fetch_add(1, memory_order_relaxed);
#endif
                }

                int position = k - 1;
                while (position >= 0 &&
                       combination[position] == m - k + position)
                    --position;
                if (position < 0) break;
                ++combination[position];
                for (int j = position + 1; j < k; ++j)
                    combination[j] = combination[j - 1] + 1;
            }
        }
#if ACMSC_ENABLE_STATS
        phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
            chrono::high_resolution_clock::now() - t0).count();
#endif
    }

    // --------------------------------------------------------
    // Phase 1：BK with Tomita pivot
    // --------------------------------------------------------
    struct AccStore {
        vector<float>    val;
        vector<uint32_t> gen;
        uint32_t         cur = 0;
        void init(int n) { val.assign(n, 0.f); gen.assign(n, 0); cur = 1; }
        inline float get(int w) const { return gen[w]==cur ? val[w] : 0.f; }
        inline void add(int w, float s) {
            if (gen[w] != cur) { val[w] = 0.f; gen[w] = cur; }
            val[w] += s;
        }
        inline void sub(int w, float s) {
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
        if (timed_out.load()) return;

        if (cands.empty() && excl.empty()) {
            process_structural_clique(C, local_cands, phase2_ns_local);
            return;
        }

        int pivot = -1, pivot_cover = -1;
        for (int p : excl) {
            int cover = 0;
            for (int c : cands) if (adjacent(p, c)) cover++;
            if (cover > pivot_cover) { pivot_cover = cover; pivot = p; }
        }
        for (int p : cands) {
            int cover = 0;
            for (int c : cands) if (c != p && adjacent(p, c)) cover++;
            if (cover > pivot_cover) { pivot_cover = cover; pivot = p; }
        }

        vector<int> to_enum;
        to_enum.reserve(cands.size());
        for (int v : cands)
            if (pivot == -1 || !adjacent(pivot, v))
                to_enum.push_back(v);

        for (int v : to_enum) {
            if (timed_out.load()) return;

            cands.erase(find(cands.begin(), cands.end(), v));

            vector<int> cands_new, excl_new;
            cands_new.reserve(cands.size());
            excl_new.reserve(excl.size());
            for (int u : cands) if (adjacent(v, u)) cands_new.push_back(u);
            for (int u : excl)  if (adjacent(v, u)) excl_new.push_back(u);

            vector<pair<int,float>> undo;
            undo.reserve(cands_new.size() + excl_new.size());
            for (int w : cands_new) { float s=lookup_sim(w,v); acc.add(w,s); undo.push_back({w,s}); }
            for (int w : excl_new)  { float s=lookup_sim(w,v); acc.add(w,s); undo.push_back({w,s}); }

            C.push_back(v);
            bk_search(C, cands_new, excl_new, acc, local_cands,
                      timed_out, phase2_ns_local);
            C.pop_back();

            for (auto& [w, s] : undo) acc.sub(w, s);
            excl.push_back(v);
        }
    }

    // --------------------------------------------------------
    // build_graph
    // --------------------------------------------------------
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

        logger.print("  [Stage1 build_graph]: %lld ms\n",
            (long long)chrono::duration_cast<chrono::milliseconds>(
                chrono::high_resolution_clock::now()-t0).count());
    }

    // --------------------------------------------------------
    // build_degeneracy_ordering
    // --------------------------------------------------------
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

    // --------------------------------------------------------
    // run_phase1_2
    // --------------------------------------------------------
    bool run_phase1_2(long long& phase1_ms, long long& phase2_ms) {
        logger.print("Stage2+3: BK + intra-clique subset enumeration...\n");
        if (timeout_seconds_ > 0)
            logger.print("  tau=%.3f  threads=%d  timeout=%ds\n",
                         tau_, omp_get_max_threads(), timeout_seconds_);
        else
            logger.print("  tau=%.3f  threads=%d  timeout=unlimited\n",
                         tau_, omp_get_max_threads());
        auto t_all = chrono::high_resolution_clock::now();
        int n = g_.V;
        int total = (int)ordering_.size();

        vector<int> order_pos(n, 0);
        for (int i = 0; i < total; i++)
            order_pos[ordering_[i]] = i;

        int nthreads = omp_get_max_threads();
        vector<AccStore>            stores(nthreads);
        vector<vector<vector<int>>> per_thread_cands(nthreads);
        for (auto& s : stores) s.init(n);

        auto last_print = chrono::high_resolution_clock::now();
        atomic<int>  progress_i{0};
        atomic<bool> timed_out{false};
        mutex        print_mutex;

#if ACMSC_ENABLE_STATS
        vector<long long> per_thread_phase1_ns(nthreads, 0);
        vector<long long> per_thread_phase2_ns(nthreads, 0);
#endif

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
#if ACMSC_ENABLE_STATS
            long long phase1_ns_local = 0;
#endif
            long long phase2_ns_local = 0;

            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < total; i++) {
                if (timed_out.load()) continue;

                auto now = chrono::high_resolution_clock::now();
                if (timeout_seconds_ > 0 &&
                    chrono::duration_cast<chrono::seconds>(now - t_all).count() >=
                        timeout_seconds_) {
                    timed_out.store(true); continue;
                }

                int v = ordering_[i];
                AccStore& acc = stores[tid];
                acc.cur++;
                if (acc.cur == 0) { fill(acc.gen.begin(), acc.gen.end(), 0); acc.cur = 1; }

                vector<int> cands, excl;
                cands.reserve(nb_[v].size());
                excl.reserve(nb_[v].size());
                for (auto& [u, s] : nb_[v]) {
                    if (order_pos[u] > i)      cands.push_back(u);
                    else if (order_pos[u] < i) excl.push_back(u);
                }
                for (auto& [u, s] : nb_[v]) acc.add(u, s);

                vector<int> C = {v};
#if ACMSC_ENABLE_STATS
                long long phase2_before = phase2_ns_local;
                auto t_bk = chrono::high_resolution_clock::now();
#endif
                bk_search(C, cands, excl, acc, per_thread_cands[tid],
                          timed_out, phase2_ns_local);
#if ACMSC_ENABLE_STATS
                long long root_total_ns = chrono::duration_cast<chrono::nanoseconds>(
                    chrono::high_resolution_clock::now() - t_bk).count();
                long long root_phase2_ns = phase2_ns_local - phase2_before;
                long long root_phase1_ns = root_total_ns - root_phase2_ns;
                if (root_phase1_ns > 0) phase1_ns_local += root_phase1_ns;
#endif

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
            }

#if ACMSC_ENABLE_STATS
            per_thread_phase1_ns[tid] = phase1_ns_local;
            per_thread_phase2_ns[tid] = phase2_ns_local;
#endif
        }

        if (timed_out.load()) {
            logger.print("  [TIMEOUT] reached %d seconds\n", timeout_seconds_);
            candidates_.clear();
            return false;
        }

#if ACMSC_ENABLE_STATS
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
#else
        phase1_ms = 0;
        phase2_ms = 0;
        phase1_ratio_ = 0.0;
        phase2_ratio_ = 0.0;
#endif

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
#if ACMSC_ENABLE_STATS
        logger.print("    BK ratio in Phase1+2: %.2f%%\n", phase1_ratio_);
        logger.print("    Subset ratio in Phase1+2: %.2f%%\n", phase2_ratio_);
#endif
        return true;
    }

    // --------------------------------------------------------
    // run_phase3
    // --------------------------------------------------------
    void run_phase3(long long& phase3_ms) {
        logger.print("Stage4: global dedup (subset pruning)...\n");
        auto t0 = chrono::high_resolution_clock::now();

        const size_t before_exact = candidates_.size();
        sort(candidates_.begin(), candidates_.end(),
             [](const vector<int>& a, const vector<int>& b) {
                 if (a.size() != b.size()) return a.size() > b.size();
                 return a < b;
             });
        candidates_.erase(
            unique(candidates_.begin(), candidates_.end()),
            candidates_.end());
        size_t after_exact = candidates_.size();
#if ACMSC_ENABLE_STATS
        exact_duplicates_ = before_exact - after_exact;
#endif
        logger.print("  after exact dedup: %zu\n", after_exact);

        vector<vector<size_t>> inverted(g_.V);
        vector<unsigned char> valid(candidates_.size(), 1);
        size_t pruned = 0;

        size_t group_begin = 0;
        while (group_begin < candidates_.size()) {
            size_t group_end = group_begin + 1;
            while (group_end < candidates_.size() &&
                   candidates_[group_end].size() == candidates_[group_begin].size())
                ++group_end;

            long long group_pruned = 0;
            #pragma omp parallel for schedule(dynamic, 256) \
                if(group_end - group_begin >= 4096) reduction(+:group_pruned)
            for (long long pos = static_cast<long long>(group_begin);
                 pos < static_cast<long long>(group_end); ++pos) {
                const size_t ci = static_cast<size_t>(pos);
                const vector<int>& C = candidates_[ci];

                int rarest = C[0];
                size_t min_len = inverted[rarest].size();
                for (int v : C) {
                    const size_t len = inverted[v].size();
                    if (len < min_len) {
                        min_len = len;
                        rarest = v;
                        if (min_len == 0) break;
                    }
                }

                for (size_t ti : inverted[rarest]) {
                    const vector<int>& T = candidates_[ti];
                    bool subset = true;
                    size_t ci2 = 0, ti2 = 0;
                    while (ci2 < C.size() && ti2 < T.size()) {
                        if      (C[ci2] == T[ti2]) { ++ci2; ++ti2; }
                        else if (C[ci2] >  T[ti2]) { ++ti2; }
                        else { subset = false; break; }
                    }
                    if (ci2 < C.size()) subset = false;
                    if (subset) {
                        valid[ci] = 0;
                        ++group_pruned;
                        break;
                    }
                }
            }

            pruned += static_cast<size_t>(group_pruned);
            for (size_t ci = group_begin; ci < group_end; ++ci)
                if (valid[ci])
                    for (int v : candidates_[ci]) inverted[v].push_back(ci);

            group_begin = group_end;
        }

        phase3_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        logger.print("  [Stage4 dedup]: %lld ms  pruned=%zu  final=%zu\n",
                     phase3_ms, pruned, candidates_.size() - pruned);
#if ACMSC_ENABLE_STATS
        global_subsets_pruned_ = pruned;
#endif

        for (size_t ci = 0; ci < candidates_.size(); ci++)
            if (valid[ci]) emit(candidates_[ci]);
    }

public:
    AVGMiner(const VectorDB& db, const Graph& g, int timeout_seconds)
        : db_(db), g_(g), tau_(0.f),
          timeout_seconds_(max(0, timeout_seconds)) {}
    ~AVGMiner() { delete bitmap_; if (sink_) fclose(sink_); }

    void set_sink(const char* fn) {
        if (sink_) { fclose(sink_); sink_ = nullptr; }
        if (fn && fn[0]) sink_ = fopen(fn, "w");
    }

    long long mine(double tau) {
        tau_ = (float)tau;
        vertex_bound_skipped_layers_.store(0, memory_order_relaxed);
#if ACMSC_ENABLE_STATS
        structural_cliques_.store(0);
        subset_tests_.store(0);
        covered_subsets_.store(0);
        feasible_local_.store(0);
        exact_duplicates_ = 0;
        global_subsets_pruned_ = 0;
#endif
        clique_count_.store(0);
        size_hist_.clear();
        candidates_.clear();
        delete bitmap_; bitmap_ = nullptr;
        nb_.clear(); sim_map_.clear(); ordering_.clear();

        logger.print("\n=== ALG3 StrSub  tau=%.3f ===\n", tau);
        auto t0 = chrono::high_resolution_clock::now();

        build_graph();
        build_degeneracy_ordering();

        long long phase1_ms = 0, phase2_ms = 0, phase3_ms = 0;
        if (!run_phase1_2(phase1_ms, phase2_ms)) {
            logger.print("[TIMEOUT] incomplete run; no result count is reported.\n");
            logger.print("Peak process memory: %.2f GiB (limit 16.00 GiB)\n",
                         acmsc_runtime::peak_memory_bytes() / 1073741824.0);
            return -1;
        }
        run_phase3(phase3_ms);

        long long total_ms = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t0).count();
        long long final_count = clique_count_.load();

        logger.print("\n=== TOTAL: %lld ms,  %lld maximal cliques ===\n",
                     total_ms, final_count);
        logger.print("Peak process memory: %.2f GiB (limit 16.00 GiB)\n",
                     acmsc_runtime::peak_memory_bytes() / 1073741824.0);
#if ACMSC_ENABLE_STATS
        logger.print("  Vertex-bound skipped layers: %llu\n",
                     static_cast<unsigned long long>(
                         vertex_bound_skipped_layers_.load(memory_order_relaxed)));
        logger.print("STAT structural_cliques=%llu subset_tests=%llu "
                     "covered_subsets=%llu feasible_local=%llu "
                     "exact_duplicates=%llu global_subsets_pruned=%llu\n",
                     structural_cliques_.load(), subset_tests_.load(),
                     covered_subsets_.load(), feasible_local_.load(),
                     static_cast<unsigned long long>(exact_duplicates_),
                     static_cast<unsigned long long>(global_subsets_pruned_));
#endif

        // 分阶段时间汇总
        logger.print("\n--- Phase timing summary ---\n");
#if ACMSC_ENABLE_STATS
        logger.print("  BK ratio in Phase1+2:         %6.2f%%\n", phase1_ratio_);
        logger.print("  Subset ratio in Phase1+2:     %6.2f%%\n", phase2_ratio_);
#endif
        logger.print("  Phase3 dedup:                 %6lld ms\n", phase3_ms);
        logger.print("----------------------------\n");

        // 团 size 分布
        logger.print("\n--- Clique size distribution ---\n");
        long long total = clique_count_.load();
        long long sum_sizes = 0;
        for (size_t sz = 1; sz < size_hist_.size(); ++sz) {
            const long long c = size_hist_[sz];
            if (c > 0) {
                logger.print("  size %3zu: %lld (%.2f%%)\n",
                             sz, c, total>0?100.0*c/total:0.0);
                sum_sizes += static_cast<long long>(sz) * c;
            }
        }
        if (total > 0)
            logger.print("  Average clique size: %.2f\n", (double)sum_sizes/total);
        logger.print("--- Total: %lld ---\n", total);

        if (sink_) fflush(sink_);
        return final_count;
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
        if (arg == "dataset" && i+1 < argc) dataset_id = atoi(argv[++i]);
        else if (arg == "log" && i+1 < argc) log_path = argv[++i];
        else if (arg == "threads" && i+1 < argc) requested_threads = atoi(argv[++i]);
        else if (arg == "timeout" && i+1 < argc) timeout_seconds = atoi(argv[++i]);
        else if (arg == "graph" && i+1 < argc) graph_override = argv[++i];
        else if (arg == "vectors" && i+1 < argc) vector_override = argv[++i];
    }
    const int thread_count = max(1, min(requested_threads, omp_get_num_procs()));
    omp_set_num_threads(thread_count);
    acmsc_runtime::select_dataset(dataset_id, graph_file, vec_index);
    if (!graph_override.empty()) graph_file = graph_override;
    if (!vector_override.empty()) vec_index = vector_override;

    if (!log_path.empty()) logger.open(log_path);

    logger.print("============================================\n");
    logger.print("  ALG3 standard (StrSub)\n");
    logger.print("  mine <tau> | sink <file> | log <file> | quit\n");
    logger.print("============================================\n");
    logger.print("Dataset: %s (ID=%d)\n", graph_file.c_str(), dataset_id);

    auto t_load = chrono::high_resolution_clock::now();
    Graph g(graph_file);
    if (g.V == 0) { fprintf(stderr, "Failed to load graph\n"); return 1; }
    vector<set<int>>().swap(g.adjSet);
    if (fs::exists(vec_index)) {
        bool ok = (vec_index.find(".index") != string::npos)
                  ? g.loadVectorsFromChunks(vec_index) : g.loadVectorsFromBinary(vec_index);
        if (!ok) { fprintf(stderr, "Failed to load vectors: %s\n", vec_index.c_str()); return 1; }
    } else { fprintf(stderr, "Vector file not found: %s\n", vec_index.c_str()); return 1; }
    g.normalizeVectors();

    long long ec = 0;
    for (int i = 0; i < g.V; i++) ec += (long long)g.adj[i].size();
    ec /= 2;
    logger.print("Graph: V=%d  E=%lld\n", g.V, ec);
    logger.print("Load: %lld ms\n\n",
        (long long)chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now()-t_load).count());

    VectorDB db; db.build(g);
    vector<vector<double>>().swap(g.semanticVectors);
    AVGMiner miner(db, g, timeout_seconds);

    logger.print("\n>> ");
    string line;
    while (getline(cin, line)) {
        if (line.empty()) { logger.print(">> "); continue; }
        istringstream iss(line); string cmd; iss >> cmd;
        if (cmd == "mine") {
            double tau = 0.2; iss >> tau; miner.mine(tau);
        } else if (cmd == "sink") {
            string fn; iss >> fn; miner.set_sink(fn.c_str());
            logger.print("Sink: %s\n", fn.c_str());
        } else if (cmd == "log") {
            string fn; iss >> fn; logger.open(fn);
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
