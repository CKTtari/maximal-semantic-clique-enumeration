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
#include <functional>

#ifndef ALG3_VERTEX_BOUND_MIN_M
#define ALG3_VERTEX_BOUND_MIN_M 12
#endif

#ifndef ACMSC_ENABLE_STATS
#define ACMSC_ENABLE_STATS 0
#endif
#ifndef ACMSC_ALG3_HOST_CANONICAL
#define ACMSC_ALG3_HOST_CANONICAL 0
#endif
#ifndef ACMSC_ALG3_HOST_FRONTIER_BOUND
#define ACMSC_ALG3_HOST_FRONTIER_BOUND 1
#endif
#ifndef ACMSC_ALG3_HOST_CANONICAL_OWNERSHIP
#define ACMSC_ALG3_HOST_CANONICAL_OWNERSHIP 1
#endif
#ifndef ACMSC_ALG3_HOST_PARALLEL
#define ACMSC_ALG3_HOST_PARALLEL 0
#endif
#ifndef ACMSC_HOST_EDGE_ENVELOPE_LIMIT
#define ACMSC_HOST_EDGE_ENVELOPE_LIMIT 24
#endif
#ifndef ACMSC_HOST_ENVELOPE_SURPLUS_LIMIT
#define ACMSC_HOST_ENVELOPE_SURPLUS_LIMIT 0.05
#endif
#ifndef ACMSC_HOST_DIRECT_LIMIT
#define ACMSC_HOST_DIRECT_LIMIT 7
#endif
#ifndef ACMSC_ALG3_HOST_CANONICAL_MASK
#define ACMSC_ALG3_HOST_CANONICAL_MASK 0
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
#ifndef ACMSC_HOST_CLOSURE_CANONICAL
#define ACMSC_HOST_CLOSURE_CANONICAL 0
#endif
#ifndef ACMSC_HOST_ROOT_CERT
#define ACMSC_HOST_ROOT_CERT 0
#endif
#ifndef ACMSC_ALG3_SUPPORT_ROOT_CERT
#define ACMSC_ALG3_SUPPORT_ROOT_CERT 0
#endif
#ifndef ACMSC_ALG3_MONOTONE_BREAK
#define ACMSC_ALG3_MONOTONE_BREAK 0
#endif
#ifndef ACMSC_ALG3_MONO_SORT_MIN_CANDIDATES
#define ACMSC_ALG3_MONO_SORT_MIN_CANDIDATES 8
#endif
#ifndef ACMSC_HOST_EXACT_FRONTIER_LIMIT
#define ACMSC_HOST_EXACT_FRONTIER_LIMIT 0
#endif
#ifndef ACMSC_HOST_VERTEX_FRONTIER_LIMIT
#define ACMSC_HOST_VERTEX_FRONTIER_LIMIT 0
#endif
#ifndef ACMSC_SUPPORT_EDGE_OWNERSHIP
#define ACMSC_SUPPORT_EDGE_OWNERSHIP 0
#endif
#ifndef ACMSC_HOST_SEM_SCHEDULE_GUIDED
#define ACMSC_HOST_SEM_SCHEDULE_GUIDED 0
#endif
#ifndef ACMSC_HOST_TOPEDGE_FRONTIER_LIMIT
#define ACMSC_HOST_TOPEDGE_FRONTIER_LIMIT 0
#endif
#ifndef ACMSC_HOST_VERTEX_DEPTH_BOUND
#define ACMSC_HOST_VERTEX_DEPTH_BOUND 0
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
#if ACMSC_HOST_CLOSURE_CANONICAL || ACMSC_ALG3_HOST_CANONICAL
    vector<float>          vertex_max_similarity_;
#endif

    BlockedSparseBitmap* bitmap_ = nullptr;
    vector<int>          ordering_;
    vector<vector<int>>  candidates_;
#if ACMSC_HOST_CLOSURE_CANONICAL
    // Structural maximal cliques are used as a read-only candidate index.
    // The semantic search itself remains one canonical reverse search over
    // the closure of all hosts; hosts are never searched independently.
    vector<vector<int>> structural_hosts_;
    vector<vector<int>> hosts_by_vertex_;
#endif
#if ACMSC_ALG3_HOST_CANONICAL
    atomic<uint64_t> host_seen_{0};
    atomic<uint64_t> host_supported_{0};
    atomic<uint64_t> host_size2_{0};
    atomic<uint64_t> host_size3_5_{0};
    atomic<uint64_t> host_size6p_{0};
#endif
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
#if ACMSC_ALG3_HOST_CANONICAL
    void process_host_canonical_mask(
        const vector<int>& M,
        vector<vector<int>>& local_cands,
        long long& phase2_ns_local
    ) {
        const auto t0 = chrono::high_resolution_clock::now();
        const int m = static_cast<int>(M.size());
        if (m < 2 || m > 63) return;
        vector<int> H = M;
        sort(H.begin(), H.end());
        vector<float> sim(static_cast<size_t>(m) * m, 0.0f);
        float top1 = -numeric_limits<float>::infinity();
        float top2 = -numeric_limits<float>::infinity();
        float top3 = -numeric_limits<float>::infinity();
        double full_sum = 0.0;
        for (int i = 0; i < m; ++i) {
            for (int j = i + 1; j < m; ++j) {
                const float s = lookup_sim(H[i], H[j]);
                sim[static_cast<size_t>(i) * m + j] = s;
                sim[static_cast<size_t>(j) * m + i] = s;
                full_sum += static_cast<double>(s);
                if (s > top1) { top3 = top2; top2 = top1; top1 = s; }
                else if (s > top2) { top3 = top2; top2 = s; }
                else if (s > top3) top3 = s;
            }
        }
        const double full_pairs = static_cast<double>(m) * (m - 1) / 2.0;
        auto emit_mask = [&](uint64_t mask) {
            vector<int> out;
            out.reserve(__builtin_popcountll(mask));
            while (mask) {
                const int i = __builtin_ctzll(mask);
                mask &= mask - 1;
                out.push_back(H[i]);
            }
            local_cands.push_back(move(out));
        };
        if (full_sum >= static_cast<double>(tau_) * full_pairs) {
            emit_mask(m == 64 ? ~0ULL : ((1ULL << m) - 1));
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
            return;
        }
        if (m >= 3 && (static_cast<double>(top1) + top2 + top3) / 3.0 <
                         static_cast<double>(tau_)) {
            for (int i = 0; i < m; ++i)
                for (int j = i + 1; j < m; ++j)
                    if (sim[static_cast<size_t>(i) * m + j] >= tau_)
                        emit_mask((1ULL << i) | (1ULL << j));
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
            return;
        }

#if ACMSC_HOST_VERTEX_DEPTH_BOUND
        int host_depth_limit = m;
        if (m >= 12) {
            vector<vector<float>> incident_sorted(m);
            for (int v = 0; v < m; ++v) {
                auto& row = incident_sorted[v];
                row.reserve(m - 1);
                for (int u = 0; u < m; ++u)
                    if (u != v) row.push_back(sim[static_cast<size_t>(v) * m + u]);
                sort(row.rbegin(), row.rend());
            }
            for (int k = 3; k <= m; ++k) {
                vector<double> caps;
                caps.reserve(m);
                for (const auto& row : incident_sorted) {
                    double s = 0.0;
                    for (int j = 0; j < k - 1; ++j) s += row[j];
                    caps.push_back(s);
                }
                if (k < m)
                    nth_element(caps.begin(), caps.begin() + k, caps.end(),
                                greater<double>());
                double cap_sum = 0.0;
                for (int i = 0; i < k; ++i) cap_sum += caps[i];
                if (cap_sum / (static_cast<double>(k) * (k - 1)) <
                    static_cast<double>(tau_) - 1e-12) {
                    host_depth_limit = k - 1;
                    break;
                }
            }
        }
#endif

        if (m <= ACMSC_HOST_DIRECT_LIMIT) {
            const uint64_t limit = 1ULL << m;
            vector<float> edge_values;
            edge_values.reserve(static_cast<size_t>(m) * (m - 1) / 2);
            for (int i = 0; i < m; ++i)
                for (int j = i + 1; j < m; ++j)
                    edge_values.push_back(sim[static_cast<size_t>(i) * m + j]);
            sort(edge_values.rbegin(), edge_values.rend());
            vector<char> layer_possible(m + 1, 0);
            double prefix = 0.0;
            for (int k = 2; k <= m; ++k) {
                const int pairs = k * (k - 1) / 2;
                if (pairs > static_cast<int>(edge_values.size())) break;
                prefix = 0.0;
                for (int e = 0; e < pairs; ++e) prefix += edge_values[e];
                layer_possible[k] = prefix / pairs + 1e-12 >= tau_;
            }
            vector<uint64_t> feasible;
            feasible.reserve(limit);
            for (uint64_t mask = 0; mask < limit; ++mask) {
                const int k = __builtin_popcountll(mask);
                if (k < 2) continue;
                if (!layer_possible[k]) continue;
                double total = 0.0;
                uint64_t bi = mask;
                while (bi) {
                    const int i = __builtin_ctzll(bi);
                    bi &= bi - 1;
                    uint64_t bj = bi;
                    while (bj) {
                        const int j = __builtin_ctzll(bj);
                        bj &= bj - 1;
                        total += static_cast<double>(
                            sim[static_cast<size_t>(i) * m + j]);
                    }
                }
                if (total + 1e-12 >= static_cast<double>(tau_) *
                    (static_cast<double>(k) * (k - 1) / 2.0))
                    feasible.push_back(mask);
            }
            sort(feasible.begin(), feasible.end(), [](uint64_t a, uint64_t b) {
                const int pa = __builtin_popcountll(a), pb = __builtin_popcountll(b);
                return pa != pb ? pa > pb : a < b;
            });
            vector<uint64_t> maximal;
            for (uint64_t mask : feasible) {
                bool covered = false;
                for (uint64_t sup : maximal)
                    if ((mask & sup) == mask) { covered = true; break; }
                if (!covered) { maximal.push_back(mask); emit_mask(mask); }
            }
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
            return;
        }

        uint64_t root_support_key = numeric_limits<uint64_t>::max();
        auto support_key = [](int a, int b) {
            const uint32_t x = static_cast<uint32_t>(min(a, b));
            const uint32_t y = static_cast<uint32_t>(max(a, b));
            return (static_cast<uint64_t>(x) << 32) | y;
        };
        function<void(uint64_t, uint64_t, double, vector<long double>&,
                      vector<double>&)> dfs;
        dfs = [&](uint64_t mask, uint64_t candidates, double sim_sum,
                  vector<long double>& incident, vector<double>& acc) {
            const int k = __builtin_popcountll(mask);
#if ACMSC_HOST_VERTEX_DEPTH_BOUND
            if (k >= host_depth_limit) {
                if (sim_sum >= static_cast<double>(tau_) *
                               (static_cast<double>(k) * (k - 1) / 2.0))
                    emit_mask(mask);
                return;
            }
#endif
            const double surplus = sim_sum - static_cast<double>(tau_) *
                (static_cast<double>(k) * (k - 1) / 2.0);
            bool has_valid_extension = false;
#if ACMSC_ALG3_MONOTONE_BREAK
            vector<int> ordered;
            ordered.reserve(__builtin_popcountll(candidates));
            uint64_t order_bits = candidates;
            while (order_bits) {
                const int v = __builtin_ctzll(order_bits);
                order_bits &= order_bits - 1;
                ordered.push_back(v);
            }
            const long double old_pairs =
                static_cast<long double>(k) * (k - 1) / 2.0L;
            const long double mono_limit = k >= 2
                ? (static_cast<long double>(sim_sum) / old_pairs) *
                  (old_pairs + k) - static_cast<long double>(sim_sum)
                : numeric_limits<long double>::infinity();
            const bool mono_sorted = true;
            sort(ordered.begin(), ordered.end(), [&](int a, int b) {
                return acc[a] != acc[b] ? acc[a] < acc[b] : H[a] < H[b];
            });
            for (const int v : ordered) {
#else
            uint64_t bits = candidates;
            while (bits) {
                const int v = __builtin_ctzll(bits);
                bits &= bits - 1;
#endif
                const double delta = acc[v];
                if (surplus + delta - static_cast<double>(tau_) * k < 0.0)
                    continue;
                has_valid_extension = true;
#if ACMSC_SUPPORT_EDGE_OWNERSHIP
                bool has_lower_support_edge = false;
                uint64_t members_for_support = mask;
                while (members_for_support) {
                    const int u = __builtin_ctzll(members_for_support);
                    members_for_support &= members_for_support - 1;
                    if (sim[static_cast<size_t>(u) * m + v] >= tau_ &&
                        support_key(H[u], H[v]) < root_support_key) {
                        has_lower_support_edge = true;
                        break;
                    }
                }
                if (has_lower_support_edge) continue;
#endif
#if ACMSC_ALG3_MONOTONE_BREAK
                // Canonical children are exactly the insertions whose
                // average does not exceed that of their parent.  Since the
                // candidates are ordered by marginal contribution, the
                // remaining suffix cannot contain a canonical child.
                if (mono_sorted && static_cast<long double>(delta) >
                              mono_limit + 1e-12L)
                    break;
#endif
                long double min_inc = static_cast<long double>(delta);
                int min_vertex = H[v];
                uint64_t members = mask;
                while (members) {
                    const int u = __builtin_ctzll(members);
                    members &= members - 1;
                    const long double ci = incident[u] +
                        static_cast<long double>(sim[static_cast<size_t>(u) * m + v]);
                    if (ci < min_inc || (ci == min_inc && H[u] < min_vertex)) {
                        min_inc = ci;
                        min_vertex = H[u];
                    }
                }
                if (min_vertex != H[v]) continue;

                vector<long double> old_inc = incident;
                vector<double> old_acc = acc;
                members = candidates & ~(1ULL << v);
                while (members) {
                    const int w = __builtin_ctzll(members);
                    members &= members - 1;
                    acc[w] += static_cast<double>(
                        sim[static_cast<size_t>(w) * m + v]);
                }
                incident.resize(m, 0.0L);
                uint64_t old_members = mask;
                while (old_members) {
                    const int u = __builtin_ctzll(old_members);
                    old_members &= old_members - 1;
                    incident[u] = old_inc[u] + static_cast<long double>(
                        sim[static_cast<size_t>(u) * m + v]);
                }
                incident[v] = static_cast<long double>(delta);
                dfs(mask | (1ULL << v), candidates & ~(1ULL << v),
                    sim_sum + delta, incident, acc);
                incident.swap(old_inc);
                acc.swap(old_acc);
            }
            if (!has_valid_extension && surplus >= 0.0)
                emit_mask(mask);
        };

        const uint64_t all = (1ULL << m) - 1;
        vector<long double> incident(m, 0.0L);
        vector<double> acc(m, 0.0);
        for (int u = 0; u < m; ++u) {
            for (int v = u + 1; v < m; ++v) {
                const float s = sim[static_cast<size_t>(u) * m + v];
                if (s < tau_) continue;
                root_support_key = support_key(H[u], H[v]);
                fill(incident.begin(), incident.end(), 0.0L);
                fill(acc.begin(), acc.end(), 0.0);
                const uint64_t root = (1ULL << u) | (1ULL << v);
                const uint64_t cand = all & ~root;
                uint64_t bits = cand;
                while (bits) {
                    const int w = __builtin_ctzll(bits);
                    bits &= bits - 1;
                    acc[w] = static_cast<double>(sim[static_cast<size_t>(w) * m + u]) +
                             static_cast<double>(sim[static_cast<size_t>(w) * m + v]);
                }
                incident[u] = incident[v] = static_cast<long double>(s);
                dfs(root, cand, static_cast<double>(s), incident, acc);
            }
        }
        phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
            chrono::high_resolution_clock::now() - t0).count();
    }

    // Canonical reverse search restricted to one structural host.  The host
    // is already a clique, so the only dynamic operation is semantic
    // feasibility; global reconciliation below removes candidates repeated
    // by overlapping hosts.
    void process_host_canonical(
        const vector<int>& M,
        vector<vector<int>>& local_cands,
        long long& phase2_ns_local
    ) {
        const auto t0 = chrono::high_resolution_clock::now();
        const int m = static_cast<int>(M.size());
        if (m < 2) return;
#if ACMSC_ALG3_HOST_CANONICAL_MASK
        if (m <= 63) {
            process_host_canonical_mask(M, local_cands, phase2_ns_local);
            return;
        }
#endif
#if ACMSC_ALG3_HOST_CANONICAL
        host_seen_.fetch_add(1, memory_order_relaxed);
        if (m == 2) host_size2_.fetch_add(1, memory_order_relaxed);
        else if (m <= 5) host_size3_5_.fetch_add(1, memory_order_relaxed);
        else host_size6p_.fetch_add(1, memory_order_relaxed);
#endif

        // Most high-selectivity structural hosts are just edges.  Handle
        // this common case without allocating a host-local matrix or
        // entering the general reverse search.
        if (m == 2) {
            if (lookup_sim(M[0], M[1]) >= tau_)
                local_cands.push_back(M);
#if ACMSC_ALG3_HOST_CANONICAL
            if (lookup_sim(M[0], M[1]) >= tau_)
                host_supported_.fetch_add(1, memory_order_relaxed);
#endif
            return;
        }

        // Necessary support-edge certificate: an average-feasible clique of
        // size at least two must contain at least one pair with similarity at
        // least tau.  Hosts without such an edge cannot contain an MSC, so
        // avoid materializing their local similarity matrix entirely.
        bool has_support_edge = false;
        for (int i = 0; i < m && !has_support_edge; ++i)
            for (int j = i + 1; j < m; ++j)
                if (lookup_sim(M[i], M[j]) >= tau_) {
                    has_support_edge = true;
                    break;
                }
        if (!has_support_edge) return;
#if ACMSC_ALG3_HOST_CANONICAL
        host_supported_.fetch_add(1, memory_order_relaxed);
#endif

        vector<vector<float>> sim(m, vector<float>(m, 0.0f));
        float top1 = -numeric_limits<float>::infinity();
        float top2 = -numeric_limits<float>::infinity();
        float top3 = -numeric_limits<float>::infinity();
        for (int i = 0; i < m; ++i)
            for (int j = i + 1; j < m; ++j)
            {
                const float s = lookup_sim(M[i], M[j]);
                sim[i][j] = sim[j][i] = s;
                if (s > top1) { top3 = top2; top2 = top1; top1 = s; }
                else if (s > top2) { top3 = top2; top2 = s; }
                else if (s > top3) top3 = s;
            }

        auto emit_local = [&](const vector<int>& local) {
            vector<int> out;
            out.reserve(local.size());
            for (int v : local) out.push_back(M[v]);
            sort(out.begin(), out.end());
            local_cands.push_back(move(out));
        };

        double full_sum = 0.0;
        for (int i = 0; i < m; ++i)
            for (int j = i + 1; j < m; ++j)
                full_sum += static_cast<double>(sim[i][j]);
        const double full_pairs = static_cast<double>(m) * (m - 1) / 2.0;
        if (full_sum >= static_cast<double>(tau_) * full_pairs) {
            vector<int> all(m);
            iota(all.begin(), all.end(), 0);
            emit_local(all);
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
            return;
        }

        // If even the three strongest internal edges cannot reach the
        // threshold, no feasible subset of size at least three exists:
        // every such subset contains at least three edges and its average is
        // bounded by the global top-three average.  The host can therefore
        // contribute only feasible pairs, which are emitted directly.
        if (m >= 3 && (static_cast<double>(top1) + top2 + top3) / 3.0 <
                         static_cast<double>(tau_)) {
            for (int i = 0; i < m; ++i)
                for (int j = i + 1; j < m; ++j)
                    if (sim[i][j] >= tau_)
                        emit_local(vector<int>{i, j});
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
            return;
        }

#if ACMSC_HOST_VERTEX_DEPTH_BOUND
        int host_depth_limit = m;
        if (m >= 12) {
            vector<vector<float>> incident_sorted(m);
            for (int v = 0; v < m; ++v) {
                auto& row = incident_sorted[v];
                row.reserve(m - 1);
                for (int u = 0; u < m; ++u)
                    if (u != v) row.push_back(sim[v][u]);
                sort(row.rbegin(), row.rend());
            }
            for (int k = 3; k <= m; ++k) {
                vector<double> caps;
                caps.reserve(m);
                for (const auto& row : incident_sorted) {
                    double s = 0.0;
                    for (int j = 0; j < k - 1; ++j) s += row[j];
                    caps.push_back(s);
                }
                if (k < m)
                    nth_element(caps.begin(), caps.begin() + k, caps.end(),
                                greater<double>());
                double cap_sum = 0.0;
                for (int i = 0; i < k; ++i) cap_sum += caps[i];
                if (cap_sum / (static_cast<double>(k) * (k - 1)) <
                    static_cast<double>(tau_) - 1e-12) {
                    host_depth_limit = k - 1;
                    break;
                }
            }
        }
#endif

        // Small infeasible hosts are common at intermediate selectivity.
        // Enumerating their bounded subset lattice directly avoids the
        // per-root allocations and recursive bookkeeping of the general
        // reverse search.  The global containment filter below keeps only
        // maximal candidates, so emitting every feasible subset here is
        // exact and does not alter the result family.
        if (m <= ACMSC_HOST_DIRECT_LIMIT) {
            const uint64_t limit = 1ULL << m;
            vector<uint64_t> feasible_masks;
            feasible_masks.reserve(limit);
            for (uint64_t mask = 0; mask < limit; ++mask) {
                const int k = __builtin_popcountll(mask);
                if (k < 2) continue;
                double sum = 0.0;
                for (int i = 0; i < m; ++i) if (mask & (1ULL << i))
                    for (int j = i + 1; j < m; ++j)
                        if (mask & (1ULL << j)) sum += sim[i][j];
                if (sum + 1e-12 >= static_cast<double>(tau_) *
                                     (static_cast<double>(k) * (k - 1) / 2.0)) {
                    feasible_masks.push_back(mask);
                }
            }
            sort(feasible_masks.begin(), feasible_masks.end(),
                 [](uint64_t a, uint64_t b) {
                     const int pa = __builtin_popcountll(a);
                     const int pb = __builtin_popcountll(b);
                     return pa != pb ? pa > pb : a < b;
                 });
            vector<uint64_t> maximal_masks;
            for (uint64_t mask : feasible_masks) {
                bool contained = false;
                for (uint64_t sup : maximal_masks)
                    if ((mask & sup) == mask) { contained = true; break; }
                if (contained) continue;
                maximal_masks.push_back(mask);
                vector<int> local;
                local.reserve(__builtin_popcountll(mask));
                for (int i = 0; i < m; ++i)
                    if (mask & (1ULL << i)) local.push_back(i);
                emit_local(local);
            }
            phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - t0).count();
            return;
        }

        auto frontier_upper_bound = [&](const vector<int>& clique,
                                        const vector<int>& candidates,
                                        const vector<double>& acc,
                                        double surplus) {
            if (candidates.empty())
                return -numeric_limits<double>::infinity();
            vector<double> margins;
            margins.reserve(candidates.size());
            const double tau_k = static_cast<double>(tau_) * clique.size();
            for (int v : candidates) margins.push_back(acc[v] - tau_k);
            sort(margins.begin(), margins.end(), greater<double>());
#if ACMSC_HOST_TOPEDGE_FRONTIER_LIMIT > 0
            if (candidates.size() <= ACMSC_HOST_TOPEDGE_FRONTIER_LIMIT) {
                vector<double> pair_surplus;
                pair_surplus.reserve(candidates.size() *
                                     (candidates.size() - 1) / 2);
                for (size_t i = 0; i < candidates.size(); ++i)
                    for (size_t j = i + 1; j < candidates.size(); ++j)
                        pair_surplus.push_back(
                            static_cast<double>(sim[candidates[i]][candidates[j]]) -
                            static_cast<double>(tau_));
                sort(pair_surplus.begin(), pair_surplus.end(), greater<double>());
                double margin_prefix = 0.0;
                double pair_prefix = 0.0;
                double best = -numeric_limits<double>::infinity();
                for (int r = 1; r <= static_cast<int>(margins.size()); ++r) {
                    margin_prefix += margins[r - 1];
                    const int need = r * (r - 1) / 2;
                    if (need > 0) pair_prefix += pair_surplus[need - 1];
                    best = max(best, surplus + margin_prefix + pair_prefix);
                }
                return best;
            }
#endif
#if ACMSC_HOST_VERTEX_FRONTIER_LIMIT > 0
            if (candidates.size() <= ACMSC_HOST_VERTEX_FRONTIER_LIMIT) {
                vector<vector<double>> incident_caps(candidates.size());
                for (size_t i = 0; i < candidates.size(); ++i) {
                    auto& row = incident_caps[i];
                    row.reserve(candidates.size() - 1);
                    for (size_t j = 0; j < candidates.size(); ++j) {
                        if (i == j) continue;
                        row.push_back(static_cast<double>(
                            sim[candidates[i]][candidates[j]]) -
                            static_cast<double>(tau_));
                    }
                    sort(row.begin(), row.end(), greater<double>());
                    for (size_t j = 1; j < row.size(); ++j)
                        row[j] += row[j - 1];
                }
                double margin_prefix = 0.0;
                double best = -numeric_limits<double>::infinity();
                for (int r = 1; r <= static_cast<int>(margins.size()); ++r) {
                    margin_prefix += margins[r - 1];
                    vector<double> caps;
                    caps.reserve(candidates.size());
                    for (const auto& row : incident_caps) {
                        const size_t idx = std::min<size_t>(
                            static_cast<size_t>(r - 1), row.size() - 1);
                        caps.push_back(row.empty() ? 0.0 : row[idx]);
                    }
                    if (r < static_cast<int>(caps.size()))
                        nth_element(caps.begin(), caps.begin() + r, caps.end(),
                                    greater<double>());
                    double cap_sum = 0.0;
                    for (int i = 0; i < r; ++i) cap_sum += caps[i];
                    best = max(best, surplus + margin_prefix + 0.5 * cap_sum);
                }
                return best;
            }
#endif
#if ACMSC_HOST_EXACT_FRONTIER_LIMIT > 0
            // For small candidate sets, retain the individual internal-edge
            // surpluses instead of replacing all of them by one global
            // maximum.  The sum of the largest C(r,2) values is a valid
            // upper bound for every r-vertex extension and is substantially
            // tighter in the intermediate-selectivity regime.
            if (candidates.size() <= ACMSC_HOST_EXACT_FRONTIER_LIMIT) {
                vector<double> pair_surpluses;
                pair_surpluses.reserve(candidates.size() * (candidates.size() - 1) / 2);
                for (size_t i = 0; i < candidates.size(); ++i)
                    for (size_t j = i + 1; j < candidates.size(); ++j)
                        pair_surpluses.push_back(
                            static_cast<double>(sim[candidates[i]][candidates[j]]) -
                            static_cast<double>(tau_));
                sort(pair_surpluses.begin(), pair_surpluses.end(), greater<double>());
                double margin_prefix = 0.0;
                double pair_prefix = 0.0;
                double best = -numeric_limits<double>::infinity();
                for (int r = 1; r <= static_cast<int>(margins.size()); ++r) {
                    margin_prefix += margins[r - 1];
                    const int need = r * (r - 1) / 2;
                    if (need > 0) pair_prefix += pair_surpluses[need - 1];
                    best = max(best, surplus + margin_prefix + pair_prefix);
                }
                return best;
            }
#endif
            double pair_surplus = max(0.0, 1.0 - static_cast<double>(tau_));
#if ACMSC_HOST_EDGE_ENVELOPE_LIMIT > 0
            if (candidates.size() <= ACMSC_HOST_EDGE_ENVELOPE_LIMIT) {
                double max_internal = -numeric_limits<double>::infinity();
                for (size_t i = 0; i < candidates.size(); ++i)
                    for (size_t j = i + 1; j < candidates.size(); ++j)
                        max_internal = max(max_internal,
                            static_cast<double>(sim[candidates[i]][candidates[j]]));
                pair_surplus = max_internal - static_cast<double>(tau_);
            } else {
                double max_internal = -numeric_limits<double>::infinity();
                for (int v : candidates)
                    max_internal = max(max_internal,
                        static_cast<double>(vertex_max_similarity_[v]));
                pair_surplus = max_internal - static_cast<double>(tau_);
            }
#endif
            double best = -numeric_limits<double>::infinity();
            double prefix = 0.0;
            for (int r = 1; r <= static_cast<int>(margins.size()); ++r) {
                prefix += margins[r - 1];
                const double pairs = static_cast<double>(r) * (r - 1) / 2.0;
                best = max(best, surplus + prefix + pair_surplus * pairs);
            }
            return best;
        };

        uint64_t root_support_key = numeric_limits<uint64_t>::max();
        auto support_key = [](int a, int b) {
            const uint32_t x = static_cast<uint32_t>(min(a, b));
            const uint32_t y = static_cast<uint32_t>(max(a, b));
            return (static_cast<uint64_t>(x) << 32) | y;
        };
        function<void(vector<int>&, vector<long double>&, vector<double>&, double,
                      vector<int>&)> dfs;
        dfs = [&](vector<int>& clique, vector<long double>& incident,
                  vector<double>& acc, double sim_sum,
                  vector<int>& candidates) {
            const int k = static_cast<int>(clique.size());
#if ACMSC_HOST_VERTEX_DEPTH_BOUND
            if (k >= host_depth_limit) {
                if (sim_sum >= static_cast<double>(tau_) *
                               (static_cast<double>(k) * (k - 1) / 2.0))
                    emit_local(clique);
                return;
            }
#endif
            const double pairs = static_cast<double>(k) * (k - 1) / 2.0;
            const double surplus = sim_sum - static_cast<double>(tau_) * pairs;
            if (ACMSC_ALG3_HOST_FRONTIER_BOUND &&
                frontier_upper_bound(clique, candidates, acc, surplus) < -1e-12) {
                if (surplus >= 0.0) emit_local(clique);
                return;
            }

            bool has_valid_extension = false;
#if ACMSC_ALG3_MONOTONE_BREAK
            vector<int> ordered = candidates;
            sort(ordered.begin(), ordered.end(), [&](int a, int b) {
                return acc[a] != acc[b] ? acc[a] < acc[b] : M[a] < M[b];
            });
            const long double old_pairs =
                static_cast<long double>(k) * (k - 1) / 2.0L;
            const long double mono_limit = k >= 2
                ? (static_cast<long double>(sim_sum) / old_pairs) *
                  (old_pairs + k) - static_cast<long double>(sim_sum)
                : numeric_limits<long double>::infinity();
            for (int v : ordered) {
#else
            for (int v : candidates) {
#endif
                const double delta = acc[v];
                if (surplus + delta - static_cast<double>(tau_) * k < 0.0)
                    continue;
                has_valid_extension = true;
#if ACMSC_SUPPORT_EDGE_OWNERSHIP
                bool has_lower_support_edge = false;
                for (int i = 0; i < k; ++i) {
                    if (sim[clique[i]][v] >= tau_ &&
                        support_key(M[clique[i]], M[v]) < root_support_key) {
                        has_lower_support_edge = true;
                        break;
                    }
                }
                if (has_lower_support_edge) continue;
#endif
#if ACMSC_ALG3_MONOTONE_BREAK
                if (k >= 2 && static_cast<long double>(delta) >
                              mono_limit + 1e-12L)
                    break;
#endif

                vector<long double> child_incident(k + 1, 0.0L);
                int minimum_vertex = M[v];
                long double minimum_incident = 0.0L;
                for (int i = 0; i < k; ++i) {
                    child_incident[i] = incident[i] +
                        static_cast<long double>(sim[clique[i]][v]);
                    if (i == 0 || child_incident[i] < minimum_incident ||
                        (child_incident[i] == minimum_incident &&
                         M[clique[i]] < minimum_vertex)) {
                        minimum_incident = child_incident[i];
                        minimum_vertex = M[clique[i]];
                    }
                }
                child_incident[k] = static_cast<long double>(delta);
                if (child_incident[k] < minimum_incident ||
                    (child_incident[k] == minimum_incident && M[v] < minimum_vertex)) {
                    minimum_incident = child_incident[k];
                    minimum_vertex = M[v];
                }
#if ACMSC_ALG3_HOST_CANONICAL_OWNERSHIP
                if (minimum_vertex != M[v]) continue;
#endif

                vector<int> child_candidates;
                child_candidates.reserve(candidates.size() - 1);
                vector<double> old_acc;
                old_acc.reserve(candidates.size() - 1);
                for (int w : candidates) {
                    if (w == v) continue;
                    child_candidates.push_back(w);
                    old_acc.push_back(acc[w]);
                    acc[w] += sim[w][v];
                }
                vector<long double> old_incident = incident;
                for (int i = 0; i < k; ++i)
                    incident[i] = child_incident[i];
                clique.push_back(v);
                incident.push_back(child_incident[k]);
                dfs(clique, incident, acc, sim_sum + delta, child_candidates);
                incident.pop_back();
                clique.pop_back();
                incident.swap(old_incident);
                for (size_t i = 0; i < child_candidates.size(); ++i)
                    acc[child_candidates[i]] = old_acc[i];
            }

            if (!has_valid_extension && surplus >= 0.0) {
                emit_local(clique);
            }
        };

        for (int u = 0; u < m; ++u) {
            for (int v = u + 1; v < m; ++v) {
                if (sim[u][v] < tau_) continue;
                root_support_key = support_key(M[u], M[v]);
                vector<int> clique{u, v};
                vector<long double> incident{
                    static_cast<long double>(sim[u][v]),
                    static_cast<long double>(sim[u][v])};
                vector<int> candidates;
                candidates.reserve(m - 2);
                vector<double> acc(m, 0.0);
                for (int w = 0; w < m; ++w) {
                    if (w == u || w == v) continue;
                    candidates.push_back(w);
                    acc[w] = static_cast<double>(sim[w][u]) +
                             static_cast<double>(sim[w][v]);
                }
                dfs(clique, incident, acc, sim[u][v], candidates);
            }
        }
        phase2_ns_local += chrono::duration_cast<chrono::nanoseconds>(
            chrono::high_resolution_clock::now() - t0).count();
    }
#endif

    void process_structural_clique(
        const vector<int>& M,
        vector<vector<int>>& local_cands,
        long long& phase2_ns_local
    ) {
#if ACMSC_ALG3_HOST_CANONICAL
        process_host_canonical(M, local_cands, phase2_ns_local);
        return;
#endif
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
#if ACMSC_HOST_CLOSURE_CANONICAL
        inline void set(int w, double s) {
            val[w] = static_cast<float>(s);
            gen[w] = cur;
        }
#endif
    };

#if ACMSC_HOST_CLOSURE_CANONICAL
    static vector<int> intersect_sorted_ids(const vector<int>& a,
                                             const vector<int>& b) {
        vector<int> out;
        out.reserve(min(a.size(), b.size()));
        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i] == b[j]) { out.push_back(a[i]); ++i; ++j; }
            else if (a[i] < b[j]) ++i;
            else ++j;
        }
        return out;
    }

    // Enumerate the semantic search space once over all structural hosts.
    // A state carries the hosts that contain it.  Its candidate domain is the
    // union of those hosts, and a child carries the corresponding host
    // intersection.  Thus the structural phase is an index for canonical
    // search rather than a collection of independent semantic searches.
    void run_host_closure(vector<vector<int>>& output) {
        structural_hosts_ = move(candidates_);
        candidates_.clear();
        for (auto& host : structural_hosts_) {
            sort(host.begin(), host.end());
            host.erase(unique(host.begin(), host.end()), host.end());
        }
        hosts_by_vertex_.assign(g_.V, {});
        for (int hid = 0; hid < static_cast<int>(structural_hosts_.size()); ++hid)
            for (int v : structural_hosts_[hid]) hosts_by_vertex_[v].push_back(hid);

        vector<unsigned char> host_feasible(structural_hosts_.size(), 0);
        vector<vector<int>> feasible_hosts;
        feasible_hosts.reserve(structural_hosts_.size());
        for (int hid = 0; hid < static_cast<int>(structural_hosts_.size()); ++hid) {
            const auto& host = structural_hosts_[hid];
            double total = 0.0;
            for (size_t i = 0; i < host.size(); ++i)
                for (size_t j = i + 1; j < host.size(); ++j)
                    total += static_cast<double>(lookup_sim(host[i], host[j]));
            const double pairs = static_cast<double>(host.size()) *
                                 static_cast<double>(host.size() - 1) / 2.0;
            if (pairs > 0 && total >= static_cast<double>(tau_) * pairs) {
                host_feasible[hid] = 1;
                feasible_hosts.push_back(host);
            }
        }

        // Only structurally maximal hosts that are themselves infeasible can
        // contain a non-maximal intermediate state.  Feasible hosts are
        // emitted once below and never need to participate in the reverse
        // search.  Keeping a filtered incidence index avoids scanning the
        // same large set of already-feasible hosts at every state.
        vector<vector<int>> bad_hosts_by_vertex(g_.V);
        for (int hid = 0; hid < static_cast<int>(structural_hosts_.size()); ++hid) {
            if (host_feasible[hid]) continue;
            for (int v : structural_hosts_[hid])
                bad_hosts_by_vertex[v].push_back(hid);
        }

        struct ClosureAcc {
            vector<double> value;
            vector<uint32_t> generation;
            uint32_t current = 1;
            void init(int n) { value.assign(n, 0.0); generation.assign(n, 0); }
            void next_generation() {
                if (++current == 0) {
                    fill(generation.begin(), generation.end(), 0);
                    current = 1;
                }
            }
            double get(int v) const {
                return generation[v] == current ? value[v] : 0.0;
            }
            void add(int v, double s) {
                if (generation[v] != current) {
                    generation[v] = current; value[v] = 0.0;
                }
                value[v] += s;
            }
            void set(int v, double s) {
                generation[v] = current; value[v] = s;
            }
        };
        const int nthreads = omp_get_max_threads();
        vector<vector<vector<int>>> per_thread(nthreads);
        vector<ClosureAcc> stores(nthreads);
        for (auto& s : stores) s.init(g_.V);
        const auto t0 = chrono::high_resolution_clock::now();
        atomic<bool> timed_out{false};
        atomic<long long> states{0};

        vector<pair<int, int>> roots;
        for (int u = 0; u < g_.V; ++u) {
            for (const auto& edge : nb_[u]) {
                const int v = edge.first;
                if (v > u && edge.second >= tau_)
                    roots.push_back({u, v});
            }
        }

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            ClosureAcc& acc = stores[tid];
            vector<int> marks(g_.V, 0), in_clique(g_.V, 0);
            int mark_token = 0;
            uint32_t tick = 0;
            function<void(vector<int>&, vector<long double>&, double,
                          vector<int>&, vector<int>&)> dfs;
            dfs = [&](vector<int>& clique, vector<long double>& incident,
                      double sim_sum, vector<int>& active_hosts,
                      vector<int>& candidates) {
                if (timed_out.load()) return;
                if (timeout_seconds_ > 0 && ((++tick & 4095U) == 0)) {
                    const auto elapsed = chrono::duration_cast<chrono::seconds>(
                        chrono::high_resolution_clock::now() - t0).count();
                    if (elapsed >= timeout_seconds_) {
                        timed_out.store(true); return;
                    }
                }
                states.fetch_add(1, memory_order_relaxed);
                const int k = static_cast<int>(clique.size());
                const double surplus = sim_sum -
                    static_cast<double>(tau_) * (static_cast<double>(k) * (k - 1) / 2.0);
                const double tau_k = static_cast<double>(tau_) * k;

#if ACMSC_HOST_EDGE_ENVELOPE_LIMIT > 0
                if (surplus <= ACMSC_HOST_ENVELOPE_SURPLUS_LIMIT &&
                    !candidates.empty()) {
                    vector<double> margins;
                    margins.reserve(candidates.size());
                    for (int w : candidates)
                        margins.push_back(acc.get(w) - tau_k);
                    sort(margins.begin(), margins.end(), greater<double>());
                    double max_internal = -numeric_limits<double>::infinity();
                    if (candidates.size() <= ACMSC_HOST_EDGE_ENVELOPE_LIMIT) {
                        for (size_t i = 0; i < candidates.size(); ++i)
                            for (size_t j = i + 1; j < candidates.size(); ++j)
                                max_internal = max(max_internal,
                                    static_cast<double>(lookup_sim(candidates[i], candidates[j])));
                    } else {
                        for (int v : candidates)
                            max_internal = max(max_internal,
                                static_cast<double>(vertex_max_similarity_[v]));
                    }
                    const double pair_surplus = max_internal - static_cast<double>(tau_);
                    double best = -numeric_limits<double>::infinity();
                    double prefix = 0.0;
                    for (int r = 1; r <= static_cast<int>(margins.size()); ++r) {
                        prefix += margins[r - 1];
                        const double pairs = static_cast<double>(r) * (r - 1) / 2.0;
                        best = max(best, surplus + prefix +
                            (r == 1 ? 0.0 : pair_surplus * pairs));
                    }
                    if (best < -1e-12) {
                        if (k >= 2 && surplus >= 0.0) {
                            vector<int> out = clique;
                            sort(out.begin(), out.end());
                            per_thread[tid].push_back(move(out));
                        }
                        return;
                    }
                }
#endif
                bool has_valid_extension = false;
                for (int v : candidates) {
                    if (timed_out.load()) return;
                    const double delta = static_cast<double>(acc.get(v));
                    if (surplus + delta - tau_k < 0.0) continue;
                    has_valid_extension = true;

                    vector<float> deltas(k);
                    long double added_incident = 0.0L;
                    for (int i = 0; i < k; ++i) {
                        deltas[i] = lookup_sim(clique[i], v);
                        added_incident += static_cast<long double>(deltas[i]);
                    }
                    int minimum_vertex = v;
                    long double minimum_incident = added_incident;
                    for (int i = 0; i < k; ++i) {
                        const long double child_incident =
                            incident[i] + static_cast<long double>(deltas[i]);
                        if (child_incident < minimum_incident ||
                            (child_incident == minimum_incident &&
                             clique[i] < minimum_vertex)) {
                            minimum_incident = child_incident;
                            minimum_vertex = clique[i];
                        }
                    }
                    if (minimum_vertex != v) continue;

#if ACMSC_HOST_ROOT_CERT
                    vector<int> child_hosts = active_hosts;
#else
                    vector<int> child_hosts = intersect_sorted_ids(
                        active_hosts, bad_hosts_by_vertex[v]);
                    if (child_hosts.empty()) continue;
#endif
                    vector<int> child_candidates;
                    child_candidates.reserve(candidates.size());
                    for (int w : candidates)
                        if (w != v && adjacent(v, w)) child_candidates.push_back(w);
                    vector<pair<int, double>> undo;
                    undo.reserve(child_candidates.size());
                    for (int w : child_candidates) {
                        const double old = acc.get(w);
                        undo.push_back({w, old});
                        acc.add(w, lookup_sim(w, v));
                    }
                    for (int i = 0; i < k; ++i)
                        incident[i] += static_cast<long double>(deltas[i]);
                    incident.push_back(added_incident);
                    clique.push_back(v); in_clique[v] = 1;
                    dfs(clique, incident, sim_sum + delta, child_hosts,
                        child_candidates);
                    in_clique[v] = 0; clique.pop_back(); incident.pop_back();
                    for (int i = 0; i < k; ++i)
                        incident[i] -= static_cast<long double>(deltas[i]);
                    for (const auto& e : undo) acc.set(e.first, e.second);
                }
                if (!timed_out.load() && !has_valid_extension &&
                    k >= 2 && surplus >= 0.0) {
                    vector<int> out = clique;
                    sort(out.begin(), out.end());
                    per_thread[tid].push_back(move(out));
                }
            };

#pragma omp for schedule(dynamic, 1)
            for (long long ri = 0; ri < static_cast<long long>(roots.size()); ++ri) {
                if (timed_out.load()) continue;
                const int u = roots[ri].first, v = roots[ri].second;
                vector<int> active = intersect_sorted_ids(
                    bad_hosts_by_vertex[u], bad_hosts_by_vertex[v]);
                if (active.empty()) continue;
#if ACMSC_HOST_ROOT_CERT
                bool all_hosts_feasible = true;
                for (int hid : active)
                    if (!host_feasible[hid]) { all_hosts_feasible = false; break; }
                if (all_hosts_feasible) continue;
#endif
                vector<int> cand;
                cand.reserve(nb_[u].size());
                for (const auto& edge : nb_[u])
                    if (edge.first != v && adjacent(v, edge.first))
                        cand.push_back(edge.first);
                acc.next_generation();
                for (int w : cand)
                    acc.add(w, lookup_sim(w, u) + lookup_sim(w, v));
                vector<int> clique{u, v};
                in_clique[u] = in_clique[v] = 1;
                vector<long double> incident{
                    static_cast<long double>(lookup_sim(u, v)),
                    static_cast<long double>(lookup_sim(u, v))};
                dfs(clique, incident, static_cast<double>(lookup_sim(u, v)),
                    active, cand);
                in_clique[u] = in_clique[v] = 0;
            }
        }
        size_t count = feasible_hosts.size();
        for (const auto& local : per_thread) count += local.size();
        candidates_.reserve(count);
        for (auto& host : feasible_hosts) candidates_.push_back(move(host));
        for (auto& local : per_thread)
            for (auto& c : local) candidates_.push_back(move(c));
        logger.print("  Host-closure canonical states: %lld; hosts=%zu; roots=%zu\n",
                     states.load(), structural_hosts_.size(), roots.size());
    }
#endif

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
#if ACMSC_HOST_CLOSURE_CANONICAL || ACMSC_ALG3_HOST_PARALLEL
            local_cands.push_back(C);
#else
            process_structural_clique(C, local_cands, phase2_ns_local);
#endif
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
#if ACMSC_HOST_CLOSURE_CANONICAL || ACMSC_ALG3_HOST_CANONICAL
        vertex_max_similarity_.assign(n, 0.0f);
        for (int u = 0; u < n; ++u)
            for (const auto& edge : nb_[u])
                vertex_max_similarity_[u] = max(vertex_max_similarity_[u], edge.second);
#endif

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
#if ACMSC_ALG3_SUPPORT_ROOT_CERT
            vector<int> support_mark(n, 0);
            int support_token = 0;
#endif

#if ACMSC_ALG3_HOST_CANONICAL
            #pragma omp for schedule(guided, 32)
#else
            #pragma omp for schedule(dynamic, 1)
#endif
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

#if ACMSC_ALG3_SUPPORT_ROOT_CERT
                // Necessary certificate: every feasible MSC contains a pair
                // whose similarity is at least tau.  The degeneracy root v
                // can only generate a host from {v} union cands; if that
                // induced set has no supported edge, the host cannot contain
                // any feasible semantic subset.
                if (++support_token == numeric_limits<int>::max()) {
                    fill(support_mark.begin(), support_mark.end(), 0);
                    support_token = 1;
                }
                support_mark[v] = support_token;
                for (int u : cands) support_mark[u] = support_token;
                bool has_support_edge = false;
                for (int x : cands) {
                    for (const auto& edge : nb_[x]) {
                        if (edge.second >= tau_ &&
                            support_mark[edge.first] == support_token) {
                            has_support_edge = true;
                            break;
                        }
                    }
                    if (has_support_edge) break;
                }
                if (!has_support_edge) {
                    for (const auto& edge : nb_[v]) {
                        if (edge.second >= tau_ &&
                            support_mark[edge.first] == support_token) {
                            has_support_edge = true;
                            break;
                        }
                    }
                }
                if (!has_support_edge) continue;
#endif

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

#if ACMSC_ALG3_HOST_PARALLEL
        vector<vector<int>> hosts = move(candidates_);
        candidates_.clear();
        vector<vector<vector<int>>> semantic_outputs(nthreads);
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
#if ACMSC_HOST_SEM_SCHEDULE_GUIDED
#pragma omp for schedule(guided, 8)
#else
#pragma omp for schedule(dynamic, 1)
#endif
            for (long long hi = 0; hi < static_cast<long long>(hosts.size()); ++hi) {
                long long phase2_ns_local = 0;
                process_host_canonical(hosts[static_cast<size_t>(hi)],
                                       semantic_outputs[tid], phase2_ns_local);
            }
        }
        size_t semantic_count = 0;
        for (const auto& local : semantic_outputs) semantic_count += local.size();
        candidates_.reserve(semantic_count);
        for (auto& local : semantic_outputs)
            for (auto& c : local) candidates_.push_back(move(c));
#endif

#if ACMSC_HOST_CLOSURE_CANONICAL
        // Replace structural hosts by the globally owned semantic states
        // before the common containment filter runs.
        run_host_closure(candidates_);
#endif

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
#if ACMSC_ALG3_HOST_CANONICAL
        host_seen_.store(0, memory_order_relaxed);
        host_supported_.store(0, memory_order_relaxed);
        host_size2_.store(0, memory_order_relaxed);
        host_size3_5_.store(0, memory_order_relaxed);
        host_size6p_.store(0, memory_order_relaxed);
#endif
        delete bitmap_; bitmap_ = nullptr;
        nb_.clear(); sim_map_.clear(); ordering_.clear();

#if ACMSC_HOST_CLOSURE_CANONICAL
        logger.print("\n=== ALG5 MSC-HostClosure canonical  tau=%.3f ===\n", tau);
#elif ACMSC_ALG3_HOST_CANONICAL
        logger.print("\n=== ALG5 MSC-HostFrontier  tau=%.3f ===\n", tau);
#else
        logger.print("\n=== ALG3 StrSub  tau=%.3f ===\n", tau);
#endif
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
#if ACMSC_ALG3_HOST_CANONICAL
        logger.print("  Supported structural hosts:   %llu / %llu\n",
                     static_cast<unsigned long long>(host_supported_.load()),
                     static_cast<unsigned long long>(host_seen_.load()));
        logger.print("  Host sizes: 2=%llu 3--5=%llu 6+=%llu\n",
                     static_cast<unsigned long long>(host_size2_.load()),
                     static_cast<unsigned long long>(host_size3_5_.load()),
                     static_cast<unsigned long long>(host_size6p_.load()));
#endif
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
#if ACMSC_HOST_CLOSURE_CANONICAL
    logger.print("  ALG5 MSC-HostClosure canonical\n");
#elif ACMSC_ALG3_HOST_CANONICAL
    logger.print("  ALG5 MSC-HostFrontier\n");
#else
    logger.print("  ALG3 standard (StrSub)\n");
#endif
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
