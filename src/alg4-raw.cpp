// ============================================================
// 平均相似度约束极大团挖掘（Monotone-path MCE）
//
// 给定无向图 G 和阈值 tau，枚举所有满足：
//   avg_{u,v in C} sim(u,v) >= tau
// 的极大团 C（maximal clique）。
//
// 优化：
//   [OPT-A] thread-local scratch buffers
//     每个线程持有按递归深度索引的 cands/undo scratch buffer，
//     进入时 clear()，退出时不 shrink，整个搜索复用同一块内存。
//
//   [OPT-C] per-thread visited 用 open-addressing flat hash set
//     FlatHashSet64：power-of-2 大小、linear probing、generation stamp，
//     全部数据在连续内存上，clear() 为 O(1)。
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
#include <array>
#include <map>

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
// [OPT-C] FlatHashSet64
// open-addressing linear probing，语义等价于 unordered_set<size_t>。
// clear() 递增 generation stamp，O(1)，不做 memset。
// ============================================================
struct FlatHashSet64
{
    static constexpr float MAX_LOAD = 0.6f;

    vector<size_t> keys;
    vector<uint32_t> gen;
    uint32_t cur_gen = 1;
    size_t mask = 0;
    size_t used = 0;

    uint64_t total_insert = 0;
    uint64_t total_lookup = 0;
    uint64_t total_probe = 0;

    FlatHashSet64() { _resize(1 << 20); }

    void clear()
    {
        ++cur_gen;
        if (cur_gen == 0)
        {
            cur_gen = 1;
            fill(gen.begin(), gen.end(), 0);
        }
        used = 0;
    }

    // 返回 true = 新插入，false = 已存在
    inline bool insert(size_t key)
    {
        ++total_insert;
        if (used + 1 > (size_t)(keys.size() * MAX_LOAD))
            _rehash();
        size_t pos = _h(key) & mask;
        for (;;)
        {
            ++total_probe;
            if (gen[pos] != cur_gen)
            {
                keys[pos] = key;
                gen[pos] = cur_gen;
                ++used;
                return true;
            }
            if (keys[pos] == key)
                return false;
            pos = (pos + 1) & mask;
        }
    }

    inline bool count(size_t key) const
    {
        ++const_cast<FlatHashSet64 *>(this)->total_lookup;
        size_t pos = _h(key) & mask;
        for (;;)
        {
            ++const_cast<FlatHashSet64 *>(this)->total_probe;
            if (gen[pos] != cur_gen)
                return false;
            if (keys[pos] == key)
                return true;
            pos = (pos + 1) & mask;
        }
    }

    size_t size() const { return used; }

private:
    static inline size_t _h(size_t k)
    {
        return k * 11400714819323198485ULL; // Fibonacci hashing
    }
    void _resize(size_t cap)
    {
        keys.assign(cap, 0);
        gen.assign(cap, 0);
        mask = cap - 1;
        used = 0;
    }
    void _rehash()
    {
        vector<size_t> live;
        live.reserve(used);
        for (size_t i = 0; i < keys.size(); i++)
            if (gen[i] == cur_gen)
                live.push_back(keys[i]);
        _resize(keys.size() * 2);
        for (size_t k : live)
        {
            size_t pos = _h(k) & mask;
            while (gen[pos] == cur_gen)
                pos = (pos + 1) & mask;
            keys[pos] = k;
            gen[pos] = cur_gen;
            ++used;
        }
    }
};

// ============================================================
// [OPT-A] ScratchBuffers — per-thread、per-depth 复用缓冲区
// ============================================================
struct ScratchBuffers
{
    static constexpr int MAX_DEPTH = 256;
    vector<vector<int>> cands_buf{MAX_DEPTH};
    vector<vector<pair<int, float>>> undo_buf{MAX_DEPTH};

    inline vector<int> &cands_at(int d)
    {
        cands_buf[d].clear();
        return cands_buf[d];
    }
    inline vector<pair<int, float>> &undo_at(int d)
    {
        undo_buf[d].clear();
        return undo_buf[d];
    }
};

// ============================================================
// BlockedSparseBitmap
// 邻接关系的 bitmap 表示，支持 O(|row|/64) 交集。
// ============================================================
class BlockedSparseBitmap
{
public:
    struct Block
    {
        uint32_t idx;
        uint64_t bits;
    };
    struct Row
    {
        vector<Block> blocks;
        void build(const vector<int> &neighbors)
        {
            blocks.clear();
            if (neighbors.empty())
                return;
            unordered_map<uint32_t, uint64_t> tmp;
            tmp.reserve(neighbors.size());
            for (int v : neighbors)
                tmp[(uint32_t)(v >> 6)] |= 1ULL << (v & 63);
            blocks.reserve(tmp.size());
            for (auto &kv : tmp)
                blocks.push_back({kv.first, kv.second});
            sort(blocks.begin(), blocks.end(),
                 [](const Block &a, const Block &b)
                 { return a.idx < b.idx; });
        }
        bool contains(int v) const
        {
            uint32_t bidx = (uint32_t)(v >> 6);
            uint64_t bbit = 1ULL << (v & 63);
            int lo = 0, hi = (int)blocks.size() - 1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >> 1;
                if (blocks[mid].idx == bidx)
                    return (blocks[mid].bits & bbit) != 0;
                blocks[mid].idx < bidx ? lo = mid + 1 : hi = mid - 1;
            }
            return false;
        }
        void intersect_to(const Row &other, vector<int> &out) const
        {
            out.clear();
            if (blocks.empty() || other.blocks.empty())
                return;
            size_t i = 0, j = 0;
            while (i < blocks.size() && j < other.blocks.size())
            {
                if (blocks[i].idx == other.blocks[j].idx)
                {
                    uint64_t common = blocks[i].bits & other.blocks[j].bits;
                    while (common)
                    {
                        out.push_back((int)(blocks[i].idx * 64 + __builtin_ctzll(common)));
                        common &= common - 1;
                    }
                    i++;
                    j++;
                }
                else if (blocks[i].idx < other.blocks[j].idx)
                    i++;
                else
                    j++;
            }
        }
        int intersect_count(const Row &other) const
        {
            if (blocks.empty() || other.blocks.empty())
                return 0;
            int cnt = 0;
            size_t i = 0, j = 0;
            while (i < blocks.size() && j < other.blocks.size())
            {
                if (blocks[i].idx == other.blocks[j].idx)
                {
                    cnt += __builtin_popcountll(blocks[i].bits & other.blocks[j].bits);
                    i++;
                    j++;
                }
                else if (blocks[i].idx < other.blocks[j].idx)
                    i++;
                else
                    j++;
            }
            return cnt;
        }
    };

private:
    vector<Row> rows_;
    int n_;

public:
    BlockedSparseBitmap() : n_(0) {}
    explicit BlockedSparseBitmap(int n) : n_(n), rows_(n) {}
    void build(int u, const vector<int> &nb) { rows_[u].build(nb); }
    bool contains(int u, int v) const { return rows_[u].contains(v); }
    const Row &get_row(int u) const { return rows_[u]; }
    int size() const { return n_; }
};

// ============================================================
// Float32VectorStore
// 节点向量的 float32 紧凑存储，支持 AVX2 加速内积。
// ============================================================
class Float32VectorStore
{
    float *vecs_ = nullptr;
    int n_ = 0, d_ = 0;

public:
    ~Float32VectorStore()
    {
        if (vecs_)
            ALIGNED_FREE(vecs_);
    }
    int dim() const { return d_; }

    void build(const Graph &g, const vector<int> &order = {})
    {
        n_ = g.V;
        d_ = (int)g.getSemanticVector(0).size();
        size_t vb = ((size_t)n_ * d_ * sizeof(float) + 63) & ~63ULL;
        vecs_ = (float *)ALIGNED_ALLOC(vb, 64);
        bool has_order = !order.empty();
#pragma omp parallel for schedule(static)
        for (int ni = 0; ni < n_; ni++)
        {
            int oi = has_order ? order[ni] : ni;
            const vector<double> &dv = g.getSemanticVector(oi);
            float *xf = vecs_ + (size_t)ni * d_;
            for (int i = 0; i < d_; i++)
                xf[i] = (float)dv[i];
        }
        cout << "  Float32VectorStore: " << n_ << "x" << d_ << "\n";
    }

    // 计算节点 nu 和 nv 的内积（重排序后的索引）
    inline float dot(int nu, int nv) const
    {
        const float *xu = vecs_ + (size_t)nu * d_;
        const float *xv = vecs_ + (size_t)nv * d_;
        float p = 0.f;
        int i = 0;
#ifdef __AVX2__
        __m256 acc = _mm256_setzero_ps();
        for (; i <= d_ - 8; i += 8)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(
                                         _mm256_loadu_ps(xu + i), _mm256_loadu_ps(xv + i)));
        {
            __m128 hi = _mm256_extractf128_ps(acc, 1),
                   lo = _mm256_castps256_ps128(acc);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            p = _mm_cvtss_f32(lo);
        }
        for (; i < d_; i++)
            p += xu[i] * xv[i];
#else
        for (; i < d_; i++)
            p += xu[i] * xv[i];
#endif
        return p;
    }
};

// ============================================================
// GraphReorder
// 按度数降序做 BFS 重排，改善向量访问局部性。
// ============================================================
struct GraphReorder
{
    vector<int> old_to_new, new_to_old;
    void build(const Graph &g)
    {
        int n = g.V;
        old_to_new.assign(n, -1);
        new_to_old.assign(n, -1);
        vector<int> order(n);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(),
             [&](int a, int b)
             { return g.adj[a].size() > g.adj[b].size(); });
        int nid = 0;
        queue<int> q;
        for (int s : order)
        {
            if (old_to_new[s] != -1)
                continue;
            old_to_new[s] = nid;
            new_to_old[nid] = s;
            nid++;
            q.push(s);
            while (!q.empty())
            {
                int u = q.front();
                q.pop();
                for (int v : g.adj[u])
                    if (old_to_new[v] == -1)
                    {
                        old_to_new[v] = nid;
                        new_to_old[nid] = v;
                        nid++;
                        q.push(v);
                    }
            }
        }
    }
};

// ============================================================
// VectorDB
// 持有重排序映射和向量存储，供相似度计算使用。
// ============================================================
struct VectorDB
{
    Float32VectorStore vecs;
    GraphReorder reorder;
    const Graph *graph = nullptr;

    void build(const Graph &g)
    {
        graph = &g;
        cout << "\n=== Building VectorDB ===\n";
        reorder.build(g);
        vecs.build(g, reorder.new_to_old);
        cout << "=== DB ready ===\n\n";
    }
};

// ============================================================
// CliqueStore
// 线程安全的候选团存储，post_validate() 去掉被包含的子团。
// ============================================================
class CliqueStore
{
public:
    struct Entry
    {
        vector<int> nodes;
        bool valid = true;
    };

private:
    vector<Entry> entries_;
    unordered_map<int, vector<size_t>> inverted_;
    mutable std::mutex mtx_;
    size_t raw_count_ = 0;
    size_t pruned_count_ = 0;

public:
    void insert(vector<int> clique)
    {
        if (!is_sorted(clique.begin(), clique.end()))
            sort(clique.begin(), clique.end());
        std::lock_guard<std::mutex> lk(mtx_);
        size_t idx = entries_.size();
        entries_.push_back({move(clique), true});
        raw_count_++;
        for (int v : entries_.back().nodes)
            inverted_[v].push_back(idx);
    }

    void post_validate()
    {
        printf("\nStage4: post-validation (exact dedup + subset check)...\n");

        sort(entries_.begin(), entries_.end(),
             [](const Entry &a, const Entry &b)
             { return a.nodes < b.nodes; });
        auto last = unique(entries_.begin(), entries_.end(),
                           [](const Entry &a, const Entry &b)
                           { return a.nodes == b.nodes; });
        entries_.erase(last, entries_.end());
        size_t after_exact = entries_.size();
        printf("  after exact dedup: %zu\n", after_exact);

        sort(entries_.begin(), entries_.end(),
             [](const Entry &a, const Entry &b){ return a.nodes.size() > b.nodes.size(); });

        unordered_map<int, vector<size_t>> inverted;
        for (size_t i = 0; i < entries_.size(); i++)
            for (int v : entries_[i].nodes)
                inverted[v].push_back(i);

        size_t subset_pruned = 0;
        for (size_t ci = 0; ci < entries_.size(); ci++) {
            if (!entries_[ci].valid) continue;
            const vector<int>& C = entries_[ci].nodes;
            int csz = (int)C.size();

            int rarest = C[0];
            size_t min_len = inverted.count(C[0]) ? inverted[C[0]].size() : 0;
            for (int v : C) {
                auto it = inverted.find(v);
                size_t len = it != inverted.end() ? it->second.size() : 0;
                if (len < min_len) { min_len = len; rarest = v; }
            }

            auto it = inverted.find(rarest);
            if (it != inverted.end()) {
                for (size_t ti : it->second) {
                    if (!entries_[ti].valid) continue;
                    const vector<int>& T = entries_[ti].nodes;
                    if ((int)T.size() <= csz) continue;
                    bool subset = true;
                    size_t ci2 = 0, ti2 = 0;
                    while (ci2 < C.size() && ti2 < T.size()) {
                        if      (C[ci2] == T[ti2]) { ci2++; ti2++; }
                        else if (C[ci2] >  T[ti2]) { ti2++; }
                        else { subset = false; break; }
                    }
                    if (ci2 < C.size()) subset = false;
                    if (subset) { entries_[ci].valid = false; subset_pruned++; break; }
                }
            }
        }

        printf("  subset check pruned: %zu\n", subset_pruned);
    }

    void write_results(FILE *fp) const
    {
        if (!fp)
            return;
        for (const auto &e : entries_)
        {
            if (!e.valid)
                continue;
            for (size_t i = 0; i < e.nodes.size(); i++)
            {
                if (i)
                    fputc(' ', fp);
                fprintf(fp, "%d", e.nodes[i]);
            }
            fputc('\n', fp);
        }
    }

    size_t raw_count() const { return raw_count_; }
    size_t valid_count() const { return raw_count_ - pruned_count_; }
    size_t pruned_count() const { return pruned_count_; }

    template <typename F>
    void foreach_valid(F &&f) const
    {
        for (const auto &e : entries_)
            if (e.valid)
                f(e.nodes);
    }

private:
    static bool is_subset_sorted(const vector<int> &A, const vector<int> &B)
    {
        size_t i = 0, j = 0;
        while (i < A.size() && j < B.size())
        {
            if (A[i] == B[j])
            {
                i++;
                j++;
            }
            else if (A[i] > B[j])
            {
                j++;
            }
            else
                return false;
        }
        return i == A.size();
    }
};

// ============================================================
// AvgSimMCEMiner
// 平均相似度约束的极大团枚举器。
// ============================================================
class AvgSimMCEMiner
{
private:
    const VectorDB &db_;
    const Graph &g_;
    float tau_;

    // nb_[u]：u 的所有邻居及对应相似度，按邻居 id 有序
    vector<vector<pair<int, float>>> nb_;

    // 节点 hash，用于团的 XOR rolling hash（两组独立函数）
    vector<size_t> node_hash_;
    vector<size_t> node_hash2_;

    BlockedSparseBitmap *bitmap_ = nullptr;
    unique_ptr<CliqueStore> store_;

    std::atomic<long long> clique_count_{0};
    static constexpr int MAX_CLIQUE_SZ = 512;
    std::atomic<long long> size_hist_[MAX_CLIQUE_SZ]{};

    // --------------------------------------------------------
    // 运行时统计
    // --------------------------------------------------------
    struct Stats
    {
        std::atomic<uint64_t> recur_calls{0};
        std::atomic<uint64_t> mono_prune{0};
        std::atomic<uint64_t> mono_prune_emit_blocked{0};
        std::atomic<uint64_t> w_prune{0};
        std::atomic<uint64_t> visited_hits{0};
        std::atomic<uint64_t> sim_lookups{0};
        std::atomic<uint64_t> acc_updates{0};
        std::atomic<uint64_t> fhs_insert{0};
        std::atomic<uint64_t> fhs_lookup{0};
        std::atomic<uint64_t> fhs_probe{0};

        void reset()
        {
            recur_calls = 0;
            mono_prune = 0;
            mono_prune_emit_blocked = 0;
            w_prune = 0;
            visited_hits = 0;
            sim_lookups = 0;
            acc_updates = 0;
            fhs_insert = 0;
            fhs_lookup = 0;
            fhs_probe = 0;
        }
        void print(float tau) const
        {
            printf("\n+----------------------------------------------------+\n");
            printf("|  AvgSim MCE  tau=%.3f                             |\n", tau);
            printf("+----------------------------------------------------+\n");
            printf("|  Recursion calls          : %12llu             |\n",
                   (unsigned long long)recur_calls.load());
            printf("|  [MONO] path prunes       : %12llu             |\n",
                   (unsigned long long)mono_prune.load());
            printf("|  [MONO] emit blocked      : %12llu             |\n",
                   (unsigned long long)mono_prune_emit_blocked.load());
            printf("|  W<=0 prunes              : %12llu             |\n",
                   (unsigned long long)w_prune.load());
            printf("|  visited hits (per-thd)   : %12llu             |\n",
                   (unsigned long long)visited_hits.load());
            printf("|  sim lookups              : %12llu             |\n",
                   (unsigned long long)sim_lookups.load());
            printf("|  acc updates              : %12llu             |\n",
                   (unsigned long long)acc_updates.load());
            printf("+-- FlatHashSet64 probe stats ------------------------+\n");
            {
                uint64_t ins = fhs_insert.load();
                uint64_t lk = fhs_lookup.load();
                uint64_t pr = fhs_probe.load();
                printf("|  inserts                  : %12llu             |\n",
                       (unsigned long long)ins);
                printf("|  lookups                  : %12llu             |\n",
                       (unsigned long long)lk);
                printf("|  avg probe length         : %12.3f             |\n",
                       (ins + lk) > 0 ? (double)pr / (ins + lk) : 0.0);
            }
            printf("+----------------------------------------------------+\n");
        }
    } mutable stats_;

    FILE *sink_ = nullptr;

    // --------------------------------------------------------
    // 内部工具
    // --------------------------------------------------------
    inline void emit_candidate(const vector<int> &C)
    {
        int sz = (int)C.size();
        if (sz <= 1)
            return;
        clique_count_.fetch_add(1, std::memory_order_relaxed);
        if (sz < MAX_CLIQUE_SZ)
            size_hist_[sz - 1].fetch_add(1, std::memory_order_relaxed);
        store_->insert(C);
    }

    void build_node_hashes()
    {
        int n = g_.V;
        node_hash_.resize(n);
        node_hash2_.resize(n);
        for (int v = 0; v < n; v++)
        {
            size_t x = (size_t)(v + 1) * 2654435761ULL;
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53ULL;
            x ^= x >> 33;
            node_hash_[v] = x ? x : 1ULL;

            size_t y = (size_t)(v + 1) * 11400714819323198485ULL;
            y ^= y >> 30;
            y *= 0xbf58476d1ce4e5b9ULL;
            y ^= y >> 27;
            y *= 0x94d049bb133111ebULL;
            y ^= y >> 31;
            node_hash2_[v] = y ? y : 1ULL;
        }
    }

    // nb_[u] 有序，二分查找 sim(u, v)
    inline float lookup_sim(int u, int v) const
    {
        stats_.sim_lookups.fetch_add(1, std::memory_order_relaxed);
        const auto &nb = nb_[u];
        int lo = 0, hi = (int)nb.size() - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            if (nb[mid].first == v)
                return nb[mid].second;
            nb[mid].first < v ? lo = mid + 1 : hi = mid - 1;
        }
        return 0.0f;
    }

    inline size_t hash_add(size_t h, int v) const { return h ^ node_hash_[v]; }

    // --------------------------------------------------------
    // AccStore：generation-stamp 懒初始化的累加器
    // acc[w] = sum_{c in C} sim(w, c)，随团 C 的扩展/回溯增减
    // --------------------------------------------------------
    struct AccStore
    {
        vector<float> val;
        vector<uint32_t> gen;
        uint32_t cur = 0;
        void init(int n)
        {
            val.assign(n, 0.f);
            gen.assign(n, 0);
            cur = 1;
        }
        inline float get(int w) const { return gen[w] == cur ? val[w] : 0.f; }
        inline void add(int w, float s)
        {
            if (gen[w] != cur)
            {
                val[w] = 0.f;
                gen[w] = cur;
            }
            val[w] += s;
        }
        inline void sub(int w, float s)
        {
            if (gen[w] == cur)
                val[w] -= s;
        }
    };

    // --------------------------------------------------------
    // search() — 递归枚举满足平均相似度约束的极大团
    //
    // C        : 当前团（不含 cands 中的节点）
    // sim_sum  : C 内所有边的相似度之和
    // cur_avg  : sim_sum / |edges(C)|，用于 monotone 剪枝
    // cur_hash : C 的 XOR rolling hash，用于 visited 去重
    // cands    : 候选扩展节点（均与 C 中每个节点相邻）
    // acc      : acc[w] = sum_{c in C} sim(w, c)
    // visited  : 本线程已访问的团 hash 集合
    // scratch  : per-depth 复用缓冲区
    // depth    : 当前递归深度，用于索引 scratch
    // --------------------------------------------------------
    void search(
        vector<int> &C,
        float sim_sum,
        float cur_avg,
        size_t cur_hash,
        vector<int> &cands,
        AccStore &acc,
        FlatHashSet64 &visited,
        ScratchBuffers &scratch,
        int depth)
    {
        stats_.recur_calls.fetch_add(1, std::memory_order_relaxed);

        if (!visited.insert(cur_hash))
        {
            stats_.visited_hits.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        int k = (int)C.size();
        float W = sim_sum - tau_ * (float)((long long)k * (k - 1) / 2);

        int new_k = k + 1;
        long long new_edges = (long long)new_k * (new_k - 1) / 2;
        float tau_k = tau_ * (float)k;
        // mono_limit：delta_v > mono_limit 意味着加入 v 后平均相似度超过
        // cur_avg，后续候选（delta 更大）同理，可以 break
        float mono_limit = cur_avg * (float)new_edges - sim_sum;

        // 按 acc[v] 升序：W 剪枝尽早触发，monotone 剪枝尽早 break
        sort(cands.begin(), cands.end(), [&](int a, int b)
             { return acc.get(a) < acc.get(b); });

        bool any_extended = false;
        bool has_valid_extension = false;

        for (int v : cands)
        {
            float delta_v = acc.get(v);

            // W 剪枝：加入 v 后仍无法满足 avg > tau
            if (W + delta_v - tau_k < 0.0f)
            {
                stats_.w_prune.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            has_valid_extension = true;

            // Monotone 剪枝：delta_v 过大说明 v 会拉高平均值，后续同理
            if (k >= 2 && delta_v > mono_limit)
            {
                stats_.mono_prune.fetch_add(1, std::memory_order_relaxed);
                break;
            }

            float new_sim_sum = sim_sum + delta_v;

            // [OPT-A] cands_new 从 scratch 取，不分配堆内存
            vector<int> &cands_new = scratch.cands_at(depth);
            cands_new.reserve(cands.size());
            const auto &vrow = bitmap_->get_row(v);
            for (int u : cands)
                if (u != v && vrow.contains(u))
                    cands_new.push_back(u);

            size_t child_hash = hash_add(cur_hash, v);
            if (visited.count(child_hash))
            {
                stats_.visited_hits.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            any_extended = true;

            // [OPT-A] undo_list 从 scratch 取，不分配堆内存
            vector<pair<int, float>> &undo_list = scratch.undo_at(depth);
            undo_list.reserve(cands_new.size());
            for (int w : cands_new)
            {
                float s = lookup_sim(w, v);
                acc.add(w, s);
                undo_list.push_back({w, s});
                stats_.acc_updates.fetch_add(1, std::memory_order_relaxed);
            }

            float new_avg = new_edges > 0
                                ? new_sim_sum / (float)new_edges
                                : 0.0f;

            C.push_back(v);
            search(C, new_sim_sum, new_avg, child_hash,
                   cands_new, acc, visited, scratch, depth + 1);
            C.pop_back();

            for (auto &[w, s] : undo_list)
                acc.sub(w, s);
        }

        if (!any_extended && !has_valid_extension && k >= 2 && W > 0.0f)
        {
            emit_candidate(C);
        }
        else if (!any_extended && has_valid_extension)
        {
            stats_.mono_prune_emit_blocked.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // --------------------------------------------------------
    // 建图：计算并缓存所有边的相似度，填充 nb_
    // --------------------------------------------------------
    void build_graph()
    {
        int n = g_.V;
        printf("Stage1: computing and caching all edge similarities...\n");
        auto t0 = chrono::high_resolution_clock::now();

        vector<vector<pair<int, float>>> half_adj(n);
        long long total_edges = 0;
        long long edges_above_tau = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+ : total_edges, edges_above_tau)
        for (int u = 0; u < n; u++)
        {
            int nu = db_.reorder.old_to_new[u];
            for (int v : g_.adj[u])
            {
                if (v <= u)
                    continue;
                int nv = db_.reorder.old_to_new[v];
                float s = db_.vecs.dot(nu, nv);
                half_adj[u].push_back({v, s});
                total_edges++;
                if (s > tau_)
                    edges_above_tau++;
            }
        }
        printf("  [Stage1] edges cached: %lld, edges with sim > %.3f: %lld\n", (long long)total_edges, (double)tau_, edges_above_tau);

        nb_.assign(n, {});
        for (int u = 0; u < n; u++)
            for (auto &[v, s] : half_adj[u])
            {
                nb_[u].push_back({v, s});
                nb_[v].push_back({u, s});
            }
        half_adj.clear();

#pragma omp parallel for schedule(dynamic, 200)
        for (int u = 0; u < n; u++)
            sort(nb_[u].begin(), nb_[u].end());

        printf("  [Stage1 total]: %lld ms\n",
               (long long)chrono::duration_cast<chrono::milliseconds>(
                   chrono::high_resolution_clock::now() - t0)
                   .count());
    }

    // --------------------------------------------------------
    // 主搜索循环：对每个根节点 v 启动 search()
    // --------------------------------------------------------
    void run_search()
    {
        printf("Stage3: MCE search...\n");
        printf("  tau=%.3f  threads=%d  timeout=600s\n", tau_, omp_get_max_threads());
        auto t0 = chrono::high_resolution_clock::now();
        int n = g_.V;

        int nthreads = omp_get_max_threads();
        vector<AccStore> stores(nthreads);
        vector<FlatHashSet64> per_thread_visited(nthreads);
        vector<ScratchBuffers> per_thread_scratch(nthreads);
        for (auto &s : stores)
            s.init(n);

        auto last_print = chrono::high_resolution_clock::now();
        std::atomic<int> progress_i{0};
        std::atomic<bool> timed_out{false};
        std::mutex print_mutex;

#pragma omp parallel for schedule(dynamic, 1)
        for (int v = 0; v < n; v++)
        {
            if (timed_out.load()) continue;

            auto now = chrono::high_resolution_clock::now();
            if (chrono::duration_cast<chrono::seconds>(now - t0).count() >= 600) {
                timed_out.store(true);
                continue;
            }

            if (nb_[v].empty())
                continue;

            int tid = omp_get_thread_num();
            AccStore &acc = stores[tid];
            FlatHashSet64 &visited = per_thread_visited[tid];
            ScratchBuffers &scratch = per_thread_scratch[tid];

            acc.cur++;
            if (acc.cur == 0)
            {
                fill(acc.gen.begin(), acc.gen.end(), 0);
                acc.cur = 1;
            }
            visited.clear();

            vector<int> cands;
            cands.reserve(nb_[v].size());
            for (auto &[u, s] : nb_[v])
            {
                cands.push_back(u);
                acc.add(u, s);
            }

            vector<int> C = {v};
            size_t init_hash = hash_add(0ULL, v);
            search(C, 0.0f, 1.0f, init_hash, cands, acc, visited, scratch, 0);

            int cur_i = progress_i.fetch_add(1, std::memory_order_relaxed) + 1;
            if (chrono::duration_cast<chrono::seconds>(now - last_print).count() >= 5)
            {
                std::lock_guard<std::mutex> lk(print_mutex);
                if (chrono::duration_cast<chrono::seconds>(
                        chrono::high_resolution_clock::now() - last_print).count() >= 5)
                {
                    long long cand_so_far = clique_count_.load();
                    printf("  progress %d/%d (%.1f%%)  candidates=%lld"
                           "  cur_clique_size=%d  elapsed=%llds\n",
                           cur_i, n, 100.0 * cur_i / n,
                           cand_so_far,
                           (int)C.size(),
                           (long long)chrono::duration_cast<chrono::seconds>(
                               now - t0).count());
                    last_print = now;
                }
            }
        }

        if (timed_out.load()) {
            printf("  [TIMEOUT] reached 600 seconds, stopping search\n");
        }

        for (auto &fhs : per_thread_visited)
        {
            stats_.fhs_insert.fetch_add(fhs.total_insert, std::memory_order_relaxed);
            stats_.fhs_lookup.fetch_add(fhs.total_lookup, std::memory_order_relaxed);
            stats_.fhs_probe.fetch_add(fhs.total_probe, std::memory_order_relaxed);
        }

        printf("  Stage3: %lld ms  candidates=%lld  recursions=%lld\n",
               (long long)chrono::duration_cast<chrono::milliseconds>(
                   chrono::high_resolution_clock::now() - t0)
                   .count(),
               (long long)clique_count_.load(),
               (long long)stats_.recur_calls.load());
    }

public:
    AvgSimMCEMiner(const VectorDB &db, const Graph &g)
        : db_(db), g_(g), tau_(0.f)
    {
        for (auto &h : size_hist_)
            h.store(0);
    }
    ~AvgSimMCEMiner()
    {
        delete bitmap_;
        if (sink_)
            fclose(sink_);
    }

    void set_sink(const char *fn)
    {
        if (sink_)
        {
            fclose(sink_);
            sink_ = nullptr;
        }
        if (fn && fn[0])
            sink_ = fopen(fn, "w");
    }

    long long mine(double tau)
    {
        tau_ = (float)tau;
        clique_count_.store(0);
        for (auto &h : size_hist_)
            h.store(0);
        stats_.reset();
        delete bitmap_;
        bitmap_ = nullptr;
        store_ = make_unique<CliqueStore>();

        printf("\n=== AvgSim MCE  tau=%.3f ===\n", tau);
        auto t0 = chrono::high_resolution_clock::now();

        build_graph();
        build_node_hashes();

        int n = g_.V;
        bitmap_ = new BlockedSparseBitmap(n);
#pragma omp parallel for schedule(static)
        for (int u = 0; u < n; u++)
        {
            vector<int> ids;
            ids.reserve(nb_[u].size());
            for (auto &[v, s] : nb_[u])
                ids.push_back(v);
            bitmap_->build(u, ids);
        }

        auto t_after_build = chrono::high_resolution_clock::now();
        run_search();
        auto t_after_search = chrono::high_resolution_clock::now();

        printf("Stage4: Post-validation...\n");
        store_->post_validate();
        auto t_after_postval = chrono::high_resolution_clock::now();

        long long final_count = (long long)store_->valid_count();

        printf("=== Total: %lld ms, %lld maximal cliques ===\n",
               (long long)chrono::duration_cast<chrono::milliseconds>(
                   t_after_postval - t0)
                   .count(),
               final_count);
        printf("  Breakdown:\n");
        printf("    Graph build              : %lld ms\n",
               (long long)chrono::duration_cast<chrono::milliseconds>(
                   t_after_build - t0)
                   .count());
        printf("    Search                   : %lld ms\n",
               (long long)chrono::duration_cast<chrono::milliseconds>(
                   t_after_search - t_after_build)
                   .count());
        printf("    Post-validation          : %lld ms\n",
               (long long)chrono::duration_cast<chrono::milliseconds>(
                   t_after_postval - t_after_search)
                   .count());
        printf("    (raw candidates: %zu, pruned: %zu)\n",
               store_->raw_count(), store_->pruned_count());

        stats_.print(tau_);

        if (sink_)
        {
            store_->write_results(sink_);
            fflush(sink_);
        }

        printf("\n--- Clique size distribution ---\n");
        std::map<int, long long> sz_dist;
        store_->foreach_valid([&](const vector<int> &c)
                              { sz_dist[(int)c.size()]++; });
        long long sum_sizes = 0;
        for (auto &[sz, cnt] : sz_dist)
        {
            printf("  size %3d: %lld (%.2f%%)\n",
                   sz, (long long)cnt,
                   final_count > 0 ? 100.0 * cnt / final_count : 0.0);
            sum_sizes += (long long)sz * cnt;
        }
        if (final_count > 0)
            printf("  Average clique size: %.2f\n", (double)sum_sizes / final_count);
        printf("--------------------------------------------------\n");

        return final_count;
    }
};

// ============================================================
// main
// ============================================================
int main(int argc, char *argv[])
{
    omp_set_num_threads(omp_get_max_threads());

    string graph_file, vec_index;
    int dataset_id = 2;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "dataset" || arg == "dataset") && i + 1 < argc) {
            dataset_id = atoi(argv[i + 1]);
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
        case 8: graph_file = "dataset/dataset/WN18RR_edges.txt"; vec_index = "dataset/vectors/WN18RR_vectors.index"; break;
        case 9: graph_file = "dataset/dataset/dblp_coauthor.txt"; vec_index = "dataset/vectors/dblp_coauthor_vectors.index"; break;
        case 10: graph_file = "dataset/dataset/products.txt"; vec_index = "dataset/vectors/products_vectors.index"; break;
        default: graph_file = "dataset/dataset/sc-nasasrb.txt"; vec_index = "dataset/vectors-256/sc-nasasrb_vectors.bin"; break;
    }

    cout << "============================================\n";
    cout << "  ALG6 (monotone-live-search)\n";
    cout << "  mine <tau>\n";
    cout << "============================================\n";
    cout << "Dataset: " << graph_file << " (ID=" << dataset_id << ")\n";

    auto t_load = chrono::high_resolution_clock::now();
    Graph g(graph_file);
    if (g.V == 0)
    {
        cerr << "Failed to load graph\n";
        return 1;
    }
    if (fs::exists(vec_index))
    {
        bool vec_loaded;
        if (vec_index.find(".index") != string::npos) {
            vec_loaded = g.loadVectorsFromChunks(vec_index);
        } else {
            vec_loaded = g.loadVectorsFromBinary(vec_index);
        }
        if (!vec_loaded) {
            cerr << "Failed to load vectors from: " << vec_index << endl;
            return 1;
        }
    }
    else
    {
        cerr << "Vector file not found: " << vec_index << endl;
        return 1;
    }
    g.normalizeVectors();

    long long ec = 0;
    for (int i = 0; i < g.V; i++)
        ec += (long long)g.adj[i].size();
    ec /= 2;
    cout << "Graph: V=" << g.V << "  E=" << ec << "\n";
    cout << "Load: "
         << chrono::duration_cast<chrono::milliseconds>(
                chrono::high_resolution_clock::now() - t_load)
                .count()
         << " ms\n\n";

    VectorDB db;
    db.build(g);

    AvgSimMCEMiner miner(db, g);

    cout << "\n>> ";
    string line;
    while (getline(cin, line))
    {
        if (line.empty())
        {
            cout << ">> ";
            continue;
        }
        istringstream iss(line);
        string cmd;
        iss >> cmd;
        if (cmd == "mine")
        {
            double tau = 0.2;
            iss >> tau;
            miner.mine(tau);
        }
        else if (cmd == "sink")
        {
            string fn;
            iss >> fn;
            miner.set_sink(fn.c_str());
            cout << "Sink set to: " << fn << "\n";
        }
        else if (cmd == "quit" || cmd == "exit")
        {
            break;
        }
        else
        {
            cout << "Commands: mine <tau> | sink <file> | quit\n";
        }
        cout << ">> ";
    }
    cout << "Bye.\n";
    return 0;
}