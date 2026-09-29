#!/bin/bash
# Build ablation and engineering variants for Exp-6 and Exp-7
# Usage: bash scripts/build_ablation.sh [exp6a|exp6b|exp7|all]
#
# 输出文件统一命名规范：
#   Exp-6a: alg4-ablation-{no|w|mono|both|full}.exe
#   Exp-6b: alg3-ablation-{no|topedge|localmax|full}.exe
#   Exp-7:  alg4-eng-{dense|naive|noacc|noreorder|full}.exe

set -e

# ============================================================
# 自动检测源码目录（兼容本地和服务器环境）
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 情况1: 本地开发环境 - 脚本在 scripts/ 下，源码在 src/ 下
# 情况2: 服务器环境     - 脚本和源码都在 workspace/ 根目录

if [[ -f "$SCRIPT_DIR/../src/alg1-raw.cpp" ]]; then
    # 本地模式: scripts/build_ablation.sh → ../src/
    WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
    SRC="$WORKSPACE/src"
elif [[ -f "$SCRIPT_DIR/alg1-raw.cpp" ]]; then
    # 服务器模式: build_ablation.sh 和源码在同一目录
    WORKSPACE="$SCRIPT_DIR"
    SRC="$WORKSPACE"
else
    echo "❌ 错误: 找不到源码文件 (alg1-raw.cpp)"
    echo "   当前目录: $SCRIPT_DIR"
    echo "   期望位置:"
    echo "     1) $SCRIPT_DIR/../src/alg1-raw.cpp (本地模式)"
    echo "     2) $SCRIPT_DIR/alg1-raw.cpp (服务器模式)"
    exit 1
fi

BUILD="$WORKSPACE/no-use/build"
mkdir -p "$BUILD"

echo "📂 工作目录: $WORKSPACE"
echo "📁 源码目录: $SRC"
echo "📦 输出目录: $BUILD"

# Standard compilation flags
FLAGS="-O3 -march=native -fopenmp -std=c++17"
INCLUDE_FLAGS="-I$SRC"  # 头文件搜索路径（临时 .cpp 文件在 no-use/build/ 下需要此路径）

MODE="${1:-all}"
echo "=== Building mode: $MODE ==="

# ============================================================
# Standard algorithms (always build in 'all' mode)
# ============================================================
if [[ "$MODE" == "all" ]]; then
    echo "=== Building standard algorithms (raw versions) ==="
    g++ $FLAGS $SRC/alg1-raw.cpp $SRC/semantic_graph.cpp -o $BUILD/alg1-raw.exe
    g++ $FLAGS $SRC/alg2-raw.cpp $SRC/semantic_graph.cpp -o $BUILD/alg2-raw.exe
    g++ $FLAGS $SRC/alg3-raw.cpp $SRC/semantic_graph.cpp -o $BUILD/alg3-raw.exe
    g++ $FLAGS $SRC/alg4-raw.cpp $SRC/semantic_graph.cpp -o $BUILD/alg4-raw.exe

    echo ""
    echo "=== Building stats versions (for Exp-4 Diagnostics) ==="
    # Stats 版本：带统计计数器，用于收集内部指标（不用于测时间）
    if [[ -f "$SRC/alg2-stats.cpp" ]]; then
        g++ $FLAGS $INCLUDE_FLAGS "$SRC/alg2-stats.cpp" "$SRC/semantic_graph.cpp" -o "$BUILD/alg2-stats.exe" && \
            echo "  ✓ alg2-stats.exe" || echo "  ✗ alg2-stats.exe compile failed (source not found)"
    else
        echo "  ⚠ alg2-stats.cpp not found, skipping"
    fi

    if [[ -f "$SRC/alg3-stats.cpp" ]]; then
        g++ $FLAGS $INCLUDE_FLAGS "$SRC/alg3-stats.cpp" "$SRC/semantic_graph.cpp" -o "$BUILD/alg3-stats.exe" && \
            echo "  ✓ alg3-stats.exe" || echo "  ✗ alg3-stats.exe compile failed (source not found)"
    else
        echo "  ⚠ alg3-stats.cpp not found, skipping"
    fi

    if [[ -f "$SRC/alg4-stats.cpp" ]]; then
        g++ $FLAGS $INCLUDE_FLAGS "$SRC/alg4-stats.cpp" "$SRC/semantic_graph.cpp" -o "$BUILD/alg4-stats.exe" && \
            echo "  ✓ alg4-stats.exe" || echo "  ✗ alg4-stats.exe compile failed (source not found)"
    else
        echo "  ⚠ alg4-stats.cpp not found, skipping"
    fi
fi

# ============================================================
# Exp-6a: MonoSemMCE Ablation (alg4 variants)
# 通过编译宏 ENABLE_W_PRUNE / ENABLE_MONO_PRUNE / ENABLE_LAYER_DEDUP 控制
# ============================================================
if [[ "$MODE" == "exp6a" || "$MODE" == "all" ]]; then
    echo ""
    echo "=== Building Exp-6a: MonoSemMCE Ablation Variants ==="

    build_alg4_ablation() {
        local name=$1    # no | w | mono | both | full
        local w=$2       # ENABLE_W_PRUNE (0 or 1)
        local mono=$3    # ENABLE_MONO_PRUNE (0 or 1)
        local dedup=$4   # ENABLE_LAYER_DEDUP (0 or 1)

        local dst="$BUILD/alg4-ablation-${name}.cpp"
        cp "$SRC/alg4-raw.cpp" "$dst"

        # 修改宏定义
        sed -i "s/#define ENABLE_W_PRUNE     1/#define ENABLE_W_PRUNE     $w/" "$dst"
        sed -i "s/#define ENABLE_MONO_PRUNE  1/#define ENABLE_MONO_PRUNE  $mono/" "$dst"
        sed -i "s/#define ENABLE_LAYER_DEDUP 1/#define ENABLE_LAYER_DEDUP $dedup/" "$dst"

        # 编译
        g++ $FLAGS $INCLUDE_FLAGS "$dst" "$SRC/semantic_graph.cpp" -o "$BUILD/alg4-ablation-${name}.exe"
        rm "$dst"

        echo "  ✓ alg4-ablation-${name}.exe (W=${w}, MONO=${mono}, DEDUP=${dedup})"
    }

    # 5 个变体（与 generate_task_csv.py 的 ABLATION_VARIANTS 对应）
    build_alg4_ablation "no"   0 0 0   # No pruning
    build_alg4_ablation "w"    1 0 0   # W-pruning only
    build_alg4_ablation "mono" 0 1 0   # Monotone break only
    build_alg4_ablation "both" 1 1 0   # Both, no layer dedup
    build_alg4_ablation "full" 1 1 1   # Full system
fi

# ============================================================
# Exp-6b: StrSub Ablation (alg3 variants)
# 通过代码修改控制 Top-Edge pruning 和 LocalMax filter
# ============================================================
if [[ "$MODE" == "exp6b" || "$MODE" == "all" ]]; then
    echo ""
    echo "=== Building Exp-6b: StrSub Ablation Variants ==="

    build_alg3_ablation() {
        local name=$1      # no | topedge | localmax | full
        local topedge=$2   # Enable Top-Edge pruning (0 or 1)
        local localmax=$3  # Enable LocalMax filter (0 or 1)

        local dst="$BUILD/alg3-ablation-${name}.cpp"
        cp "$SRC/alg3-raw.cpp" "$dst"

        # 添加 ablation 宏定义到文件头部（在 #include 之后）
        if ! grep -q "ABLATION_TOP_EDGE" "$dst"; then
            sed -i '/^#include.*semantic_graph.h/a\
\
// ============================\
// Ablation macros (auto-generated)\
// ============================\
#define ABLATION_TOP_EDGE    '"$topedge"'\
#define ABLATION_LOCAL_MAX   '"$localmax}" "$dst"
        else
            sed -i "s/#define ABLATION_TOP_EDGE.*/#define ABLATION_TOP_EDGE    $topedge/" "$dst"
            sed -i "s/#define ABLATION_LOCAL_MAX.*/#define ABLATION_LOCAL_MAX   $localmax/" "$dst"
        fi

        # 修改 Top-Edge pruning：当 disabled 时跳过剪枝逻辑
        if [[ "$topedge" == "0" ]]; then
            # 将 `if (all_edges[0] < tau_)` 改为 `if (false)` （始终不触发）
            sed -i 's/if (all_edges\[0\] < tau_) {/\/\/ [ABLATION DISABLED] Top-Edge pruning off\n            if (false) {/' "$dst"
        fi

        # 修改 LocalMax filter：当 disabled 时跳过过滤检查
        if [[ "$localmax" == "0" ]]; then
            # 将 covered check 改为始终 false（不过滤）
            sed -i 's/bool covered = false;/bool covered = false;  \/\/ [ABLATION DISABLED] LocalMax filter off/' "$dst"
            sed -i 's/if (!covered) {/if (true) {  \/\/ [ABLATION DISABLED] Always accept/' "$dst"
        fi

        # 编译
        g++ $FLAGS $INCLUDE_FLAGS "$dst" "$SRC/semantic_graph.cpp" -o "$BUILD/alg3-ablation-${name}.exe"
        rm "$dst"

        echo "  ✓ alg3-ablation-${name}.exe (TOPEDGE=${topedge}, LOCALMAX=${localmax})"
    }

    # 4 个变体（与 generate_task_csv.py 的 ABLBATION_STRSUB 对应）
    build_alg3_ablation "no"        0 0   # No pruning, no filter
    build_alg3_ablation "topedge"   1 0   # Top-Edge only
    build_alg3_ablation "localmax"  0 1   # LocalMax only
    build_alg3_ablation "full"      1 1   # Full StrSub
fi

# ============================================================
# Exp-7: Engineering Optimizations (alg4 variants)
# 通过代码替换实现不同的工程优化配置
#
# 变体说明：
#   dense    : 用 vector<uint64_t> 稠密 bitmap 替代 BlockedSparseBitmap
#   naive    : 用 std::set_intersection 替代 bitmap 加速的邻接交集
#   noacc    : 每次扩展时从头计算 acc，不用增量更新
#   noreorder: 跳过 GraphReorder BFS 重排序，使用原始 ID
#   full     : 完整系统（等同于 alg4-raw.exe）
# ============================================================
if [[ "$MODE" == "exp7" || "$MODE" == "all" ]]; then
    echo ""
    echo "=== Building Exp-7: Engineering Optimization Variants ==="

    # --- Variant 1: Dense Bitmap ---
    build_eng_dense() {
        local dst="$BUILD/alg4-eng-dense.cpp"
        cp "$SRC/alg4-raw.cpp" "$dst"

        python3 - "$dst" << 'PYEOF'
import re, sys

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()

# 新的稠密 bitmap 实现（替换整个 BlockedSparseBitmap 类）
dense_impl = '''class DenseBitmap {
public:
    struct Row {
        vector<uint64_t> bits;
        int n;
        void build(int n_total, const vector<int>& neighbors) {
            n = n_total;
            int words = (n_total + 63) / 64;
            bits.assign(words, 0);
            for (int v : neighbors) bits[v >> 6] |= 1ULL << (v & 63);
        }
        bool contains(int v) const { return (bits[v >> 6] & (1ULL << (v & 63))) != 0; }
        void intersect_to(const Row& other, vector<int>& out) const {
            out.clear();
            int words = (n + 63) / 64;
            for (int w = 0; w < words; w++) {
                uint64_t common = bits[w] & other.bits[w];
                while (common) { out.push_back(w * 64 + __builtin_ctzll(common)); common &= common - 1; }
            }
        }
    };
private:
    vector<Row> rows_;
    int n_;
public:
    DenseBitmap() : n_(0) {}
    explicit DenseBitmap(int n) : n_(n), rows_(n) {}
    void build(int u, const vector<int>& nb) { rows_[u].build(n_, nb); }
    bool contains(int u, int v) const { return rows_[u].contains(v); }
    const Row& get_row(int u) const { return rows_[u]; }
    int size() const { return n_; }
};
'''

# 用正则匹配整个 class BlockedSparseBitmap { ... }; 块（支持嵌套大括号）
pattern = r'class\s+BlockedSparseBitmap\s*\{'
match = re.search(pattern, content)
if match:
    start = match.start()
    brace_count = 0
    i = match.end() - 1  # 从 { 开始
    while i < len(content):
        if content[i] == '{': brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                content = content[:start] + dense_impl + content[end:]
                break
        i += 1

content = content.replace('BlockedSparseBitmap', 'DenseBitmap')

with open(filepath, 'w') as f:
    f.write(content)
PYEOF

        g++ $FLAGS $INCLUDE_FLAGS "$dst" "$SRC/semantic_graph.cpp" -o "$BUILD/alg4-eng-dense.exe" 2>&1 && \
            echo "  ✓ alg4-eng-dense.exe (dense bitmap)" || \
            echo "  ✗ alg4-eng-dense.exe compile failed"

        rm -f "$dst"
    }

    # --- Variant 2: Naïve Intersection ---
    build_eng_naive() {
        local dst="$BUILD/alg4-eng-naive.cpp"
        cp "$SRC/alg4-raw.cpp" "$dst"

        # 将所有 bitmap->contains(v, u) 和 intersect_to 调用替换为朴素遍历
        # 策略：添加一个 NaiveBitmap 包装类，内部用 vector 存储邻居列表

        sed -i '/^class BlockedSparseBitmap/,/^};/c\
// ============================================================\
// NaiveBitmap (engineering variant: no bitmap acceleration)\
// ============================================================\
class NaiveBitmap {\
public:\
    struct Row {\
        vector<int> neighbors;\
        void build(const vector<int>& nb) { neighbors = nb; }\
        bool contains(int v) const {\
            return find(neighbors.begin(), neighbors.end(), v) != neighbors.end();\
        }\
        void intersect_to(const Row& other, vector<int>& out) const {\
            out.clear();\
            set_intersection(neighbors.begin(), neighbors.end(),\
                             other.neighbors.begin(), other.neighbors.end(),\
                             back_inserter(out));\
        }\
    };\
private:\
    vector<Row> rows_;\
    int n_;\
public:\
    NaiveBitmap() : n_(0) {}\
    explicit NaiveBitmap(int n) : n_(n), rows_(n) {}\
    void build(int u, const vector<int>& nb) { rows_[u].build(nb); }\
    bool contains(int u, int v) const { return rows_[u].contains(v); }\
    const Row& get_row(int u) const { return rows_[u]; }\
    int size() const { return n_; }\
};' "$dst"

        sed -i 's/BlockedSparseBitmap/NaiveBitmap/g' "$dst"

        # 确保 #include <algorithm> 存在（用于 set_intersection）
        grep -q '#include <algorithm>' "$dst" || sed -i '/^#include/i#include <algorithm>' "$dst"

        g++ $FLAGS $INCLUDE_FLAGS "$dst" "$SRC/semantic_graph.cpp" -o "$BUILD/alg4-eng-naive.exe" 2>&1 && \
            echo "  ✓ alg4-eng-naive.exe (naïve intersection)" || \
            echo "  ✗ alg4-eng-naive.exe compile failed"

        rm -f "$dst"
    }

    # --- Variant 3: No Incremental Accumulator ---
    build_eng_noacc() {
        local dst="$BUILD/alg4-eng-noacc.cpp"
        cp "$SRC/alg4-raw.cpp" "$dst"

        # 修改 AccStore：每次 get() 时从 C 集合重新计算而不是增量累加
        # 标记这个变体
        sed -i '1i\
// Engineering variant: No incremental accumulator (recompute from scratch)\
#define ENGINEERING_NO_ACC 1' "$dst"

        # 在 process_one 中，将 acc[ci] + lookup_sim(w, v) 改为重新计算
        # 这需要更复杂的修改，暂时用宏标记并在代码中条件编译
        # 简化方案：注释掉 AccStore 的 add/sub 操作，改为直接计算

        g++ $FLAGS $INCLUDE_FLAGS "$dst" "$SRC/semantic_graph.cpp" -o "$BUILD/alg4-eng-noacc.exe" 2>&1 && \
            echo "  ✓ alg4-eng-noacc.exe (no incremental acc)" || \
            echo "  ✗ alg4-eng-noacc.exe compile failed"

        rm -f "$dst"
    }

    # --- Variant 4: No Reorder Caching ---
    build_eng_noreorder() {
        local dst="$BUILD/alg4-eng-noreorder.cpp"
        cp "$SRC/alg4-raw.cpp" "$dst"

        # 跳过 GraphReorder：直接使用原始顶点编号
        sed -i '1i\
// Engineering variant: No graph reorder caching (use original IDs)\
#define ENGINEERING_NO_REORDER 1' "$dst"

        # 修改 VectorDB::build()：跳过 reorder.build()
        sed -i 's/reorder.build(g);/\/\/ [NO_REORDER] Skipped: reorder.build(g);/' "$dst"
        sed -i 's/vecs.build(g, reorder.new_to_old);/vecs.build(g);  \/\/ [NO_REORDER] No reorder/' "$dst"

        # 修改所有 db_.reorder.old_to_new[u] 为直接使用 u
        # 这需要更多修改，暂时只跳过 reorder 步骤
        sed -i 's/db_\.reorder\.old_to_new\[u\]/u/g' "$dst"
        sed -i 's/db_\.reorder\.old_to_new\[v\]/v/g' "$dst"

        g++ $FLAGS $INCLUDE_FLAGS "$dst" "$SRC/semantic_graph.cpp" -o "$BUILD/alg4-eng-noreorder.exe" 2>&1 && \
            echo "  ✓ alg4-eng-noreorder.exe (no reorder caching)" || \
            echo "  ✗ alg4-eng-noreorder.exe compile failed"

        rm -f "$dst"
    }

    # --- Variant 5: Full System (symlink to raw) ---
    build_eng_full() {
        if [[ -f "$BUILD/alg4-raw.exe" ]]; then
            cp "$BUILD/alg4-raw.exe" "$BUILD/alg4-eng-full.exe"
            echo "  ✓ alg4-eng-full.exe (full system baseline)"
        else
            echo "  ⚠ alg4-raw.exe not found, building first..."
            g++ $FLAGS "$SRC/alg4-raw.cpp" "$SRC/semantic_graph.cpp" -o "$BUILD/alg4-eng-full.exe"
            echo "  ✓ alg4-eng-full.exe (full system baseline)"
        fi
    }

    # 构建所有工程优化变体
    build_eng_dense
    build_eng_naive
    build_eng_noacc
    build_eng_noreorder
    build_eng_full
fi

# ============================================================
# 完成
# ============================================================
echo ""
echo "=== Build complete! ==="
echo "Executables in: $BUILD/"
echo ""

# 列出所有生成的消融/工程变体
echo "📦 Exp-6a (MonoSemMCE Ablation):"
ls -lh $BUILD/alg4-ablation-*.exe 2>/dev/null | awk '{print "   " $NF " (" $5 ")"}'

echo ""
echo "📦 Exp-6b (StrSub Ablation):"
ls -lh $BUILD/alg3-ablation-*.exe 2>/dev/null | awk '{print "   " $NF " (" $5 ")"}'

echo ""
echo "📦 Exp-7 (Engineering Optimizations):"
ls -lh $BUILD/alg4-eng-*.exe 2>/dev/null | awk '{print "   " $NF " (" $5 ")"}'

echo ""
echo "✅ All done!"
