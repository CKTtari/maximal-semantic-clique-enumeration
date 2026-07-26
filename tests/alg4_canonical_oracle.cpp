#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <vector>

using namespace std;

struct Instance {
    int n;
    int tau;
    vector<uint64_t> adjacency;
    vector<vector<int>> weight;
};

int edge_count(uint64_t mask) {
    const int k = __builtin_popcountll(mask);
    return k * (k - 1) / 2;
}

bool is_clique(const Instance& instance, uint64_t mask) {
    for (int u = 0; u < instance.n; ++u) {
        if (!(mask & (1ULL << u))) continue;
        const uint64_t others = mask & ~(1ULL << u);
        if (others & ~instance.adjacency[u]) return false;
    }
    return true;
}

int similarity_sum(const Instance& instance, uint64_t mask) {
    int sum = 0;
    for (int u = 0; u < instance.n; ++u)
        if (mask & (1ULL << u))
            for (int v = u + 1; v < instance.n; ++v)
                if (mask & (1ULL << v)) sum += instance.weight[u][v];
    return sum;
}

bool feasible(const Instance& instance, uint64_t mask) {
    if (__builtin_popcountll(mask) < 2 || !is_clique(instance, mask)) return false;
    return similarity_sum(instance, mask) >= instance.tau * edge_count(mask);
}

int canonical_deletion(const Instance& instance, uint64_t mask) {
    int best_vertex = -1;
    int best_incident = 0;
    for (int v = 0; v < instance.n; ++v) {
        if (!(mask & (1ULL << v))) continue;
        int incident = 0;
        for (int u = 0; u < instance.n; ++u)
            if (u != v && (mask & (1ULL << u))) incident += instance.weight[u][v];
        if (best_vertex == -1 || incident < best_incident ||
            (incident == best_incident && v < best_vertex)) {
            best_vertex = v;
            best_incident = incident;
        }
    }
    return best_vertex;
}

set<uint64_t> brute_force_feasible(const Instance& instance) {
    set<uint64_t> result;
    for (uint64_t mask = 0; mask < (1ULL << instance.n); ++mask)
        if (feasible(instance, mask)) result.insert(mask);
    return result;
}

set<uint64_t> brute_force_maximal(const set<uint64_t>& all_feasible) {
    set<uint64_t> result;
    for (uint64_t candidate : all_feasible) {
        bool maximal = true;
        for (uint64_t larger : all_feasible) {
            if (candidate != larger && (candidate & larger) == candidate) {
                maximal = false;
                break;
            }
        }
        if (maximal) result.insert(candidate);
    }
    return result;
}

void canonical_dfs(const Instance& instance,
                   uint64_t mask,
                   set<uint64_t>& states,
                   set<uint64_t>& terminals,
                   bool& duplicate) {
    if (!states.insert(mask).second) duplicate = true;

    bool has_valid_extension = false;
    for (int v = 0; v < instance.n; ++v) {
        if (mask & (1ULL << v)) continue;
        const uint64_t child = mask | (1ULL << v);
        if (!feasible(instance, child)) continue;
        has_valid_extension = true;
        if (canonical_deletion(instance, child) == v)
            canonical_dfs(instance, child, states, terminals, duplicate);
    }
    if (!has_valid_extension) terminals.insert(mask);
}

set<uint64_t> canonical_maximal(const Instance& instance,
                                set<uint64_t>& states,
                                bool& duplicate) {
    set<uint64_t> terminals;
    for (int u = 0; u < instance.n; ++u) {
        for (int v = u + 1; v < instance.n; ++v) {
            const uint64_t edge = (1ULL << u) | (1ULL << v);
            if (feasible(instance, edge))
                canonical_dfs(instance, edge, states, terminals, duplicate);
        }
    }

    set<uint64_t> result;
    for (uint64_t candidate : terminals) {
        bool contained = false;
        for (uint64_t larger : terminals) {
            if (candidate != larger && (candidate & larger) == candidate) {
                contained = true;
                break;
            }
        }
        if (!contained) result.insert(candidate);
    }
    return result;
}

void verify(const Instance& instance) {
    const set<uint64_t> expected_states = brute_force_feasible(instance);
    const set<uint64_t> expected_maximal = brute_force_maximal(expected_states);
    set<uint64_t> actual_states;
    bool duplicate = false;
    const set<uint64_t> actual_maximal =
        canonical_maximal(instance, actual_states, duplicate);

    if (duplicate || expected_states != actual_states ||
        expected_maximal != actual_maximal) {
        cerr << "Counterexample: n=" << instance.n << " tau=" << instance.tau
             << " duplicate=" << duplicate << '\n';
        cerr << "feasible expected/actual: " << expected_states.size() << '/'
             << actual_states.size() << '\n';
        cerr << "maximal expected/actual: " << expected_maximal.size() << '/'
             << actual_maximal.size() << '\n';
        exit(1);
    }
}

Instance random_instance(mt19937& rng, int n) {
    uniform_int_distribution<int> edge_probability(0, 99);
    uniform_int_distribution<int> similarity(-5, 10);
    uniform_int_distribution<int> threshold(-2, 9);
    Instance instance{n, threshold(rng), vector<uint64_t>(n, 0),
                      vector<vector<int>>(n, vector<int>(n, 0))};
    for (int u = 0; u < n; ++u) {
        for (int v = u + 1; v < n; ++v) {
            if (edge_probability(rng) < 72) {
                instance.adjacency[u] |= 1ULL << v;
                instance.adjacency[v] |= 1ULL << u;
            }
            const int value = similarity(rng);
            instance.weight[u][v] = value;
            instance.weight[v][u] = value;
        }
    }
    return instance;
}

int main() {
    mt19937 rng(20260723);
    uint64_t cases = 0;
    for (int n = 2; n <= 9; ++n) {
        for (int trial = 0; trial < 5000; ++trial) {
            verify(random_instance(rng, n));
            ++cases;
        }
    }

    // Equal incident sums and threshold-adjacent integer values occur often in
    // the random grid above. This complete graph forces a valid four-vertex
    // recovery although both one-vertex extensions of {0,1} are invalid.
    // It also exercises the inclusive threshold boundary:
    // 6 >= 6, (6+5+5)/3 < 6, and (6+4*5+10)/6 = 6.
    Instance recovery{4, 6, vector<uint64_t>(4, 0),
                      vector<vector<int>>(4, vector<int>(4, 0))};
    for (int u = 0; u < 4; ++u)
        for (int v = u + 1; v < 4; ++v) {
            recovery.adjacency[u] |= 1ULL << v;
            recovery.adjacency[v] |= 1ULL << u;
        }
    const int values[4][4] = {
        {0, 6, 5, 5}, {6, 0, 5, 5}, {5, 5, 0, 10}, {5, 5, 10, 0}};
    for (int u = 0; u < 4; ++u)
        for (int v = 0; v < 4; ++v) recovery.weight[u][v] = values[u][v];
    verify(recovery);
    ++cases;

    cout << "PASS: " << cases
         << " exact-oracle instances; no missing state, duplicate state, or "
            "maximal-output mismatch.\n";
    return 0;
}
