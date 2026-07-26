#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace std;

int main() {
    mt19937 rng(20260723);
    uniform_int_distribution<int> weight_dist(-20, 20);
    uint64_t cases = 0;

    for (int n = 2; n <= 10; ++n) {
        for (int trial = 0; trial < 5000; ++trial) {
            vector<vector<int>> weight(n, vector<int>(n, 0));
            for (int u = 0; u < n; ++u) {
                for (int v = u + 1; v < n; ++v) {
                    weight[u][v] = weight[v][u] = weight_dist(rng);
                }
            }

            vector<vector<int64_t>> incident_prefix(n, vector<int64_t>(n, 0));
            for (int v = 0; v < n; ++v) {
                vector<int> incident;
                for (int u = 0; u < n; ++u)
                    if (u != v) incident.push_back(weight[v][u]);
                sort(incident.rbegin(), incident.rend());
                for (int r = 1; r < n; ++r)
                    incident_prefix[v][r] =
                        incident_prefix[v][r - 1] + incident[r - 1];
            }

            for (int k = 2; k <= n; ++k) {
                int64_t exact_best = numeric_limits<int64_t>::min();
                for (uint64_t mask = 0; mask < (1ULL << n); ++mask) {
                    if (__builtin_popcountll(mask) != k) continue;
                    int64_t sum = 0;
                    for (int u = 0; u < n; ++u)
                        if (mask & (1ULL << u))
                            for (int v = u + 1; v < n; ++v)
                                if (mask & (1ULL << v)) sum += weight[u][v];
                    exact_best = max(exact_best, sum);
                }

                vector<int64_t> caps(n);
                for (int v = 0; v < n; ++v)
                    caps[v] = incident_prefix[v][k - 1];
                sort(caps.rbegin(), caps.rend());
                int64_t cap_sum = 0;
                for (int i = 0; i < k; ++i) cap_sum += caps[i];

                // Twice the exact subset edge sum equals the sum of its
                // member incident sums, each bounded by the corresponding
                // top-(k-1) incident cap.
                if (2 * exact_best > cap_sum) {
                    cerr << "Counterexample n=" << n << " k=" << k
                         << " best=" << exact_best
                         << " cap_sum=" << cap_sum << '\n';
                    return 1;
                }
                ++cases;
            }
        }
    }

    cout << "PASS: " << cases
         << " exact size-layer bounds; no underestimated optimum.\n";
    return 0;
}
