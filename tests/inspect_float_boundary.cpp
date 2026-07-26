#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

static float avx_dot(const vector<float>& a, const vector<float>& b) {
    float result = 0.0f;
    int j = 0;
#ifdef __AVX2__
    __m256 sum = _mm256_setzero_ps();
    for (; j <= static_cast<int>(a.size()) - 8; j += 8)
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(a.data() + j),
                                              _mm256_loadu_ps(b.data() + j)));
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    result = _mm_cvtss_f32(lo);
#endif
    for (; j < static_cast<int>(a.size()); ++j) result += a[j] * b[j];
    return result;
}

static double double_dot(const vector<double>& a, const vector<double>& b) {
    double result = 0.0;
    for (size_t j = 0; j < a.size(); ++j) result += a[j] * b[j];
    return result;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "usage: inspect_float_boundary VECTORS.bin TAU VERTEX...\n";
        return 2;
    }
    const string vector_path = argv[1];
    const float tau = static_cast<float>(stod(argv[2]));
    vector<int> ids;
    for (int i = 3; i < argc; ++i) ids.push_back(stoi(argv[i]));

    ifstream input(vector_path, ios::binary);
    int vertex_count = 0;
    int dimension = 0;
    input.read(reinterpret_cast<char*>(&vertex_count), sizeof(vertex_count));
    input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    if (!input || vertex_count <= 0 || dimension <= 0)
        throw runtime_error("invalid vector header");

    vector<vector<double>> doubles(ids.size(), vector<double>(dimension));
    vector<vector<float>> floats(ids.size(), vector<float>(dimension));
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= vertex_count) throw runtime_error("bad vertex ID");
        const streamoff offset = 2 * sizeof(int) +
                                 static_cast<streamoff>(ids[i]) * dimension * sizeof(double);
        input.seekg(offset);
        input.read(reinterpret_cast<char*>(doubles[i].data()),
                   static_cast<streamsize>(dimension) * sizeof(double));
        double norm2 = 0.0;
        for (double value : doubles[i]) norm2 += value * value;
        const double norm = sqrt(norm2);
        for (int j = 0; j < dimension; ++j) {
            doubles[i][j] /= norm;
            floats[i][j] = static_cast<float>(doubles[i][j]);
        }
    }

    const int n = static_cast<int>(ids.size());
    vector<vector<float>> sim(n, vector<float>(n));
    double high_precision_sum = 0.0;
    cout << setprecision(10) << scientific;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            sim[i][j] = sim[j][i] = avx_dot(floats[i], floats[j]);
            const double precise = double_dot(doubles[i], doubles[j]);
            high_precision_sum += precise;
            cout << ids[i] << " " << ids[j] << " float=" << sim[i][j]
                 << " double=" << precise << "\n";
        }
    }

    float supplied_order_sum = 0.0f;
    for (int added = 1; added < n; ++added)
        for (int prior = 0; prior < added; ++prior)
            supplied_order_sum += sim[added][prior];

    vector<int> permutation(n);
    iota(permutation.begin(), permutation.end(), 0);
    float minimum_sum = numeric_limits<float>::infinity();
    float maximum_sum = -numeric_limits<float>::infinity();
    do {
        float sum = 0.0f;
        for (int added = 1; added < n; ++added)
            for (int prior = 0; prior < added; ++prior)
                sum += sim[permutation[added]][permutation[prior]];
        minimum_sum = min(minimum_sum, sum);
        maximum_sum = max(maximum_sum, sum);
    } while (next_permutation(permutation.begin(), permutation.end()));

    const int edge_count = n * (n - 1) / 2;
    const float threshold_sum = tau * static_cast<float>(edge_count);
    cout << "tau_float=" << tau << " edge_count=" << edge_count << "\n";
    cout << "threshold_sum_float=" << threshold_sum << "\n";
    cout << "supplied_order_sum_float=" << supplied_order_sum
         << " surplus=" << supplied_order_sum - threshold_sum << "\n";
    cout << "permutation_sum_float_min=" << minimum_sum
         << " surplus=" << minimum_sum - threshold_sum << "\n";
    cout << "permutation_sum_float_max=" << maximum_sum
         << " surplus=" << maximum_sum - threshold_sum << "\n";
    cout << "sum_of_float_edges_in_double=";
    double float_edge_sum = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            float_edge_sum += static_cast<double>(sim[i][j]);
    cout << float_edge_sum << " surplus="
         << float_edge_sum - static_cast<double>(tau) * edge_count << "\n";
    cout << "normalized_double_sum=" << high_precision_sum << " surplus="
         << high_precision_sum - static_cast<double>(tau) * edge_count << "\n";
}
