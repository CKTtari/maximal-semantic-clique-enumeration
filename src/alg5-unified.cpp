// Unified exact search:
// structural BK/pivot hosts feed one semantic search organization.  Each
// host uses the same minimum-contribution canonical-parent rule, the same
// frontier bound, and the same monotone marginal certificate; the fixed
// small-host subset kernel is only the bounded base case of that search, not
// a threshold-dependent route to another algorithm.
#define ACMSC_ALG3_HOST_CANONICAL 1
#define ACMSC_ALG3_HOST_PARALLEL 1
#define ACMSC_ALG3_HOST_CANONICAL_MASK 1
#define ACMSC_ALG3_SUPPORT_ROOT_CERT 1
#define ACMSC_ALG3_MONOTONE_BREAK 1
#define ACMSC_HOST_EDGE_ENVELOPE_LIMIT 0
#define ACMSC_HOST_DIRECT_LIMIT 5
#include "alg3-raw.cpp"
