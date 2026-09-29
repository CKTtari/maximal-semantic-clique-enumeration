// MSC framework baseline.
//
// This variant keeps the structural BK host decomposition, canonical-parent
// ownership, and the exact feasibility/maximality logic used by the unified
// candidate.  It disables the unified candidate's optional frontier, root,
// monotone-break, and host-envelope pruning so that the ablation isolates the
// contribution of those search reductions.  The bounded direct kernel remains
// an exact base case rather than a threshold-dependent algorithm route.
#define ACMSC_ALG3_HOST_CANONICAL 1
#define ACMSC_ALG3_HOST_PARALLEL 1
#define ACMSC_ALG3_HOST_CANONICAL_MASK 1
#define ACMSC_ALG3_HOST_CANONICAL_OWNERSHIP 1
#define ACMSC_ALG3_HOST_FRONTIER_BOUND 0
#define ACMSC_ALG3_SUPPORT_ROOT_CERT 0
#define ACMSC_ALG3_MONOTONE_BREAK 0
#define ACMSC_HOST_EDGE_ENVELOPE_LIMIT 0
#define ACMSC_HOST_VERTEX_DEPTH_BOUND 0
#define ACMSC_HOST_EXACT_FRONTIER_LIMIT 0
#define ACMSC_HOST_VERTEX_FRONTIER_LIMIT 0
#define ACMSC_HOST_TOPEDGE_FRONTIER_LIMIT 0
#define ACMSC_SUPPORT_EDGE_OWNERSHIP 0
#define ACMSC_HOST_DIRECT_LIMIT 5
#include "alg3-raw.cpp"
