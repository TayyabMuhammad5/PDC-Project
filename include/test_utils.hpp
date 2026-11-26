#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace test_utils {

// Exact O(V^3) brute-force triangle count on a symmetric, zero-indexed adjacency list (validates V <= 200)
int64_t brute_force_triangles(const std::vector<std::vector<int64_t>>& adj);

// Deterministic in-memory graph generators (returns clean, sorted, zero-indexed symmetric adj lists)
std::vector<std::vector<int64_t>> make_complete_graph(int64_t n);
std::vector<std::vector<int64_t>> make_star_graph(int64_t n);
std::vector<std::vector<int64_t>> make_gnp_graph(int64_t n, double p, uint64_t seed);

// Skewed RMAT generator (a=0.57, b=0.19, c=0.19, d=0.05) for Step 3 B_max memory stress tests
// Note: Do NOT check large RMAT counts against P=1; validate generator logic on small sizes via brute_force_triangles.
std::vector<std::vector<int64_t>> make_rmat_graph(int scale, int edge_factor, uint64_t seed);

// Parse generator specs like:
//   GEN:KN:5
//   GEN:STAR:10
//   GEN:GNP:40:0.25:42
//   GEN:RMAT:12:16:99
bool is_generator_spec(const std::string& spec);
std::vector<std::vector<int64_t>> generate_from_spec(const std::string& spec);

} // namespace test_utils
