#include "test_utils.hpp"
#include <algorithm>
#include <random>
#include <sstream>
#include <stdexcept>

namespace test_utils {

static void clean_adj(std::vector<std::vector<int64_t>>& adj) {
    for (auto& nbrs : adj) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
}

int64_t brute_force_triangles(const std::vector<std::vector<int64_t>>& adj) {
    int64_t n = static_cast<int64_t>(adj.size());
    if (n > 200) {
        throw std::invalid_argument("brute_force_triangles is restricted to |V| <= 200");
    }
    std::vector<std::vector<uint8_t>> mat(n, std::vector<uint8_t>(n, 0));
    for (int64_t u = 0; u < n; ++u) {
        for (int64_t v : adj[u]) {
            if (u != v && v >= 0 && v < n) {
                mat[u][v] = 1;
                mat[v][u] = 1;
            }
        }
    }
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            if (!mat[i][j]) continue;
            for (int64_t k = j + 1; k < n; ++k) {
                if (mat[i][k] && mat[j][k]) {
                    count++;
                }
            }
        }
    }
    return count;
}

std::vector<std::vector<int64_t>> make_complete_graph(int64_t n) {
    std::vector<std::vector<int64_t>> adj(n);
    for (int64_t u = 0; u < n; ++u) {
        for (int64_t v = 0; v < n; ++v) {
            if (u != v) adj[u].push_back(v);
        }
    }
    return adj;
}

std::vector<std::vector<int64_t>> make_star_graph(int64_t n) {
    std::vector<std::vector<int64_t>> adj(n);
    for (int64_t v = 1; v < n; ++v) {
        adj[0].push_back(v);
        adj[v].push_back(0);
    }
    clean_adj(adj);
    return adj;
}

std::vector<std::vector<int64_t>> make_gnp_graph(int64_t n, double p, uint64_t seed) {
    std::vector<std::vector<int64_t>> adj(n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int64_t u = 0; u < n; ++u) {
        for (int64_t v = u + 1; v < n; ++v) {
            if (dist(rng) < p) {
                adj[u].push_back(v);
                adj[v].push_back(u);
            }
        }
    }
    clean_adj(adj);
    return adj;
}

std::vector<std::vector<int64_t>> make_rmat_graph(int scale, int edge_factor, uint64_t seed) {
    int64_t n = int64_t{1} << scale;
    int64_t m = n * edge_factor;
    std::vector<std::vector<int64_t>> adj(n);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double a = 0.57, b = 0.19, c = 0.19; // d = 0.05

    for (int64_t e = 0; e < m; ++e) {
        int64_t u = 0, v = 0;
        int64_t step = n >> 1;
        for (int s = 0; s < scale; ++s) {
            double r = dist(rng);
            if (r < a) {
                // top-left (0, 0)
            } else if (r < a + b) {
                v += step;
            } else if (r < a + b + c) {
                u += step;
            } else {
                u += step;
                v += step;
            }
            step >>= 1;
        }
        if (u != v) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }
    clean_adj(adj);
    return adj;
}

bool is_generator_spec(const std::string& spec) {
    return spec.rfind("GEN:", 0) == 0;
}

std::vector<std::vector<int64_t>> generate_from_spec(const std::string& spec) {
    std::vector<std::string> parts;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() < 3 || parts[0] != "GEN") {
        throw std::invalid_argument("Invalid generator spec: " + spec);
    }
    if (parts[1] == "KN") {
        return make_complete_graph(std::stoll(parts[2]));
    } else if (parts[1] == "STAR") {
        return make_star_graph(std::stoll(parts[2]));
    } else if (parts[1] == "GNP" && parts.size() == 5) {
        return make_gnp_graph(std::stoll(parts[2]), std::stod(parts[3]), std::stoull(parts[4]));
    } else if (parts[1] == "RMAT" && parts.size() == 5) {
        return make_rmat_graph(std::stoi(parts[2]), std::stoi(parts[3]), std::stoull(parts[4]));
    }
    throw std::invalid_argument("Unsupported generator spec: " + spec);
}

} // namespace test_utils
