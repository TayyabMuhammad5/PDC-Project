#include "dist_graph.hpp"
#include "test_utils.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <random>
#include <numeric>
#include <stdexcept>
#include <algorithm>

static DistCSR build_dist_csr_from_adj(std::vector<std::vector<int64_t>>& full_adj,
                                       bool random_relabel, uint64_t seed, MPI_Comm comm) {
    DistCSR g;
    MPI_Comm_rank(comm, &g.rank);
    MPI_Comm_size(comm, &g.num_ranks);

    int64_t n = static_cast<int64_t>(full_adj.size());
    if (random_relabel && n > 0) {
        std::vector<int64_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937_64 rng(seed);
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<std::vector<int64_t>> relabeled(n);
        for (int64_t u = 0; u < n; ++u) {
            int64_t nu = perm[u];
            relabeled[nu].reserve(full_adj[u].size());
            for (int64_t v : full_adj[u]) {
                relabeled[nu].push_back(perm[v]);
            }
            std::sort(relabeled[nu].begin(), relabeled[nu].end());
        }
        full_adj = std::move(relabeled);
    }

    int64_t total_directed_edges = 0;
    for (int64_t u = 0; u < n; ++u) {
        std::sort(full_adj[u].begin(), full_adj[u].end());
        full_adj[u].erase(std::unique(full_adj[u].begin(), full_adj[u].end()), full_adj[u].end());
        total_directed_edges += static_cast<int64_t>(full_adj[u].size());
    }

    g.global_n = n;
    g.global_m = total_directed_edges / 2;

    // 1D Edge-balanced partitioning via prefix sums over vertex degrees (~2|E|/P edges per rank)
    g.vtx_dist.assign(g.num_ranks + 1, 0);
    int64_t cum_edges = 0;
    int cur_rank = 1;
    for (int64_t u = 0; u < n && cur_rank < g.num_ranks; ++u) {
        cum_edges += static_cast<int64_t>(full_adj[u].size());
        int64_t target = (total_directed_edges * cur_rank) / g.num_ranks;
        if (cum_edges >= target) {
            g.vtx_dist[cur_rank] = u + 1;
            cur_rank++;
        }
    }
    while (cur_rank <= g.num_ranks) {
        g.vtx_dist[cur_rank++] = n;
    }

    int64_t start_v = g.vtx_dist[g.rank];
    int64_t end_v = g.vtx_dist[g.rank + 1];
    g.row_ptr.push_back(0);
    for (int64_t v = start_v; v < end_v; ++v) {
        for (int64_t nb : full_adj[v]) {
            g.adj.push_back(nb);
        }
        g.local_deg.push_back(static_cast<int64_t>(full_adj[v].size()));
        g.row_ptr.push_back(static_cast<int64_t>(g.adj.size()));
    }
    return g;
}

DistCSR make_mock_k4(MPI_Comm comm) {
    std::vector<std::vector<int64_t>> adj = {
        {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}
    };
    return build_dist_csr_from_adj(adj, false, 0, comm);
}

DistCSR load_and_partition_graph(const std::string& filepath, bool random_relabel, uint64_t seed, MPI_Comm comm) {
    if (test_utils::is_generator_spec(filepath)) {
        auto adj = test_utils::generate_from_spec(filepath);
        return build_dist_csr_from_adj(adj, random_relabel, seed, comm);
    }
    if (filepath == "MOCK_K4") {
        std::vector<std::vector<int64_t>> adj = {
            {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}
        };
        return build_dist_csr_from_adj(adj, random_relabel, seed, comm);
    }

    std::ifstream in(filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open graph file: " + filepath);
    }

    std::unordered_map<int64_t, int64_t> id_map;
    id_map.reserve(1 << 18);
    auto get_id = [&](int64_t raw) -> int64_t {
        auto it = id_map.find(raw);
        if (it != id_map.end()) return it->second;
        int64_t next_id = static_cast<int64_t>(id_map.size());
        id_map.emplace(raw, next_id);
        return next_id;
    };

    std::vector<std::pair<int64_t, int64_t>> edges;
    edges.reserve(1 << 20);

    bool is_mtx = false;
    bool mtx_header_skipped = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '%') {
            if (line.find("MatrixMarket") != std::string::npos) is_mtx = true;
            continue;
        }
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        if (is_mtx && !mtx_header_skipped) {
            mtx_header_skipped = true;
            continue;
        }
        int64_t raw_u, raw_v;
        if (!(iss >> raw_u >> raw_v)) continue;
        int64_t u = get_id(raw_u);
        int64_t v = get_id(raw_v);
        if (u == v) continue; // Strip self-loops after registering vertices
        edges.emplace_back(u, v);
        edges.emplace_back(v, u);
    }

    int64_t n = static_cast<int64_t>(id_map.size());
    std::vector<std::vector<int64_t>> full_adj(n);
    for (const auto& e : edges) {
        full_adj[e.first].push_back(e.second);
    }
    edges.clear();
    edges.shrink_to_fit();

    return build_dist_csr_from_adj(full_adj, random_relabel, seed, comm);
}
