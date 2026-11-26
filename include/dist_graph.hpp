#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>
#include <mpi.h>

struct DistCSR {
    int rank = 0;
    int num_ranks = 1;
    int64_t global_n = 0;              // Total vertices |V|
    int64_t global_m = 0;              // Total undirected edges |E|
    std::vector<int64_t> vtx_dist;     // Size P+1: rank r owns [vtx_dist[r], vtx_dist[r+1])
    std::vector<int64_t> row_ptr;      // Local CSR offsets (size: local_n + 1)
    std::vector<int64_t> adj;          // Sorted global neighbor IDs (local + ghost)
    std::vector<int64_t> local_deg;    // Degrees of local vertices (size: local_n)

    int64_t local_n() const { return vtx_dist[rank + 1] - vtx_dist[rank]; }

    int owner_of(int64_t global_v) const {
        auto it = std::upper_bound(vtx_dist.begin(), vtx_dist.end(), global_v);
        int r = static_cast<int>(it - vtx_dist.begin()) - 1;
        if (r < 0) return 0;
        if (r >= num_ranks) return num_ranks - 1;
        return r;
    }
};

struct RunMetrics {
    uint64_t triangles = 0;
    uint64_t wedges_examined = 0;
    uint64_t bytes_sent = 0;
    uint64_t messages_sent = 0;
    double comm_time_sec = 0.0;
    double mpi_wait_sec = 0.0;
    double total_time_sec = 0.0;
    size_t peak_rss_kb = 0;
};

DistCSR make_mock_k4(MPI_Comm comm);
DistCSR load_and_partition_graph(const std::string& filepath, bool random_relabel, uint64_t seed, MPI_Comm comm);
RunMetrics run_tric_lite(const DistCSR& g, MPI_Comm comm);
RunMetrics run_ditric_step1(const DistCSR& g, bool use_degree_order, MPI_Comm comm);
RunMetrics run_ditric_step2(const DistCSR& g, MPI_Comm comm);
RunMetrics run_ditric_step3(const DistCSR& g, size_t max_buf_bytes, MPI_Comm comm);
