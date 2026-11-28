#include "dist_graph.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sys/resource.h>
static size_t get_peak_rss_kb() {
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<size_t>(ru.ru_maxrss);
}
static bool has_edge(const DistCSR &g, int64_t u, int64_t w) {
    int64_t local_u = u - g.vtx_dist[g.rank];
    auto first = g.adj.begin() + g.row_ptr[local_u];
    auto last = g.adj.begin() + g.row_ptr[local_u + 1];
    return std::binary_search(first, last, w);
}
static void check_mpi_count(size_t count, MPI_Comm comm) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "TriC-lite: MPI buffer exceeds INT_MAX\n");
        MPI_Abort(comm, 1);
        std::abort();
    }
}
RunMetrics run_tric_lite(const DistCSR &g, MPI_Comm comm) {
    RunMetrics m;
    const int rank = g.rank;
    const int P = g.num_ranks;
    uint64_t local_hits = 0, local_wedges = 0, local_bytes = 0, local_messages = 0;
    double comm_time = 0.0, wait_time = 0.0;
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    std::vector<std::vector<int64_t>> queries(P);
    for (int64_t v = 0; v < g.local_n(); ++v) {
        int64_t begin = g.row_ptr[v];
        int64_t end = g.row_ptr[v + 1];
        for (int64_t i = begin; i < end; ++i) {
            int64_t u = g.adj[i];
            int64_t k = end - i - 1;
            if (k == 0)
                continue;
            int owner = g.owner_of(u);
            local_wedges += static_cast<uint64_t>(k);
            if (owner == rank) {
                for (int64_t j = i + 1; j < end; ++j) {
                    if (has_edge(g, u, g.adj[j]))
                        ++local_hits;
                }
            } else {
                auto &q = queries[owner];
                q.push_back(u);
                q.push_back(k);
                q.insert(q.end(), g.adj.begin() + i + 1, g.adj.begin() + end);
            }
        }
    }
    std::vector<int> send_counts(P, 0), recv_counts(P, 0), send_displs(P, 0), recv_displs(P, 0);
    size_t total_send = 0, total_recv = 0;
    for (int r = 0; r < P; ++r) {
        check_mpi_count(queries[r].size(), comm);
        send_displs[r] = static_cast<int>(total_send);
        send_counts[r] = static_cast<int>(queries[r].size());
        total_send += queries[r].size();
        check_mpi_count(total_send, comm);
        if (r != rank && send_counts[r] > 0) {
            local_bytes += queries[r].size() * sizeof(int64_t);
            ++local_messages;
        }
    }
    std::vector<int64_t> send_buffer(std::max<size_t>(1, total_send));
    for (int r = 0; r < P; ++r)
        std::copy(queries[r].begin(), queries[r].end(), send_buffer.begin() + send_displs[r]);
    double ts = MPI_Wtime();
    MPI_Barrier(comm);
    wait_time += MPI_Wtime() - ts;
    ts = MPI_Wtime();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    comm_time += MPI_Wtime() - ts;
    for (int r = 0; r < P; ++r) {
        recv_displs[r] = static_cast<int>(total_recv);
        total_recv += static_cast<size_t>(recv_counts[r]);
        check_mpi_count(total_recv, comm);
    }
    std::vector<int64_t> recv_buffer(std::max<size_t>(1, total_recv));
    ts = MPI_Wtime();
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_INT64_T, recv_buffer.data(),
                  recv_counts.data(), recv_displs.data(), MPI_INT64_T, comm);
    comm_time += MPI_Wtime() - ts;
    std::vector<uint64_t> replies(P, 0), received_replies(P, 0);
    for (int src = 0; src < P; ++src) {
        size_t pos = recv_displs[src];
        size_t end = pos + recv_counts[src];
        while (pos < end) {
            if (end - pos < 2) {
                std::fprintf(stderr, "Invalid query header\n");
                MPI_Abort(comm, 2);
                std::abort();
            }
            int64_t u = recv_buffer[pos++];
            int64_t k = recv_buffer[pos++];
            if (k < 0 || static_cast<uint64_t>(k) > end - pos || u < g.vtx_dist[rank] ||
                u >= g.vtx_dist[rank + 1]) {
                std::fprintf(stderr, "Invalid query packet\n");
                MPI_Abort(comm, 2);
                std::abort();
            }
            for (int64_t j = 0; j < k; ++j) {
                int64_t w = recv_buffer[pos++];
                if (has_edge(g, u, w))
                    ++replies[src];
            }
        }
    }
    ts = MPI_Wtime();
    MPI_Barrier(comm);
    wait_time += MPI_Wtime() - ts;
    ts = MPI_Wtime();
    MPI_Alltoall(replies.data(), 1, MPI_UINT64_T, received_replies.data(), 1, MPI_UINT64_T, comm);
    comm_time += MPI_Wtime() - ts;
    // A reply is a real message only to ranks that sent us queries (recv_counts[r] > 0); count it only then,
    // so bytes/messages are comparable with Umair's steps (the Alltoall itself is a fixed-size collective).
    for (int r = 0; r < P; ++r) {
        local_hits += received_replies[r];
        if (r != rank && recv_counts[r] > 0) {
            local_bytes += sizeof(uint64_t);
            ++local_messages;
        }
    }
    uint64_t global_hits = 0;
    ts = MPI_Wtime();
    MPI_Allreduce(&local_hits, &global_hits, 1, MPI_UINT64_T, MPI_SUM, comm);
    comm_time += MPI_Wtime() - ts;
    double local_time = MPI_Wtime() - t0;
    uint64_t local_counters[3] = {local_wedges, local_bytes, local_messages};
    uint64_t global_counters[3] = {0, 0, 0};
    MPI_Allreduce(local_counters, global_counters, 3, MPI_UINT64_T, MPI_SUM, comm);
    double local_times[3] = {local_time, comm_time, wait_time};
    double global_times[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local_times, global_times, 3, MPI_DOUBLE, MPI_MAX, comm);
    uint64_t local_rss = get_peak_rss_kb();
    uint64_t global_rss = 0;
    MPI_Allreduce(&local_rss, &global_rss, 1, MPI_UINT64_T, MPI_MAX, comm);
    // Each triangle is found from all 3 corners, so the sum must be a multiple of 3.
    if (global_hits % 3 != 0) {
        std::fprintf(stderr, "TriC-lite: sum %llu is not divisible by 3\n", (unsigned long long)global_hits);
        MPI_Abort(comm, 3);
        std::abort();
    }
    { // every rank returns the same globally reduced metrics (interface contract)
        m.triangles = global_hits / 3;
        m.wedges_examined = global_counters[0];
        m.bytes_sent = global_counters[1];
        m.messages_sent = global_counters[2];
        m.total_time_sec = global_times[0];
        m.comm_time_sec = global_times[1];
        m.mpi_wait_sec = global_times[2];
        m.peak_rss_kb = static_cast<size_t>(global_rss);
    }
    return m;
}
