#include "dist_graph.hpp"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
#include <utility>
#include <vector>
namespace {
size_t get_peak_rss_kb() {
    struct rusage ru {};
    if (getrusage(RUSAGE_SELF, &ru) != 0)
        return 0;
    return static_cast<size_t>(ru.ru_maxrss);
}
int checked_mpi_count(size_t n, const char *what) {
    if (n > static_cast<size_t>(INT_MAX))
        throw std::runtime_error(std::string("MPI count exceeds INT_MAX in ") + what);
    return static_cast<int>(n);
}
int build_displacements(const std::vector<int> &counts, std::vector<int> &displs, const char *what) {
    int64_t total = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        if (total > INT_MAX)
            throw std::runtime_error(std::string("MPI displacement exceeds INT_MAX in ") + what);
        displs[i] = static_cast<int>(total);
        total += counts[i];
    }
    if (total > INT_MAX)
        throw std::runtime_error(std::string("MPI total count exceeds INT_MAX in ") + what);
    return static_cast<int>(total);
}
// Time spent waiting in a barrier placed right before a collective (= load imbalance).
void timed_barrier(MPI_Comm comm, double &wait_sec) {
    const double ts = MPI_Wtime();
    MPI_Barrier(comm);
    wait_sec += MPI_Wtime() - ts;
}
struct DegreeInfo {
    std::unordered_map<int64_t, int64_t> ghost_degree;
    int64_t get(const DistCSR &g, int64_t v) const {
        int64_t begin = g.vtx_dist[g.rank];
        int64_t end = g.vtx_dist[g.rank + 1];
        if (v >= begin && v < end)
            return g.local_deg.at(static_cast<size_t>(v - begin));
        auto it = ghost_degree.find(v);
        if (it == ghost_degree.end())
            throw std::runtime_error("Missing ghost degree for vertex " + std::to_string(v));
        return it->second;
    }
};
DegreeInfo exchange_ghost_degrees(const DistCSR &g, MPI_Comm comm, uint64_t &bytes_sent,
                                  uint64_t &messages_sent, double &wait_sec, double &comm_sec) {
    DegreeInfo result;
    const int p = g.num_ranks;
    std::vector<std::vector<int64_t>> requests(p);
    for (int64_t v = g.vtx_dist[g.rank]; v < g.vtx_dist[g.rank + 1]; ++v) {
        int64_t li = v - g.vtx_dist[g.rank];
        for (int64_t k = g.row_ptr[li]; k < g.row_ptr[li + 1]; ++k) {
            int64_t u = g.adj[k];
            int owner = g.owner_of(u);
            if (owner != g.rank)
                requests[owner].push_back(u);
        }
    }
    std::vector<int> send_counts(p, 0), recv_counts(p, 0), send_displs(p, 0), recv_displs(p, 0);
    for (int r = 0; r < p; ++r) {
        auto &ids = requests[r];
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        send_counts[r] = checked_mpi_count(ids.size(), "ghost requests");
    }
    int send_total = build_displacements(send_counts, send_displs, "ghost requests");
    timed_barrier(comm, wait_sec);
    double ts = MPI_Wtime();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    comm_sec += MPI_Wtime() - ts;
    int recv_total = build_displacements(recv_counts, recv_displs, "ghost responses");
    std::vector<int64_t> send_ids(send_total);
    for (int r = 0; r < p; ++r)
        std::copy(requests[r].begin(), requests[r].end(), send_ids.begin() + send_displs[r]);
    std::vector<int64_t> received_ids(recv_total);
    int64_t dummy = 0;
    ts = MPI_Wtime();
    MPI_Alltoallv(send_ids.empty() ? &dummy : send_ids.data(), send_counts.data(), send_displs.data(),
                  MPI_INT64_T, received_ids.empty() ? &dummy : received_ids.data(), recv_counts.data(),
                  recv_displs.data(), MPI_INT64_T, comm);
    comm_sec += MPI_Wtime() - ts;
    std::vector<int64_t> returned_degrees(recv_total);
    int64_t local_begin = g.vtx_dist[g.rank];
    int64_t local_end = g.vtx_dist[g.rank + 1];
    for (int i = 0; i < recv_total; ++i) {
        int64_t v = received_ids[i];
        if (v < local_begin || v >= local_end)
            throw std::runtime_error("Received request for a vertex not owned by this rank");
        returned_degrees[i] = g.local_deg[v - local_begin];
    }
    std::vector<int64_t> ghost_degrees(send_total);
    ts = MPI_Wtime();
    MPI_Alltoallv(returned_degrees.empty() ? &dummy : returned_degrees.data(), recv_counts.data(),
                  recv_displs.data(), MPI_INT64_T, ghost_degrees.empty() ? &dummy : ghost_degrees.data(),
                  send_counts.data(), send_displs.data(), MPI_INT64_T, comm);
    comm_sec += MPI_Wtime() - ts;
    for (int r = 0; r < p; ++r) {
        int offset = send_displs[r];
        for (size_t j = 0; j < requests[r].size(); ++j)
            result.ghost_degree.emplace(requests[r][j], ghost_degrees[offset + j]);
    }
    uint64_t items = static_cast<uint64_t>(send_total) + static_cast<uint64_t>(recv_total);
    bytes_sent += items * sizeof(int64_t);
    for (int r = 0; r < p; ++r) {
        if (send_counts[r] > 0)
            ++messages_sent;
        if (recv_counts[r] > 0)
            ++messages_sent;
    }
    return result;
}
bool comes_before(int64_t u, int64_t v, const DegreeInfo &degrees, const DistCSR &g, bool use_degree_order) {
    if (use_degree_order) {
        int64_t du = degrees.get(g, u);
        int64_t dv = degrees.get(g, v);
        if (du != dv)
            return du < dv;
    }
    return u < v;
}
using ForwardAdj = std::vector<std::vector<int64_t>>;
ForwardAdj build_forward_adjacency(const DistCSR &g, const DegreeInfo &degrees, bool use_degree_order) {
    int64_t begin = g.vtx_dist[g.rank];
    int64_t nlocal = g.local_n();
    ForwardAdj forward(nlocal);
    auto less_by_order = [&](int64_t a, int64_t b) {
        return comes_before(a, b, degrees, g, use_degree_order);
    };
    for (int64_t li = 0; li < nlocal; ++li) {
        int64_t v = begin + li;
        auto &out = forward[li];
        for (int64_t k = g.row_ptr[li]; k < g.row_ptr[li + 1]; ++k) {
            int64_t u = g.adj[k];
            if (comes_before(v, u, degrees, g, use_degree_order))
                out.push_back(u);
        }
        std::sort(out.begin(), out.end(), less_by_order);
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }
    return forward;
}
} // namespace
RunMetrics run_ditric_step1(const DistCSR &g, bool use_degree_order, MPI_Comm comm) {
    RunMetrics m;
    const int p = g.num_ranks;
    double wait_sec = 0.0, comm_sec = 0.0;
    MPI_Barrier(comm); // common start line for all ranks (not counted as wait)
    const double t0 = MPI_Wtime();
    const int64_t local_begin = g.vtx_dist[g.rank];
    const int64_t nlocal = g.local_n();
    DegreeInfo degrees;
    if (use_degree_order)
        degrees = exchange_ghost_degrees(g, comm, m.bytes_sent, m.messages_sent, wait_sec, comm_sec);
    ForwardAdj forward = build_forward_adjacency(g, degrees, use_degree_order);
    uint64_t local_triangles = 0;
    uint64_t local_wedges = 0;
    std::vector<std::vector<int64_t>> outgoing(p);
    for (int64_t li = 0; li < nlocal; ++li) {
        const auto &nv = forward[li];
        for (size_t i = 0; i < nv.size(); ++i) {
            const int64_t u = nv[i];
            const size_t k = nv.size() - i - 1; // only vertices AFTER u can close a triangle with (v,u)
            if (k == 0)
                continue;
            local_wedges += k; // candidate pairs {u,w} tested
            const int owner = g.owner_of(u);
            if (owner == g.rank) {
                const int64_t ui = u - local_begin;
                const int64_t *b = g.adj.data() + g.row_ptr[ui];
                const int64_t *e = g.adj.data() + g.row_ptr[ui + 1];
                for (size_t j = i + 1; j < nv.size(); ++j)
                    if (std::binary_search(b, e, nv[j]))
                        ++local_triangles;
            } else {
                auto &d = outgoing[owner];
                d.push_back(u);
                d.push_back(static_cast<int64_t>(k));
                d.insert(d.end(), nv.begin() + i + 1, nv.end());
            }
        }
    }
    std::vector<int> send_counts(p, 0), recv_counts(p, 0), send_displs(p, 0), recv_displs(p, 0);
    for (int r = 0; r < p; ++r)
        send_counts[r] = checked_mpi_count(outgoing[r].size(), "Step 1 candidate packets");
    int send_total = build_displacements(send_counts, send_displs, "Step 1 send packets");
    timed_barrier(comm, wait_sec);
    double tc = MPI_Wtime();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    comm_sec += MPI_Wtime() - tc;
    int recv_total = build_displacements(recv_counts, recv_displs, "Step 1 receive packets");
    std::vector<int64_t> send_buffer(send_total);
    for (int r = 0; r < p; ++r)
        std::copy(outgoing[r].begin(), outgoing[r].end(), send_buffer.begin() + send_displs[r]);
    std::vector<int64_t> recv_buffer(recv_total);
    int64_t dummy = 0;
    tc = MPI_Wtime();
    MPI_Alltoallv(send_buffer.empty() ? &dummy : send_buffer.data(), send_counts.data(), send_displs.data(),
                  MPI_INT64_T, recv_buffer.empty() ? &dummy : recv_buffer.data(), recv_counts.data(),
                  recv_displs.data(), MPI_INT64_T, comm);
    comm_sec += MPI_Wtime() - tc;
    m.bytes_sent += static_cast<uint64_t>(send_total) * sizeof(int64_t);
    for (int r = 0; r < p; ++r)
        if (send_counts[r] > 0)
            ++m.messages_sent;
    for (int src = 0; src < p; ++src) {
        size_t pos = recv_displs[src];
        size_t end = pos + recv_counts[src];
        while (pos < end) {
            if (end - pos < 2)
                throw std::runtime_error("Malformed Step 1 packet header");
            int64_t u = recv_buffer[pos++];
            int64_t len64 = recv_buffer[pos++];
            if (len64 < 0 || static_cast<uint64_t>(len64) > end - pos)
                throw std::runtime_error("Malformed Step 1 packet length");
            int64_t ui = u - local_begin;
            if (ui < 0 || ui >= nlocal)
                throw std::runtime_error("Packet delivered to incorrect owner");
            const int64_t *b = g.adj.data() + g.row_ptr[ui];
            const int64_t *e = g.adj.data() + g.row_ptr[ui + 1];
            for (int64_t j = 0; j < len64; ++j)
                if (std::binary_search(b, e, recv_buffer[pos + j]))
                    ++local_triangles;
            // wedges for this packet were already counted by the sender (local_wedges += k)
            pos += static_cast<size_t>(len64);
        }
    }
    uint64_t global_triangles = 0, global_wedges = 0, global_bytes = 0, global_messages = 0;
    MPI_Allreduce(&local_triangles, &global_triangles, 1, MPI_UINT64_T, MPI_SUM, comm);
    MPI_Allreduce(&local_wedges, &global_wedges, 1, MPI_UINT64_T, MPI_SUM, comm);
    MPI_Allreduce(&m.bytes_sent, &global_bytes, 1, MPI_UINT64_T, MPI_SUM, comm);
    MPI_Allreduce(&m.messages_sent, &global_messages, 1, MPI_UINT64_T, MPI_SUM, comm);
    const double local_elapsed = MPI_Wtime() - t0;
    double max_elapsed = 0.0;
    MPI_Allreduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);
    double local_t[3] = {local_elapsed, comm_sec, wait_sec}, max_t[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local_t, max_t, 3, MPI_DOUBLE, MPI_MAX, comm);
    max_elapsed = max_t[0];
    unsigned long long local_rss = static_cast<unsigned long long>(get_peak_rss_kb());
    unsigned long long max_rss = 0;
    MPI_Allreduce(&local_rss, &max_rss, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    // Every rank returns the same globally reduced metrics (interface contract).
    m.triangles = global_triangles;
    m.wedges_examined = global_wedges;
    m.bytes_sent = global_bytes;
    m.messages_sent = global_messages;
    m.total_time_sec = max_elapsed;
    m.comm_time_sec = max_t[1];
    m.mpi_wait_sec = max_t[2];
    m.peak_rss_kb = static_cast<size_t>(max_rss);
    return m;
}
RunMetrics run_ditric_step2(const DistCSR &g, MPI_Comm comm) {
    (void)g;
    (void)comm;
    RunMetrics m;
    return m;
}
RunMetrics run_ditric_step3(const DistCSR &g, size_t max_buf_bytes, MPI_Comm comm) {
    (void)g;
    (void)max_buf_bytes;
    (void)comm;
    RunMetrics m;
    return m;
}
