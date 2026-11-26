#include "dist_graph.hpp"
#include <sys/resource.h>

static size_t get_peak_rss_kb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<size_t>(ru.ru_maxrss);
}

RunMetrics run_ditric_step1(const DistCSR& g, bool use_degree_order, MPI_Comm comm) {
    RunMetrics m;
    double t0 = MPI_Wtime();
    // TODO (Umair - Days 1-2): +A Ghost-degree exchange + (deg, ID) DAG orientation + 1-round count
    m.total_time_sec = MPI_Wtime() - t0;
    m.peak_rss_kb = get_peak_rss_kb();
    return m;
}

RunMetrics run_ditric_step2(const DistCSR& g, MPI_Comm comm) {
    RunMetrics m;
    double t0 = MPI_Wtime();
    // TODO (Umair - Days 2-3): +B Surrogate deduplication (send N+(v) once per destination rank)
    m.total_time_sec = MPI_Wtime() - t0;
    m.peak_rss_kb = get_peak_rss_kb();
    return m;
}

RunMetrics run_ditric_step3(const DistCSR& g, size_t max_buf_bytes, MPI_Comm comm) {
    RunMetrics m;
    double t0 = MPI_Wtime();
    // TODO (Umair - Day 4): +C Bounded buffered aggregation (cap buffer RAM at max_buf_bytes)
    m.total_time_sec = MPI_Wtime() - t0;
    m.peak_rss_kb = get_peak_rss_kb();
    return m;
}
