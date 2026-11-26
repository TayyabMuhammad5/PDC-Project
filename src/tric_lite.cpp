#include "dist_graph.hpp"
#include <sys/resource.h>

static size_t get_peak_rss_kb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<size_t>(ru.ru_maxrss);
}

RunMetrics run_tric_lite(const DistCSR& g, MPI_Comm comm) {
    RunMetrics m;
    double t0 = MPI_Wtime();
    // TODO (Saad - Days 1-3): Algorithm 1 (flat-ID packing, Alltoallv query, Alltoall reply, Reduce / 3)
    m.total_time_sec = MPI_Wtime() - t0;
    m.peak_rss_kb = get_peak_rss_kb();
    return m;
}
