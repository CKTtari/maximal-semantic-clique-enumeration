#pragma once

// Shared process limits and dataset mapping for the four standard executables.
// Include this header before other standard-library headers on Windows.

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/resource.h>
#endif

#include <cstddef>
#include <string>

namespace acmsc_runtime {

inline constexpr std::size_t kMemoryLimitBytes =
    16ULL * 1024ULL * 1024ULL * 1024ULL;
inline constexpr int kDefaultThreads = 32;
inline constexpr int kDefaultTimeoutSeconds = 0;

#ifdef _WIN32
inline HANDLE& process_memory_job() {
    static HANDLE handle = nullptr;
    return handle;
}

inline bool enforce_memory_limit() {
    HANDLE& job = process_memory_job();
    if (job) return true;
    job = CreateJobObjectW(nullptr, nullptr);
    if (!job) return false;

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    info.ProcessMemoryLimit = static_cast<SIZE_T>(kMemoryLimitBytes);
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                 &info, sizeof(info)))
        return false;
    return AssignProcessToJobObject(job, GetCurrentProcess()) != FALSE;
}

inline std::size_t peak_memory_bytes() {
    HANDLE job = process_memory_job();
    if (!job) return 0;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    if (!QueryInformationJobObject(job, JobObjectExtendedLimitInformation,
                                   &info, sizeof(info), nullptr))
        return 0;
    return static_cast<std::size_t>(info.PeakProcessMemoryUsed);
}
#else
inline bool enforce_memory_limit() {
    rlimit limit{};
    limit.rlim_cur = static_cast<rlim_t>(kMemoryLimitBytes);
    limit.rlim_max = static_cast<rlim_t>(kMemoryLimitBytes);
    return setrlimit(RLIMIT_AS, &limit) == 0;
}

inline std::size_t peak_memory_bytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#ifdef __APPLE__
    return static_cast<std::size_t>(usage.ru_maxrss);
#else
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024ULL;
#endif
}
#endif

inline void select_dataset(int id, std::string& graph_file,
                           std::string& vector_file) {
    switch (id) {
        case 1:
            graph_file = "dataset/dataset/sc-ldoor.txt";
            vector_file = "dataset/vectors-256/sc-ldoor_vectors.bin";
            break;
        case 2:
            graph_file = "dataset/dataset/sc-nasasrb.txt";
            vector_file = "dataset/vectors-256/sc-nasasrb_vectors.bin";
            break;
        case 3:
            graph_file = "dataset/dataset/sc-pkustk11.txt";
            vector_file = "dataset/vectors-256/sc-pkustk11_vectors.bin";
            break;
        case 4:
            graph_file = "dataset/dataset/soc-buzznet.txt";
            vector_file = "dataset/vectors-256/soc-buzznet_vectors.bin";
            break;
        case 5:
            graph_file = "dataset/dataset/soc-digg.txt";
            vector_file = "dataset/vectors-256/soc-digg_vectors.bin";
            break;
        case 6:
            graph_file = "dataset/dataset/tech-as-skitter.txt";
            vector_file = "dataset/vectors-256/tech-as-skitter_vectors.bin";
            break;
        case 7:
            graph_file = "dataset/dataset/FB15K-237_edges.txt";
            vector_file = "dataset/vectors/FB15K-237_vectors.index";
            break;
        case 8:
            graph_file = "dataset/dataset/WN18RR_edges.txt";
            vector_file = "dataset/vectors/WN18RR_vectors.index";
            break;
        case 9:
            graph_file = "dataset/dataset/dblp_coauthor.txt";
            vector_file = "dataset/vectors/dblp_coauthor_vectors.index";
            break;
        case 10:
            graph_file = "dataset/dataset/products.txt";
            vector_file = "dataset/vectors/products_vectors.index";
            break;
        case 11:
            graph_file = "dataset/dataset/ogbn-arxiv.txt";
            vector_file = "dataset/vectors/ogbn-arxiv_vectors.bin";
            break;
        case 12:
            graph_file = "dataset/dataset/com-amazon.txt";
            vector_file = "dataset/vectors-256/com-amazon_vectors.bin";
            break;
        case 13:
            graph_file = "dataset/dataset/ca-dblp-2012.txt";
            vector_file = "dataset/vectors-256/ca-dblp-2012_vectors.bin";
            break;
        case 14:
            graph_file = "dataset/dataset/sc-pwtk.txt";
            vector_file = "dataset/vectors-256/sc-pwtk_vectors.bin";
            break;
        case 16:
            graph_file = "dataset/Processed/email-Enron/graph.txt";
            vector_file = "dataset/Processed/email-Enron/email-Enron_vectors.bin";
            break;
        default:
            select_dataset(2, graph_file, vector_file);
            break;
    }
}

}  // namespace acmsc_runtime
