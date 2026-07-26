#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/resource.h>
#endif

#include <cstddef>
#include <cstdio>

#define main alg4_layered_original_main
#include "../no-use/legacy-alg4/alg4-layered-reference.cpp"
#undef main

namespace {
constexpr size_t kMemoryLimitBytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;

#ifdef _WIN32
HANDLE memory_job = nullptr;

bool enforce_memory_limit() {
    memory_job = CreateJobObjectW(nullptr, nullptr);
    if (!memory_job) return false;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    info.ProcessMemoryLimit = static_cast<SIZE_T>(kMemoryLimitBytes);
    if (!SetInformationJobObject(memory_job, JobObjectExtendedLimitInformation,
                                 &info, sizeof(info)))
        return false;
    return AssignProcessToJobObject(memory_job, GetCurrentProcess()) != FALSE;
}
#else
bool enforce_memory_limit() {
    rlimit limit{};
    limit.rlim_cur = static_cast<rlim_t>(kMemoryLimitBytes);
    limit.rlim_max = static_cast<rlim_t>(kMemoryLimitBytes);
    return setrlimit(RLIMIT_AS, &limit) == 0;
}
#endif
}  // namespace

int main(int argc, char* argv[]) {
    if (!enforce_memory_limit()) {
        fprintf(stderr, "Failed to enforce the 16 GiB process memory limit.\n");
        return 2;
    }
    return alg4_layered_original_main(argc, argv);
}
