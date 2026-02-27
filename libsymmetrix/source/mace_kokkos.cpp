#include <fstream>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <cstdlib>
#include <sstream>
#include <type_traits>
#include <mutex>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <cctype>
#include <utility>

// TODO: remove some of these headers?
#include "KokkosBatched_Util.hpp"
#include "KokkosBlas.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "nlohmann/json.hpp"
#include "sphericart.hpp"
#include "sphericart_cuda.hpp"

#include "tools_kokkos.hpp"
#include "mace_kokkos.hpp"
#include "mace.hpp"

#if defined(KOKKOS_ENABLE_CUDA) && defined(SYMMETRIX_HAVE_CUBLAS)
#include <cublas_v2.h>
#include <cublasLt.h>
#endif

using Kokkos::ALL;
using Kokkos::LayoutRight;
using Kokkos::make_pair;
using Kokkos::MemoryUnmanaged;
using Kokkos::parallel_for;
using Kokkos::PerTeam;
using Kokkos::subview;
using Kokkos::TeamPolicy;
using Kokkos::TeamVectorRange;
using Kokkos::TeamVectorMDRange;
using Kokkos::View;

namespace {

auto rrnlb_env_flag(const char* name, const bool default_value = false) -> bool
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return default_value;
    if (env[0] == '0') return false;
    if (env[0] == 'f' || env[0] == 'F') return false;
    if (env[0] == 'n' || env[0] == 'N') return false;
    return true;
}

auto rrnlb_env_int(const char* name, const int default_value) -> int
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return default_value;
    int value = std::atoi(env);
    if (value < 0) value = 0;
    return value;
}

template <typename F>
class ScopeExit {
public:
    explicit ScopeExit(F&& fn) : fn_(std::forward<F>(fn)), active_(true) {}
    ~ScopeExit() { if (active_) fn_(); }
    ScopeExit(const ScopeExit&) = delete;
    auto operator=(const ScopeExit&) -> ScopeExit& = delete;
    ScopeExit(ScopeExit&& other) noexcept
        : fn_(std::move(other.fn_)), active_(other.active_)
    {
        other.active_ = false;
    }
private:
    F fn_;
    bool active_;
};

template <typename F>
auto make_scope_exit(F&& fn) -> ScopeExit<F>
{
    return ScopeExit<F>(std::forward<F>(fn));
}

enum class RRNLBGroupPackedGemmMode {
    Off,
    On,
    Auto
};

enum class RRNLBInteractionCompactMode {
    Off,
    On,
    Auto
};

enum class RRNLBNonlinearAblationMode {
    Off,
    GateIdentity
};

enum class RRNLBEdgeTopologyRefreshMode {
    Always,
    LegacyEpoch
};

enum class RRNLBMpiApNativeM0Impl {
    NodeReduce,
    AtomicLegacy,
    NodeReduceV2
};

auto rrnlb_nonlinear_ablation_mode() -> RRNLBNonlinearAblationMode
{
    static RRNLBNonlinearAblationMode mode = []() -> RRNLBNonlinearAblationMode {
        const char* mode_env = std::getenv("SYMMETRIX_RRNLB_ABLATION");
        if (mode_env == nullptr || mode_env[0] == '\0') {
            return RRNLBNonlinearAblationMode::Off;
        }
        std::string mode(mode_env);
        std::transform(
            mode.begin(),
            mode.end(),
            mode.begin(),
            [] (unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (mode == "off") return RRNLBNonlinearAblationMode::Off;
        if (mode == "gate_identity" || mode == "identity_gate") {
            return RRNLBNonlinearAblationMode::GateIdentity;
        }
        std::ostringstream oss;
        oss << "Unsupported SYMMETRIX_RRNLB_ABLATION mode '" << mode
            << "'. Supported values: off, gate_identity.";
        throw std::runtime_error(oss.str());
    }();
    return mode;
}

auto rrnlb_reverse_vector_length() -> int
{
    static int value = []() -> int {
        const char* env = std::getenv("SYMMETRIX_RRNLB_REVERSE_VECTOR_LENGTH");
        if (env == nullptr || env[0] == '\0') return 1;
        int parsed = std::atoi(env);
        if (parsed < 1) parsed = 1;
        if (parsed > 32) parsed = 32;
        return parsed;
    }();
    return value;
}

auto rrnlb_forward_vector_length() -> int
{
    static int value = []() -> int {
        const char* env = std::getenv("SYMMETRIX_RRNLB_FORWARD_VECTOR_LENGTH");
        if (env == nullptr || env[0] == '\0') return 1;
        int parsed = std::atoi(env);
        if (parsed < 1) parsed = 1;
        if (parsed > 32) parsed = 32;
        return parsed;
    }();
    return value;
}

auto rrnlb_forward_adaptive_mode() -> RRNLBForwardAdaptiveMode
{
    static RRNLBForwardAdaptiveMode mode = []() -> RRNLBForwardAdaptiveMode {
        const char* mode_env = std::getenv("SYMMETRIX_RRNLB_FORWARD_ADAPTIVE_MODE");
        if (mode_env == nullptr || mode_env[0] == '\0') {
            return RRNLBForwardAdaptiveMode::Auto;
        }
        std::string mode(mode_env);
        std::transform(
            mode.begin(),
            mode.end(),
            mode.begin(),
            [] (unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (mode == "auto") return RRNLBForwardAdaptiveMode::Auto;
        if (mode == "force_fused") return RRNLBForwardAdaptiveMode::ForceFused;
        if (mode == "force_split") return RRNLBForwardAdaptiveMode::ForceSplit;
        std::ostringstream oss;
        oss << "Unsupported SYMMETRIX_RRNLB_FORWARD_ADAPTIVE_MODE value '" << mode
            << "'. Supported values: auto, force_fused, force_split.";
        throw std::runtime_error(oss.str());
    }();
    return mode;
}

auto rrnlb_forward_split_min_nodes() -> int
{
    static int value = []() -> int {
        const char* env = std::getenv("SYMMETRIX_RRNLB_FORWARD_SPLIT_MIN_NODES");
        if (env == nullptr || env[0] == '\0') return 384;
        int parsed = std::atoi(env);
        if (parsed < 1) parsed = 1;
        return parsed;
    }();
    return value;
}

auto rrnlb_forward_should_use_split(
    const RRNLBForwardAdaptiveMode mode,
    const int num_nodes,
    const bool receiver_parallel_enabled,
    const bool use_edge_parallel) -> bool
{
    if (!receiver_parallel_enabled || !use_edge_parallel) return false;
    switch (mode) {
        case RRNLBForwardAdaptiveMode::ForceFused:
            return false;
        case RRNLBForwardAdaptiveMode::ForceSplit:
            return true;
        case RRNLBForwardAdaptiveMode::Auto:
        default:
            return num_nodes < rrnlb_forward_split_min_nodes();
    }
}

auto rrnlb_edge_topology_refresh_mode() -> RRNLBEdgeTopologyRefreshMode
{
    static RRNLBEdgeTopologyRefreshMode mode = []() -> RRNLBEdgeTopologyRefreshMode {
        const char* mode_env = std::getenv("SYMMETRIX_RRNLB_EDGE_TOPOLOGY_REFRESH");
        if (mode_env == nullptr || mode_env[0] == '\0') {
            return RRNLBEdgeTopologyRefreshMode::Always;
        }
        std::string mode(mode_env);
        std::transform(
            mode.begin(),
            mode.end(),
            mode.begin(),
            [] (unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (mode == "always") return RRNLBEdgeTopologyRefreshMode::Always;
        if (mode == "legacy_epoch" || mode == "legacy") {
            return RRNLBEdgeTopologyRefreshMode::LegacyEpoch;
        }
        std::ostringstream oss;
        oss << "Unsupported SYMMETRIX_RRNLB_EDGE_TOPOLOGY_REFRESH value '" << mode
            << "'. Supported values: always, legacy_epoch.";
        throw std::runtime_error(oss.str());
    }();
    return mode;
}

auto rrnlb_mpi_ap_native_m0_impl() -> RRNLBMpiApNativeM0Impl
{
    static RRNLBMpiApNativeM0Impl mode = []() -> RRNLBMpiApNativeM0Impl {
        const char* mode_env = std::getenv("SYMMETRIX_RRNLB_MPI_AP_NATIVE_M0_IMPL");
        if (mode_env == nullptr || mode_env[0] == '\0') {
            return RRNLBMpiApNativeM0Impl::NodeReduce;
        }
        std::string mode(mode_env);
        std::transform(
            mode.begin(),
            mode.end(),
            mode.begin(),
            [] (unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (mode == "node_reduce" || mode == "node") {
            return RRNLBMpiApNativeM0Impl::NodeReduce;
        }
        if (mode == "atomic_legacy" || mode == "legacy" || mode == "atomic") {
            return RRNLBMpiApNativeM0Impl::AtomicLegacy;
        }
        if (mode == "node_reduce_v2" || mode == "node_v2" || mode == "v2") {
            return RRNLBMpiApNativeM0Impl::NodeReduceV2;
        }
        std::ostringstream oss;
        oss << "Unsupported SYMMETRIX_RRNLB_MPI_AP_NATIVE_M0_IMPL value '" << mode
            << "'. Supported values: node_reduce, atomic_legacy, node_reduce_v2.";
        throw std::runtime_error(oss.str());
    }();
    return mode;
}

auto rrnlb_group_packed_gemm_mode() -> RRNLBGroupPackedGemmMode
{
    static RRNLBGroupPackedGemmMode mode = []() -> RRNLBGroupPackedGemmMode {
        const char* mode_env = std::getenv("SYMMETRIX_RRNLB_GROUP_PACKED_GEMM_MODE");
        if (mode_env != nullptr && mode_env[0] != '\0') {
            std::string mode(mode_env);
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [] (unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (mode == "off") return RRNLBGroupPackedGemmMode::Off;
            if (mode == "on") return RRNLBGroupPackedGemmMode::On;
            if (mode == "auto") return RRNLBGroupPackedGemmMode::Auto;
        }

        // Backward-compatible fallback for older env knob.
        const char* legacy_env = std::getenv("SYMMETRIX_RRNLB_GROUP_PACKED_GEMM");
        if (legacy_env != nullptr) {
            const bool legacy_on = legacy_env[0] != '\0' && legacy_env[0] != '0';
            return legacy_on ? RRNLBGroupPackedGemmMode::On : RRNLBGroupPackedGemmMode::Off;
        }

        // New default: adaptive auto mode.
        return RRNLBGroupPackedGemmMode::Auto;
    }();
    return mode;
}

auto rrnlb_group_packed_min_rows() -> int
{
    static int value = []() -> int {
        const char* env = std::getenv("SYMMETRIX_RRNLB_GROUP_PACKED_MIN_ROWS");
        if (env == nullptr || env[0] == '\0') return 1024;
        int parsed = std::atoi(env);
        if (parsed < 1) parsed = 1;
        return parsed;
    }();
    return value;
}

auto rrnlb_group_packed_auto_should_run(
    const int num_nodes,
    const std::vector<int>& group_ir_dim,
    const std::vector<int>& group_lhs_dim,
    const std::vector<int>& group_rhs_dim) -> bool
{
    if (num_nodes <= 0) return false;
    if (group_ir_dim.empty()) return false;
    if (group_lhs_dim.size() != group_ir_dim.size()
        || group_rhs_dim.size() != group_ir_dim.size()) {
        return false;
    }

    const int min_rows = rrnlb_group_packed_min_rows();
    int max_rows = 0;
    long long total_work = 0;
    for (int g = 0; g < static_cast<int>(group_ir_dim.size()); ++g) {
        const int ir_dim = group_ir_dim[g];
        const int lhs_dim = group_lhs_dim[g];
        const int rhs_dim = group_rhs_dim[g];
        if (ir_dim <= 0 || lhs_dim <= 0 || rhs_dim <= 0) continue;
        const int rows = num_nodes * ir_dim;
        max_rows = std::max(max_rows, rows);
        const long long work = static_cast<long long>(rows) * lhs_dim * rhs_dim;
        if (work > 0) {
            if (total_work > std::numeric_limits<long long>::max() - work) {
                total_work = std::numeric_limits<long long>::max();
            } else {
                total_work += work;
            }
        }
    }
    return max_rows >= min_rows && total_work > 0;
}

auto rrnlb_group_packed_should_run(
    const int num_nodes,
    const std::vector<int>& group_ir_dim,
    const std::vector<int>& group_lhs_dim,
    const std::vector<int>& group_rhs_dim) -> bool
{
    const auto mode = rrnlb_group_packed_gemm_mode();
    if (mode == RRNLBGroupPackedGemmMode::Off) return false;
    if (mode == RRNLBGroupPackedGemmMode::On) return true;
#if defined(KOKKOS_ENABLE_CUDA) && defined(SYMMETRIX_HAVE_CUBLAS)
    // In auto mode on CUDA, prefer the vendor strided-batched path by default.
    // Packed portable GEMM remains available via SYMMETRIX_RRNLB_GROUP_PACKED_GEMM_MODE=on
    // and is still used automatically when cuBLAS is explicitly disabled.
    const char* disable_cublas_env = std::getenv("SYMMETRIX_DISABLE_RRNLB_CUBLAS");
    const bool cublas_enabled =
        disable_cublas_env == nullptr
        || disable_cublas_env[0] == '\0'
        || disable_cublas_env[0] == '0';
    if (cublas_enabled) return false;
#endif
    return rrnlb_group_packed_auto_should_run(
        num_nodes, group_ir_dim, group_lhs_dim, group_rhs_dim);
}

auto rrnlb_group_packed_gemm_strict_enabled() -> bool
{
    static bool enabled = []() -> bool {
        const char* env = std::getenv("SYMMETRIX_RRNLB_GROUP_PACKED_GEMM_STRICT");
        if (env == nullptr) return false;
        return env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

auto rrnlb_interaction_compact_mode() -> RRNLBInteractionCompactMode
{
    static RRNLBInteractionCompactMode mode = []() -> RRNLBInteractionCompactMode {
        const char* mode_env = std::getenv("SYMMETRIX_RRNLB_INTERACTION_COMPACT_MODE");
        if (mode_env != nullptr && mode_env[0] != '\0') {
            std::string mode(mode_env);
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [] (unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (mode == "off") return RRNLBInteractionCompactMode::Off;
            if (mode == "on") return RRNLBInteractionCompactMode::On;
            if (mode == "auto") return RRNLBInteractionCompactMode::Auto;
        }

        // Backward-compatible fallback for older env knob.
        const char* legacy_env = std::getenv("SYMMETRIX_RRNLB_INTERACTION_COMPACT");
        if (legacy_env != nullptr) {
            const bool legacy_on = legacy_env[0] != '\0' && legacy_env[0] != '0';
            return legacy_on ? RRNLBInteractionCompactMode::On : RRNLBInteractionCompactMode::Off;
        }

        return RRNLBInteractionCompactMode::Auto;
    }();
    return mode;
}

auto rrnlb_receiver_tiled_forward_override() -> int
{
    static int mode = []() -> int {
        const char* env = std::getenv("SYMMETRIX_RRNLB_RECEIVER_TILED_FWD");
        if (env == nullptr) return -1;
        return (env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return mode;
}

auto rrnlb_sender_tiled_reverse_override() -> int
{
    static int mode = []() -> int {
        const char* env = std::getenv("SYMMETRIX_RRNLB_SENDER_TILED_REV");
        if (env == nullptr) return -1;
        return (env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return mode;
}

auto rrnlb_sender_segment_size() -> int
{
    static int segment_size = []() -> int {
        const char* env = std::getenv("SYMMETRIX_RRNLB_SENDER_SEGMENT_SIZE");
        if (env == nullptr || env[0] == '\0') return 64;
        const int parsed = std::atoi(env);
        return parsed > 0 ? parsed : 64;
    }();
    return segment_size;
}

auto rrnlb_epoch_topology_validate_enabled() -> bool
{
    static bool enabled = []() -> bool {
        const char* env = std::getenv("SYMMETRIX_RRNLB_EPOCH_TOPOLOGY_VALIDATE");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

auto rrnlb_group_packed_work_fits_int(
    const int num_nodes,
    const int ir_dim,
    const int packed_width) -> bool
{
    if (num_nodes <= 0 || ir_dim <= 0 || packed_width <= 0) return false;
    const long long work =
        static_cast<long long>(num_nodes) * ir_dim * packed_width;
    return work > 0 && work <= std::numeric_limits<int>::max();
}

#if defined(KOKKOS_ENABLE_CUDA) && defined(SYMMETRIX_HAVE_CUBLAS)
auto rrnlb_cublas_tf32_enabled() -> bool
{
    static bool enabled = []() -> bool {
        const char* env = std::getenv("SYMMETRIX_RRNLB_CUBLAS_TF32");
        // Default-on for NVIDIA CUDA path; can be disabled explicitly with
        // SYMMETRIX_RRNLB_CUBLAS_TF32=0 for diagnostic/validation runs.
        if (env == nullptr) return true;
        return env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

auto rrnlb_cublaslt_strict_enabled() -> bool
{
    static bool enabled = []() -> bool {
        const char* env = std::getenv("SYMMETRIX_RRNLB_CUBLASLT_STRICT");
        if (env == nullptr) return false;
        return env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

auto rrnlb_get_cublas_handle() -> cublasHandle_t
{
    static cublasHandle_t handle = []() -> cublasHandle_t {
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("RRNLB cuBLAS init failed (cublasCreate).");
        }
        if (rrnlb_cublas_tf32_enabled()) {
            cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
        } else {
            cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
        }
        return h;
    }();
    return handle;
}

auto rrnlb_get_cublaslt_handle() -> cublasLtHandle_t
{
    static cublasLtHandle_t handle = []() -> cublasLtHandle_t {
        cublasLtHandle_t h = nullptr;
        if (cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("RRNLB cuBLASLt init failed (cublasLtCreate).");
        }
        return h;
    }();
    return handle;
}

auto rrnlb_throw_cublas_error(cublasStatus_t status, const char* where) -> void
{
    if (status == CUBLAS_STATUS_SUCCESS) return;
    std::ostringstream oss;
    oss << where << " failed with cuBLAS status " << static_cast<int>(status);
    throw std::runtime_error(oss.str());
}

struct RRNLBCublasLtPlanKey {
    int op_a = 0;
    int op_b = 0;
    int m = 0;
    int n = 0;
    int k = 0;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    long long stride_a = 0;
    long long stride_b = 0;
    long long stride_c = 0;
    int batch_count = 0;

    auto operator==(const RRNLBCublasLtPlanKey& other) const -> bool
    {
        return op_a == other.op_a
            && op_b == other.op_b
            && m == other.m
            && n == other.n
            && k == other.k
            && lda == other.lda
            && ldb == other.ldb
            && ldc == other.ldc
            && stride_a == other.stride_a
            && stride_b == other.stride_b
            && stride_c == other.stride_c
            && batch_count == other.batch_count;
    }
};

struct RRNLBCublasLtPlanKeyHash {
    auto operator()(const RRNLBCublasLtPlanKey& key) const -> std::size_t
    {
        auto seed = std::size_t{0};
        auto hash_combine = [&seed](std::size_t v) {
            seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        hash_combine(std::hash<int>{}(key.op_a));
        hash_combine(std::hash<int>{}(key.op_b));
        hash_combine(std::hash<int>{}(key.m));
        hash_combine(std::hash<int>{}(key.n));
        hash_combine(std::hash<int>{}(key.k));
        hash_combine(std::hash<int>{}(key.lda));
        hash_combine(std::hash<int>{}(key.ldb));
        hash_combine(std::hash<int>{}(key.ldc));
        hash_combine(std::hash<long long>{}(key.stride_a));
        hash_combine(std::hash<long long>{}(key.stride_b));
        hash_combine(std::hash<long long>{}(key.stride_c));
        hash_combine(std::hash<int>{}(key.batch_count));
        return seed;
    }
};

struct RRNLBCublasLtPlan {
    bool available = false;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatrixLayout_t d_layout = nullptr;
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

auto rrnlb_destroy_cublaslt_plan(RRNLBCublasLtPlan& plan) -> void
{
    if (plan.d_layout != nullptr) cublasLtMatrixLayoutDestroy(plan.d_layout);
    if (plan.c_layout != nullptr) cublasLtMatrixLayoutDestroy(plan.c_layout);
    if (plan.b_layout != nullptr) cublasLtMatrixLayoutDestroy(plan.b_layout);
    if (plan.a_layout != nullptr) cublasLtMatrixLayoutDestroy(plan.a_layout);
    if (plan.op_desc != nullptr) cublasLtMatmulDescDestroy(plan.op_desc);
    plan = RRNLBCublasLtPlan{};
}

auto rrnlb_create_cublaslt_plan(const RRNLBCublasLtPlanKey& key) -> RRNLBCublasLtPlan
{
    RRNLBCublasLtPlan plan;
    auto lt_handle = rrnlb_get_cublaslt_handle();

    if (cublasLtMatmulDescCreate(&plan.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        rrnlb_destroy_cublaslt_plan(plan);
        return plan;
    }

    const auto op_a = static_cast<cublasOperation_t>(key.op_a);
    const auto op_b = static_cast<cublasOperation_t>(key.op_b);
    cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    if (cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulDescSetAttribute(
            plan.op_desc,
            CUBLASLT_MATMUL_DESC_POINTER_MODE,
            &pointer_mode,
            sizeof(pointer_mode)) != CUBLAS_STATUS_SUCCESS) {
        rrnlb_destroy_cublaslt_plan(plan);
        return plan;
    }

    const int64_t a_rows = (op_a == CUBLAS_OP_N) ? key.m : key.k;
    const int64_t a_cols = (op_a == CUBLAS_OP_N) ? key.k : key.m;
    const int64_t b_rows = (op_b == CUBLAS_OP_N) ? key.k : key.n;
    const int64_t b_cols = (op_b == CUBLAS_OP_N) ? key.n : key.k;
    const int64_t c_rows = key.m;
    const int64_t c_cols = key.n;
    const int64_t batch_count = key.batch_count;
    const int64_t stride_a = key.stride_a;
    const int64_t stride_b = key.stride_b;
    const int64_t stride_c = key.stride_c;
    const int64_t lda = key.lda;
    const int64_t ldb = key.ldb;
    const int64_t ldc = key.ldc;

    if (cublasLtMatrixLayoutCreate(&plan.a_layout, CUDA_R_32F, a_rows, a_cols, lda) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&plan.b_layout, CUDA_R_32F, b_rows, b_cols, ldb) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&plan.c_layout, CUDA_R_32F, c_rows, c_cols, ldc) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&plan.d_layout, CUDA_R_32F, c_rows, c_cols, ldc) != CUBLAS_STATUS_SUCCESS) {
        rrnlb_destroy_cublaslt_plan(plan);
        return plan;
    }

    if (cublasLtMatrixLayoutSetAttribute(
            plan.a_layout,
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count,
            sizeof(batch_count)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(
            plan.b_layout,
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count,
            sizeof(batch_count)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(
            plan.c_layout,
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count,
            sizeof(batch_count)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(
            plan.d_layout,
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count,
            sizeof(batch_count)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(
            plan.a_layout,
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_a,
            sizeof(stride_a)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(
            plan.b_layout,
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_b,
            sizeof(stride_b)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(
            plan.c_layout,
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_c,
            sizeof(stride_c)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(
            plan.d_layout,
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_c,
            sizeof(stride_c)) != CUBLAS_STATUS_SUCCESS) {
        rrnlb_destroy_cublaslt_plan(plan);
        return plan;
    }

    cublasLtMatmulPreference_t preference = nullptr;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
        rrnlb_destroy_cublaslt_plan(plan);
        return plan;
    }
    const size_t max_workspace_bytes = []() -> size_t {
        const char* env = std::getenv("SYMMETRIX_RRNLB_CUBLASLT_MAX_WORKSPACE_MB");
        if (env == nullptr || env[0] == '\0') return static_cast<size_t>(8) * 1024 * 1024;
        const long mb = std::atol(env);
        if (mb <= 0) return 0;
        return static_cast<size_t>(mb) * 1024 * 1024;
    }();
    if (cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &max_workspace_bytes,
            sizeof(max_workspace_bytes)) != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulPreferenceDestroy(preference);
        rrnlb_destroy_cublaslt_plan(plan);
        return plan;
    }

    constexpr int max_heuristics = 32;
    cublasLtMatmulHeuristicResult_t heuristics[max_heuristics];
    int returned_results = 0;
    const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle,
        plan.op_desc,
        plan.a_layout,
        plan.b_layout,
        plan.c_layout,
        plan.d_layout,
        preference,
        max_heuristics,
        heuristics,
        &returned_results);
    cublasLtMatmulPreferenceDestroy(preference);

    if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
        rrnlb_destroy_cublaslt_plan(plan);
        return plan;
    }

    int best_idx = 0;
    float best_waves = std::numeric_limits<float>::max();
    for (int i = 0; i < returned_results; ++i) {
        if (heuristics[i].wavesCount < best_waves) {
            best_waves = heuristics[i].wavesCount;
            best_idx = i;
        }
    }
    plan.algo = heuristics[best_idx].algo;
    plan.workspace_bytes = heuristics[best_idx].workspaceSize;
    plan.available = true;
    return plan;
}

auto rrnlb_get_cublaslt_plan(const RRNLBCublasLtPlanKey& key) -> RRNLBCublasLtPlan
{
    static auto* plan_cache = new std::unordered_map<RRNLBCublasLtPlanKey, RRNLBCublasLtPlan, RRNLBCublasLtPlanKeyHash>();
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> guard(cache_mutex);
    auto it = plan_cache->find(key);
    if (it == plan_cache->end()) {
        it = plan_cache->emplace(key, rrnlb_create_cublaslt_plan(key)).first;
    }
    return it->second;
}

auto rrnlb_get_cublaslt_workspace(const size_t required_bytes) -> void*
{
    if (required_bytes == 0) return nullptr;
    struct WorkspaceState {
        void* ptr = nullptr;
        size_t size = 0;
        std::mutex mtx;
    };
    static auto* state = new WorkspaceState();
    std::lock_guard<std::mutex> guard(state->mtx);
    if (state->size >= required_bytes) return state->ptr;
    if (state->ptr != nullptr) {
        cudaFree(state->ptr);
        state->ptr = nullptr;
        state->size = 0;
    }
    if (cudaMalloc(&state->ptr, required_bytes) != cudaSuccess) {
        return nullptr;
    }
    state->size = required_bytes;
    return state->ptr;
}

auto rrnlb_cublas_enabled() -> bool
{
    const char* disable = std::getenv("SYMMETRIX_DISABLE_RRNLB_CUBLAS");
    if (disable == nullptr) return true;
    return disable[0] == '\0' || disable[0] == '0';
}

auto rrnlb_cublaslt_gemm_strided_batched_fp32(
    cublasLtHandle_t lt_handle,
    cudaStream_t stream,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* a,
    int lda,
    long long int stride_a,
    const float* b,
    int ldb,
    long long int stride_b,
    const float* beta,
    float* c,
    int ldc,
    long long int stride_c,
    int batch_count) -> cublasStatus_t
{
    const RRNLBCublasLtPlanKey key{
        static_cast<int>(op_a),
        static_cast<int>(op_b),
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        stride_a,
        stride_b,
        stride_c,
        batch_count
    };
    const RRNLBCublasLtPlan plan = rrnlb_get_cublaslt_plan(key);
    if (!plan.available) {
        if (rrnlb_cublaslt_strict_enabled()) {
            std::ostringstream oss;
            oss << "RRNLB cublasLt plan unavailable for "
                << "opA=" << key.op_a
                << " opB=" << key.op_b
                << " m=" << key.m
                << " n=" << key.n
                << " k=" << key.k
                << " lda=" << key.lda
                << " ldb=" << key.ldb
                << " ldc=" << key.ldc
                << " strideA=" << key.stride_a
                << " strideB=" << key.stride_b
                << " strideC=" << key.stride_c
                << " batch=" << key.batch_count;
            throw std::runtime_error(oss.str());
        }
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    void* workspace = rrnlb_get_cublaslt_workspace(plan.workspace_bytes);
    if (plan.workspace_bytes > 0 && workspace == nullptr) {
        if (rrnlb_cublaslt_strict_enabled()) {
            throw std::runtime_error("RRNLB cublasLt workspace allocation failed.");
        }
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    return cublasLtMatmul(
        lt_handle,
        plan.op_desc,
        alpha,
        a,
        plan.a_layout,
        b,
        plan.b_layout,
        beta,
        c,
        plan.c_layout,
        c,
        plan.d_layout,
        &plan.algo,
        workspace,
        plan.workspace_bytes,
        stream);
}

template <typename Precision>
auto rrnlb_cublas_gemm_strided_batched(
    cublasHandle_t handle,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int m,
    int n,
    int k,
    const Precision* alpha,
    const Precision* a,
    int lda,
    long long int stride_a,
    const Precision* b,
    int ldb,
    long long int stride_b,
    const Precision* beta,
    Precision* c,
    int ldc,
    long long int stride_c,
    int batch_count) -> cublasStatus_t;

template <>
auto rrnlb_cublas_gemm_strided_batched<double>(
    cublasHandle_t handle,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int m,
    int n,
    int k,
    const double* alpha,
    const double* a,
    int lda,
    long long int stride_a,
    const double* b,
    int ldb,
    long long int stride_b,
    const double* beta,
    double* c,
    int ldc,
    long long int stride_c,
    int batch_count) -> cublasStatus_t
{
    return cublasDgemmStridedBatched(
        handle, op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_count);
}

template <>
auto rrnlb_cublas_gemm_strided_batched<float>(
    cublasHandle_t handle,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* a,
    int lda,
    long long int stride_a,
    const float* b,
    int ldb,
    long long int stride_b,
    const float* beta,
    float* c,
    int ldc,
    long long int stride_c,
    int batch_count) -> cublasStatus_t
{
    return cublasSgemmStridedBatched(
        handle, op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_count);
}
#endif

template <typename T>
KOKKOS_INLINE_FUNCTION
auto rrnlb_sigmoid(const T x) -> T
{
    if (x >= static_cast<T>(0.0)) {
        const T z = std::exp(-x);
        return static_cast<T>(1.0) / (static_cast<T>(1.0) + z);
    }
    const T z = std::exp(x);
    return z / (static_cast<T>(1.0) + z);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
auto rrnlb_silu(const T x) -> T
{
    return x * rrnlb_sigmoid(x);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
auto rrnlb_silu_deriv(const T x) -> T
{
    const T s = rrnlb_sigmoid(x);
    return s + x * s * (static_cast<T>(1.0) - s);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
auto rrnlb_eval_spline(
    const Kokkos::View<T****,Kokkos::LayoutRight>& coeffs,
    const int pair_index,
    const int interval,
    const int comp,
    const T x,
    const T xx,
    const T xxx) -> T
{
    const T c0 = coeffs(pair_index, interval, 0, comp);
    const T c1 = coeffs(pair_index, interval, 1, comp);
    const T c2 = coeffs(pair_index, interval, 2, comp);
    const T c3 = coeffs(pair_index, interval, 3, comp);
    return c0 + c1 * x + c2 * xx + c3 * xxx;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
auto rrnlb_eval_spline_deriv(
    const Kokkos::View<T****,Kokkos::LayoutRight>& coeffs,
    const int pair_index,
    const int interval,
    const int comp,
    const T x,
    const T xx) -> T
{
    const T c1 = coeffs(pair_index, interval, 1, comp);
    const T c2 = coeffs(pair_index, interval, 2, comp);
    const T c3 = coeffs(pair_index, interval, 3, comp);
    return c1 + static_cast<T>(2.0) * c2 * x + static_cast<T>(3.0) * c3 * xx;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
auto rrnlb_eval_spline_scalar(
    const Kokkos::View<T***,Kokkos::LayoutRight>& coeffs,
    const int pair_index,
    const int interval,
    const T x,
    const T xx,
    const T xxx) -> T
{
    const T c0 = coeffs(pair_index, interval, 0);
    const T c1 = coeffs(pair_index, interval, 1);
    const T c2 = coeffs(pair_index, interval, 2);
    const T c3 = coeffs(pair_index, interval, 3);
    return c0 + c1 * x + c2 * xx + c3 * xxx;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
auto rrnlb_eval_spline_scalar_deriv(
    const Kokkos::View<T***,Kokkos::LayoutRight>& coeffs,
    const int pair_index,
    const int interval,
    const T x,
    const T xx) -> T
{
    const T c1 = coeffs(pair_index, interval, 1);
    const T c2 = coeffs(pair_index, interval, 2);
    const T c3 = coeffs(pair_index, interval, 3);
    return c1 + static_cast<T>(2.0) * c2 * x + static_cast<T>(3.0) * c3 * xx;
}

template <typename T>
struct RRNLBTermReduction {
    T dE_dr;
    T y_adj;

    KOKKOS_INLINE_FUNCTION
    RRNLBTermReduction()
        : dE_dr(static_cast<T>(0.0)),
          y_adj(static_cast<T>(0.0))
    {}

    KOKKOS_INLINE_FUNCTION
    auto operator+=(const RRNLBTermReduction& rhs) -> void
    {
        dE_dr += rhs.dE_dr;
        y_adj += rhs.y_adj;
    }
};

template <typename Precision, typename AccumPrecision>
auto make_rrnlb_linear_kokkos(
    const MACE::RRNLBLinear& src,
    const std::string& label)
    -> typename MACEKokkos<Precision, AccumPrecision>::RRNLBLinearKokkos
{
    using Linear = typename MACEKokkos<Precision, AccumPrecision>::RRNLBLinearKokkos;
    Linear dst;
    dst.dim_in = src.dim_in;
    dst.dim_out = src.dim_out;
    dst.h_parts_in_offset.resize(src.parts_in.size());
    dst.h_parts_in_mul.resize(src.parts_in.size());
    dst.h_parts_in_l.resize(src.parts_in.size());
    dst.h_parts_out_offset.resize(src.parts_out.size());
    dst.h_parts_out_mul.resize(src.parts_out.size());
    dst.h_parts_out_l.resize(src.parts_out.size());

    const int num_parts_in = static_cast<int>(src.parts_in.size());
    const int num_parts_out = static_cast<int>(src.parts_out.size());
    Kokkos::realloc(dst.parts_in_offset, num_parts_in);
    Kokkos::realloc(dst.parts_in_mul, num_parts_in);
    Kokkos::realloc(dst.parts_in_l, num_parts_in);
    Kokkos::realloc(dst.parts_in_dim, num_parts_in);
    Kokkos::realloc(dst.parts_out_offset, num_parts_out);
    Kokkos::realloc(dst.parts_out_mul, num_parts_out);
    Kokkos::realloc(dst.parts_out_l, num_parts_out);
    Kokkos::realloc(dst.parts_out_dim, num_parts_out);

    auto h_parts_in_offset = Kokkos::create_mirror_view(dst.parts_in_offset);
    auto h_parts_in_mul = Kokkos::create_mirror_view(dst.parts_in_mul);
    auto h_parts_in_l = Kokkos::create_mirror_view(dst.parts_in_l);
    auto h_parts_in_dim = Kokkos::create_mirror_view(dst.parts_in_dim);
    auto h_parts_out_offset = Kokkos::create_mirror_view(dst.parts_out_offset);
    auto h_parts_out_mul = Kokkos::create_mirror_view(dst.parts_out_mul);
    auto h_parts_out_l = Kokkos::create_mirror_view(dst.parts_out_l);
    auto h_parts_out_dim = Kokkos::create_mirror_view(dst.parts_out_dim);
    for (int p = 0; p < num_parts_in; ++p) {
        h_parts_in_offset(p) = src.parts_in[p].offset;
        h_parts_in_mul(p) = src.parts_in[p].mul;
        h_parts_in_l(p) = src.parts_in[p].l;
        h_parts_in_dim(p) = src.parts_in[p].dim;
        dst.h_parts_in_offset[p] = src.parts_in[p].offset;
        dst.h_parts_in_mul[p] = src.parts_in[p].mul;
        dst.h_parts_in_l[p] = src.parts_in[p].l;
    }
    for (int p = 0; p < num_parts_out; ++p) {
        h_parts_out_offset(p) = src.parts_out[p].offset;
        h_parts_out_mul(p) = src.parts_out[p].mul;
        h_parts_out_l(p) = src.parts_out[p].l;
        h_parts_out_dim(p) = src.parts_out[p].dim;
        dst.h_parts_out_offset[p] = src.parts_out[p].offset;
        dst.h_parts_out_mul[p] = src.parts_out[p].mul;
        dst.h_parts_out_l[p] = src.parts_out[p].l;
    }
    Kokkos::deep_copy(dst.parts_in_offset, h_parts_in_offset);
    Kokkos::deep_copy(dst.parts_in_mul, h_parts_in_mul);
    Kokkos::deep_copy(dst.parts_in_l, h_parts_in_l);
    Kokkos::deep_copy(dst.parts_in_dim, h_parts_in_dim);
    Kokkos::deep_copy(dst.parts_out_offset, h_parts_out_offset);
    Kokkos::deep_copy(dst.parts_out_mul, h_parts_out_mul);
    Kokkos::deep_copy(dst.parts_out_l, h_parts_out_l);
    Kokkos::deep_copy(dst.parts_out_dim, h_parts_out_dim);

    std::vector<unsigned char> active_in_mask(dst.dim_in, static_cast<unsigned char>(0));

    const int num_ins = static_cast<int>(src.instructions.size());
    dst.num_ins = num_ins;
    dst.h_ins_in_offset.resize(num_ins);
    dst.h_ins_out_offset.resize(num_ins);
    dst.h_ins_mul_in.resize(num_ins);
    dst.h_ins_mul_out.resize(num_ins);
    dst.h_ins_ir_dim.resize(num_ins);
    dst.h_ins_weight_offset.resize(num_ins);
    dst.h_ins_path_weight.resize(num_ins);
    Kokkos::realloc(dst.ins_in_offset, num_ins);
    Kokkos::realloc(dst.ins_out_offset, num_ins);
    Kokkos::realloc(dst.ins_mul_in, num_ins);
    Kokkos::realloc(dst.ins_mul_out, num_ins);
    Kokkos::realloc(dst.ins_ir_dim, num_ins);
    Kokkos::realloc(dst.ins_weight_offset, num_ins);
    Kokkos::realloc(dst.ins_path_weight, num_ins);
    auto h_ins_in_offset = Kokkos::create_mirror_view(dst.ins_in_offset);
    auto h_ins_out_offset = Kokkos::create_mirror_view(dst.ins_out_offset);
    auto h_ins_mul_in = Kokkos::create_mirror_view(dst.ins_mul_in);
    auto h_ins_mul_out = Kokkos::create_mirror_view(dst.ins_mul_out);
    auto h_ins_ir_dim = Kokkos::create_mirror_view(dst.ins_ir_dim);
    auto h_ins_weight_offset = Kokkos::create_mirror_view(dst.ins_weight_offset);
    auto h_ins_path_weight = Kokkos::create_mirror_view(dst.ins_path_weight);
    int total_weights = 0;
    for (const auto& ins : src.instructions) {
        total_weights += static_cast<int>(ins.weights.size());
    }
    Kokkos::realloc(dst.ins_weights, total_weights);
    auto h_ins_weights = Kokkos::create_mirror_view(dst.ins_weights);
    int offset = 0;
    for (int q = 0; q < num_ins; ++q) {
        const auto& ins = src.instructions[q];
        const auto& in_part = src.parts_in[ins.i_in];
        const auto& out_part = src.parts_out[ins.i_out];
        const int ir_dim = 2 * in_part.l + 1;
        h_ins_in_offset(q) = in_part.offset;
        h_ins_out_offset(q) = out_part.offset;
        h_ins_mul_in(q) = ins.mul_in;
        h_ins_mul_out(q) = ins.mul_out;
        h_ins_ir_dim(q) = ir_dim;
        h_ins_weight_offset(q) = offset;
        h_ins_path_weight(q) = static_cast<Precision>(ins.path_weight);
        dst.h_ins_in_offset[q] = in_part.offset;
        dst.h_ins_out_offset[q] = out_part.offset;
        dst.h_ins_mul_in[q] = ins.mul_in;
        dst.h_ins_mul_out[q] = ins.mul_out;
        dst.h_ins_ir_dim[q] = ir_dim;
        dst.h_ins_weight_offset[q] = offset;
        dst.h_ins_path_weight[q] = static_cast<Precision>(ins.path_weight);
        for (int w = 0; w < static_cast<int>(ins.weights.size()); ++w) {
            h_ins_weights(offset + w) = static_cast<Precision>(ins.weights[w]);
        }
        offset += static_cast<int>(ins.weights.size());
    }
    Kokkos::deep_copy(dst.ins_in_offset, h_ins_in_offset);
    Kokkos::deep_copy(dst.ins_out_offset, h_ins_out_offset);
    Kokkos::deep_copy(dst.ins_mul_in, h_ins_mul_in);
    Kokkos::deep_copy(dst.ins_mul_out, h_ins_mul_out);
    Kokkos::deep_copy(dst.ins_ir_dim, h_ins_ir_dim);
    Kokkos::deep_copy(dst.ins_weight_offset, h_ins_weight_offset);
    Kokkos::deep_copy(dst.ins_path_weight, h_ins_path_weight);
    Kokkos::deep_copy(dst.ins_weights, h_ins_weights);

    for (int q = 0; q < num_ins; ++q) {
        const int in_offset = dst.h_ins_in_offset[q];
        const int mul_in = dst.h_ins_mul_in[q];
        const int ir_dim = dst.h_ins_ir_dim[q];
        for (int k = 0; k < mul_in; ++k) {
            for (int m = 0; m < ir_dim; ++m) {
                const int in_idx = in_offset + k * ir_dim + m;
                if (in_idx >= 0 && in_idx < dst.dim_in) {
                    active_in_mask[in_idx] = static_cast<unsigned char>(1);
                }
            }
        }
    }
    std::vector<int> active_in_indices;
    active_in_indices.reserve(dst.dim_in);
    for (int p = 0; p < dst.dim_in; ++p) {
        if (active_in_mask[p] != static_cast<unsigned char>(0)) {
            active_in_indices.push_back(p);
        }
    }
    if (active_in_indices.empty()) {
        active_in_indices.resize(dst.dim_in);
        for (int p = 0; p < dst.dim_in; ++p) active_in_indices[p] = p;
    }
    dst.active_in_count = static_cast<int>(active_in_indices.size());
    dst.active_in_indices = toKokkosView(
        (label + "_active_in_indices").c_str(),
        active_in_indices);

    std::vector<unsigned char> active_out_mask(dst.dim_out, static_cast<unsigned char>(0));
    for (int q = 0; q < num_ins; ++q) {
        const int out_offset = dst.h_ins_out_offset[q];
        const int mul_out = dst.h_ins_mul_out[q];
        const int ir_dim = dst.h_ins_ir_dim[q];
        for (int k = 0; k < mul_out; ++k) {
            for (int m = 0; m < ir_dim; ++m) {
                const int out_idx = out_offset + k * ir_dim + m;
                if (out_idx >= 0 && out_idx < dst.dim_out) {
                    active_out_mask[out_idx] = static_cast<unsigned char>(1);
                }
            }
        }
    }

    // Build grouped instruction plans for portable batched kernels:
    // - forward grouped by output part (exclusive y-block ownership)
    // - transpose grouped by input part (exclusive x_adj-block ownership)
    std::vector<std::vector<int>> fwd_lists(num_parts_out);
    std::vector<std::vector<int>> rev_lists(num_parts_in);
    for (int q = 0; q < num_ins; ++q) {
        const auto& ins = src.instructions[q];
        fwd_lists[ins.i_out].push_back(q);
        rev_lists[ins.i_in].push_back(q);
    }

    std::vector<int> fwd_group_out_offset;
    std::vector<int> fwd_group_mul_out;
    std::vector<int> fwd_group_ir_dim;
    std::vector<int> fwd_group_first_ins;
    std::vector<int> fwd_group_num_ins;
    std::vector<int> fwd_group_ins_index;
    for (int p = 0; p < num_parts_out; ++p) {
        const auto& group = fwd_lists[p];
        if (group.empty()) continue;
        const int mul_out = dst.h_ins_mul_out[group.front()];
        const int ir_dim = dst.h_ins_ir_dim[group.front()];
        fwd_group_out_offset.push_back(src.parts_out[p].offset);
        fwd_group_mul_out.push_back(mul_out);
        fwd_group_ir_dim.push_back(ir_dim);
        fwd_group_first_ins.push_back(static_cast<int>(fwd_group_ins_index.size()));
        fwd_group_num_ins.push_back(static_cast<int>(group.size()));
        fwd_group_ins_index.insert(fwd_group_ins_index.end(), group.begin(), group.end());
    }
    dst.fwd_group_out_offset = toKokkosView((label + "_fwd_group_out_offset").c_str(), fwd_group_out_offset);
    dst.fwd_group_mul_out = toKokkosView((label + "_fwd_group_mul_out").c_str(), fwd_group_mul_out);
    dst.fwd_group_ir_dim = toKokkosView((label + "_fwd_group_ir_dim").c_str(), fwd_group_ir_dim);
    dst.fwd_group_first_ins = toKokkosView((label + "_fwd_group_first_ins").c_str(), fwd_group_first_ins);
    dst.fwd_group_num_ins = toKokkosView((label + "_fwd_group_num_ins").c_str(), fwd_group_num_ins);
    dst.fwd_group_ins_index = toKokkosView((label + "_fwd_group_ins_index").c_str(), fwd_group_ins_index);
    dst.h_fwd_group_out_offset = fwd_group_out_offset;
    dst.h_fwd_group_mul_out = fwd_group_mul_out;
    dst.h_fwd_group_ir_dim = fwd_group_ir_dim;
    dst.h_fwd_group_first_ins = fwd_group_first_ins;
    dst.h_fwd_group_num_ins = fwd_group_num_ins;
    dst.h_fwd_group_ins_index = fwd_group_ins_index;

    std::vector<int> rev_group_in_offset;
    std::vector<int> rev_group_mul_in;
    std::vector<int> rev_group_ir_dim;
    std::vector<int> rev_group_first_ins;
    std::vector<int> rev_group_num_ins;
    std::vector<int> rev_group_ins_index;
    for (int p = 0; p < num_parts_in; ++p) {
        const auto& group = rev_lists[p];
        if (group.empty()) continue;
        const int mul_in = dst.h_ins_mul_in[group.front()];
        const int ir_dim = dst.h_ins_ir_dim[group.front()];
        rev_group_in_offset.push_back(src.parts_in[p].offset);
        rev_group_mul_in.push_back(mul_in);
        rev_group_ir_dim.push_back(ir_dim);
        rev_group_first_ins.push_back(static_cast<int>(rev_group_ins_index.size()));
        rev_group_num_ins.push_back(static_cast<int>(group.size()));
        rev_group_ins_index.insert(rev_group_ins_index.end(), group.begin(), group.end());
    }
    dst.rev_group_in_offset = toKokkosView((label + "_rev_group_in_offset").c_str(), rev_group_in_offset);
    dst.rev_group_mul_in = toKokkosView((label + "_rev_group_mul_in").c_str(), rev_group_mul_in);
    dst.rev_group_ir_dim = toKokkosView((label + "_rev_group_ir_dim").c_str(), rev_group_ir_dim);
    dst.rev_group_first_ins = toKokkosView((label + "_rev_group_first_ins").c_str(), rev_group_first_ins);
    dst.rev_group_num_ins = toKokkosView((label + "_rev_group_num_ins").c_str(), rev_group_num_ins);
    dst.rev_group_ins_index = toKokkosView((label + "_rev_group_ins_index").c_str(), rev_group_ins_index);
    dst.h_rev_group_in_offset = rev_group_in_offset;
    dst.h_rev_group_mul_in = rev_group_mul_in;
    dst.h_rev_group_ir_dim = rev_group_ir_dim;
    dst.h_rev_group_first_ins = rev_group_first_ins;
    dst.h_rev_group_num_ins = rev_group_num_ins;
    dst.h_rev_group_ins_index = rev_group_ins_index;

    // Build packed per-group weights used by the grouped dense GEMM path.
    // Forward groups are packed as [K_total, N_out] where K_total is the
    // concatenation of all instruction mul_in blocks for the output group.
    const int num_fwd_groups = static_cast<int>(dst.h_fwd_group_first_ins.size());
    dst.h_fwd_group_k_total.assign(num_fwd_groups, 0);
    dst.h_fwd_group_pack_weight_offset.assign(num_fwd_groups, 0);
    dst.h_fwd_group_ins_k_offset.assign(dst.h_fwd_group_ins_index.size(), 0);
    int total_fwd_pack_weights = 0;
    for (int g = 0; g < num_fwd_groups; ++g) {
        const int first = dst.h_fwd_group_first_ins[g];
        const int count = dst.h_fwd_group_num_ins[g];
        int k_total = 0;
        for (int t = 0; t < count; ++t) {
            const int idx = first + t;
            const int q = dst.h_fwd_group_ins_index[idx];
            dst.h_fwd_group_ins_k_offset[idx] = k_total;
            k_total += dst.h_ins_mul_in[q];
        }
        dst.h_fwd_group_k_total[g] = k_total;
        dst.h_fwd_group_pack_weight_offset[g] = total_fwd_pack_weights;
        total_fwd_pack_weights += k_total * dst.h_fwd_group_mul_out[g];
    }
    Kokkos::realloc(dst.fwd_group_pack_weights, total_fwd_pack_weights);
    auto h_fwd_group_pack_weights = Kokkos::create_mirror_view(dst.fwd_group_pack_weights);
    for (int g = 0; g < num_fwd_groups; ++g) {
        const int first = dst.h_fwd_group_first_ins[g];
        const int count = dst.h_fwd_group_num_ins[g];
        const int n_out = dst.h_fwd_group_mul_out[g];
        const int base = dst.h_fwd_group_pack_weight_offset[g];
        for (int t = 0; t < count; ++t) {
            const int idx = first + t;
            const int q = dst.h_fwd_group_ins_index[idx];
            const int k_off = dst.h_fwd_group_ins_k_offset[idx];
            const int mul_in = dst.h_ins_mul_in[q];
            const int mul_out = dst.h_ins_mul_out[q];
            const int w_off = dst.h_ins_weight_offset[q];
            const Precision alpha = dst.h_ins_path_weight[q];
            if (mul_out != n_out) {
                throw std::runtime_error("RRNLB forward packed GEMM group has inconsistent mul_out.");
            }
            for (int kin = 0; kin < mul_in; ++kin) {
                for (int out = 0; out < n_out; ++out) {
                    h_fwd_group_pack_weights(base + (k_off + kin) * n_out + out) =
                        alpha * h_ins_weights(w_off + kin * mul_out + out);
                }
            }
        }
    }
    Kokkos::deep_copy(dst.fwd_group_pack_weights, h_fwd_group_pack_weights);

    // Reverse groups are packed as [N_total, K_in] where N_total is the
    // concatenation of all instruction mul_out blocks for the input group.
    const int num_rev_groups = static_cast<int>(dst.h_rev_group_first_ins.size());
    dst.h_rev_group_n_total.assign(num_rev_groups, 0);
    dst.h_rev_group_pack_weight_offset.assign(num_rev_groups, 0);
    dst.h_rev_group_ins_n_offset.assign(dst.h_rev_group_ins_index.size(), 0);
    int total_rev_pack_weights = 0;
    for (int g = 0; g < num_rev_groups; ++g) {
        const int first = dst.h_rev_group_first_ins[g];
        const int count = dst.h_rev_group_num_ins[g];
        int n_total = 0;
        for (int t = 0; t < count; ++t) {
            const int idx = first + t;
            const int q = dst.h_rev_group_ins_index[idx];
            dst.h_rev_group_ins_n_offset[idx] = n_total;
            n_total += dst.h_ins_mul_out[q];
        }
        dst.h_rev_group_n_total[g] = n_total;
        dst.h_rev_group_pack_weight_offset[g] = total_rev_pack_weights;
        total_rev_pack_weights += n_total * dst.h_rev_group_mul_in[g];
    }
    Kokkos::realloc(dst.rev_group_pack_weights, total_rev_pack_weights);
    auto h_rev_group_pack_weights = Kokkos::create_mirror_view(dst.rev_group_pack_weights);
    for (int g = 0; g < num_rev_groups; ++g) {
        const int first = dst.h_rev_group_first_ins[g];
        const int count = dst.h_rev_group_num_ins[g];
        const int k_in = dst.h_rev_group_mul_in[g];
        const int base = dst.h_rev_group_pack_weight_offset[g];
        for (int t = 0; t < count; ++t) {
            const int idx = first + t;
            const int q = dst.h_rev_group_ins_index[idx];
            const int n_off = dst.h_rev_group_ins_n_offset[idx];
            const int mul_in = dst.h_ins_mul_in[q];
            const int mul_out = dst.h_ins_mul_out[q];
            const int w_off = dst.h_ins_weight_offset[q];
            const Precision alpha = dst.h_ins_path_weight[q];
            if (mul_in != k_in) {
                throw std::runtime_error("RRNLB reverse packed GEMM group has inconsistent mul_in.");
            }
            for (int out = 0; out < mul_out; ++out) {
                for (int kin = 0; kin < k_in; ++kin) {
                    h_rev_group_pack_weights(base + (n_off + out) * k_in + kin) =
                        alpha * h_ins_weights(w_off + kin * mul_out + out);
                }
            }
        }
    }
    Kokkos::deep_copy(dst.rev_group_pack_weights, h_rev_group_pack_weights);

    // Validate packed metadata once at load time so hot loops do not recheck it.
    dst.fwd_group_pack_valid =
        dst.h_fwd_group_ins_k_offset.size() == dst.h_fwd_group_ins_index.size();
    for (int g = 0; dst.fwd_group_pack_valid && g < num_fwd_groups; ++g) {
        const int ir_dim = dst.h_fwd_group_ir_dim[g];
        const int k_total = dst.h_fwd_group_k_total[g];
        const int n_out = dst.h_fwd_group_mul_out[g];
        const int first = dst.h_fwd_group_first_ins[g];
        const int count = dst.h_fwd_group_num_ins[g];
        if (ir_dim <= 0 || k_total <= 0 || n_out <= 0 || first < 0 || count <= 0
            || first + count > static_cast<int>(dst.h_fwd_group_ins_index.size())) {
            dst.fwd_group_pack_valid = false;
            break;
        }
        dst.fwd_group_max_ir_dim = std::max(dst.fwd_group_max_ir_dim, ir_dim);
        dst.fwd_group_max_k_total = std::max(dst.fwd_group_max_k_total, k_total);
        dst.fwd_group_max_mul_out = std::max(dst.fwd_group_max_mul_out, n_out);
        for (int t = 0; t < count; ++t) {
            const int idx = first + t;
            const int q = dst.h_fwd_group_ins_index[idx];
            if (q < 0 || q >= num_ins) {
                dst.fwd_group_pack_valid = false;
                break;
            }
            const int mul_in = dst.h_ins_mul_in[q];
            const int q_ir_dim = dst.h_ins_ir_dim[q];
            const int in_offset = dst.h_ins_in_offset[q];
            const int out_offset = dst.h_ins_out_offset[q];
            const int k_offset = dst.h_fwd_group_ins_k_offset[idx];
            if (mul_in <= 0
                || q_ir_dim != ir_dim
                || in_offset < 0
                || out_offset < 0
                || in_offset + mul_in * ir_dim > dst.dim_in
                || out_offset + n_out * ir_dim > dst.dim_out
                || k_offset < 0
                || k_offset + mul_in > k_total) {
                dst.fwd_group_pack_valid = false;
                break;
            }
        }
    }

    dst.rev_group_pack_valid =
        dst.h_rev_group_ins_n_offset.size() == dst.h_rev_group_ins_index.size();
    for (int g = 0; dst.rev_group_pack_valid && g < num_rev_groups; ++g) {
        const int ir_dim = dst.h_rev_group_ir_dim[g];
        const int n_total = dst.h_rev_group_n_total[g];
        const int k_in = dst.h_rev_group_mul_in[g];
        const int first = dst.h_rev_group_first_ins[g];
        const int count = dst.h_rev_group_num_ins[g];
        if (ir_dim <= 0 || n_total <= 0 || k_in <= 0 || first < 0 || count <= 0
            || first + count > static_cast<int>(dst.h_rev_group_ins_index.size())) {
            dst.rev_group_pack_valid = false;
            break;
        }
        dst.rev_group_max_ir_dim = std::max(dst.rev_group_max_ir_dim, ir_dim);
        dst.rev_group_max_n_total = std::max(dst.rev_group_max_n_total, n_total);
        dst.rev_group_max_mul_in = std::max(dst.rev_group_max_mul_in, k_in);
        for (int t = 0; t < count; ++t) {
            const int idx = first + t;
            const int q = dst.h_rev_group_ins_index[idx];
            if (q < 0 || q >= num_ins) {
                dst.rev_group_pack_valid = false;
                break;
            }
            const int mul_out = dst.h_ins_mul_out[q];
            const int q_ir_dim = dst.h_ins_ir_dim[q];
            const int in_offset = dst.h_rev_group_in_offset[g];
            const int out_offset = dst.h_ins_out_offset[q];
            const int n_offset = dst.h_rev_group_ins_n_offset[idx];
            if (mul_out <= 0
                || q_ir_dim != ir_dim
                || in_offset < 0
                || out_offset < 0
                || in_offset + k_in * ir_dim > dst.dim_in
                || out_offset + mul_out * ir_dim > dst.dim_out
                || n_offset < 0
                || n_offset + mul_out > n_total) {
                dst.rev_group_pack_valid = false;
                break;
            }
        }
    }

    std::vector<int> bias_indices;
    std::vector<Precision> bias_values;
    if (!src.bias.empty()) {
        int bias_offset = 0;
        for (const auto& out_part : src.parts_out) {
            if (out_part.l != 0) continue;
            for (int k = 0; k < out_part.mul; ++k) {
                bias_indices.push_back(out_part.offset + k);
                bias_values.push_back(static_cast<Precision>(src.bias[bias_offset + k]));
            }
            bias_offset += out_part.mul;
        }
    }
    dst.bias_indices = toKokkosView((label + "_bias_indices").c_str(), bias_indices);
    dst.bias_values = toKokkosView((label + "_bias_values").c_str(), bias_values);
    for (const int idx : bias_indices) {
        if (idx >= 0 && idx < dst.dim_out) {
            active_out_mask[idx] = static_cast<unsigned char>(1);
        }
    }
    std::vector<int> active_out_indices;
    active_out_indices.reserve(dst.dim_out);
    for (int p = 0; p < dst.dim_out; ++p) {
        if (active_out_mask[p] != static_cast<unsigned char>(0)) {
            active_out_indices.push_back(p);
        }
    }
    if (active_out_indices.empty()) {
        active_out_indices.resize(dst.dim_out);
        for (int p = 0; p < dst.dim_out; ++p) active_out_indices[p] = p;
    }
    dst.active_out_count = static_cast<int>(active_out_indices.size());
    dst.active_out_indices = toKokkosView(
        (label + "_active_out_indices").c_str(),
        active_out_indices);
    // Sparse reset is enabled for known sparse-output linears whose inactive
    // channels are expected to remain zero once initialized.
    dst.sparse_output_reset_enabled =
        label.find("_linear_res") != std::string::npos
        || label.find("_skip_tp") != std::string::npos;
    return dst;
}

} // namespace

template <typename Precision, typename AccumPrecision>
MACEKokkos<Precision, AccumPrecision>::MACEKokkos(std::string filename)
{
    load_from_json(filename);
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_record_comm_pack(double seconds)
{
    rrnlb_phase_counters.comm_pack_seconds += seconds;
    rrnlb_phase_counters.comm_pack_calls += 1;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_record_comm_unpack(double seconds)
{
    rrnlb_phase_counters.comm_unpack_seconds += seconds;
    rrnlb_phase_counters.comm_unpack_calls += 1;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_record_workspace_reset(double seconds)
{
    rrnlb_phase_counters.workspace_reset_seconds += seconds;
    rrnlb_phase_counters.workspace_reset_calls += 1;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_record_forward_interaction(double seconds)
{
    rrnlb_phase_counters.forward_interaction_seconds += seconds;
    rrnlb_phase_counters.forward_interaction_calls += 1;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_record_reverse_interaction(double seconds)
{
    rrnlb_phase_counters.reverse_interaction_seconds += seconds;
    rrnlb_phase_counters.reverse_interaction_calls += 1;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_record_linear_forward(double seconds)
{
    rrnlb_phase_counters.linear_forward_seconds += seconds;
    rrnlb_phase_counters.linear_forward_calls += 1;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_record_linear_transpose(double seconds)
{
    rrnlb_phase_counters.linear_transpose_seconds += seconds;
    rrnlb_phase_counters.linear_transpose_calls += 1;
}

template <typename Precision, typename AccumPrecision>
auto MACEKokkos<Precision, AccumPrecision>::rrnlb_take_phase_counters()
    -> RRNLBPhaseCounters
{
    const auto counters = rrnlb_phase_counters;
    rrnlb_phase_counters = RRNLBPhaseCounters{};
    return counters;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_reset_phase_counters()
{
    rrnlb_phase_counters = RRNLBPhaseCounters{};
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_set_neighbor_epoch(long long epoch)
{
    rrnlb_neighbor_epoch = epoch;
}

template <typename Precision, typename AccumPrecision>
auto MACEKokkos<Precision, AccumPrecision>::rrnlb_epoch_topology_fastpath_enabled() const -> bool
{
    return !rrnlb_epoch_topology_validate_enabled();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::ensure_rrnlb_model_static_cache()
{
    if (!interaction_mode_rrnlb) return;
    if (rrnlb_model_static_cache.initialized
        && rrnlb_model_static_cache.revision == rrnlb_model_revision) {
        rrnlb_cache_stats.l0_hits += 1;
        return;
    }
    Kokkos::Timer timer;
    rrnlb_model_static_cache.initialized = true;
    rrnlb_model_static_cache.revision = rrnlb_model_revision;
    rrnlb_cache_stamp.model_rev = rrnlb_model_revision;
    rrnlb_cache_stats.l0_rebuilds += 1;
    rrnlb_cache_stats.l0_rebuild_us += static_cast<long long>(timer.seconds() * 1.0e6);
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::ensure_rrnlb_system_static_cache(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    if (!interaction_mode_rrnlb) return;
    if (rrnlb_system_static_cache.initialized
        && rrnlb_system_static_cache.revision == rrnlb_system_revision) {
        rrnlb_cache_stats.l1_hits += 1;
        return;
    }
    Kokkos::Timer timer;
    int max_type = 0;
    if (num_nodes > 0) {
        Kokkos::parallel_reduce(
            "rrnlb_system_max_type",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_nodes),
            KOKKOS_LAMBDA (const int i, int& lmax) {
                if (node_types(i) > lmax) lmax = node_types(i);
            },
            Kokkos::Max<int>(max_type));
    }
    rrnlb_system_static_cache.initialized = true;
    rrnlb_system_static_cache.revision = rrnlb_system_revision;
    rrnlb_system_static_cache.max_lammps_type_seen = max_type;
    rrnlb_cache_stamp.system_rev = rrnlb_system_revision;
    rrnlb_cache_stats.l1_rebuilds += 1;
    rrnlb_cache_stats.l1_rebuild_us += static_cast<long long>(timer.seconds() * 1.0e6);
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::ensure_rrnlb_epoch_topology_cache(
    const int num_nodes,
    const int total_edges,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> first_neigh_input,
    Kokkos::View<const int*> edge_to_receiver_input,
    bool need_sender_maps,
    int sender_nodes,
    bool force_refresh)
{
    if (!interaction_mode_rrnlb) return;
    if (sender_nodes < 0) sender_nodes = num_nodes;
    const bool force_refresh_enabled =
        force_refresh
        && rrnlb_edge_topology_refresh_mode() == RRNLBEdgeTopologyRefreshMode::Always;
    const int sender_segment_edges = std::max(1, rrnlb_sender_segment_size());
    const bool need_sender_segments = need_sender_maps && total_edges > 0;
    const long long epoch_key =
        rrnlb_neighbor_epoch >= 0 ? rrnlb_neighbor_epoch : (rrnlb_cache_stamp.epoch_rev + 1);
    const bool validate_topology = !rrnlb_epoch_topology_fastpath_enabled();
    const bool has_first_neigh_input = first_neigh_input.extent_int(0) >= num_nodes;
    const bool has_edge_to_receiver_input = edge_to_receiver_input.extent_int(0) >= total_edges;
    unsigned long long topology_sig_nodes = 0;
    unsigned long long topology_sig_edges = 0;
    if (validate_topology && num_nodes > 0) {
        Kokkos::parallel_reduce(
            "rrnlb_epoch_topology_sig_nodes",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_nodes),
            KOKKOS_LAMBDA (const int i, unsigned long long& lsum) {
                const unsigned long long neigh =
                    static_cast<unsigned long long>(num_neigh(i) + 1);
                const unsigned long long first =
                    has_first_neigh_input
                        ? static_cast<unsigned long long>(first_neigh_input(i) + 1)
                        : 0ull;
                const unsigned long long idx = static_cast<unsigned long long>(i + 1);
                const unsigned long long mix =
                    (neigh * 11400714819323198485ull)
                    ^ (first * 14029467366897019727ull)
                    ^ (idx * 1609587929392839161ull);
                lsum += mix;
            },
            topology_sig_nodes);
    }
    if (validate_topology && total_edges > 0) {
        Kokkos::parallel_reduce(
            "rrnlb_epoch_topology_sig_edges",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, total_edges),
            KOKKOS_LAMBDA (const int ij, unsigned long long& lsum) {
                const unsigned long long sender =
                    static_cast<unsigned long long>(neigh_indices(ij) + 1);
                const unsigned long long receiver =
                    has_edge_to_receiver_input
                        ? static_cast<unsigned long long>(edge_to_receiver_input(ij) + 1)
                        : 0ull;
                const unsigned long long edge = static_cast<unsigned long long>(ij + 1);
                const unsigned long long mix =
                    (sender * 9650029242287828579ull)
                    ^ (receiver * 2870177450012600261ull)
                    ^ (edge * 11604937117756816743ull);
                lsum += mix;
            },
            topology_sig_edges);
    }

    const bool cache_hit =
        !force_refresh_enabled
        &&
        rrnlb_epoch_topology_cache.initialized
        && rrnlb_epoch_topology_cache.epoch == epoch_key
        && rrnlb_epoch_topology_cache.num_nodes == num_nodes
        && rrnlb_epoch_topology_cache.sender_nodes == sender_nodes
        && rrnlb_epoch_topology_cache.total_edges == total_edges
        && (!validate_topology
            || (rrnlb_epoch_topology_cache.topology_sig_nodes == topology_sig_nodes
                && rrnlb_epoch_topology_cache.topology_sig_edges == topology_sig_edges))
        && (!need_sender_maps
            || rrnlb_epoch_topology_cache.sender_edge_offsets.extent_int(0) >= sender_nodes + 1
            || total_edges == 0)
        && (!need_sender_maps
            || rrnlb_epoch_topology_cache.sender_edge_indices.extent_int(0) >= total_edges
            || total_edges == 0)
        && (!need_sender_segments
            || (rrnlb_epoch_topology_cache.sender_segment_edges == sender_segment_edges
                && rrnlb_epoch_topology_cache.sender_segment_offsets.extent_int(0) >= sender_nodes + 1
                && rrnlb_epoch_topology_cache.sender_segment_to_sender.extent_int(0)
                    >= rrnlb_epoch_topology_cache.total_sender_segments));
    if (cache_hit) {
        rrnlb_cache_stats.l2_hits += 1;
        rrnlb_first_neigh = rrnlb_epoch_topology_cache.first_neigh;
        rrnlb_edge_to_receiver = rrnlb_epoch_topology_cache.edge_to_receiver;
        rrnlb_sender_edge_counts = rrnlb_epoch_topology_cache.sender_edge_counts;
        rrnlb_sender_edge_offsets = rrnlb_epoch_topology_cache.sender_edge_offsets;
        rrnlb_sender_edge_cursor = rrnlb_epoch_topology_cache.sender_edge_cursor;
        rrnlb_sender_edge_indices = rrnlb_epoch_topology_cache.sender_edge_indices;
        rrnlb_total_edges = total_edges;
        return;
    }

    Kokkos::Timer timer;

    if (rrnlb_epoch_topology_cache.first_neigh.extent_int(0) < num_nodes) {
        Kokkos::realloc(rrnlb_epoch_topology_cache.first_neigh, num_nodes);
    }
    auto first_neigh_cache =
        Kokkos::subview(rrnlb_epoch_topology_cache.first_neigh, Kokkos::make_pair(0, num_nodes));
    if (first_neigh_input.extent_int(0) >= num_nodes) {
        auto first_neigh_src = Kokkos::subview(first_neigh_input, Kokkos::make_pair(0, num_nodes));
        Kokkos::deep_copy(first_neigh_cache, first_neigh_src);
    } else if (num_nodes > 0) {
        Kokkos::parallel_scan(
            "rrnlb_epoch_first_neigh",
            num_nodes,
            KOKKOS_LAMBDA (const int i, int& update, const bool final) {
                if (final) first_neigh_cache(i) = update;
                update += num_neigh(i);
            });
    }

    if (total_edges > 0) {
        if (rrnlb_epoch_topology_cache.edge_to_receiver.extent_int(0) < total_edges) {
            Kokkos::realloc(rrnlb_epoch_topology_cache.edge_to_receiver, total_edges);
        }
        auto edge_recv_cache = Kokkos::subview(
            rrnlb_epoch_topology_cache.edge_to_receiver, Kokkos::make_pair(0, total_edges));
        if (edge_to_receiver_input.extent_int(0) >= total_edges) {
            auto edge_recv_src =
                Kokkos::subview(edge_to_receiver_input, Kokkos::make_pair(0, total_edges));
            Kokkos::deep_copy(edge_recv_cache, edge_recv_src);
        } else {
            Kokkos::parallel_for(
                "rrnlb_epoch_edge_to_receiver",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_nodes),
                KOKKOS_LAMBDA (const int i) {
                    const int ij0 = first_neigh_cache(i);
                    const int n = num_neigh(i);
                    for (int jj = 0; jj < n; ++jj) {
                        edge_recv_cache(ij0 + jj) = i;
                    }
                });
        }
    }

    if (need_sender_maps && sender_nodes > 0) {
        if (rrnlb_epoch_topology_cache.sender_edge_counts.extent_int(0) < sender_nodes) {
            Kokkos::realloc(rrnlb_epoch_topology_cache.sender_edge_counts, sender_nodes);
        }
        auto sender_counts = Kokkos::subview(
            rrnlb_epoch_topology_cache.sender_edge_counts, Kokkos::make_pair(0, sender_nodes));
        Kokkos::deep_copy(sender_counts, 0);
        if (total_edges > 0) {
            Kokkos::parallel_for(
                "rrnlb_epoch_sender_counts",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, total_edges),
                KOKKOS_LAMBDA (const int ij) {
                    const int sender = neigh_indices(ij);
                    if (sender >= 0 && sender < sender_nodes) {
                        Kokkos::atomic_add(&sender_counts(sender), 1);
                    }
                });
        }

        if (rrnlb_epoch_topology_cache.sender_edge_offsets.extent_int(0) < sender_nodes + 1) {
            Kokkos::realloc(rrnlb_epoch_topology_cache.sender_edge_offsets, sender_nodes + 1);
        }
        auto sender_offsets = Kokkos::subview(
            rrnlb_epoch_topology_cache.sender_edge_offsets, Kokkos::make_pair(0, sender_nodes + 1));
        Kokkos::parallel_scan(
            "rrnlb_epoch_sender_offsets",
            sender_nodes + 1,
            KOKKOS_LAMBDA (const int i, int& update, const bool final) {
                if (final) sender_offsets(i) = update;
                if (i < sender_nodes) update += sender_counts(i);
            });

        if (rrnlb_epoch_topology_cache.sender_edge_cursor.extent_int(0) < sender_nodes) {
            Kokkos::realloc(rrnlb_epoch_topology_cache.sender_edge_cursor, sender_nodes);
        }
        auto sender_cursor = Kokkos::subview(
            rrnlb_epoch_topology_cache.sender_edge_cursor, Kokkos::make_pair(0, sender_nodes));
        Kokkos::deep_copy(
            sender_cursor,
            Kokkos::subview(sender_offsets, Kokkos::make_pair(0, sender_nodes)));

        if (rrnlb_epoch_topology_cache.sender_edge_indices.extent_int(0) < total_edges) {
            Kokkos::realloc(rrnlb_epoch_topology_cache.sender_edge_indices, total_edges);
        }
        auto sender_edge_indices = Kokkos::subview(
            rrnlb_epoch_topology_cache.sender_edge_indices, Kokkos::make_pair(0, total_edges));
        if (total_edges > 0) {
            Kokkos::parallel_for(
                "rrnlb_epoch_sender_edges",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, total_edges),
                KOKKOS_LAMBDA (const int ij) {
                    const int sender = neigh_indices(ij);
                    if (sender >= 0 && sender < sender_nodes) {
                        const int pos = Kokkos::atomic_fetch_add(&sender_cursor(sender), 1);
                        sender_edge_indices(pos) = ij;
                    }
                });
        }

        if (rrnlb_epoch_topology_cache.sender_segment_counts.extent_int(0) < sender_nodes) {
            Kokkos::realloc(rrnlb_epoch_topology_cache.sender_segment_counts, sender_nodes);
        }
        auto sender_segment_counts = Kokkos::subview(
            rrnlb_epoch_topology_cache.sender_segment_counts, Kokkos::make_pair(0, sender_nodes));
        Kokkos::deep_copy(sender_segment_counts, 0);
        Kokkos::parallel_for(
            "rrnlb_epoch_sender_segment_counts",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, sender_nodes),
            KOKKOS_LAMBDA (const int sender) {
                const int edge_begin = sender_offsets(sender);
                const int edge_end = sender_offsets(sender + 1);
                const int edge_count = edge_end - edge_begin;
                sender_segment_counts(sender) =
                    (edge_count + sender_segment_edges - 1) / sender_segment_edges;
            });

        if (rrnlb_epoch_topology_cache.sender_segment_offsets.extent_int(0) < sender_nodes + 1) {
            Kokkos::realloc(rrnlb_epoch_topology_cache.sender_segment_offsets, sender_nodes + 1);
        }
        auto sender_segment_offsets = Kokkos::subview(
            rrnlb_epoch_topology_cache.sender_segment_offsets, Kokkos::make_pair(0, sender_nodes + 1));
        Kokkos::deep_copy(sender_segment_offsets, 0);
        Kokkos::parallel_scan(
            "rrnlb_epoch_sender_segment_offsets",
            sender_nodes + 1,
            KOKKOS_LAMBDA (const int i, int& update, const bool final) {
                if (final) sender_segment_offsets(i) = update;
                if (i < sender_nodes) update += sender_segment_counts(i);
            });

        int total_sender_segments = 0;
        if (sender_nodes > 0) {
            auto tail = Kokkos::subview(sender_segment_offsets, sender_nodes);
            auto h_tail = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tail);
            total_sender_segments = h_tail();
        }
        rrnlb_epoch_topology_cache.total_sender_segments = total_sender_segments;
        if (rrnlb_epoch_topology_cache.sender_segment_to_sender.extent_int(0) < total_sender_segments) {
            Kokkos::realloc(rrnlb_epoch_topology_cache.sender_segment_to_sender, total_sender_segments);
        }
        if (total_sender_segments > 0) {
            auto sender_segment_to_sender = Kokkos::subview(
                rrnlb_epoch_topology_cache.sender_segment_to_sender,
                Kokkos::make_pair(0, total_sender_segments));
            Kokkos::parallel_for(
                "rrnlb_epoch_sender_segment_map",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, sender_nodes),
                KOKKOS_LAMBDA (const int sender) {
                    const int segment_begin = sender_segment_offsets(sender);
                    const int segment_end = sender_segment_offsets(sender + 1);
                    for (int segment = segment_begin; segment < segment_end; ++segment) {
                        sender_segment_to_sender(segment) = sender;
                    }
                });
        }
    } else {
        rrnlb_epoch_topology_cache.total_sender_segments = 0;
    }
    if (!need_sender_maps) {
        rrnlb_epoch_topology_cache.total_sender_segments = 0;
    }

    rrnlb_epoch_topology_cache.initialized = true;
    rrnlb_epoch_topology_cache.epoch = epoch_key;
    rrnlb_epoch_topology_cache.topology_sig_nodes = validate_topology ? topology_sig_nodes : 0ull;
    rrnlb_epoch_topology_cache.topology_sig_edges = validate_topology ? topology_sig_edges : 0ull;
    rrnlb_epoch_topology_cache.num_nodes = num_nodes;
    rrnlb_epoch_topology_cache.sender_nodes = sender_nodes;
    rrnlb_epoch_topology_cache.total_edges = total_edges;
    rrnlb_epoch_topology_cache.sender_segment_edges = sender_segment_edges;
    rrnlb_cache_stamp.epoch_rev = epoch_key;
    rrnlb_cache_stats.l2_rebuilds += 1;
    rrnlb_cache_stats.l2_rebuild_us += static_cast<long long>(timer.seconds() * 1.0e6);

    rrnlb_first_neigh = rrnlb_epoch_topology_cache.first_neigh;
    rrnlb_edge_to_receiver = rrnlb_epoch_topology_cache.edge_to_receiver;
    rrnlb_sender_edge_counts = rrnlb_epoch_topology_cache.sender_edge_counts;
    rrnlb_sender_edge_offsets = rrnlb_epoch_topology_cache.sender_edge_offsets;
    rrnlb_sender_edge_cursor = rrnlb_epoch_topology_cache.sender_edge_cursor;
    rrnlb_sender_edge_indices = rrnlb_epoch_topology_cache.sender_edge_indices;
    rrnlb_total_edges = total_edges;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::ensure_rrnlb_scratch_capacity(
    int num_nodes,
    int total_edges,
    int sender_nodes)
{
    if (!interaction_mode_rrnlb) return;
    bool grew = false;
    if (num_nodes > rrnlb_scratch_cache.max_nodes) {
        rrnlb_scratch_cache.max_nodes = num_nodes;
        grew = true;
    }
    if (total_edges > rrnlb_scratch_cache.max_edges) {
        rrnlb_scratch_cache.max_edges = total_edges;
        grew = true;
    }
    if (sender_nodes > rrnlb_scratch_cache.max_sender_nodes) {
        rrnlb_scratch_cache.max_sender_nodes = sender_nodes;
        grew = true;
    }
    if (grew) {
        rrnlb_cache_stats.capacity_growths += 1;
        rrnlb_cache_stamp.capacity_rev += 1;
    }
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::ensure_rrnlb_layer_workspace_capacity(
    RRNLBLayerCacheKokkos& cache,
    int num_nodes,
    int sender_nodes,
    int total_edges)
{
    if (num_nodes > cache.capacity_nodes) cache.capacity_nodes = num_nodes;
    if (sender_nodes > cache.capacity_sender_nodes) cache.capacity_sender_nodes = sender_nodes;
    if (total_edges > cache.capacity_edges) cache.capacity_edges = total_edges;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::bind_rrnlb_layer_active_views(
    RRNLBLayerCacheKokkos& cache,
    int num_nodes,
    int sender_nodes,
    int total_edges)
{
    cache.active_nodes = num_nodes;
    cache.active_sender_nodes = sender_nodes;
    cache.active_edges = total_edges;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::ensure_rrnlb_layer_dispatch_cache(
    const RRNLBLayerKokkos& layer,
    RRNLBLayerCacheKokkos& cache,
    int sender_nodes,
    int total_edges)
{
    const long long epoch_id = rrnlb_cache_stamp.epoch_rev;
    const bool same_epoch =
        cache.dispatch_cache.valid
        && cache.dispatch_cache.epoch_id == epoch_id
        && cache.dispatch_cache.sender_nodes == sender_nodes
        && cache.dispatch_cache.total_edges == total_edges
        && cache.dispatch_cache.total_sender_segments
            == rrnlb_epoch_topology_cache.total_sender_segments;
    if (same_epoch) return;

    // Canonical dispatch metadata is model-shape static; bind once per epoch
    // stamp instead of re-copying to temporary cache buffers.
    cache.dispatch_cache.fwd_active_out_indices = layer.conv_active_out_indices;
    cache.dispatch_cache.fwd_active_out_inverse = layer.conv_active_out_inverse;
    cache.dispatch_cache.rev_active_in_indices = layer.conv_active_in_indices;
    cache.dispatch_cache.rev_active_in_inverse = layer.conv_active_in_inverse;

    cache.dispatch_cache.valid = true;
    cache.dispatch_cache.epoch_id = epoch_id;
    cache.dispatch_cache.sender_nodes = sender_nodes;
    cache.dispatch_cache.total_edges = total_edges;
    cache.dispatch_cache.total_sender_segments = rrnlb_epoch_topology_cache.total_sender_segments;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::precompute_rrnlb_edge_radial_descriptors(
    const int layer_index,
    const int total_edges,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> edge_to_receiver,
    Kokkos::View<const double*> r)
{
    const auto& layer = rrnlb_layers_kokkos[layer_index];
    const int tp_weight_numel = layer.tp_weight_numel;
    const auto tp_spline_coeff = layer.tp_spline_coeff;
    const auto density_spline_coeff = layer.density_spline_coeff;
    const double radial_h = layer.radial_h;
    const int radial_num_intervals = layer.radial_num_intervals;
    const int num_elements_local = this->num_elements;

    // High-water realloc: only grow, never shrink.
    auto& tp_vals = rrnlb_edge_tp_values[layer_index];
    auto& tp_ders = rrnlb_edge_tp_derivs[layer_index];
    auto& den_val = rrnlb_edge_density_value[layer_index];
    auto& den_der = rrnlb_edge_density_deriv[layer_index];

    if (tp_vals.extent_int(0) < total_edges || tp_vals.extent_int(1) < tp_weight_numel) {
        const int alloc_edges = std::max(tp_vals.extent_int(0), total_edges);
        const int alloc_weights = std::max(tp_vals.extent_int(1), tp_weight_numel);
        Kokkos::realloc(tp_vals, alloc_edges, alloc_weights);
        Kokkos::realloc(tp_ders, alloc_edges, alloc_weights);
    }
    if (den_val.extent_int(0) < total_edges) {
        const int alloc_edges = std::max(den_val.extent_int(0), total_edges);
        Kokkos::realloc(den_val, alloc_edges);
        Kokkos::realloc(den_der, alloc_edges);
    }

    const auto edge_recv = edge_to_receiver;
    Kokkos::parallel_for(
        "RRNLB precompute edge radial descriptors",
        Kokkos::TeamPolicy<>(total_edges, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int ij = team_member.league_rank();
            const int i = edge_recv(ij);
            const int pair_index = neigh_types(ij) * num_elements_local + node_types(i);
            const Precision r_ij = static_cast<Precision>(r(ij));
            int interval = static_cast<int>(r_ij / static_cast<Precision>(radial_h));
            if (interval < 0) interval = 0;
            if (interval >= radial_num_intervals) interval = radial_num_intervals - 1;
            const Precision x = r_ij - static_cast<Precision>(radial_h) * interval;
            const Precision xx = x * x;
            const Precision xxx = xx * x;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, tp_weight_numel),
                [&] (const int w) {
                    tp_vals(ij, w) = rrnlb_eval_spline(
                        tp_spline_coeff, pair_index, interval, w, x, xx, xxx);
                    tp_ders(ij, w) = rrnlb_eval_spline_deriv(
                        tp_spline_coeff, pair_index, interval, w, x, xx);
                });

            Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                den_val(ij) = rrnlb_eval_spline_scalar(
                    density_spline_coeff, pair_index, interval, x, xx, xxx);
                den_der(ij) = rrnlb_eval_spline_scalar_deriv(
                    density_spline_coeff, pair_index, interval, x, xx);
            });
        });
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_reset_cache_stats()
{
    rrnlb_cache_stats = RrnlbCacheStats{};
}

template <typename Precision, typename AccumPrecision>
MACEKokkos<Precision, AccumPrecision>::~MACEKokkos()
{
    Kokkos::fence();

    M0_monomials = Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    M0_weights = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    M0_poly_spec = Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    M0_poly_coeff = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    M0_poly_values = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    M0_poly_adjoints = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    rrnlb_M0_poly_adjoints_ap =
        Kokkos::View<Kokkos::View<AccumPrecision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();

    A1_weights = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    A1_weights_trans = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_apply_linear_forward(
    const RRNLBLinearKokkos& linear,
    const int num_nodes,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> x,
    Kokkos::View<Precision**,Kokkos::LayoutRight> y)
{
    if (x.extent(0) < num_nodes || x.extent(1) != linear.dim_in) {
        throw std::runtime_error("RRNLB linear forward input has invalid shape.");
    }
    if (y.extent(0) < num_nodes || y.extent(1) != linear.dim_out) {
        throw std::runtime_error("RRNLB linear forward output has invalid shape.");
    }

    Kokkos::deep_copy(
        Kokkos::subview(y, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
        static_cast<Precision>(0.0));

    const bool want_group_packed_forward =
        num_nodes > 0
        && !linear.h_fwd_group_first_ins.empty()
        && rrnlb_group_packed_should_run(
            num_nodes,
            linear.h_fwd_group_ir_dim,
            linear.h_fwd_group_k_total,
            linear.h_fwd_group_mul_out);
    const bool use_group_packed_forward =
        want_group_packed_forward && linear.fwd_group_pack_valid;
    if (want_group_packed_forward && !linear.fwd_group_pack_valid
        && rrnlb_group_packed_gemm_strict_enabled()) {
        throw std::runtime_error(
            "RRNLB grouped packed GEMM forward path validation failed.");
    }
    if (use_group_packed_forward) {
        const bool packed_work_overflow =
            !rrnlb_group_packed_work_fits_int(
                num_nodes,
                linear.fwd_group_max_ir_dim,
                linear.fwd_group_max_k_total)
            || !rrnlb_group_packed_work_fits_int(
                num_nodes,
                linear.fwd_group_max_ir_dim,
                linear.fwd_group_max_mul_out);
        if (packed_work_overflow) {
            if (rrnlb_group_packed_gemm_strict_enabled()) {
                throw std::runtime_error(
                    "RRNLB grouped packed GEMM forward path work overflow.");
            }
        } else {
            const int num_groups = static_cast<int>(linear.h_fwd_group_first_ins.size());
            const int max_rows = num_nodes * linear.fwd_group_max_ir_dim;
            if (rrnlb_fwd_pack_x_workspace.extent_int(0) < max_rows
                || rrnlb_fwd_pack_x_workspace.extent_int(1) < linear.fwd_group_max_k_total) {
                Kokkos::realloc(
                    rrnlb_fwd_pack_x_workspace,
                    max_rows,
                    linear.fwd_group_max_k_total);
            }
            if (rrnlb_fwd_pack_y_workspace.extent_int(0) < max_rows
                || rrnlb_fwd_pack_y_workspace.extent_int(1) < linear.fwd_group_max_mul_out) {
                Kokkos::realloc(
                    rrnlb_fwd_pack_y_workspace,
                    max_rows,
                    linear.fwd_group_max_mul_out);
            }

            for (int g = 0; g < num_groups; ++g) {
                const int ir_dim = linear.h_fwd_group_ir_dim[g];
                const int k_total = linear.h_fwd_group_k_total[g];
                const int n_out = linear.h_fwd_group_mul_out[g];
                const int out_offset = linear.h_fwd_group_out_offset[g];
                const int first = linear.h_fwd_group_first_ins[g];
                const int count = linear.h_fwd_group_num_ins[g];
                const int rows = num_nodes * ir_dim;
                auto xg = Kokkos::subview(
                    rrnlb_fwd_pack_x_workspace,
                    make_pair(0, rows),
                    make_pair(0, k_total));
                auto yg = Kokkos::subview(
                    rrnlb_fwd_pack_y_workspace,
                    make_pair(0, rows),
                    make_pair(0, n_out));

                for (int t = 0; t < count; ++t) {
                    const int idx = first + t;
                    const int q = linear.h_fwd_group_ins_index[idx];
                    const int in_offset = linear.h_ins_in_offset[q];
                    const int mul_in = linear.h_ins_mul_in[q];
                    const int k_offset = linear.h_fwd_group_ins_k_offset[idx];
                    const int pack_work = num_nodes * mul_in * ir_dim;
                    Kokkos::parallel_for(
                        "RRNLB fwd pack input",
                        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, pack_work),
                        KOKKOS_LAMBDA (const int ikm) {
                            const int i = ikm / (mul_in * ir_dim);
                            const int rem = ikm - i * mul_in * ir_dim;
                            const int kin = rem / ir_dim;
                            const int lm = rem - kin * ir_dim;
                            const int row = i * ir_dim + lm;
                            xg(row, k_offset + kin) = x(i, in_offset + kin * ir_dim + lm);
                        });
                }

                const int w_base = linear.h_fwd_group_pack_weight_offset[g];
                const auto wg = Kokkos::View<const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                    linear.fwd_group_pack_weights.data() + w_base,
                    k_total,
                    n_out);
                KokkosBlas::gemm(
                    "N",
                    "N",
                    static_cast<Precision>(1.0),
                    xg,
                    wg,
                    static_cast<Precision>(0.0),
                    yg);

                Kokkos::parallel_for(
                    "RRNLB fwd unpack output",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_nodes * n_out * ir_dim),
                    KOKKOS_LAMBDA (const int iom) {
                        const int i = iom / (n_out * ir_dim);
                        const int rem = iom - i * n_out * ir_dim;
                        const int out = rem / ir_dim;
                        const int lm = rem - out * ir_dim;
                        const int row = i * ir_dim + lm;
                        y(i, out_offset + out * ir_dim + lm) = yg(row, out);
                    });
            }

            const auto bias_indices = linear.bias_indices;
            const auto bias_values = linear.bias_values;
            const int num_bias = bias_indices.extent_int(0);
            if (num_bias > 0) {
                Kokkos::parallel_for(
                    "RRNLB linear forward bias (packed path)",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_nodes * num_bias),
                    KOKKOS_LAMBDA (const int ib) {
                        const int i = ib / num_bias;
                        const int b = ib % num_bias;
                        y(i, bias_indices(b)) += bias_values(b);
                    });
            }
            return;
        }
    }

#if defined(KOKKOS_ENABLE_CUDA) && defined(SYMMETRIX_HAVE_CUBLAS)
    if (rrnlb_cublas_enabled() &&
        Kokkos::SpaceAccessibility<Kokkos::CudaSpace, typename decltype(x)::memory_space>::accessible
               && Kokkos::SpaceAccessibility<Kokkos::CudaSpace, typename decltype(y)::memory_space>::accessible) {
        if (num_nodes > 0) {
            auto exec = Kokkos::DefaultExecutionSpace();
            auto handle = rrnlb_get_cublas_handle();
            cublasLtHandle_t lt_handle = nullptr;
            if constexpr (std::is_same_v<Precision, float>) {
                if (!rrnlb_cublas_tf32_enabled()) {
                    lt_handle = rrnlb_get_cublaslt_handle();
                }
            }
            rrnlb_throw_cublas_error(
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                "rrnlb_apply_linear_forward(cublasSetPointerMode)");
            rrnlb_throw_cublas_error(
                cublasSetStream(handle, exec.cuda_stream()),
                "rrnlb_apply_linear_forward(cublasSetStream)");

            const int x_row_stride = x.extent_int(1);
            const int y_row_stride = y.extent_int(1);
            const auto& h_ins_in_offset = linear.h_ins_in_offset;
            const auto& h_ins_out_offset = linear.h_ins_out_offset;
            const auto& h_ins_mul_in = linear.h_ins_mul_in;
            const auto& h_ins_mul_out = linear.h_ins_mul_out;
            const auto& h_ins_ir_dim = linear.h_ins_ir_dim;
            const auto& h_ins_weight_offset = linear.h_ins_weight_offset;
            const auto& h_ins_path_weight = linear.h_ins_path_weight;
            const auto ins_weights = linear.ins_weights;
            const int num_ins = static_cast<int>(h_ins_in_offset.size());
            const Precision beta = static_cast<Precision>(1.0);

            for (int q = 0; q < num_ins; ++q) {
                const int mul_in = h_ins_mul_in[q];
                const int mul_out = h_ins_mul_out[q];
                const int ir_dim = h_ins_ir_dim[q];
                const int in_offset = h_ins_in_offset[q];
                const int out_offset = h_ins_out_offset[q];
                const int w_offset = h_ins_weight_offset[q];
                const Precision alpha = h_ins_path_weight[q];
                const Precision* x_base = x.data() + in_offset;
                const Precision* w_base = ins_weights.data() + w_offset;
                Precision* y_base = y.data() + out_offset;
                cublasStatus_t gemm_status = CUBLAS_STATUS_NOT_SUPPORTED;
                if constexpr (std::is_same_v<Precision, float>) {
                    if (!rrnlb_cublas_tf32_enabled() && lt_handle != nullptr) {
                        gemm_status = rrnlb_cublaslt_gemm_strided_batched_fp32(
                            lt_handle,
                            exec.cuda_stream(),
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            ir_dim,
                            mul_out,
                            mul_in,
                            &alpha,
                            x_base,
                            ir_dim,
                            static_cast<long long int>(x_row_stride),
                            w_base,
                            mul_out,
                            0,
                            &beta,
                            y_base,
                            ir_dim,
                            static_cast<long long int>(y_row_stride),
                            num_nodes);
                        if (rrnlb_cublaslt_strict_enabled() && gemm_status != CUBLAS_STATUS_SUCCESS) {
                            rrnlb_throw_cublas_error(
                                gemm_status,
                                "rrnlb_apply_linear_forward(cublasLtMatmul)");
                        }
                    }
                }
                if (gemm_status != CUBLAS_STATUS_SUCCESS) {
                    gemm_status = rrnlb_cublas_gemm_strided_batched<Precision>(
                        handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_T,
                        ir_dim,
                        mul_out,
                        mul_in,
                        &alpha,
                        x_base,
                        ir_dim,
                        static_cast<long long int>(x_row_stride),
                        w_base,
                        mul_out,
                        0,
                        &beta,
                        y_base,
                        ir_dim,
                        static_cast<long long int>(y_row_stride),
                        num_nodes);
                }
                rrnlb_throw_cublas_error(
                    gemm_status,
                    "rrnlb_apply_linear_forward(cublasGemmStridedBatched)");
            }
        }

        const auto bias_indices = linear.bias_indices;
        const auto bias_values = linear.bias_values;
        const int num_bias = bias_indices.extent_int(0);
        if (num_bias > 0 && num_nodes > 0) {
            Kokkos::parallel_for(
                "RRNLB linear forward bias (cuBLAS path)",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_nodes * num_bias),
                KOKKOS_LAMBDA (const int ib) {
                    const int i = ib / num_bias;
                    const int b = ib % num_bias;
                    y(i, bias_indices(b)) += bias_values(b);
                });
        }
        return;
    }
#endif

    const auto ins_in_offset = linear.ins_in_offset;
    const auto ins_out_offset = linear.ins_out_offset;
    const auto ins_mul_in = linear.ins_mul_in;
    const auto ins_mul_out = linear.ins_mul_out;
    const auto ins_ir_dim = linear.ins_ir_dim;
    const auto ins_weight_offset = linear.ins_weight_offset;
    const auto ins_path_weight = linear.ins_path_weight;
    const auto ins_weights = linear.ins_weights;
    const auto fwd_group_first_ins = linear.fwd_group_first_ins;
    const auto fwd_group_num_ins = linear.fwd_group_num_ins;
    const auto fwd_group_ins_index = linear.fwd_group_ins_index;
    const auto bias_indices = linear.bias_indices;
    const auto bias_values = linear.bias_values;
    const int num_groups = fwd_group_first_ins.extent_int(0);
    if (num_nodes > 0 && num_groups > 0) {
        Kokkos::parallel_for(
            "RRNLB linear forward",
            Kokkos::TeamPolicy<>(num_nodes * num_groups, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int league = team_member.league_rank();
                const int g = league / num_nodes;
                const int i = league - g * num_nodes;
                auto x_i = Kokkos::subview(x, i, Kokkos::ALL);
                auto y_i = Kokkos::subview(y, i, Kokkos::ALL);
                const int first = fwd_group_first_ins(g);
                const int count = fwd_group_num_ins(g);
                for (int t = 0; t < count; ++t) {
                    const int q = fwd_group_ins_index(first + t);
                    const int mul_in = ins_mul_in(q);
                    const int mul_out = ins_mul_out(q);
                    const int ir_dim = ins_ir_dim(q);
                    const int in_offset = ins_in_offset(q);
                    const int out_offset = ins_out_offset(q);
                    const int w_offset = ins_weight_offset(q);
                    const auto W = Kokkos::View<const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                        ins_weights.data() + w_offset, mul_in, mul_out);
                    const auto X = Kokkos::View<const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                        x_i.data() + in_offset, mul_in, ir_dim);
                    auto Y = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                        y_i.data() + out_offset, mul_out, ir_dim);
                    KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                            KokkosBatched::Trans::Transpose,
                                            KokkosBatched::Trans::NoTranspose,
                                            KokkosBatched::Algo::Gemm::Blocked>
                        ::invoke(
                            team_member,
                            ins_path_weight(q),
                            W,
                            X,
                            static_cast<Precision>(1.0),
                            Y);
                    team_member.team_barrier();
                }
            });
    }

    const int num_bias = bias_indices.extent_int(0);
    if (num_bias > 0 && num_nodes > 0) {
        Kokkos::parallel_for(
            "RRNLB linear forward bias",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_nodes * num_bias),
            KOKKOS_LAMBDA (const int ib) {
                const int i = ib / num_bias;
                const int b = ib % num_bias;
                y(i, bias_indices(b)) += bias_values(b);
            });
    }
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_apply_linear_transpose(
    const RRNLBLinearKokkos& linear,
    const int num_nodes,
    Kokkos::View<const AccumPrecision**,Kokkos::LayoutRight> y_adj,
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> x_adj)
{
    if (y_adj.extent(0) < num_nodes || y_adj.extent(1) != linear.dim_out) {
        throw std::runtime_error("RRNLB linear transpose input has invalid shape.");
    }
    if (x_adj.extent(0) < num_nodes || x_adj.extent(1) != linear.dim_in) {
        throw std::runtime_error("RRNLB linear transpose output has invalid shape.");
    }

    Kokkos::deep_copy(
        Kokkos::subview(x_adj, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
        static_cast<AccumPrecision>(0.0));

    auto run_transpose_precision = [&] (
        Kokkos::View<const Precision**,Kokkos::LayoutStride> y_adj_work,
        Kokkos::View<Precision**,Kokkos::LayoutStride> x_adj_work) {
    const bool want_group_packed_transpose =
        num_nodes > 0
        && !linear.h_rev_group_first_ins.empty()
        && rrnlb_group_packed_should_run(
            num_nodes,
            linear.h_rev_group_ir_dim,
            linear.h_rev_group_n_total,
            linear.h_rev_group_mul_in);
    const bool use_group_packed_transpose =
        want_group_packed_transpose && linear.rev_group_pack_valid;
    if (want_group_packed_transpose && !linear.rev_group_pack_valid
        && rrnlb_group_packed_gemm_strict_enabled()) {
        throw std::runtime_error(
            "RRNLB grouped packed GEMM transpose path validation failed.");
    }
    if (use_group_packed_transpose) {
        const bool packed_work_overflow =
            !rrnlb_group_packed_work_fits_int(
                num_nodes,
                linear.rev_group_max_ir_dim,
                linear.rev_group_max_n_total)
            || !rrnlb_group_packed_work_fits_int(
                num_nodes,
                linear.rev_group_max_ir_dim,
                linear.rev_group_max_mul_in);
        if (packed_work_overflow) {
            if (rrnlb_group_packed_gemm_strict_enabled()) {
                throw std::runtime_error(
                    "RRNLB grouped packed GEMM transpose path work overflow.");
            }
        } else {
            const int num_groups = static_cast<int>(linear.h_rev_group_first_ins.size());
            const int max_rows = num_nodes * linear.rev_group_max_ir_dim;
            if (rrnlb_rev_pack_y_workspace.extent_int(0) < max_rows
                || rrnlb_rev_pack_y_workspace.extent_int(1) < linear.rev_group_max_n_total) {
                Kokkos::realloc(
                    rrnlb_rev_pack_y_workspace,
                    max_rows,
                    linear.rev_group_max_n_total);
            }
            if (rrnlb_rev_pack_x_workspace.extent_int(0) < max_rows
                || rrnlb_rev_pack_x_workspace.extent_int(1) < linear.rev_group_max_mul_in) {
                Kokkos::realloc(
                    rrnlb_rev_pack_x_workspace,
                    max_rows,
                    linear.rev_group_max_mul_in);
            }

            for (int g = 0; g < num_groups; ++g) {
                const int ir_dim = linear.h_rev_group_ir_dim[g];
                const int n_total = linear.h_rev_group_n_total[g];
                const int k_in = linear.h_rev_group_mul_in[g];
                const int in_offset = linear.h_rev_group_in_offset[g];
                const int first = linear.h_rev_group_first_ins[g];
                const int count = linear.h_rev_group_num_ins[g];
                const int rows = num_nodes * ir_dim;
                auto yg = Kokkos::subview(
                    rrnlb_rev_pack_y_workspace,
                    make_pair(0, rows),
                    make_pair(0, n_total));
                auto xg = Kokkos::subview(
                    rrnlb_rev_pack_x_workspace,
                    make_pair(0, rows),
                    make_pair(0, k_in));

                for (int t = 0; t < count; ++t) {
                    const int idx = first + t;
                    const int q = linear.h_rev_group_ins_index[idx];
                    const int out_offset = linear.h_ins_out_offset[q];
                    const int mul_out = linear.h_ins_mul_out[q];
                    const int n_offset = linear.h_rev_group_ins_n_offset[idx];
                    const int pack_work = num_nodes * mul_out * ir_dim;
                    Kokkos::parallel_for(
                        "RRNLB rev pack input",
                        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, pack_work),
                        KOKKOS_LAMBDA (const int iom) {
                            const int i = iom / (mul_out * ir_dim);
                            const int rem = iom - i * mul_out * ir_dim;
                            const int out = rem / ir_dim;
                            const int lm = rem - out * ir_dim;
                            const int row = i * ir_dim + lm;
                            yg(row, n_offset + out) = y_adj_work(i, out_offset + out * ir_dim + lm);
                        });
                }

                const int w_base = linear.h_rev_group_pack_weight_offset[g];
                const auto wg = Kokkos::View<const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                    linear.rev_group_pack_weights.data() + w_base,
                    n_total,
                    k_in);
                KokkosBlas::gemm(
                    "N",
                    "N",
                    static_cast<Precision>(1.0),
                    yg,
                    wg,
                    static_cast<Precision>(0.0),
                    xg);

                Kokkos::parallel_for(
                    "RRNLB rev unpack output",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_nodes * k_in * ir_dim),
                    KOKKOS_LAMBDA (const int ikm) {
                        const int i = ikm / (k_in * ir_dim);
                        const int rem = ikm - i * k_in * ir_dim;
                        const int kin = rem / ir_dim;
                        const int lm = rem - kin * ir_dim;
                        const int row = i * ir_dim + lm;
                        x_adj_work(i, in_offset + kin * ir_dim + lm) = xg(row, kin);
                    });
            }
            return;
        }
    }

#if defined(KOKKOS_ENABLE_CUDA) && defined(SYMMETRIX_HAVE_CUBLAS)
    if (rrnlb_cublas_enabled() &&
        Kokkos::SpaceAccessibility<Kokkos::CudaSpace, typename decltype(y_adj_work)::memory_space>::accessible
               && Kokkos::SpaceAccessibility<Kokkos::CudaSpace, typename decltype(x_adj_work)::memory_space>::accessible) {
        if (num_nodes > 0) {
            auto exec = Kokkos::DefaultExecutionSpace();
            auto handle = rrnlb_get_cublas_handle();
            cublasLtHandle_t lt_handle = nullptr;
            if constexpr (std::is_same_v<Precision, float>) {
                if (!rrnlb_cublas_tf32_enabled()) {
                    lt_handle = rrnlb_get_cublaslt_handle();
                }
            }
            rrnlb_throw_cublas_error(
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                "rrnlb_apply_linear_transpose(cublasSetPointerMode)");
            rrnlb_throw_cublas_error(
                cublasSetStream(handle, exec.cuda_stream()),
                "rrnlb_apply_linear_transpose(cublasSetStream)");

            const std::size_t y_row_stride_sz = y_adj_work.stride(0);
            const std::size_t y_col_stride_sz = y_adj_work.stride(1);
            const std::size_t x_row_stride_sz = x_adj_work.stride(0);
            const std::size_t x_col_stride_sz = x_adj_work.stride(1);
            if (y_col_stride_sz != 1 || x_col_stride_sz != 1) {
                throw std::runtime_error(
                    "RRNLB linear transpose requires contiguous column stride.");
            }
            const long long int y_row_stride = static_cast<long long int>(y_row_stride_sz);
            const long long int x_row_stride = static_cast<long long int>(x_row_stride_sz);
            const auto& h_ins_in_offset = linear.h_ins_in_offset;
            const auto& h_ins_out_offset = linear.h_ins_out_offset;
            const auto& h_ins_mul_in = linear.h_ins_mul_in;
            const auto& h_ins_mul_out = linear.h_ins_mul_out;
            const auto& h_ins_ir_dim = linear.h_ins_ir_dim;
            const auto& h_ins_weight_offset = linear.h_ins_weight_offset;
            const auto& h_ins_path_weight = linear.h_ins_path_weight;
            const auto ins_weights = linear.ins_weights;
            const int num_ins = static_cast<int>(h_ins_in_offset.size());
            const Precision beta = static_cast<Precision>(1.0);

            for (int q = 0; q < num_ins; ++q) {
                const int mul_in = h_ins_mul_in[q];
                const int mul_out = h_ins_mul_out[q];
                const int ir_dim = h_ins_ir_dim[q];
                const int in_offset = h_ins_in_offset[q];
                const int out_offset = h_ins_out_offset[q];
                const int w_offset = h_ins_weight_offset[q];
                const Precision alpha = h_ins_path_weight[q];
                const Precision* y_base = y_adj_work.data() + out_offset;
                const Precision* w_base = ins_weights.data() + w_offset;
                Precision* x_base = x_adj_work.data() + in_offset;
                cublasStatus_t gemm_status = CUBLAS_STATUS_NOT_SUPPORTED;
                if constexpr (std::is_same_v<Precision, float>) {
                    if (!rrnlb_cublas_tf32_enabled() && lt_handle != nullptr) {
                        gemm_status = rrnlb_cublaslt_gemm_strided_batched_fp32(
                            lt_handle,
                            exec.cuda_stream(),
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            ir_dim,
                            mul_in,
                            mul_out,
                            &alpha,
                            y_base,
                            ir_dim,
                            y_row_stride,
                            w_base,
                            mul_out,
                            0,
                            &beta,
                            x_base,
                            ir_dim,
                            x_row_stride,
                            num_nodes);
                        if (rrnlb_cublaslt_strict_enabled() && gemm_status != CUBLAS_STATUS_SUCCESS) {
                            rrnlb_throw_cublas_error(
                                gemm_status,
                                "rrnlb_apply_linear_transpose(cublasLtMatmul)");
                        }
                    }
                }
                if (gemm_status != CUBLAS_STATUS_SUCCESS) {
                    gemm_status = rrnlb_cublas_gemm_strided_batched<Precision>(
                        handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        ir_dim,
                        mul_in,
                        mul_out,
                        &alpha,
                        y_base,
                        ir_dim,
                        y_row_stride,
                        w_base,
                        mul_out,
                        0,
                        &beta,
                        x_base,
                        ir_dim,
                        x_row_stride,
                        num_nodes);
                }
                rrnlb_throw_cublas_error(
                    gemm_status,
                    "rrnlb_apply_linear_transpose(cublasGemmStridedBatched)");
            }
        }
        return;
    }
#endif

    const auto ins_in_offset = linear.ins_in_offset;
    const auto ins_out_offset = linear.ins_out_offset;
    const auto ins_mul_in = linear.ins_mul_in;
    const auto ins_mul_out = linear.ins_mul_out;
    const auto ins_ir_dim = linear.ins_ir_dim;
    const auto ins_weight_offset = linear.ins_weight_offset;
    const auto ins_path_weight = linear.ins_path_weight;
    const auto ins_weights = linear.ins_weights;
    const auto rev_group_first_ins = linear.rev_group_first_ins;
    const auto rev_group_num_ins = linear.rev_group_num_ins;
    const auto rev_group_ins_index = linear.rev_group_ins_index;
    const int num_groups = rev_group_first_ins.extent_int(0);

    if (num_nodes > 0 && num_groups > 0) {
        Kokkos::parallel_for(
            "RRNLB linear transpose",
            Kokkos::TeamPolicy<>(num_nodes * num_groups, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int league = team_member.league_rank();
                const int g = league / num_nodes;
                const int i = league - g * num_nodes;
                auto y_i = Kokkos::subview(y_adj_work, i, Kokkos::ALL);
                auto x_i = Kokkos::subview(x_adj_work, i, Kokkos::ALL);
                const int first = rev_group_first_ins(g);
                const int count = rev_group_num_ins(g);
                for (int t = 0; t < count; ++t) {
                    const int q = rev_group_ins_index(first + t);
                    const int in_offset = ins_in_offset(q);
                    const int mul_in = ins_mul_in(q);
                    const int mul_out = ins_mul_out(q);
                    const int ir_dim = ins_ir_dim(q);
                    const int out_offset = ins_out_offset(q);
                    const int w_offset = ins_weight_offset(q);
                    auto X = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                        x_i.data() + in_offset, mul_in, ir_dim);
                    const auto W = Kokkos::View<const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                        ins_weights.data() + w_offset, mul_in, mul_out);
                    const auto Y = Kokkos::View<const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                        y_i.data() + out_offset, mul_out, ir_dim);
                    KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                            KokkosBatched::Trans::NoTranspose,
                                            KokkosBatched::Trans::NoTranspose,
                                            KokkosBatched::Algo::Gemm::Blocked>
                        ::invoke(
                            team_member,
                            ins_path_weight(q),
                            W,
                            Y,
                            static_cast<Precision>(1.0),
                            X);
                    team_member.team_barrier();
                }
            });
    }
    };

    if constexpr (!std::is_same_v<Precision, AccumPrecision>) {
        if (rrnlb_transpose_y_precision_workspace.extent_int(0) < num_nodes
            || rrnlb_transpose_y_precision_workspace.extent_int(1) < linear.dim_out) {
            Kokkos::realloc(
                rrnlb_transpose_y_precision_workspace,
                num_nodes,
                linear.dim_out);
        }
        if (rrnlb_transpose_x_precision_workspace.extent_int(0) < num_nodes
            || rrnlb_transpose_x_precision_workspace.extent_int(1) < linear.dim_in) {
            Kokkos::realloc(
                rrnlb_transpose_x_precision_workspace,
                num_nodes,
                linear.dim_in);
        }

        auto y_adj_precision = Kokkos::subview(
            rrnlb_transpose_y_precision_workspace,
            Kokkos::make_pair(0, num_nodes),
            Kokkos::make_pair(0, linear.dim_out));

        Kokkos::parallel_for(
            "RRNLB transpose cast-in",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_nodes * linear.dim_out),
            KOKKOS_LAMBDA (const int ip) {
                const int i = ip / linear.dim_out;
                const int p = ip % linear.dim_out;
                y_adj_precision(i, p) = static_cast<Precision>(y_adj(i, p));
            });

        const bool want_group_packed_transpose =
            num_nodes > 0
            && !linear.h_rev_group_first_ins.empty()
            && rrnlb_group_packed_should_run(
                num_nodes,
                linear.h_rev_group_ir_dim,
                linear.h_rev_group_n_total,
                linear.h_rev_group_mul_in);
        const bool use_group_packed_transpose =
            want_group_packed_transpose && linear.rev_group_pack_valid;
        if (want_group_packed_transpose && !linear.rev_group_pack_valid
            && rrnlb_group_packed_gemm_strict_enabled()) {
            throw std::runtime_error(
                "RRNLB grouped packed GEMM transpose path validation failed.");
        }

        if (use_group_packed_transpose) {
            const bool packed_work_overflow =
                !rrnlb_group_packed_work_fits_int(
                    num_nodes,
                    linear.rev_group_max_ir_dim,
                    linear.rev_group_max_n_total)
                || !rrnlb_group_packed_work_fits_int(
                    num_nodes,
                    linear.rev_group_max_ir_dim,
                    linear.rev_group_max_mul_in);
            if (packed_work_overflow) {
                if (rrnlb_group_packed_gemm_strict_enabled()) {
                    throw std::runtime_error(
                        "RRNLB grouped packed GEMM transpose path work overflow.");
                }
            } else {
                const int num_groups = static_cast<int>(linear.h_rev_group_first_ins.size());
                const int max_rows = num_nodes * linear.rev_group_max_ir_dim;
                if (rrnlb_rev_pack_y_workspace.extent_int(0) < max_rows
                    || rrnlb_rev_pack_y_workspace.extent_int(1) < linear.rev_group_max_n_total) {
                    Kokkos::realloc(
                        rrnlb_rev_pack_y_workspace,
                        max_rows,
                        linear.rev_group_max_n_total);
                }
                if (rrnlb_rev_pack_x_workspace.extent_int(0) < max_rows
                    || rrnlb_rev_pack_x_workspace.extent_int(1) < linear.rev_group_max_mul_in) {
                    Kokkos::realloc(
                        rrnlb_rev_pack_x_workspace,
                        max_rows,
                        linear.rev_group_max_mul_in);
                }

                for (int g = 0; g < num_groups; ++g) {
                    const int ir_dim = linear.h_rev_group_ir_dim[g];
                    const int n_total = linear.h_rev_group_n_total[g];
                    const int k_in = linear.h_rev_group_mul_in[g];
                    const int in_offset = linear.h_rev_group_in_offset[g];
                    const int first = linear.h_rev_group_first_ins[g];
                    const int count = linear.h_rev_group_num_ins[g];
                    const int rows = num_nodes * ir_dim;
                    auto yg = Kokkos::subview(
                        rrnlb_rev_pack_y_workspace,
                        make_pair(0, rows),
                        make_pair(0, n_total));
                    auto xg = Kokkos::subview(
                        rrnlb_rev_pack_x_workspace,
                        make_pair(0, rows),
                        make_pair(0, k_in));

                    for (int t = 0; t < count; ++t) {
                        const int idx = first + t;
                        const int q = linear.h_rev_group_ins_index[idx];
                        const int out_offset = linear.h_ins_out_offset[q];
                        const int mul_out = linear.h_ins_mul_out[q];
                        const int n_offset = linear.h_rev_group_ins_n_offset[idx];
                        const int pack_work = num_nodes * mul_out * ir_dim;
                        Kokkos::parallel_for(
                            "RRNLB rev pack input (mixed)",
                            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, pack_work),
                            KOKKOS_LAMBDA (const int iom) {
                                const int i = iom / (mul_out * ir_dim);
                                const int rem = iom - i * mul_out * ir_dim;
                                const int out = rem / ir_dim;
                                const int lm = rem - out * ir_dim;
                                const int row = i * ir_dim + lm;
                                yg(row, n_offset + out) =
                                    y_adj_precision(i, out_offset + out * ir_dim + lm);
                            });
                    }

                    const int w_base = linear.h_rev_group_pack_weight_offset[g];
                    const auto wg =
                        Kokkos::View<const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                            linear.rev_group_pack_weights.data() + w_base,
                            n_total,
                            k_in);
                    KokkosBlas::gemm(
                        "N",
                        "N",
                        static_cast<Precision>(1.0),
                        yg,
                        wg,
                        static_cast<Precision>(0.0),
                        xg);

                    Kokkos::parallel_for(
                        "RRNLB rev unpack output (mixed)",
                        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                            0,
                            num_nodes * k_in * ir_dim),
                        KOKKOS_LAMBDA (const int ikm) {
                            const int i = ikm / (k_in * ir_dim);
                            const int rem = ikm - i * k_in * ir_dim;
                            const int kin = rem / ir_dim;
                            const int lm = rem - kin * ir_dim;
                            const int row = i * ir_dim + lm;
                            x_adj(i, in_offset + kin * ir_dim + lm) +=
                                static_cast<AccumPrecision>(xg(row, kin));
                        });
                }
                return;
            }
        }

#if defined(KOKKOS_ENABLE_CUDA) && defined(SYMMETRIX_HAVE_CUBLAS)
        if (rrnlb_cublas_enabled()
            && Kokkos::SpaceAccessibility<
                Kokkos::CudaSpace,
                typename decltype(y_adj_precision)::memory_space>::accessible
            && Kokkos::SpaceAccessibility<
                Kokkos::CudaSpace,
                typename decltype(rrnlb_transpose_x_precision_workspace)::memory_space>::accessible) {
            if (num_nodes > 0) {
                auto exec = Kokkos::DefaultExecutionSpace();
                auto handle = rrnlb_get_cublas_handle();
                cublasLtHandle_t lt_handle = nullptr;
                if constexpr (std::is_same_v<Precision, float>) {
                    if (!rrnlb_cublas_tf32_enabled()) {
                        lt_handle = rrnlb_get_cublaslt_handle();
                    }
                }
                rrnlb_throw_cublas_error(
                    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                    "rrnlb_apply_linear_transpose(mixed cublasSetPointerMode)");
                rrnlb_throw_cublas_error(
                    cublasSetStream(handle, exec.cuda_stream()),
                    "rrnlb_apply_linear_transpose(mixed cublasSetStream)");

                const std::size_t y_row_stride_sz = y_adj_precision.stride(0);
                const std::size_t y_col_stride_sz = y_adj_precision.stride(1);
                if (y_col_stride_sz != 1) {
                    throw std::runtime_error(
                        "RRNLB mixed linear transpose requires contiguous Y column stride.");
                }
                const long long int y_row_stride = static_cast<long long int>(y_row_stride_sz);
                const auto& h_ins_out_offset = linear.h_ins_out_offset;
                const auto& h_ins_mul_out = linear.h_ins_mul_out;
                const auto& h_ins_weight_offset = linear.h_ins_weight_offset;
                const auto& h_ins_path_weight = linear.h_ins_path_weight;
                const auto ins_weights = linear.ins_weights;
                const int num_groups = static_cast<int>(linear.h_rev_group_first_ins.size());
                const Precision beta = static_cast<Precision>(1.0);

                for (int g = 0; g < num_groups; ++g) {
                    const int ir_dim = linear.h_rev_group_ir_dim[g];
                    const int k_in = linear.h_rev_group_mul_in[g];
                    const int in_offset = linear.h_rev_group_in_offset[g];
                    const int first = linear.h_rev_group_first_ins[g];
                    const int count = linear.h_rev_group_num_ins[g];
                    const int cols = k_in * ir_dim;

                    auto x_group = Kokkos::subview(
                        rrnlb_transpose_x_precision_workspace,
                        Kokkos::make_pair(0, num_nodes),
                        Kokkos::make_pair(0, cols));
                    Kokkos::deep_copy(x_group, static_cast<Precision>(0.0));

                    const std::size_t x_row_stride_sz = x_group.stride(0);
                    const std::size_t x_col_stride_sz = x_group.stride(1);
                    if (x_col_stride_sz != 1) {
                        throw std::runtime_error(
                            "RRNLB mixed linear transpose requires contiguous X column stride.");
                    }
                    const long long int x_row_stride = static_cast<long long int>(x_row_stride_sz);

                    for (int t = 0; t < count; ++t) {
                        const int idx = first + t;
                        const int q = linear.h_rev_group_ins_index[idx];
                        const int mul_out = h_ins_mul_out[q];
                        const int out_offset = h_ins_out_offset[q];
                        const int w_offset = h_ins_weight_offset[q];
                        const Precision alpha = h_ins_path_weight[q];
                        const Precision* y_base = y_adj_precision.data() + out_offset;
                        const Precision* w_base = ins_weights.data() + w_offset;
                        Precision* x_base = x_group.data();
                        cublasStatus_t gemm_status = CUBLAS_STATUS_NOT_SUPPORTED;
                        if constexpr (std::is_same_v<Precision, float>) {
                            if (!rrnlb_cublas_tf32_enabled() && lt_handle != nullptr) {
                                gemm_status = rrnlb_cublaslt_gemm_strided_batched_fp32(
                                    lt_handle,
                                    exec.cuda_stream(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    ir_dim,
                                    k_in,
                                    mul_out,
                                    &alpha,
                                    y_base,
                                    ir_dim,
                                    y_row_stride,
                                    w_base,
                                    mul_out,
                                    0,
                                    &beta,
                                    x_base,
                                    ir_dim,
                                    x_row_stride,
                                    num_nodes);
                                if (rrnlb_cublaslt_strict_enabled()
                                    && gemm_status != CUBLAS_STATUS_SUCCESS) {
                                    rrnlb_throw_cublas_error(
                                        gemm_status,
                                        "rrnlb_apply_linear_transpose(mixed cublasLtMatmul)");
                                }
                            }
                        }
                        if (gemm_status != CUBLAS_STATUS_SUCCESS) {
                            gemm_status = rrnlb_cublas_gemm_strided_batched<Precision>(
                                handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                ir_dim,
                                k_in,
                                mul_out,
                                &alpha,
                                y_base,
                                ir_dim,
                                y_row_stride,
                                w_base,
                                mul_out,
                                0,
                                &beta,
                                x_base,
                                ir_dim,
                                x_row_stride,
                                num_nodes);
                        }
                        rrnlb_throw_cublas_error(
                            gemm_status,
                            "rrnlb_apply_linear_transpose(mixed cublasGemmStridedBatched)");
                    }

                    Kokkos::parallel_for(
                        "RRNLB mixed transpose accum (cublas)",
                        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                            0,
                            num_nodes * cols),
                        KOKKOS_LAMBDA (const int ik) {
                            const int i = ik / cols;
                            const int rem = ik - i * cols;
                            const int kin = rem / ir_dim;
                            const int lm = rem - kin * ir_dim;
                            x_adj(i, in_offset + kin * ir_dim + lm) +=
                                static_cast<AccumPrecision>(x_group(i, rem));
                        });
                }
            }
            return;
        }
#endif

        const auto ins_out_offset = linear.ins_out_offset;
        const auto ins_mul_in = linear.ins_mul_in;
        const auto ins_mul_out = linear.ins_mul_out;
        const auto ins_ir_dim = linear.ins_ir_dim;
        const auto ins_weight_offset = linear.ins_weight_offset;
        const auto ins_path_weight = linear.ins_path_weight;
        const auto ins_weights = linear.ins_weights;
        const auto rev_group_first_ins = linear.rev_group_first_ins;
        const auto rev_group_num_ins = linear.rev_group_num_ins;
        const auto rev_group_ins_index = linear.rev_group_ins_index;
        const int num_groups = rev_group_first_ins.extent_int(0);

        for (int g = 0; g < num_groups; ++g) {
            const int ir_dim = linear.h_rev_group_ir_dim[g];
            const int k_in = linear.h_rev_group_mul_in[g];
            const int in_offset = linear.h_rev_group_in_offset[g];
            const int cols = k_in * ir_dim;
            auto x_group = Kokkos::subview(
                rrnlb_transpose_x_precision_workspace,
                Kokkos::make_pair(0, num_nodes),
                Kokkos::make_pair(0, cols));
            Kokkos::deep_copy(x_group, static_cast<Precision>(0.0));

            if (num_nodes > 0) {
                Kokkos::parallel_for(
                    "RRNLB linear transpose mixed fallback",
                    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
                    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                        const int i = team_member.league_rank();
                        auto y_i = Kokkos::subview(y_adj_precision, i, Kokkos::ALL);
                        auto x_i = Kokkos::subview(x_group, i, Kokkos::ALL);
                        const int first = rev_group_first_ins(g);
                        const int count = rev_group_num_ins(g);
                        for (int t = 0; t < count; ++t) {
                            const int q = rev_group_ins_index(first + t);
                            const int mul_in = ins_mul_in(q);
                            const int mul_out = ins_mul_out(q);
                            const int ir_dim_q = ins_ir_dim(q);
                            const int out_offset = ins_out_offset(q);
                            const int w_offset = ins_weight_offset(q);
                            auto X =
                                Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                                    x_i.data(),
                                    mul_in,
                                    ir_dim_q);
                            const auto W = Kokkos::View<
                                const Precision**,
                                Kokkos::LayoutRight,
                                Kokkos::MemoryUnmanaged>(
                                    ins_weights.data() + w_offset,
                                    mul_in,
                                    mul_out);
                            const auto Y = Kokkos::View<
                                const Precision**,
                                Kokkos::LayoutRight,
                                Kokkos::MemoryUnmanaged>(
                                    y_i.data() + out_offset,
                                    mul_out,
                                    ir_dim_q);
                            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                                    KokkosBatched::Trans::NoTranspose,
                                                    KokkosBatched::Trans::NoTranspose,
                                                    KokkosBatched::Algo::Gemm::Blocked>
                                ::invoke(
                                    team_member,
                                    ins_path_weight(q),
                                    W,
                                    Y,
                                    static_cast<Precision>(1.0),
                                    X);
                            team_member.team_barrier();
                        }
                    });
            }

            Kokkos::parallel_for(
                "RRNLB mixed transpose accum (fallback)",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_nodes * cols),
                KOKKOS_LAMBDA (const int ik) {
                    const int i = ik / cols;
                    const int rem = ik - i * cols;
                    const int kin = rem / ir_dim;
                    const int lm = rem - kin * ir_dim;
                    x_adj(i, in_offset + kin * ir_dim + lm) +=
                        static_cast<AccumPrecision>(x_group(i, rem));
                });
        }
        return;
    } else {
        run_transpose_precision(y_adj, x_adj);
    }
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_apply_gate_forward(
    const RRNLBLayerKokkos& layer,
    const int num_nodes,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> x,
    Kokkos::View<Precision**,Kokkos::LayoutRight> y)
{
    if (x.extent(0) < num_nodes || x.extent(1) != layer.linear_1.dim_out) {
        throw std::runtime_error("RRNLB gate forward input has invalid shape.");
    }
    if (y.extent(0) < num_nodes || y.extent(1) != layer.linear_2.dim_in) {
        throw std::runtime_error("RRNLB gate forward output has invalid shape.");
    }
    Kokkos::deep_copy(y, static_cast<Precision>(0.0));

    const auto target_offset = layer.target_offset;
    const auto target_mul = layer.target_mul;
    const auto target_l = layer.target_l;
    const auto nonlin_offset = layer.nonlin_offset;
    const auto nonlin_mul = layer.nonlin_mul;
    const auto nonlin_l = layer.nonlin_l;
    const auto gate_gate_cst = layer.gate_gate_cst;
    const Precision gate_scalar_cst = static_cast<Precision>(layer.gate_scalar_cst);
    const bool gate_identity_mode =
        rrnlb_nonlinear_ablation_mode() == RRNLBNonlinearAblationMode::GateIdentity;

    Kokkos::parallel_for(
        "RRNLB gate forward",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            auto x_i = Kokkos::subview(x, i, Kokkos::ALL);
            auto y_i = Kokkos::subview(y, i, Kokkos::ALL);

            const int scalar_mul = target_mul(0);
            const int scalar_nonlin_offset = nonlin_offset(0);
            const int scalar_target_offset = target_offset(0);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, scalar_mul),
                [&] (const int k) {
                    const Precision v = x_i(scalar_nonlin_offset + k);
                    y_i(scalar_target_offset + k) = gate_identity_mode
                        ? gate_scalar_cst * v
                        : gate_scalar_cst * rrnlb_silu(v);
                });
            team_member.team_barrier();

            int gate_offset = scalar_nonlin_offset + scalar_mul;
            for (int p = 1; p < target_offset.extent(0); ++p) {
                const int out_offset = target_offset(p);
                const int in_offset = nonlin_offset(p);
                const int mul = target_mul(p);
                const int ir_dim = 2 * target_l(p) + 1;
                const Precision cst =
                    p - 1 < gate_gate_cst.extent(0) ? gate_gate_cst(p - 1) : static_cast<Precision>(1.0);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, mul),
                    [&] (const int k) {
                        const Precision g = gate_identity_mode
                            ? cst
                            : cst * rrnlb_sigmoid(x_i(gate_offset + k));
                        for (int m = 0; m < ir_dim; ++m) {
                            y_i(out_offset + k * ir_dim + m) = g * x_i(in_offset + k * ir_dim + m);
                        }
                    });
                team_member.team_barrier();
                gate_offset += mul;
            }
        });
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::rrnlb_apply_gate_reverse(
    const RRNLBLayerKokkos& layer,
    const int num_nodes,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> x,
    Kokkos::View<const AccumPrecision**,Kokkos::LayoutRight> y_adj,
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> x_adj)
{
    if (x.extent(0) < num_nodes || x.extent(1) != layer.linear_1.dim_out) {
        throw std::runtime_error("RRNLB gate reverse input has invalid shape.");
    }
    if (y_adj.extent(0) < num_nodes || y_adj.extent(1) != layer.linear_2.dim_in) {
        throw std::runtime_error("RRNLB gate reverse output adjoint has invalid shape.");
    }
    if (x_adj.extent(0) < num_nodes || x_adj.extent(1) != layer.linear_1.dim_out) {
        throw std::runtime_error("RRNLB gate reverse input adjoint has invalid shape.");
    }
    Kokkos::deep_copy(x_adj, static_cast<AccumPrecision>(0.0));

    const auto target_offset = layer.target_offset;
    const auto target_mul = layer.target_mul;
    const auto target_l = layer.target_l;
    const auto nonlin_offset = layer.nonlin_offset;
    const auto nonlin_mul = layer.nonlin_mul;
    const auto nonlin_l = layer.nonlin_l;
    const auto gate_gate_cst = layer.gate_gate_cst;
    const AccumPrecision gate_scalar_cst = static_cast<AccumPrecision>(layer.gate_scalar_cst);
    const bool gate_identity_mode =
        rrnlb_nonlinear_ablation_mode() == RRNLBNonlinearAblationMode::GateIdentity;

    Kokkos::parallel_for(
        "RRNLB gate reverse",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            auto x_i = Kokkos::subview(x, i, Kokkos::ALL);
            auto y_adj_i = Kokkos::subview(y_adj, i, Kokkos::ALL);
            auto x_adj_i = Kokkos::subview(x_adj, i, Kokkos::ALL);

            const int scalar_mul = target_mul(0);
            const int scalar_nonlin_offset = nonlin_offset(0);
            const int scalar_target_offset = target_offset(0);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, scalar_mul),
                [&] (const int k) {
                    const int in_idx = scalar_nonlin_offset + k;
                    const int out_idx = scalar_target_offset + k;
                    x_adj_i(in_idx) += y_adj_i(out_idx) * gate_scalar_cst
                        * (gate_identity_mode
                            ? static_cast<AccumPrecision>(1.0)
                            : static_cast<AccumPrecision>(rrnlb_silu_deriv(x_i(in_idx))));
                });
            team_member.team_barrier();

            int gate_offset = scalar_nonlin_offset + scalar_mul;
            for (int p = 1; p < target_offset.extent(0); ++p) {
                const int out_offset = target_offset(p);
                const int in_offset = nonlin_offset(p);
                const int mul = target_mul(p);
                const int ir_dim = 2 * target_l(p) + 1;
                const AccumPrecision cst =
                    p - 1 < gate_gate_cst.extent(0)
                        ? static_cast<AccumPrecision>(gate_gate_cst(p - 1))
                        : static_cast<AccumPrecision>(1.0);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, mul),
                    [&] (const int k) {
                        const AccumPrecision sig = static_cast<AccumPrecision>(rrnlb_sigmoid(x_i(gate_offset + k)));
                        const AccumPrecision g = gate_identity_mode
                            ? cst
                            : cst * sig;
                        AccumPrecision dg = static_cast<AccumPrecision>(0.0);
                        for (int m = 0; m < ir_dim; ++m) {
                            const int out_idx = out_offset + k * ir_dim + m;
                            const int in_idx = in_offset + k * ir_dim + m;
                            x_adj_i(in_idx) += y_adj_i(out_idx) * g;
                            if (!gate_identity_mode) {
                                dg += y_adj_i(out_idx) * static_cast<AccumPrecision>(x_i(in_idx));
                            }
                        }
                        if (!gate_identity_mode) {
                            x_adj_i(gate_offset + k) += dg * cst * sig
                                * (static_cast<AccumPrecision>(1.0) - sig);
                        }
                    });
                team_member.team_barrier();
                gate_offset += mul;
            }
        });
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_rrnlb_interaction_layer_forward(
    const int layer_index,
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r,
    Kokkos::View<const int*> first_neigh,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> node_feats_in,
    Kokkos::View<Precision**,Kokkos::LayoutRight> layer_output,
    Kokkos::View<Precision**,Kokkos::LayoutRight> layer_skip,
    RRNLBLayerCacheKokkos& cache,
    int num_sender_nodes,
    Kokkos::View<const int*> target_node_indices,
    int total_edges,
    Kokkos::View<const int*> edge_to_receiver)
{
    if (layer_index < 0 || layer_index >= static_cast<int>(rrnlb_layers_kokkos.size())) {
        throw std::runtime_error("RRNLB layer index out of bounds.");
    }
    const auto& layer = rrnlb_layers_kokkos[layer_index];
    const int in_dim = layer.linear_up.dim_in;
    const int sender_nodes = (num_sender_nodes > 0) ? num_sender_nodes : num_nodes;
    if (node_feats_in.extent(0) < sender_nodes || node_feats_in.extent(1) != in_dim) {
        throw std::runtime_error("RRNLB interaction forward input has invalid shape.");
    }
    if (layer_output.extent(0) < num_nodes || layer_output.extent(1) != layer.linear_2.dim_out) {
        throw std::runtime_error("RRNLB interaction forward output has invalid shape.");
    }
    if (layer_skip.extent(0) < num_nodes || layer_skip.extent(1) != layer.skip_tp.dim_out) {
        throw std::runtime_error("RRNLB interaction forward skip has invalid shape.");
    }
    const bool has_target_map = target_node_indices.extent(0) > 0;
    if (has_target_map && target_node_indices.extent(0) < num_nodes) {
        throw std::runtime_error("RRNLB interaction forward target index map has invalid shape.");
    }
    const bool use_edge_parallel = total_edges > 0 && edge_to_receiver.extent(0) >= total_edges;
    ensure_rrnlb_layer_workspace_capacity(cache, num_nodes, sender_nodes, total_edges);
    bind_rrnlb_layer_active_views(cache, num_nodes, sender_nodes, total_edges);
    ensure_rrnlb_layer_dispatch_cache(layer, cache, sender_nodes, total_edges);
    const int cap_nodes = std::max(cache.capacity_nodes, num_nodes);
    const int cap_sender_nodes = std::max(cache.capacity_sender_nodes, sender_nodes);

    auto ensure_cache_2d = [](auto& view, const int d0, const int d1) {
        if (view.extent(0) < static_cast<std::size_t>(d0)
            || view.extent(1) < static_cast<std::size_t>(d1)) {
            Kokkos::realloc(view, d0, d1);
        }
    };
    auto ensure_cache_1d = [](auto& view, const int d0) {
        if (view.extent(0) < static_cast<std::size_t>(d0)) {
            Kokkos::realloc(view, d0);
        }
    };

    ensure_cache_2d(cache.h_up, cap_sender_nodes, layer.linear_up.dim_out);
    ensure_cache_2d(cache.h_up_targets, cap_nodes, layer.linear_up.dim_out);
    ensure_cache_2d(cache.x_targets, cap_nodes, in_dim);
    ensure_cache_2d(cache.h_res, cap_nodes, layer.linear_res.dim_out);
    ensure_cache_2d(cache.conv_accum, cap_nodes, layer.linear_1.dim_in);
    ensure_cache_1d(cache.density, cap_nodes);
    ensure_cache_2d(cache.lin1_raw, cap_nodes, layer.linear_1.dim_out);
    ensure_cache_2d(cache.pre_gate, cap_nodes, layer.linear_1.dim_out);
    ensure_cache_2d(cache.gated, cap_nodes, layer.linear_2.dim_in);

    auto h_up_targets = cache.h_up_targets;
    auto x_targets = cache.x_targets;
    auto h_res = cache.h_res;
    auto conv_accum = cache.conv_accum;
    auto gated = cache.gated;

    rrnlb_apply_linear_forward(layer.linear_up, sender_nodes, node_feats_in, cache.h_up);
    const auto target_map = target_node_indices;
    const auto h_up_all = cache.h_up;
    Kokkos::parallel_for(
        "RRNLB gather targets",
        num_nodes,
        KOKKOS_LAMBDA (const int i) {
            const int sender_i = has_target_map ? target_map(i) : i;
            for (int p = 0; p < layer.linear_up.dim_out; ++p) {
                h_up_targets(i, p) = h_up_all(sender_i, p);
            }
            for (int p = 0; p < in_dim; ++p) {
                x_targets(i, p) = node_feats_in(sender_i, p);
            }
        });
    rrnlb_apply_linear_forward(layer.linear_res, num_nodes, h_up_targets, h_res);
    rrnlb_apply_linear_forward(layer.skip_tp, num_nodes, x_targets, layer_skip);
    auto conv_accum_active = Kokkos::subview(
        conv_accum, Kokkos::make_pair(0, num_nodes), Kokkos::ALL);
    auto density_active = Kokkos::subview(
        cache.density, Kokkos::make_pair(0, num_nodes));
    const int recv_tiled_fwd_override = rrnlb_receiver_tiled_forward_override();
    const bool receiver_parallel_fwd_enabled =
        recv_tiled_fwd_override >= 0 ? recv_tiled_fwd_override != 0 : true;
    const auto forward_adaptive_mode = rrnlb_forward_adaptive_mode();
    switch (forward_adaptive_mode) {
        case RRNLBForwardAdaptiveMode::ForceFused:
            rrnlb_phase_counters.forward_adaptive_mode_force_fused_calls++;
            break;
        case RRNLBForwardAdaptiveMode::ForceSplit:
            rrnlb_phase_counters.forward_adaptive_mode_force_split_calls++;
            break;
        case RRNLBForwardAdaptiveMode::Auto:
        default:
            rrnlb_phase_counters.forward_adaptive_mode_auto_calls++;
            break;
    }
    const bool receiver_forward_will_run = use_edge_parallel && receiver_parallel_fwd_enabled;
    const bool use_split_receiver_forward = rrnlb_forward_should_use_split(
        forward_adaptive_mode, num_nodes, receiver_parallel_fwd_enabled, use_edge_parallel);
    const bool fused_forward_uses_scratch =
        receiver_forward_will_run
        && !use_split_receiver_forward
        && layer.fusion_storage_mode_fwd == RRNLBFusionStorageMode::ScratchTiled;
    // Forward scratch-tiled fusion stages conv_row in shared memory, so conv_accum
    // is unused and can skip the clear; staged/global and fallback paths still need it.
    if (!fused_forward_uses_scratch) {
        Kokkos::deep_copy(conv_accum_active, static_cast<Precision>(0.0));
    }
    // Density is written directly by receiver-parallel forward paths (full-fused or split);
    // only non-receiver paths need the atomic-accumulation zero init.
    if (!receiver_forward_will_run) {
        Kokkos::deep_copy(density_active, static_cast<Precision>(0.0));
    }

    const auto conv_in_offset = layer.conv_in_offset;
    const auto conv_out_offset = layer.conv_out_offset;
    const auto conv_mul = layer.conv_mul;
    const auto conv_weight_offset = layer.conv_weight_offset;
    const auto conv_in_ir_dim = layer.conv_in_ir_dim;
    const auto conv_out_ir_dim = layer.conv_out_ir_dim;
    const auto conv_term_offset = layer.conv_term_offset;
    const auto conv_term_count = layer.conv_term_count;
    const auto conv_term_m_out = layer.conv_term_m_out;
    const auto conv_term_m_in1 = layer.conv_term_m_in1;
    const auto conv_term_y_lm = layer.conv_term_y_lm;
    const auto conv_term_coeff = layer.conv_term_coeff;
    const int num_conv_ins = conv_in_offset.extent(0);
    const auto tp_spline_coeff = layer.tp_spline_coeff;
    const auto density_spline_coeff = layer.density_spline_coeff;
    const double radial_h = layer.radial_h;
    const int radial_num_intervals = layer.radial_num_intervals;
    const int tp_weight_numel = layer.tp_weight_numel;
    const int conv_out_dim = layer.linear_1.dim_in;
    const int num_elements = this->num_elements;
    const auto Y = this->Y;
    const int num_lm = this->num_lm;
    const bool use_cached_forward_dispatch =
        cache.dispatch_cache.valid
        && cache.dispatch_cache.epoch_id == rrnlb_cache_stamp.epoch_rev
        && layer.conv_active_out_count > 0
        && cache.dispatch_cache.fwd_active_out_indices.extent_int(0) >= layer.conv_active_out_count
        && cache.dispatch_cache.fwd_active_out_inverse.extent_int(0) >= conv_out_dim;
    const auto conv_active_out_indices = use_cached_forward_dispatch
        ? cache.dispatch_cache.fwd_active_out_indices
        : layer.conv_active_out_indices;
    const auto conv_active_out_inverse = use_cached_forward_dispatch
        ? cache.dispatch_cache.fwd_active_out_inverse
        : layer.conv_active_out_inverse;

    // Work table references for flattened convolution.
    const auto work_out_idx = layer.conv_work_out_idx;
    const auto work_in_idx = layer.conv_work_in_idx;
    const auto work_w_idx = layer.conv_work_w_idx;
    const auto work_y_lm = layer.conv_work_y_lm;
    const auto work_coeff = layer.conv_work_coeff;
    const auto work_out_slot = layer.conv_work_out_slot;
    const int num_work_items = layer.conv_work_table_size;

    bool did_fused_forward = false;
    bool did_split_forward = false;
    if (use_edge_parallel) {
        // Precompute per-edge radial descriptors (shared by forward + reverse).
        precompute_rrnlb_edge_radial_descriptors(
            layer_index, total_edges, neigh_types, node_types, edge_to_receiver, r);
        const auto edge_tp_vals = rrnlb_edge_tp_values[layer_index];
        const auto edge_den_val = rrnlb_edge_density_value[layer_index];

        // Gate: prefer receiver-parallel (zero global atomics) when enabled.
        const bool use_receiver_parallel_fwd = receiver_parallel_fwd_enabled;

        auto density_view = cache.density;
        const int compact_out_count =
            layer.conv_active_out_count > 0 ? layer.conv_active_out_count : conv_out_dim;
        const bool compact_auto = compact_out_count * 5 <= conv_out_dim * 4;
        const auto compact_mode = rrnlb_interaction_compact_mode();
        const bool use_compact_out =
            compact_mode == RRNLBInteractionCompactMode::On
                ? true
                : (compact_mode == RRNLBInteractionCompactMode::Auto && compact_auto);

        if (use_receiver_parallel_fwd && !use_split_receiver_forward) {
            // Fused forward mega-kernel: conv + linear_1 + normalize + gate + linear_2.
            // Uses scratch for conv/gate rows when available, with global-staged
            // fallback for oversized layers.
            did_fused_forward = true;
            rrnlb_phase_counters.forward_full_fused_calls++;

            const int lin1_dim = layer.linear_1.dim_out;
            const int gate_dim = layer.linear_2.dim_in;
            const int lin2_out_dim = layer.linear_2.dim_out;

            // Linear_1 metadata (device views).
            const auto l1_ins_in_off = layer.linear_1.ins_in_offset;
            const auto l1_ins_out_off = layer.linear_1.ins_out_offset;
            const auto l1_ins_mul_in = layer.linear_1.ins_mul_in;
            const auto l1_ins_mul_out = layer.linear_1.ins_mul_out;
            const auto l1_ins_ir_dim = layer.linear_1.ins_ir_dim;
            const auto l1_ins_w_off = layer.linear_1.ins_weight_offset;
            const auto l1_ins_pw = layer.linear_1.ins_path_weight;
            const auto l1_weights = layer.linear_1.ins_weights;
            const auto l1_bias_idx = layer.linear_1.bias_indices;
            const auto l1_bias_val = layer.linear_1.bias_values;
            const int l1_num_ins = layer.linear_1.num_ins;
            const int l1_num_bias = l1_bias_idx.extent_int(0);

            // Linear_2 metadata (device views).
            const auto l2_ins_in_off = layer.linear_2.ins_in_offset;
            const auto l2_ins_out_off = layer.linear_2.ins_out_offset;
            const auto l2_ins_mul_in = layer.linear_2.ins_mul_in;
            const auto l2_ins_mul_out = layer.linear_2.ins_mul_out;
            const auto l2_ins_ir_dim = layer.linear_2.ins_ir_dim;
            const auto l2_ins_w_off = layer.linear_2.ins_weight_offset;
            const auto l2_ins_pw = layer.linear_2.ins_path_weight;
            const auto l2_weights = layer.linear_2.ins_weights;
            const auto l2_bias_idx = layer.linear_2.bias_indices;
            const auto l2_bias_val = layer.linear_2.bias_values;
            const int l2_num_ins = layer.linear_2.num_ins;
            const int l2_num_bias = l2_bias_idx.extent_int(0);

            // Gate metadata.
            const auto g_target_off = layer.target_offset;
            const auto g_target_mul = layer.target_mul;
            const auto g_target_l = layer.target_l;
            const auto g_nonlin_off = layer.nonlin_offset;
            const auto g_gate_cst = layer.gate_gate_cst;
            const Precision g_scalar_cst = static_cast<Precision>(layer.gate_scalar_cst);
            const int g_num_parts = layer.num_gate_parts;
            const bool gate_identity_mode =
                rrnlb_nonlinear_ablation_mode() == RRNLBNonlinearAblationMode::GateIdentity;

            // Normalize.
            const double fused_alpha = layer.alpha;
            const double fused_beta = layer.beta;

            // Cache views for intermediate storage (needed by reverse pass, and for
            // global-staged fused fallback when scratch is too large).
            const auto conv_accum_cache = conv_accum;
            const auto lin1_raw_cache = cache.lin1_raw;
            const auto pre_gate_cache = cache.pre_gate;
            const auto gated_cache = cache.gated;

            const bool use_scratch_fwd =
                layer.fusion_storage_mode_fwd == RRNLBFusionStorageMode::ScratchTiled;
            const int fwd_scratch_bytes = use_scratch_fwd
                ? (conv_out_dim + gate_dim) * sizeof(Precision)
                : 0;
            if (use_scratch_fwd) {
                rrnlb_phase_counters.fused_forward_scratch_tiled_calls++;
            } else {
                rrnlb_phase_counters.fused_forward_global_staged_calls++;
            }

            Kokkos::parallel_for(
                "RRNLB fused forward",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, rrnlb_forward_vector_length())
                    .set_scratch_size(0, Kokkos::PerTeam(fwd_scratch_bytes)),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                    const int i = team_member.league_rank();

                    Precision* conv_row_ptr = nullptr;
                    Precision* gate_row_ptr = nullptr;
                    if (use_scratch_fwd) {
                        // Kokkos ScratchMemorySpace is a sequential allocator: each
                        // View constructor advances the internal pointer, so conv_row
                        // and gate_row are allocated contiguously.
                        using ScratchSpace =
                            Kokkos::TeamPolicy<>::member_type::scratch_memory_space;
                        auto scratch = team_member.team_scratch(0);
                        Kokkos::View<Precision*, ScratchSpace, Kokkos::MemoryUnmanaged>
                            conv_row_scratch(scratch, conv_out_dim);
                        Kokkos::View<Precision*, ScratchSpace, Kokkos::MemoryUnmanaged>
                            gate_row_scratch(scratch, gate_dim);
                        conv_row_ptr = conv_row_scratch.data();
                        gate_row_ptr = gate_row_scratch.data();
                    } else {
                        auto conv_row_global = Kokkos::subview(conv_accum_cache, i, Kokkos::ALL);
                        auto gate_row_global = Kokkos::subview(gated_cache, i, Kokkos::ALL);
                        conv_row_ptr = conv_row_global.data();
                        gate_row_ptr = gate_row_global.data();
                    }

                    // lin1_row and pre_gate_row stay in global (needed by reverse cache).
                    auto lin1_row = Kokkos::subview(lin1_raw_cache, i, Kokkos::ALL);
                    auto pre_gate_row = Kokkos::subview(pre_gate_cache, i, Kokkos::ALL);
                    auto out_row = Kokkos::subview(layer_output, i, Kokkos::ALL);

                    // === PHASE 1: Convolution ===
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, conv_out_dim),
                        [&] (const int p) { conv_row_ptr[p] = static_cast<Precision>(0.0); });
                    team_member.team_barrier();

                    const int edge_begin = first_neigh(i);
                    const int edge_end = (i + 1 < num_nodes) ? first_neigh(i + 1) : total_edges;

                    Precision density_i = static_cast<Precision>(0.0);
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team_member, edge_end - edge_begin),
                        [&] (const int j, Precision& den) {
                            den += edge_den_val(edge_begin + j);
                        },
                        density_i);

                    for (int ij = edge_begin; ij < edge_end; ++ij) {
                        const int sender = neigh_indices(ij);
                        const auto up_sender = Kokkos::subview(h_up_all, sender, Kokkos::ALL);
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, num_work_items),
                            [&] (const int w) {
                                const Precision val =
                                    work_coeff(w)
                                    * Y(ij * num_lm + work_y_lm(w))
                                    * edge_tp_vals(ij, work_w_idx(w))
                                    * up_sender(work_in_idx(w));
                                Kokkos::atomic_add(&conv_row_ptr[work_out_idx(w)], val);
                            });
                    }
                    team_member.team_barrier();

                    // Write density to global cache (needed for reverse).
                    Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                        density_view(i) = density_i;
                    });

                    // === PHASE 2: Linear_1 GEMV ===
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, lin1_dim),
                        [&] (const int p) { lin1_row(p) = static_cast<Precision>(0.0); });
                    team_member.team_barrier();

                    for (int q = 0; q < l1_num_ins; ++q) {
                        const int in_off = l1_ins_in_off(q);
                        const int out_off = l1_ins_out_off(q);
                        const int mul_in = l1_ins_mul_in(q);
                        const int mul_out = l1_ins_mul_out(q);
                        const int ir_d = l1_ins_ir_dim(q);
                        const Precision pw = l1_ins_pw(q);
                        const int w_base = l1_ins_w_off(q);
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, mul_out * ir_d),
                            [&] (const int p) {
                                const int k = p / ir_d;
                                const int m = p % ir_d;
                                Precision sum = static_cast<Precision>(0.0);
                                for (int j = 0; j < mul_in; ++j) {
                                    sum += l1_weights(w_base + k + j * mul_out)
                                         * conv_row_ptr[in_off + j * ir_d + m];
                                }
                                lin1_row(out_off + k * ir_d + m) += pw * sum;
                            });
                        team_member.team_barrier();
                    }

                    if (l1_num_bias > 0) {
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, l1_num_bias),
                            [&] (const int b) {
                                lin1_row(l1_bias_idx(b)) += l1_bias_val(b);
                            });
                        team_member.team_barrier();
                    }

                    // === PHASE 3: Normalize + Residual ===
                    const Precision denom = static_cast<Precision>(fused_alpha)
                        + static_cast<Precision>(fused_beta) * density_i;
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, lin1_dim),
                        [&] (const int p) {
                            pre_gate_row(p) = lin1_row(p) / denom + h_res(i, p);
                        });
                    team_member.team_barrier();

                    // === PHASE 4: Gate ===
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, gate_dim),
                        [&] (const int p) { gate_row_ptr[p] = static_cast<Precision>(0.0); });
                    team_member.team_barrier();

                    // Scalar channel (p=0): SiLU.
                    {
                        const int scalar_mul = g_target_mul(0);
                        const int scalar_nonlin_off = g_nonlin_off(0);
                        const int scalar_target_off = g_target_off(0);
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, scalar_mul),
                            [&] (const int k) {
                                const Precision v = pre_gate_row(scalar_nonlin_off + k);
                                gate_row_ptr[scalar_target_off + k] = gate_identity_mode
                                    ? g_scalar_cst * v
                                    : g_scalar_cst * rrnlb_silu(v);
                            });
                        team_member.team_barrier();
                    }

                    // Higher-l channels: sigmoid gate.
                    {
                        int gate_off = g_nonlin_off(0) + g_target_mul(0);
                        for (int gp = 1; gp < g_num_parts; ++gp) {
                            const int out_off = g_target_off(gp);
                            const int in_off = g_nonlin_off(gp);
                            const int mul = g_target_mul(gp);
                            const int ir_d = 2 * g_target_l(gp) + 1;
                            const Precision cst =
                                gp - 1 < g_gate_cst.extent_int(0)
                                    ? g_gate_cst(gp - 1)
                                    : static_cast<Precision>(1.0);
                            Kokkos::parallel_for(
                                Kokkos::TeamThreadRange(team_member, mul),
                                [&] (const int k) {
                                    const Precision g = gate_identity_mode
                                        ? cst
                                        : cst * rrnlb_sigmoid(pre_gate_row(gate_off + k));
                                    for (int m = 0; m < ir_d; ++m) {
                                        gate_row_ptr[out_off + k * ir_d + m] =
                                            g * pre_gate_row(in_off + k * ir_d + m);
                                    }
                                });
                            team_member.team_barrier();
                            gate_off += mul;
                        }
                    }

                    // === PHASE 5: Linear_2 GEMV → layer_output ===
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, lin2_out_dim),
                        [&] (const int p) {
                            out_row(p) = static_cast<Precision>(0.0);
                        });
                    team_member.team_barrier();

                    for (int q = 0; q < l2_num_ins; ++q) {
                        const int in_off = l2_ins_in_off(q);
                        const int out_off = l2_ins_out_off(q);
                        const int mul_in = l2_ins_mul_in(q);
                        const int mul_out = l2_ins_mul_out(q);
                        const int ir_d = l2_ins_ir_dim(q);
                        const Precision pw = l2_ins_pw(q);
                        const int w_base = l2_ins_w_off(q);
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, mul_out * ir_d),
                            [&] (const int p) {
                                const int k = p / ir_d;
                                const int m = p % ir_d;
                                Precision sum = static_cast<Precision>(0.0);
                                for (int j = 0; j < mul_in; ++j) {
                                    sum += l2_weights(w_base + k + j * mul_out)
                                         * gate_row_ptr[in_off + j * ir_d + m];
                                }
                                out_row(out_off + k * ir_d + m) += pw * sum;
                            });
                        team_member.team_barrier();
                    }

                    if (l2_num_bias > 0) {
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, l2_num_bias),
                            [&] (const int b) {
                                out_row(l2_bias_idx(b)) += l2_bias_val(b);
                            });
                    }
                });
        } else if (use_receiver_parallel_fwd) {
            did_split_forward = true;
            rrnlb_phase_counters.forward_split_calls++;
            rrnlb_phase_counters.forward_split_conv_stage_calls++;

            Kokkos::parallel_for(
                "RRNLB receiver conv forward (split)",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, rrnlb_forward_vector_length()),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                    const int i = team_member.league_rank();
                    auto conv_i = Kokkos::subview(conv_accum, i, Kokkos::ALL);

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, conv_out_dim),
                        [&] (const int p) { conv_i(p) = static_cast<Precision>(0.0); });
                    team_member.team_barrier();

                    const int edge_begin = first_neigh(i);
                    const int edge_end = (i + 1 < num_nodes) ? first_neigh(i + 1) : total_edges;

                    Precision density_i = static_cast<Precision>(0.0);
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team_member, edge_end - edge_begin),
                        [&] (const int j, Precision& den) {
                            den += edge_den_val(edge_begin + j);
                        },
                        density_i);

                    for (int ij = edge_begin; ij < edge_end; ++ij) {
                        const int sender = neigh_indices(ij);
                        const auto up_sender = Kokkos::subview(h_up_all, sender, Kokkos::ALL);
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, num_work_items),
                            [&] (const int w) {
                                const Precision val =
                                    work_coeff(w)
                                    * Y(ij * num_lm + work_y_lm(w))
                                    * edge_tp_vals(ij, work_w_idx(w))
                                    * up_sender(work_in_idx(w));
                                Kokkos::atomic_add(&conv_i(work_out_idx(w)), val);
                            });
                    }
                    team_member.team_barrier();

                    Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                        density_view(i) = density_i;
                    });
                });

            rrnlb_apply_linear_forward(layer.linear_1, num_nodes, conv_accum, cache.lin1_raw);

            rrnlb_phase_counters.forward_split_norm_gate_stage_calls++;
            const int lin1_dim = layer.linear_1.dim_out;
            const int gate_dim = layer.linear_2.dim_in;
            const auto g_target_off = layer.target_offset;
            const auto g_target_mul = layer.target_mul;
            const auto g_target_l = layer.target_l;
            const auto g_nonlin_off = layer.nonlin_offset;
            const auto g_gate_cst = layer.gate_gate_cst;
            const Precision g_scalar_cst = static_cast<Precision>(layer.gate_scalar_cst);
            const int g_num_parts = layer.num_gate_parts;
            const bool gate_identity_mode =
                rrnlb_nonlinear_ablation_mode() == RRNLBNonlinearAblationMode::GateIdentity;
            const double split_alpha = layer.alpha;
            const double split_beta = layer.beta;
            const auto lin1_raw_cache = cache.lin1_raw;
            const auto pre_gate_cache = cache.pre_gate;
            const auto gated_cache = cache.gated;

            Kokkos::parallel_for(
                "RRNLB fused normalize+gate forward (split)",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, rrnlb_forward_vector_length()),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                    const int i = team_member.league_rank();
                    auto lin1_row = Kokkos::subview(lin1_raw_cache, i, Kokkos::ALL);
                    auto pre_gate_row = Kokkos::subview(pre_gate_cache, i, Kokkos::ALL);
                    auto gate_row = Kokkos::subview(gated_cache, i, Kokkos::ALL);
                    const Precision denom = static_cast<Precision>(split_alpha)
                        + static_cast<Precision>(split_beta) * density_view(i);

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, lin1_dim),
                        [&] (const int p) {
                            pre_gate_row(p) = lin1_row(p) / denom + h_res(i, p);
                        });
                    team_member.team_barrier();

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, gate_dim),
                        [&] (const int p) { gate_row(p) = static_cast<Precision>(0.0); });
                    team_member.team_barrier();

                    const int scalar_mul = g_target_mul(0);
                    const int scalar_nonlin_off = g_nonlin_off(0);
                    const int scalar_target_off = g_target_off(0);
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, scalar_mul),
                        [&] (const int k) {
                            const Precision v = pre_gate_row(scalar_nonlin_off + k);
                            gate_row(scalar_target_off + k) = gate_identity_mode
                                ? g_scalar_cst * v
                                : g_scalar_cst * rrnlb_silu(v);
                        });
                    team_member.team_barrier();

                    int gate_off = g_nonlin_off(0) + g_target_mul(0);
                    for (int gp = 1; gp < g_num_parts; ++gp) {
                        const int out_off = g_target_off(gp);
                        const int in_off = g_nonlin_off(gp);
                        const int mul = g_target_mul(gp);
                        const int ir_d = 2 * g_target_l(gp) + 1;
                        const Precision cst =
                            gp - 1 < g_gate_cst.extent_int(0)
                                ? g_gate_cst(gp - 1)
                                : static_cast<Precision>(1.0);
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, mul),
                            [&] (const int k) {
                                const Precision g = gate_identity_mode
                                    ? cst
                                    : cst * rrnlb_sigmoid(pre_gate_row(gate_off + k));
                                for (int m = 0; m < ir_d; ++m) {
                                    gate_row(out_off + k * ir_d + m) =
                                        g * pre_gate_row(in_off + k * ir_d + m);
                                }
                            });
                        team_member.team_barrier();
                        gate_off += mul;
                    }
                });

            rrnlb_apply_linear_forward(layer.linear_2, num_nodes, cache.gated, layer_output);
        } else {
            // Edge-parallel forward: one team per edge for maximum GPU occupancy.
            const auto edge_recv = edge_to_receiver;
            if (use_compact_out) {
                Kokkos::parallel_for(
                    "RRNLB conv forward (edge-parallel compact-out)",
                    Kokkos::TeamPolicy<>(total_edges, Kokkos::AUTO, rrnlb_forward_vector_length())
                        .set_scratch_size(
                            0,
                            Kokkos::PerTeam(compact_out_count * sizeof(Precision))),
                    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                        const int ij = team_member.league_rank();
                        const int i = edge_recv(ij);
                        const int sender = neigh_indices(ij);
                        auto conv_edge_local = Kokkos::View<Precision*,Kokkos::MemoryUnmanaged>(
                            team_member.team_scratch(0), compact_out_count);

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, compact_out_count),
                            [&] (const int p) { conv_edge_local(p) = static_cast<Precision>(0.0); });
                        team_member.team_barrier();

                        Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                            Kokkos::atomic_add(&density_view(i), edge_den_val(ij));
                        });

                        const auto up_sender = Kokkos::subview(h_up_all, sender, Kokkos::ALL);
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, num_work_items),
                            [&] (const int w) {
                                const int out_s = work_out_slot(w);
                                if (out_s >= 0) {
                                    const Precision val =
                                        work_coeff(w)
                                        * Y(ij * num_lm + work_y_lm(w))
                                        * edge_tp_vals(ij, work_w_idx(w))
                                        * up_sender(work_in_idx(w));
                                    Kokkos::atomic_add(&conv_edge_local(out_s), val);
                                }
                            });
                        team_member.team_barrier();

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, compact_out_count),
                            [&] (const int p_local) {
                                const Precision val = conv_edge_local(p_local);
                                if (val != static_cast<Precision>(0.0)) {
                                    const int p = conv_active_out_indices(p_local);
                                    Kokkos::atomic_add(&conv_accum(i, p), val);
                                }
                            });
                    });
            } else {
                Kokkos::parallel_for(
                    "RRNLB conv forward (edge-parallel)",
                    Kokkos::TeamPolicy<>(total_edges, Kokkos::AUTO, rrnlb_forward_vector_length()),
                    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                        const int ij = team_member.league_rank();
                        const int i = edge_recv(ij);
                        const int sender = neigh_indices(ij);

                        Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                            Kokkos::atomic_add(&density_view(i), edge_den_val(ij));
                        });

                        const auto up_sender = Kokkos::subview(h_up_all, sender, Kokkos::ALL);
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, num_work_items),
                            [&] (const int w) {
                                const Precision val =
                                    work_coeff(w)
                                    * Y(ij * num_lm + work_y_lm(w))
                                    * edge_tp_vals(ij, work_w_idx(w))
                                    * up_sender(work_in_idx(w));
                                Kokkos::atomic_add(&conv_accum(i, work_out_idx(w)), val);
                            });
                    });
            }
        }
    } else {
        // Fallback: original node-parallel forward (for MPI paths without edge map)
        const int compact_out_count =
            layer.conv_active_out_count > 0 ? layer.conv_active_out_count : conv_out_dim;
        const bool compact_auto_node =
            compact_out_count * 5 <= conv_out_dim * 4 && compact_out_count <= 2048;
        const auto compact_mode = rrnlb_interaction_compact_mode();
        const bool use_compact_out_node =
            compact_mode == RRNLBInteractionCompactMode::On
                ? (compact_out_count <= 2048)
                : (compact_mode == RRNLBInteractionCompactMode::Auto && compact_auto_node);
        if (use_compact_out_node) {
            Kokkos::parallel_for(
                "RRNLB conv forward (node-parallel compact-out)",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, rrnlb_forward_vector_length())
                    .set_scratch_size(
                        0,
                        Kokkos::PerTeam((tp_weight_numel + compact_out_count) * sizeof(Precision))),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                    const int i = team_member.league_rank();
                    auto scratch = Kokkos::View<Precision*,Kokkos::MemoryUnmanaged>(
                        team_member.team_scratch(0), tp_weight_numel + compact_out_count);
                    auto tp_values = Kokkos::subview(scratch, Kokkos::make_pair(0, tp_weight_numel));
                    auto conv_local =
                        Kokkos::subview(scratch, Kokkos::make_pair(tp_weight_numel, tp_weight_numel + compact_out_count));
                    auto conv_i = Kokkos::subview(conv_accum, i, Kokkos::ALL);
                    Precision density_i = static_cast<Precision>(0.0);

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, compact_out_count),
                        [&] (const int p_local) {
                            conv_local(p_local) = static_cast<Precision>(0.0);
                        });
                    team_member.team_barrier();

                    const int i0 = first_neigh(i);
                    for (int j = 0; j < num_neigh(i); ++j) {
                        const int ij = i0 + j;
                        const int sender = neigh_indices(ij);
                        const int pair_index = neigh_types(ij) * num_elements + node_types(i);
                        const Precision r_ij = static_cast<Precision>(r(ij));
                        int interval = static_cast<int>(r_ij / static_cast<Precision>(radial_h));
                        if (interval < 0) interval = 0;
                        if (interval >= radial_num_intervals) interval = radial_num_intervals - 1;
                        const Precision x = r_ij - static_cast<Precision>(radial_h) * interval;
                        const Precision xx = x * x;
                        const Precision xxx = xx * x;

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, tp_weight_numel),
                            [&] (const int w) {
                                tp_values(w) = rrnlb_eval_spline(tp_spline_coeff, pair_index, interval, w, x, xx, xxx);
                            });
                        team_member.team_barrier();

                        density_i += rrnlb_eval_spline_scalar(density_spline_coeff, pair_index, interval, x, xx, xxx);
                        const auto up_sender = Kokkos::subview(h_up_all, sender, Kokkos::ALL);

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, num_work_items),
                            [&] (const int w) {
                                const int out_s = work_out_slot(w);
                                if (out_s >= 0) {
                                    const Precision val =
                                        work_coeff(w)
                                        * Y(ij * num_lm + work_y_lm(w))
                                        * tp_values(work_w_idx(w))
                                        * up_sender(work_in_idx(w));
                                    Kokkos::atomic_add(&conv_local(out_s), val);
                                }
                            });
                    }

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, compact_out_count),
                        [&] (const int p_local) {
                            const Precision val = conv_local(p_local);
                            if (val != static_cast<Precision>(0.0)) {
                                conv_i(conv_active_out_indices(p_local)) += val;
                            }
                        });

                    Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                        cache.density(i) = density_i;
                    });
                });
        } else {
            Kokkos::parallel_for(
                "RRNLB conv forward",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, rrnlb_forward_vector_length())
                    .set_scratch_size(0, Kokkos::PerTeam(tp_weight_numel * sizeof(Precision))),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                    const int i = team_member.league_rank();
                    auto tp_values = Kokkos::View<Precision*,Kokkos::MemoryUnmanaged>(
                        team_member.team_scratch(0), tp_weight_numel);
                    auto conv_i = Kokkos::subview(conv_accum, i, Kokkos::ALL);
                    Precision density_i = static_cast<Precision>(0.0);

                    const int i0 = first_neigh(i);
                    for (int j = 0; j < num_neigh(i); ++j) {
                        const int ij = i0 + j;
                        const int sender = neigh_indices(ij);
                        const int pair_index = neigh_types(ij) * num_elements + node_types(i);
                        const Precision r_ij = static_cast<Precision>(r(ij));
                        int interval = static_cast<int>(r_ij / static_cast<Precision>(radial_h));
                        if (interval < 0) interval = 0;
                        if (interval >= radial_num_intervals) interval = radial_num_intervals - 1;
                        const Precision x = r_ij - static_cast<Precision>(radial_h) * interval;
                        const Precision xx = x * x;
                        const Precision xxx = xx * x;

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, tp_weight_numel),
                            [&] (const int w) {
                                tp_values(w) = rrnlb_eval_spline(tp_spline_coeff, pair_index, interval, w, x, xx, xxx);
                            });
                        team_member.team_barrier();

                        density_i += rrnlb_eval_spline_scalar(density_spline_coeff, pair_index, interval, x, xx, xxx);
                        const auto up_sender = Kokkos::subview(h_up_all, sender, Kokkos::ALL);

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, num_work_items),
                            [&] (const int w) {
                                const Precision val =
                                    work_coeff(w)
                                    * Y(ij * num_lm + work_y_lm(w))
                                    * tp_values(work_w_idx(w))
                                    * up_sender(work_in_idx(w));
                                Kokkos::atomic_add(&conv_i(work_out_idx(w)), val);
                            });
                    }

                    Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                        cache.density(i) = density_i;
                    });
                });
        }
    }
    if (!did_fused_forward && !did_split_forward) {
        rrnlb_apply_linear_forward(layer.linear_1, num_nodes, conv_accum, cache.lin1_raw);
        const auto density = cache.density;
        const auto lin1_raw = cache.lin1_raw;
        const auto pre_gate = cache.pre_gate;
        const double alpha = layer.alpha;
        const double beta = layer.beta;
        Kokkos::parallel_for(
            "RRNLB normalize/residual",
            num_nodes * layer.linear_1.dim_out,
            KOKKOS_LAMBDA (const int ip) {
                const int i = ip / layer.linear_1.dim_out;
                const int p = ip % layer.linear_1.dim_out;
                const Precision denom = static_cast<Precision>(alpha) + static_cast<Precision>(beta) * density(i);
                pre_gate(i, p) = lin1_raw(i, p) / denom + h_res(i, p);
            });
        rrnlb_apply_gate_forward(layer, num_nodes, cache.pre_gate, gated);
        rrnlb_apply_linear_forward(layer.linear_2, num_nodes, gated, layer_output);
    }
}

template <typename T>
struct RRNLBReverseReduceVal {
    T dE_dr, fx, fy, fz;
    KOKKOS_INLINE_FUNCTION RRNLBReverseReduceVal()
        : dE_dr(0), fx(0), fy(0), fz(0) {}
    KOKKOS_INLINE_FUNCTION void operator+=(const RRNLBReverseReduceVal& rhs) {
        dE_dr += rhs.dE_dr; fx += rhs.fx; fy += rhs.fy; fz += rhs.fz;
    }
};

namespace Kokkos {
template <typename T>
struct reduction_identity<RRNLBReverseReduceVal<T>> {
    KOKKOS_FORCEINLINE_FUNCTION static RRNLBReverseReduceVal<T> sum() {
        return RRNLBReverseReduceVal<T>();
    }
};
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_rrnlb_interaction_layer(
    const int layer_index,
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const int*> first_neigh,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> node_feats_in,
    RRNLBLayerCacheKokkos& cache,
    Kokkos::View<const AccumPrecision**,Kokkos::LayoutRight> layer_output_adj,
    Kokkos::View<const AccumPrecision**,Kokkos::LayoutRight> layer_skip_adj,
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> node_feats_in_adj,
    int num_sender_nodes,
    Kokkos::View<const int*> target_node_indices,
    int total_edges,
    Kokkos::View<const int*> edge_to_receiver,
    Kokkos::View<const int*> sender_edge_offsets,
    Kokkos::View<const int*> sender_edge_indices,
    Kokkos::View<const int*> sender_segment_offsets,
    Kokkos::View<const int*> sender_segment_to_sender,
    int total_sender_segments)
{
    if (layer_index < 0 || layer_index >= static_cast<int>(rrnlb_layers_kokkos.size())) {
        throw std::runtime_error("RRNLB reverse layer index out of bounds.");
    }
    const auto& layer = rrnlb_layers_kokkos[layer_index];
    const int in_dim = layer.linear_up.dim_in;
    const int sender_nodes = (num_sender_nodes > 0) ? num_sender_nodes : num_nodes;
    const bool has_target_map = target_node_indices.extent(0) > 0;
    ensure_rrnlb_layer_workspace_capacity(cache, num_nodes, sender_nodes, total_edges);
    bind_rrnlb_layer_active_views(cache, num_nodes, sender_nodes, total_edges);
    ensure_rrnlb_layer_dispatch_cache(layer, cache, sender_nodes, total_edges);
    const int cap_nodes = std::max(cache.capacity_nodes, num_nodes);
    const int cap_sender_nodes = std::max(cache.capacity_sender_nodes, sender_nodes);

    if (node_feats_in.extent(0) < sender_nodes || node_feats_in.extent(1) != in_dim) {
        throw std::runtime_error("RRNLB reverse input has invalid shape.");
    }
    if (node_feats_in_adj.extent(0) < sender_nodes || node_feats_in_adj.extent(1) != in_dim) {
        throw std::runtime_error("RRNLB reverse input adjoint has invalid shape.");
    }
    if (has_target_map && target_node_indices.extent(0) < num_nodes) {
        throw std::runtime_error("RRNLB reverse target index map has invalid shape.");
    }
    if (cache.h_up.extent(0) < sender_nodes
        || cache.h_up.extent(1) != layer.linear_up.dim_out
        || cache.density.extent(0) < num_nodes
        || cache.lin1_raw.extent(0) < num_nodes
        || cache.lin1_raw.extent(1) != layer.linear_1.dim_out
        || cache.pre_gate.extent(0) < num_nodes
        || cache.pre_gate.extent(1) != layer.linear_1.dim_out) {
        throw std::runtime_error("RRNLB reverse cache has invalid shape.");
    }
    auto ensure_cache_2d = [](auto& view, const int d0, const int d1) {
        if (view.extent(0) < static_cast<std::size_t>(d0)
            || view.extent(1) < static_cast<std::size_t>(d1)) {
            Kokkos::realloc(view, d0, d1);
        }
    };
    auto ensure_cache_1d = [](auto& view, const int d0) {
        if (view.extent(0) < static_cast<std::size_t>(d0)) {
            Kokkos::realloc(view, d0);
        }
    };

    ensure_cache_2d(cache.gated_adj, cap_nodes, layer.linear_2.dim_in);
    ensure_cache_2d(cache.pre_gate_adj, cap_nodes, layer.linear_1.dim_out);
    ensure_cache_2d(cache.lin1_raw_adj, cap_nodes, layer.linear_1.dim_out);
    ensure_cache_2d(cache.h_res_adj, cap_nodes, layer.linear_res.dim_out);
    ensure_cache_2d(cache.conv_adj, cap_nodes, layer.linear_1.dim_in);
    ensure_cache_2d(cache.h_up_adj, cap_sender_nodes, layer.linear_up.dim_out);
    ensure_cache_2d(cache.h_up_adj_targets, cap_nodes, layer.linear_up.dim_out);
    ensure_cache_2d(cache.skip_input_adj_targets, cap_nodes, in_dim);
    ensure_cache_1d(cache.density_adj, cap_nodes);
    ensure_cache_2d(cache.x_up_adj, cap_sender_nodes, in_dim);

    auto gated_adj = cache.gated_adj;
    auto pre_gate_adj = cache.pre_gate_adj;
    auto lin1_raw_adj = cache.lin1_raw_adj;
    auto h_res_adj = cache.h_res_adj;
    auto conv_adj = cache.conv_adj;
    auto h_up_adj = cache.h_up_adj;
    auto h_up_adj_targets = cache.h_up_adj_targets;
    auto skip_input_adj_targets = cache.skip_input_adj_targets;
    auto density_adj = cache.density_adj;

    // h_up_adj is an accumulation buffer across multiple reverse stages inside this
    // call, but must start at zero for each invocation.
    Kokkos::deep_copy(
        Kokkos::subview(h_up_adj, Kokkos::make_pair(0, sender_nodes), Kokkos::ALL),
        static_cast<AccumPrecision>(0.0));
    // lin1_raw_adj, h_res_adj, density_adj: write-before-read in the normalize
    // reverse kernel below -- no zeroing needed.

    // Partially-fused reverse: cuBLAS linear_2^T → fused gate+normalize reverse → cuBLAS linear_1^T.
    // Restores cuBLAS batched GEMM for the two linear transposes (eliminating hand-written
    // GEMV register pressure), and fuses only the lightweight gate_reverse + normalize_reverse
    // into a single kernel with scratch-tiled pre_gate_adj and global fallback.
    {
        const int lin1_dim = layer.linear_1.dim_out;

        // Stage 1: cuBLAS linear_2^T (layer_output_adj → gated_adj).
        rrnlb_apply_linear_transpose(layer.linear_2, num_nodes, layer_output_adj, gated_adj);

        // Stage 2: Fused gate_reverse + normalize_reverse kernel.
        // ~15 captures vs 40+ in the full fusion → much better register pressure/occupancy.
        // Reads gated_adj, pre_gate_cache, lin1_raw, density from global.
        // Writes lin1_raw_adj, h_res_adj, density_adj to global.
        // pre_gate_adj is scratch-tiled when possible, with global fallback.
        {
            const auto g_target_off = layer.target_offset;
            const auto g_target_mul = layer.target_mul;
            const auto g_target_l = layer.target_l;
            const auto g_nonlin_off = layer.nonlin_offset;
            const auto g_gate_cst = layer.gate_gate_cst;
            const AccumPrecision g_scalar_cst = static_cast<AccumPrecision>(layer.gate_scalar_cst);
            const int g_num_parts = layer.num_gate_parts;
            const bool gate_identity_mode =
                rrnlb_nonlinear_ablation_mode() == RRNLBNonlinearAblationMode::GateIdentity;

            const double rev_alpha = layer.alpha;
            const double rev_beta = layer.beta;

            const auto pre_gate_cache_view = cache.pre_gate;
            const auto lin1_raw = cache.lin1_raw;
            const auto density = cache.density;

            const bool use_scratch_rev =
                layer.fusion_storage_mode_rev == RRNLBFusionStorageMode::ScratchTiled;
            const int rev_scratch_bytes = use_scratch_rev
                ? lin1_dim * sizeof(AccumPrecision)
                : 0;
            if (use_scratch_rev) {
                rrnlb_phase_counters.fused_reverse_scratch_tiled_calls++;
            } else {
                rrnlb_phase_counters.fused_reverse_global_staged_calls++;
            }

            Kokkos::parallel_for(
                "RRNLB fused gate+normalize reverse",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, rrnlb_reverse_vector_length())
                    .set_scratch_size(0, Kokkos::PerTeam(rev_scratch_bytes)),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                    const int i = team_member.league_rank();
                    auto gated_adj_row = Kokkos::subview(gated_adj, i, Kokkos::ALL);
                    AccumPrecision* pregate_adj_ptr = nullptr;
                    if (use_scratch_rev) {
                        using ScratchSpace =
                            Kokkos::TeamPolicy<>::member_type::scratch_memory_space;
                        auto scratch = team_member.team_scratch(0);
                        Kokkos::View<AccumPrecision*, ScratchSpace, Kokkos::MemoryUnmanaged>
                            pregate_adj_row_scratch(scratch, lin1_dim);
                        pregate_adj_ptr = pregate_adj_row_scratch.data();
                    } else {
                        auto pregate_adj_row_global = Kokkos::subview(pre_gate_adj, i, Kokkos::ALL);
                        pregate_adj_ptr = pregate_adj_row_global.data();
                    }

                    auto lin1raw_adj_row = Kokkos::subview(lin1_raw_adj, i, Kokkos::ALL);

                    // === Gate Reverse ===
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, lin1_dim),
                        [&] (const int p) { pregate_adj_ptr[p] = static_cast<AccumPrecision>(0.0); });
                    team_member.team_barrier();

                    // Scalar channel: SiLU derivative.
                    {
                        const int scalar_mul = g_target_mul(0);
                        const int scalar_nonlin_off = g_nonlin_off(0);
                        const int scalar_target_off = g_target_off(0);
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, scalar_mul),
                            [&] (const int k) {
                                const int in_idx = scalar_nonlin_off + k;
                                const int out_idx = scalar_target_off + k;
                                pregate_adj_ptr[in_idx] += gated_adj_row(out_idx)
                                    * g_scalar_cst
                                    * (gate_identity_mode
                                        ? static_cast<AccumPrecision>(1.0)
                                        : static_cast<AccumPrecision>(
                                              rrnlb_silu_deriv(pre_gate_cache_view(i, in_idx))));
                            });
                        team_member.team_barrier();
                    }

                    // Higher-l: sigmoid gate derivative.
                    {
                        int gate_off = g_nonlin_off(0) + g_target_mul(0);
                        for (int gp = 1; gp < g_num_parts; ++gp) {
                            const int out_off = g_target_off(gp);
                            const int in_off = g_nonlin_off(gp);
                            const int mul = g_target_mul(gp);
                            const int ir_d = 2 * g_target_l(gp) + 1;
                            const AccumPrecision cst =
                                gp - 1 < g_gate_cst.extent_int(0)
                                    ? static_cast<AccumPrecision>(g_gate_cst(gp - 1))
                                    : static_cast<AccumPrecision>(1.0);
                            Kokkos::parallel_for(
                                Kokkos::TeamThreadRange(team_member, mul),
                                [&] (const int k) {
                                    const AccumPrecision sig = static_cast<AccumPrecision>(
                                        rrnlb_sigmoid(pre_gate_cache_view(i, gate_off + k)));
                                    const AccumPrecision g = gate_identity_mode
                                        ? cst : cst * sig;
                                    AccumPrecision dg = static_cast<AccumPrecision>(0.0);
                                    for (int m = 0; m < ir_d; ++m) {
                                        const int out_idx = out_off + k * ir_d + m;
                                        const int in_idx = in_off + k * ir_d + m;
                                        pregate_adj_ptr[in_idx] += gated_adj_row(out_idx) * g;
                                        if (!gate_identity_mode) {
                                            dg += gated_adj_row(out_idx)
                                                * static_cast<AccumPrecision>(
                                                      pre_gate_cache_view(i, in_idx));
                                        }
                                    }
                                    if (!gate_identity_mode) {
                                        pregate_adj_ptr[gate_off + k] += dg * cst * sig
                                            * (static_cast<AccumPrecision>(1.0) - sig);
                                    }
                                });
                            team_member.team_barrier();
                            gate_off += mul;
                        }
                    }

                    // === Normalize Reverse ===
                    const AccumPrecision norm_denom =
                        static_cast<AccumPrecision>(rev_alpha)
                        + static_cast<AccumPrecision>(rev_beta)
                          * static_cast<AccumPrecision>(density(i));
                    const AccumPrecision norm_inv =
                        static_cast<AccumPrecision>(1.0) / norm_denom;
                    const AccumPrecision norm_inv2 = norm_inv * norm_inv;
                    AccumPrecision density_contrib = static_cast<AccumPrecision>(0.0);
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team_member, lin1_dim),
                        [&] (const int p, AccumPrecision& lsum) {
                            const AccumPrecision upstream = pregate_adj_ptr[p];
                            lin1raw_adj_row(p) = upstream * norm_inv;
                            h_res_adj(i, p) = upstream;
                            lsum += upstream
                                * (-static_cast<AccumPrecision>(rev_beta)
                                   * static_cast<AccumPrecision>(lin1_raw(i, p)) * norm_inv2);
                        },
                        density_contrib);
                    Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                        density_adj(i) = density_contrib;
                    });
                });
        }

        // Stage 3: cuBLAS linear_1^T (lin1_raw_adj → conv_adj).
        rrnlb_apply_linear_transpose(layer.linear_1, num_nodes, lin1_raw_adj, conv_adj);
    }
    rrnlb_apply_linear_transpose(layer.linear_res, num_nodes, h_res_adj, h_up_adj_targets);
    const auto target_map = target_node_indices;
    if (!has_target_map) {
        Kokkos::parallel_for(
            "RRNLB scatter h_up_adj (unique)",
            num_nodes,
            KOKKOS_LAMBDA (const int i) {
                const int sender_i = i;
                for (int p = 0; p < h_up_adj_targets.extent_int(1); ++p) {
                    h_up_adj(sender_i, p) += h_up_adj_targets(i, p);
                }
            });
    } else {
        Kokkos::parallel_for(
            "RRNLB scatter h_up_adj",
            num_nodes,
            KOKKOS_LAMBDA (const int i) {
                const int sender_i = target_map(i);
                for (int p = 0; p < h_up_adj_targets.extent_int(1); ++p) {
                    Kokkos::atomic_add(&h_up_adj(sender_i, p), h_up_adj_targets(i, p));
                }
            });
    }
    // linear_1^T is Stage 3 of the partially-fused reverse pipeline above.

    const auto conv_in_offset = layer.conv_in_offset;
    const auto conv_out_offset = layer.conv_out_offset;
    const auto conv_mul = layer.conv_mul;
    const auto conv_weight_offset = layer.conv_weight_offset;
    const auto conv_in_ir_dim = layer.conv_in_ir_dim;
    const auto conv_out_ir_dim = layer.conv_out_ir_dim;
    const auto conv_term_offset = layer.conv_term_offset;
    const auto conv_term_count = layer.conv_term_count;
    const auto conv_term_m_out = layer.conv_term_m_out;
    const auto conv_term_m_in1 = layer.conv_term_m_in1;
    const auto conv_term_y_lm = layer.conv_term_y_lm;
    const auto conv_term_coeff = layer.conv_term_coeff;
    const int num_conv_ins = conv_in_offset.extent(0);
    const auto tp_spline_coeff = layer.tp_spline_coeff;
    const auto density_spline_coeff = layer.density_spline_coeff;
    const double radial_h = layer.radial_h;
    const int radial_num_intervals = layer.radial_num_intervals;
    const int tp_weight_numel = layer.tp_weight_numel;
    const int tp_local_numel = layer.conv_max_mul > 0 ? layer.conv_max_mul : tp_weight_numel;
    const int h_up_dim = layer.linear_up.dim_out;
    const int num_elements = this->num_elements;
    const int num_lm = this->num_lm;
    const auto Y = this->Y;
    const auto Y_grad = this->Y_grad;
    auto node_forces = this->node_forces;
    using ForceAccumPrecision = AccumPrecision;
    const auto h_up = cache.h_up;
    const bool use_cached_reverse_dispatch =
        cache.dispatch_cache.valid
        && cache.dispatch_cache.epoch_id == rrnlb_cache_stamp.epoch_rev
        && layer.conv_active_in_count > 0
        && cache.dispatch_cache.rev_active_in_indices.extent_int(0) >= layer.conv_active_in_count
        && cache.dispatch_cache.rev_active_in_inverse.extent_int(0) >= h_up_dim;
    const auto conv_active_in_indices = use_cached_reverse_dispatch
        ? cache.dispatch_cache.rev_active_in_indices
        : layer.conv_active_in_indices;
    const auto conv_active_in_inverse = use_cached_reverse_dispatch
        ? cache.dispatch_cache.rev_active_in_inverse
        : layer.conv_active_in_inverse;
    const bool use_edge_parallel_rev = total_edges > 0 && edge_to_receiver.extent(0) >= total_edges;
    const bool has_sender_maps =
        sender_edge_offsets.extent_int(0) >= sender_nodes + 1
        && sender_edge_indices.extent_int(0) >= total_edges
        && sender_segment_offsets.extent_int(0) >= sender_nodes + 1
        && sender_segment_to_sender.extent_int(0) > 0;
    const int sender_tiled_reverse_override = rrnlb_sender_tiled_reverse_override();
    const bool use_sender_tiled_reverse =
        use_edge_parallel_rev
        && has_sender_maps
        && (sender_tiled_reverse_override >= 0
                ? sender_tiled_reverse_override != 0
                : false);

    // Work table and precomputed descriptor references for flattened reverse conv.
    const auto work_out_idx = layer.conv_work_out_idx;
    const auto work_in_idx = layer.conv_work_in_idx;
    const auto work_in_local = layer.conv_work_in_local_idx;
    const auto work_w_idx = layer.conv_work_w_idx;
    const auto work_y_lm = layer.conv_work_y_lm;
    const auto work_coeff = layer.conv_work_coeff;
    const int num_work_items = layer.conv_work_table_size;

    if (use_edge_parallel_rev && use_sender_tiled_reverse) {
        // Precomputed descriptors (forward already called precompute for edge-parallel).
        const auto edge_tp_vals = rrnlb_edge_tp_values[layer_index];
        const auto edge_tp_ders = rrnlb_edge_tp_derivs[layer_index];
        const auto edge_den_der = rrnlb_edge_density_deriv[layer_index];

        const int compact_active_count = layer.conv_active_in_count > 0 ? layer.conv_active_in_count : h_up_dim;
        const bool use_compact_active = compact_active_count * 5 <= h_up_dim * 4;
        const int active_in_count = use_compact_active ? compact_active_count : h_up_dim;
        const auto edge_recv = edge_to_receiver;

        if (sender_edge_offsets.extent_int(0) < sender_nodes + 1
            || sender_edge_indices.extent_int(0) < total_edges) {
            throw std::runtime_error(
                "RRNLB reverse sender-tiled path requires precomputed sender edge maps.");
        }
        const auto sender_edge_offsets_view = sender_edge_offsets;
        const auto sender_edge_indices_view = sender_edge_indices;

        const int sender_segment_edges = std::max(1, rrnlb_sender_segment_size());
        if (sender_segment_offsets.extent_int(0) < sender_nodes + 1) {
            throw std::runtime_error(
                "RRNLB reverse sender-tiled path requires precomputed sender segment offsets.");
        }
        const auto sender_segment_offsets_view = sender_segment_offsets;
        int total_sender_segments_local = 0;
        if (total_sender_segments >= 0) {
            total_sender_segments_local = total_sender_segments;
        } else {
            auto segment_total =
                Kokkos::subview(sender_segment_offsets_view, Kokkos::make_pair(sender_nodes, sender_nodes + 1));
            auto segment_total_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), segment_total);
            total_sender_segments_local = segment_total_host(0);
        }
        if (total_sender_segments_local <= 0) {
            throw std::runtime_error(
                "RRNLB reverse sender-tiled path produced no sender segments for nonzero edge count.");
        }
        if (sender_segment_to_sender.extent_int(0) < total_sender_segments_local) {
            throw std::runtime_error(
                "RRNLB reverse sender-tiled path requires precomputed sender->segment map.");
        }
        const auto sender_segment_to_sender_view = sender_segment_to_sender;

        const int scratch_size = active_in_count;
        Kokkos::parallel_for(
            "RRNLB conv reverse (edge-parallel sender-tiled)",
            Kokkos::TeamPolicy<>(total_sender_segments_local, Kokkos::AUTO, rrnlb_reverse_vector_length())
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size * sizeof(AccumPrecision))),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int sender_segment = team_member.league_rank();
                const int sender = sender_segment_to_sender_view(sender_segment);
                if (sender < 0 || sender >= sender_nodes) return;
                const int sender_segment_begin = sender_segment_offsets_view(sender);
                const int sender_segment_local = sender_segment - sender_segment_begin;
                const int sender_segment_count =
                    sender_segment_offsets_view(sender + 1) - sender_segment_begin;
                const bool sender_has_single_segment = sender_segment_count == 1;
                const int edge_begin =
                    sender_edge_offsets_view(sender) + sender_segment_local * sender_segment_edges;
                const int edge_end = Kokkos::min(
                    edge_begin + sender_segment_edges,
                    sender_edge_offsets_view(sender + 1));
                if (edge_begin >= edge_end) return;

                auto scratch = Kokkos::View<AccumPrecision*,Kokkos::MemoryUnmanaged>(
                    team_member.team_scratch(0), scratch_size);
                auto sender_adj_local = Kokkos::subview(
                    scratch, Kokkos::make_pair(0, active_in_count));

                const auto h_up_sender = Kokkos::subview(h_up, sender, Kokkos::ALL);
                auto h_up_sender_adj = Kokkos::subview(h_up_adj, sender, Kokkos::ALL);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, active_in_count),
                    [&] (const int p) {
                        sender_adj_local(p) = static_cast<AccumPrecision>(0.0);
                    });
                team_member.team_barrier();

                for (int edge_pos = edge_begin; edge_pos < edge_end; ++edge_pos) {
                    const int ij = sender_edge_indices_view(edge_pos);
                    const int i = edge_recv(ij);
                    const Precision r_ij = static_cast<Precision>(r(ij));

                    auto conv_adj_i = Kokkos::subview(conv_adj, i, Kokkos::ALL);

                    // Flattened work table sweep — parallel_reduce for dE_dr + force contributions
                    RRNLBReverseReduceVal<AccumPrecision> rev_reduce;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team_member, num_work_items),
                        [&] (const int w, RRNLBReverseReduceVal<AccumPrecision>& vals) {
                            const AccumPrecision upstream = conv_adj_i(work_out_idx(w));
                            const AccumPrecision tp_val = static_cast<AccumPrecision>(edge_tp_vals(ij, work_w_idx(w)));
                            const AccumPrecision tp_der = static_cast<AccumPrecision>(edge_tp_ders(ij, work_w_idx(w)));
                            const AccumPrecision up_val = static_cast<AccumPrecision>(h_up_sender(work_in_idx(w)));
                            const AccumPrecision y_val = static_cast<AccumPrecision>(Y(ij * num_lm + work_y_lm(w)));
                            const AccumPrecision contrib = upstream * static_cast<AccumPrecision>(work_coeff(w));

                            if (use_compact_active) {
                                Kokkos::atomic_add(&sender_adj_local(work_in_local(w)),
                                    contrib * y_val * tp_val);
                            } else {
                                Kokkos::atomic_add(&sender_adj_local(work_in_idx(w)),
                                    contrib * y_val * tp_val);
                            }
                            vals.dE_dr += contrib * y_val * up_val * tp_der;
                            const AccumPrecision y_force_base = contrib * tp_val * up_val;
                            const int lm = work_y_lm(w);
                            vals.fx += y_force_base * static_cast<AccumPrecision>(Y_grad(ij * 3 * num_lm + lm));
                            vals.fy += y_force_base * static_cast<AccumPrecision>(Y_grad(ij * 3 * num_lm + num_lm + lm));
                            vals.fz += y_force_base * static_cast<AccumPrecision>(Y_grad(ij * 3 * num_lm + 2 * num_lm + lm));
                        },
                        rev_reduce);
                    team_member.team_barrier();

                    Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                        const ForceAccumPrecision inv_r =
                            static_cast<ForceAccumPrecision>(1.0) / static_cast<ForceAccumPrecision>(r_ij);
                        const ForceAccumPrecision dE_dr_total =
                            static_cast<ForceAccumPrecision>(
                                density_adj(i) * static_cast<AccumPrecision>(edge_den_der(ij))
                                + rev_reduce.dE_dr);
                        ForceAccumPrecision dxyz_x =
                            dE_dr_total * static_cast<ForceAccumPrecision>(xyz(3 * ij + 0)) * inv_r
                            + static_cast<ForceAccumPrecision>(rev_reduce.fx);
                        ForceAccumPrecision dxyz_y =
                            dE_dr_total * static_cast<ForceAccumPrecision>(xyz(3 * ij + 1)) * inv_r
                            + static_cast<ForceAccumPrecision>(rev_reduce.fy);
                        ForceAccumPrecision dxyz_z =
                            dE_dr_total * static_cast<ForceAccumPrecision>(xyz(3 * ij + 2)) * inv_r
                            + static_cast<ForceAccumPrecision>(rev_reduce.fz);
                        node_forces(3 * ij + 0) -= static_cast<double>(dxyz_x);
                        node_forces(3 * ij + 1) -= static_cast<double>(dxyz_y);
                        node_forces(3 * ij + 2) -= static_cast<double>(dxyz_z);
                    });
                    team_member.team_barrier();
                }

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, active_in_count),
                    [&] (const int p_local) {
                        const AccumPrecision val = sender_adj_local(p_local);
                        if (val != static_cast<AccumPrecision>(0.0)) {
                            if (use_compact_active) {
                                const int p = conv_active_in_indices(p_local);
                                if (sender_has_single_segment) {
                                    h_up_sender_adj(p) += val;
                                } else {
                                    Kokkos::atomic_add(&h_up_sender_adj(p), val);
                                }
                            } else {
                                if (sender_has_single_segment) {
                                    h_up_sender_adj(p_local) += val;
                                } else {
                                    Kokkos::atomic_add(&h_up_sender_adj(p_local), val);
                                }
                            }
                        }
                    });
            });
    } else if (use_edge_parallel_rev) {
        // Precomputed descriptors (forward already called precompute for edge-parallel).
        const auto edge_tp_vals = rrnlb_edge_tp_values[layer_index];
        const auto edge_tp_ders = rrnlb_edge_tp_derivs[layer_index];
        const auto edge_den_der = rrnlb_edge_density_deriv[layer_index];

        const auto conv_active_in_indices = layer.conv_active_in_indices;
        const auto conv_active_in_inverse = layer.conv_active_in_inverse;
        const int compact_active_count = layer.conv_active_in_count > 0 ? layer.conv_active_in_count : h_up_dim;
        const bool use_compact_active = compact_active_count * 5 <= h_up_dim * 4;
        const int active_in_count = use_compact_active ? compact_active_count : h_up_dim;
        const auto edge_recv = edge_to_receiver;
        const int scratch_size = active_in_count;
        Kokkos::parallel_for(
            "RRNLB conv reverse (edge-parallel)",
            Kokkos::TeamPolicy<>(total_edges, Kokkos::AUTO, rrnlb_reverse_vector_length())
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size * sizeof(AccumPrecision))),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int ij = team_member.league_rank();
                const int i = edge_recv(ij);
                const int sender = neigh_indices(ij);
                const Precision r_ij = static_cast<Precision>(r(ij));

                auto scratch = Kokkos::View<AccumPrecision*,Kokkos::MemoryUnmanaged>(
                    team_member.team_scratch(0), scratch_size);
                auto sender_adj_local = Kokkos::subview(
                    scratch, Kokkos::make_pair(0, active_in_count));

                auto conv_adj_i = Kokkos::subview(conv_adj, i, Kokkos::ALL);
                const auto h_up_sender = Kokkos::subview(h_up, sender, Kokkos::ALL);
                auto h_up_sender_adj = Kokkos::subview(h_up_adj, sender, Kokkos::ALL);

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, active_in_count),
                    [&] (const int p) {
                        sender_adj_local(p) = static_cast<AccumPrecision>(0.0);
                    });
                team_member.team_barrier();

                // Flattened work table sweep — parallel_reduce for dE_dr + force contributions
                RRNLBReverseReduceVal<AccumPrecision> rev_reduce;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team_member, num_work_items),
                    [&] (const int w, RRNLBReverseReduceVal<AccumPrecision>& vals) {
                        const AccumPrecision upstream = conv_adj_i(work_out_idx(w));
                        const AccumPrecision tp_val = static_cast<AccumPrecision>(edge_tp_vals(ij, work_w_idx(w)));
                        const AccumPrecision tp_der = static_cast<AccumPrecision>(edge_tp_ders(ij, work_w_idx(w)));
                        const AccumPrecision up_val = static_cast<AccumPrecision>(h_up_sender(work_in_idx(w)));
                        const AccumPrecision y_val = static_cast<AccumPrecision>(Y(ij * num_lm + work_y_lm(w)));
                        const AccumPrecision contrib = upstream * static_cast<AccumPrecision>(work_coeff(w));

                        if (use_compact_active) {
                            Kokkos::atomic_add(&sender_adj_local(work_in_local(w)),
                                contrib * y_val * tp_val);
                        } else {
                            Kokkos::atomic_add(&sender_adj_local(work_in_idx(w)),
                                contrib * y_val * tp_val);
                        }
                        vals.dE_dr += contrib * y_val * up_val * tp_der;
                        const AccumPrecision y_force_base = contrib * tp_val * up_val;
                        const int lm = work_y_lm(w);
                        vals.fx += y_force_base * static_cast<AccumPrecision>(Y_grad(ij * 3 * num_lm + lm));
                        vals.fy += y_force_base * static_cast<AccumPrecision>(Y_grad(ij * 3 * num_lm + num_lm + lm));
                        vals.fz += y_force_base * static_cast<AccumPrecision>(Y_grad(ij * 3 * num_lm + 2 * num_lm + lm));
                    },
                    rev_reduce);
                team_member.team_barrier();

                // Force computation
                Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                    const ForceAccumPrecision inv_r =
                        static_cast<ForceAccumPrecision>(1.0) / static_cast<ForceAccumPrecision>(r_ij);
                    const ForceAccumPrecision dE_dr_total =
                        static_cast<ForceAccumPrecision>(
                            density_adj(i) * static_cast<AccumPrecision>(edge_den_der(ij))
                            + rev_reduce.dE_dr);
                    ForceAccumPrecision dxyz_x =
                        dE_dr_total * static_cast<ForceAccumPrecision>(xyz(3 * ij + 0)) * inv_r
                        + static_cast<ForceAccumPrecision>(rev_reduce.fx);
                    ForceAccumPrecision dxyz_y =
                        dE_dr_total * static_cast<ForceAccumPrecision>(xyz(3 * ij + 1)) * inv_r
                        + static_cast<ForceAccumPrecision>(rev_reduce.fy);
                    ForceAccumPrecision dxyz_z =
                        dE_dr_total * static_cast<ForceAccumPrecision>(xyz(3 * ij + 2)) * inv_r
                        + static_cast<ForceAccumPrecision>(rev_reduce.fz);
                    node_forces(3 * ij + 0) -= static_cast<double>(dxyz_x);
                    node_forces(3 * ij + 1) -= static_cast<double>(dxyz_y);
                    node_forces(3 * ij + 2) -= static_cast<double>(dxyz_z);
                });

                // Scatter sender_adj_local to h_up_adj
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, active_in_count),
                    [&] (const int p_local) {
                        const AccumPrecision val = sender_adj_local(p_local);
                        if (val != static_cast<AccumPrecision>(0.0)) {
                            if (use_compact_active) {
                                const int p = conv_active_in_indices(p_local);
                                Kokkos::atomic_add(&h_up_sender_adj(p), val);
                            } else {
                                Kokkos::atomic_add(&h_up_sender_adj(p_local), val);
                            }
                        }
                    });
            });
    } else {
        // Fallback: original node-parallel reverse (for MPI paths without edge map)
        const int compact_active_count = layer.conv_active_in_count > 0 ? layer.conv_active_in_count : h_up_dim;
        const bool use_compact_active = compact_active_count * 5 <= h_up_dim * 4;
        const int active_in_count = use_compact_active ? compact_active_count : h_up_dim;
        Kokkos::parallel_for(
            "RRNLB conv reverse",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, rrnlb_reverse_vector_length())
                .set_scratch_size(
                    0,
                    Kokkos::PerTeam((2 * tp_weight_numel + num_lm + active_in_count + 1) * sizeof(AccumPrecision))),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank();
                auto scratch = Kokkos::View<AccumPrecision*,Kokkos::MemoryUnmanaged>(
                    team_member.team_scratch(0), 2 * tp_weight_numel + num_lm + active_in_count + 1);
                auto tp_values = Kokkos::subview(scratch, Kokkos::make_pair(0, tp_weight_numel));
                auto tp_derivs = Kokkos::subview(
                    scratch,
                    Kokkos::make_pair(tp_weight_numel, 2 * tp_weight_numel));
                auto y_adj_s = Kokkos::subview(
                    scratch,
                    Kokkos::make_pair(2 * tp_weight_numel, 2 * tp_weight_numel + num_lm));
                auto sender_adj_local = Kokkos::subview(
                    scratch,
                    Kokkos::make_pair(2 * tp_weight_numel + num_lm, 2 * tp_weight_numel + num_lm + active_in_count));
                auto dE_dr_shared = Kokkos::View<AccumPrecision*,Kokkos::MemoryUnmanaged>(
                    scratch.data() + (2 * tp_weight_numel + num_lm + active_in_count), 1);

                const int i0 = first_neigh(i);
                auto conv_adj_i = Kokkos::subview(conv_adj, i, Kokkos::ALL);
                for (int j = 0; j < num_neigh(i); ++j) {
                    const int ij = i0 + j;
                    const int sender = neigh_indices(ij);
                    const int pair_index = neigh_types(ij) * num_elements + node_types(i);
                    const Precision r_ij = static_cast<Precision>(r(ij));
                    int interval = static_cast<int>(r_ij / static_cast<Precision>(radial_h));
                    if (interval < 0) interval = 0;
                    if (interval >= radial_num_intervals) interval = radial_num_intervals - 1;
                    const Precision x_val = r_ij - static_cast<Precision>(radial_h) * interval;
                    const Precision xx = x_val * x_val;
                    const Precision xxx = xx * x_val;

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, active_in_count),
                        [&] (const int p) {
                            sender_adj_local(p) = static_cast<AccumPrecision>(0.0);
                        });
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, num_lm),
                        [&] (const int lm) {
                            y_adj_s(lm) = static_cast<AccumPrecision>(0.0);
                        });
                    Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                        dE_dr_shared(0) = density_adj(i)
                            * static_cast<AccumPrecision>(
                                rrnlb_eval_spline_scalar_deriv(
                                    density_spline_coeff, pair_index, interval, x_val, xx));
                    });

                    const auto h_up_sender = Kokkos::subview(h_up, sender, Kokkos::ALL);
                    auto h_up_sender_adj = Kokkos::subview(h_up_adj, sender, Kokkos::ALL);

                    // Node-parallel uses per-edge spline evaluation (no precompute buffer)
                    // but flattened work table for the inner (q, t, k) loops.
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, tp_weight_numel),
                        [&] (const int w) {
                            tp_values(w) = static_cast<AccumPrecision>(
                                rrnlb_eval_spline(
                                    tp_spline_coeff, pair_index, interval, w, x_val, xx, xxx));
                            tp_derivs(w) = static_cast<AccumPrecision>(
                                rrnlb_eval_spline_deriv(
                                    tp_spline_coeff, pair_index, interval, w, x_val, xx));
                        });
                    team_member.team_barrier();

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, num_work_items),
                        [&] (const int w) {
                            const AccumPrecision upstream = conv_adj_i(work_out_idx(w));
                            const AccumPrecision tp_val = tp_values(work_w_idx(w));
                            const AccumPrecision tp_der = tp_derivs(work_w_idx(w));
                            const AccumPrecision up_val = static_cast<AccumPrecision>(h_up_sender(work_in_idx(w)));
                            const AccumPrecision y_val = static_cast<AccumPrecision>(Y(ij * num_lm + work_y_lm(w)));
                            const AccumPrecision contrib = upstream * static_cast<AccumPrecision>(work_coeff(w));

                            if (use_compact_active) {
                                Kokkos::atomic_add(&sender_adj_local(work_in_local(w)),
                                    contrib * y_val * tp_val);
                            } else {
                                Kokkos::atomic_add(&sender_adj_local(work_in_idx(w)),
                                    contrib * y_val * tp_val);
                            }
                            Kokkos::atomic_add(&dE_dr_shared(0),
                                contrib * y_val * up_val * tp_der);
                            Kokkos::atomic_add(&y_adj_s(work_y_lm(w)),
                                contrib * tp_val * up_val);
                        });
                    team_member.team_barrier();

                    // Scatter sender_adj
                    if (use_compact_active) {
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, active_in_count),
                            [&] (const int p_local) {
                                const AccumPrecision val = sender_adj_local(p_local);
                                if (val != static_cast<AccumPrecision>(0.0)) {
                                    const int p = conv_active_in_indices(p_local);
                                    Kokkos::atomic_add(&h_up_sender_adj(p), val);
                                }
                            });
                    } else {
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team_member, active_in_count),
                            [&] (const int p) {
                                const AccumPrecision val = sender_adj_local(p);
                                if (val != static_cast<AccumPrecision>(0.0)) {
                                    Kokkos::atomic_add(&h_up_sender_adj(p), val);
                                }
                            });
                    }

                    Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                        const ForceAccumPrecision inv_r =
                            static_cast<ForceAccumPrecision>(1.0) / static_cast<ForceAccumPrecision>(r_ij);
                        const ForceAccumPrecision dE_dr_total =
                            static_cast<ForceAccumPrecision>(dE_dr_shared(0));
                        ForceAccumPrecision dxyz_x =
                            dE_dr_total * static_cast<ForceAccumPrecision>(xyz(3 * ij + 0)) * inv_r;
                        ForceAccumPrecision dxyz_y =
                            dE_dr_total * static_cast<ForceAccumPrecision>(xyz(3 * ij + 1)) * inv_r;
                        ForceAccumPrecision dxyz_z =
                            dE_dr_total * static_cast<ForceAccumPrecision>(xyz(3 * ij + 2)) * inv_r;
                        for (int lm = 0; lm < num_lm; ++lm) {
                            dxyz_x += static_cast<ForceAccumPrecision>(y_adj_s(lm))
                                * static_cast<ForceAccumPrecision>(Y_grad(ij * 3 * num_lm + lm));
                            dxyz_y += static_cast<ForceAccumPrecision>(y_adj_s(lm))
                                * static_cast<ForceAccumPrecision>(Y_grad(ij * 3 * num_lm + num_lm + lm));
                            dxyz_z += static_cast<ForceAccumPrecision>(y_adj_s(lm))
                                * static_cast<ForceAccumPrecision>(Y_grad(ij * 3 * num_lm + 2 * num_lm + lm));
                        }
                        node_forces(3 * ij + 0) -= static_cast<double>(dxyz_x);
                        node_forces(3 * ij + 1) -= static_cast<double>(dxyz_y);
                        node_forces(3 * ij + 2) -= static_cast<double>(dxyz_z);
                    });
                }
            });
    }
    rrnlb_apply_linear_transpose(layer.skip_tp, num_nodes, layer_skip_adj, skip_input_adj_targets);
    if (!has_target_map) {
        Kokkos::parallel_for(
            "RRNLB scatter skip adj (unique)",
            num_nodes,
            KOKKOS_LAMBDA (const int i) {
                const int sender_i = i;
                for (int p = 0; p < skip_input_adj_targets.extent_int(1); ++p) {
                    node_feats_in_adj(sender_i, p) += skip_input_adj_targets(i, p);
                }
            });
    } else {
        Kokkos::parallel_for(
            "RRNLB scatter skip adj",
            num_nodes,
            KOKKOS_LAMBDA (const int i) {
                const int sender_i = target_map(i);
                for (int p = 0; p < skip_input_adj_targets.extent_int(1); ++p) {
                    Kokkos::atomic_add(&node_feats_in_adj(sender_i, p), skip_input_adj_targets(i, p));
                }
            });
    }
    auto x_up_adj = cache.x_up_adj;
    rrnlb_apply_linear_transpose(layer.linear_up, sender_nodes, h_up_adj, x_up_adj);
    Kokkos::parallel_for(
        "RRNLB sum input adjoints",
        sender_nodes * in_dim,
        KOKKOS_LAMBDA (const int ip) {
            const int i = ip / in_dim;
            const int p = ip % in_dim;
            node_feats_in_adj(i, p) += x_up_adj(i, p);
        });
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_rrnlb_node_energies_forces(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const int*> first_neigh_input,
    int total_edges_input,
    Kokkos::View<const int*> edge_to_receiver_input)
{
    static bool rrnlb_gpu_debug_once = false;
    const bool rrnlb_gpu_debug = std::getenv("SYMMETRIX_RRNLB_GPU_DEBUG") != nullptr;
    auto rrnlb_debug = [&](const char* msg) {
        if (!rrnlb_gpu_debug) return;
        if (!rrnlb_gpu_debug_once) {
            std::cerr << "[RRNLB-GPU-DEBUG] " << msg << std::endl;
        }
    };

    if (rrnlb_layers_kokkos.size() != 2) {
        throw std::runtime_error("RRNLB Kokkos path currently expects exactly two interaction layers.");
    }
    if (num_nodes == 0) return;
    const int num_channels_local = num_channels;
    const int num_lm_local = num_lm;
    const int num_LM_local = num_LM;
    const int product0_dim_in = rrnlb_product_linear_0.dim_in;
    const int product0_dim_out = rrnlb_product_linear_0.dim_out;
    const int product1_dim_in = rrnlb_product_linear_1.dim_in;
    const int product1_dim_out = rrnlb_product_linear_1.dim_out;

    // Use epoch topology cache to avoid recomputing first_neigh / edge_to_receiver
    // every step when the neighbor list has not changed.
    int total_edges_local = total_edges_input;
    if (total_edges_local < 0) {
        Kokkos::parallel_reduce(
            "rrnlb_total_edges",
            num_nodes,
            KOKKOS_LAMBDA (const int i, int& lsum) {
                lsum += num_neigh(i);
            },
            total_edges_local);
    }
    const bool edge_parallel_topology_active =
        total_edges_local > 0
        && edge_to_receiver_input.extent_int(0) >= total_edges_local;
    ensure_rrnlb_epoch_topology_cache(
        num_nodes,
        total_edges_local,
        num_neigh,
        neigh_indices,
        first_neigh_input,
        edge_to_receiver_input,
        false,
        -1,
        edge_parallel_topology_active);
    ensure_rrnlb_scratch_capacity(num_nodes, total_edges_local, num_nodes);
    rrnlb_total_edges = total_edges_local;
    auto rrnlb_first_neigh_view = Kokkos::subview(
        rrnlb_epoch_topology_cache.first_neigh, Kokkos::make_pair(0, num_nodes));
    rrnlb_debug("epoch topology cache done");

    // Grow-only workspace helper: reallocates only when capacity is insufficient.
    auto ensure_sr_2d = [](auto& view, const int d0, const int d1) -> bool {
        if (view.extent_int(0) < d0 || view.extent_int(1) < d1) {
            Kokkos::realloc(view, d0, d1);
            return true;
        }
        return false;
    };
    auto ensure_sr_2d_double = [](auto& view, const int d0, const int d1) -> bool {
        if (view.extent_int(0) < d0 || view.extent_int(1) < d1) {
            Kokkos::realloc(view, d0, d1);
            return true;
        }
        return false;
    };
    auto ensure_sr_1d_double = [](auto& view, const int d0) -> bool {
        if (view.extent_int(0) < d0) {
            Kokkos::realloc(view, d0);
            return true;
        }
        return false;
    };
    const int ws_nodes = std::max(num_nodes, rrnlb_scratch_cache.max_nodes);

    // Persistent workspace allocations (grow-only, no per-step malloc/free).
    ensure_sr_2d(rrnlb_sr_node_embed, ws_nodes, num_channels_local);
    auto node_embed = rrnlb_sr_node_embed;
    const auto node_embedding = rrnlb_node_embedding;
    rrnlb_debug("node_embed begin");
    Kokkos::parallel_for(
        "rrnlb_embed",
        num_nodes * num_channels_local,
        KOKKOS_LAMBDA (const int ik) {
            const int i = ik / num_channels_local;
            const int k = ik % num_channels_local;
            node_embed(i, k) = node_embedding(node_types(i), k);
        });
    rrnlb_debug("node_embed done");

    const auto& layer0 = rrnlb_layers_kokkos[0];
    const auto& layer1 = rrnlb_layers_kokkos[1];
    ensure_sr_2d(rrnlb_sr_interaction0_out, ws_nodes, layer0.linear_2.dim_out);
    ensure_sr_2d(rrnlb_sr_skip0, ws_nodes, layer0.skip_tp.dim_out);
    ensure_sr_2d(rrnlb_sr_interaction1_out, ws_nodes, layer1.linear_2.dim_out);
    ensure_sr_2d(rrnlb_sr_skip1, ws_nodes, layer1.skip_tp.dim_out);
    auto interaction0_out = rrnlb_sr_interaction0_out;
    auto skip0 = rrnlb_sr_skip0;
    auto interaction1_out = rrnlb_sr_interaction1_out;
    auto skip1 = rrnlb_sr_skip1;

    auto& cache0 = rrnlb_cache_0;
    auto& cache1 = rrnlb_cache_1;
    rrnlb_debug("layer0 forward begin");
    compute_rrnlb_interaction_layer_forward(
        0, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r, rrnlb_first_neigh_view,
        node_embed, interaction0_out, skip0, cache0, -1, Kokkos::View<const int*>(),
        rrnlb_total_edges, rrnlb_edge_to_receiver);
    rrnlb_debug("layer0 forward done");

    // Use fused M0 forward path: compute_M0_from_rrnlb_layer0_out reads
    // interaction0_out directly, eliminating the A0 alloc+zero+scatter.
    compute_M0_from_rrnlb_layer0_out(num_nodes, node_types, interaction0_out);
    rrnlb_debug("M0 forward done");

    const bool product0_resized = ensure_sr_2d(rrnlb_sr_product0_in, ws_nodes, product0_dim_in);
    auto product0_in = rrnlb_sr_product0_in;
    if (product0_resized) {
        Kokkos::deep_copy(
            Kokkos::subview(product0_in, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
            static_cast<Precision>(0.0));
    }
    auto M0_view = M0;
    const auto& h_p0_in_offset = rrnlb_product_linear_0.h_parts_in_offset;
    const auto& h_p0_in_mul = rrnlb_product_linear_0.h_parts_in_mul;
    const auto& h_p0_in_l = rrnlb_product_linear_0.h_parts_in_l;
    const auto p0_in_offset = rrnlb_product_linear_0.parts_in_offset;
    const auto p0_in_mul = rrnlb_product_linear_0.parts_in_mul;
    const auto p0_in_l = rrnlb_product_linear_0.parts_in_l;
    const int p0_num_parts = static_cast<int>(h_p0_in_offset.size());
    Kokkos::parallel_for(
        "rrnlb_M0_to_product0",
        num_nodes * p0_num_parts,
        KOKKOS_LAMBDA (const int ipart) {
            const int i = ipart / p0_num_parts;
            const int pp = ipart % p0_num_parts;
            const int offset = p0_in_offset(pp);
            const int mul = p0_in_mul(pp);
            const int l = p0_in_l(pp);
            const int ir_dim = 2 * l + 1;
            const int lm0 = l * l;
            for (int k = 0; k < mul; ++k) {
                for (int m = 0; m < ir_dim; ++m) {
                    product0_in(i, offset + k * ir_dim + m) = M0_view(i, lm0 + m, k);
                }
            }
        });
    ensure_sr_2d(rrnlb_sr_feat0, ws_nodes, product0_dim_out);
    auto feat0 = rrnlb_sr_feat0;
    rrnlb_apply_linear_forward(rrnlb_product_linear_0, num_nodes, product0_in, feat0);
    Kokkos::parallel_for(
        "rrnlb_add_skip0",
        num_nodes * product0_dim_out,
        KOKKOS_LAMBDA (const int ip) {
            const int i = ip / product0_dim_out;
            const int p = ip % product0_dim_out;
            feat0(i, p) += skip0(i, p);
        });
    rrnlb_debug("layer1 forward begin");
    compute_rrnlb_interaction_layer_forward(
        1, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r, rrnlb_first_neigh_view,
        feat0, interaction1_out, skip1, cache1, -1, Kokkos::View<const int*>(),
        rrnlb_total_edges, rrnlb_edge_to_receiver);
    rrnlb_debug("layer1 forward done");

    // Use fused M1 forward path: compute_M1_from_rrnlb_layer1_out reads
    // interaction1_out directly, eliminating the A1 alloc+zero+scatter.
    compute_M1_from_rrnlb_layer1_out(num_nodes, node_types, interaction1_out);
    rrnlb_debug("M1 forward done");

    const bool product1_resized = ensure_sr_2d(rrnlb_sr_product1_in, ws_nodes, product1_dim_in);
    auto product1_in = rrnlb_sr_product1_in;
    if (product1_resized) {
        Kokkos::deep_copy(
            Kokkos::subview(product1_in, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
            static_cast<Precision>(0.0));
    }
    auto M1_view = M1;
    const auto& h_p1_in_offset = rrnlb_product_linear_1.h_parts_in_offset;
    const auto& h_p1_in_mul = rrnlb_product_linear_1.h_parts_in_mul;
    const auto& h_p1_in_l = rrnlb_product_linear_1.h_parts_in_l;
    for (std::size_t p = 0; p < h_p1_in_l.size(); ++p) {
        if (h_p1_in_l[p] != 0) {
            throw std::runtime_error("RRNLB Kokkos path currently expects scalar-only product_linear_1 input.");
        }
    }
    const auto p1_in_offset = rrnlb_product_linear_1.parts_in_offset;
    const auto p1_in_mul = rrnlb_product_linear_1.parts_in_mul;
    const auto p1_in_l = rrnlb_product_linear_1.parts_in_l;
    const int p1_num_parts = static_cast<int>(h_p1_in_offset.size());
    Kokkos::parallel_for(
        "rrnlb_M1_to_product1",
        num_nodes * p1_num_parts,
        KOKKOS_LAMBDA (const int ipart) {
            const int i = ipart / p1_num_parts;
            const int pp = ipart % p1_num_parts;
            const int offset = p1_in_offset(pp);
            const int mul = p1_in_mul(pp);
            const int l = p1_in_l(pp);
            const int ir_dim = 2 * l + 1;
            for (int k = 0; k < mul; ++k) {
                product1_in(i, offset + k * ir_dim) = M1_view(i, k);
            }
        });
    ensure_sr_2d(rrnlb_sr_feat1, ws_nodes, product1_dim_out);
    auto feat1 = rrnlb_sr_feat1;
    rrnlb_apply_linear_forward(rrnlb_product_linear_1, num_nodes, product1_in, feat1);
    Kokkos::parallel_for(
        "rrnlb_add_skip1",
        num_nodes * product1_dim_out,
        KOKKOS_LAMBDA (const int ip) {
            const int i = ip / product1_dim_out;
            const int p = ip % product1_dim_out;
            feat1(i, p) += skip1(i, p);
        });

    // Adjoint workspaces (write-before-read in readout kernels below).
    ensure_sr_2d(rrnlb_sr_feat0_adj, ws_nodes, product0_dim_out);
    ensure_sr_2d(rrnlb_sr_feat1_adj, ws_nodes, product1_dim_out);
    auto feat0_adj = rrnlb_sr_feat0_adj;
    auto feat1_adj = rrnlb_sr_feat1_adj;
    auto node_energies_view = node_energies;
    auto atomic_energies_view = atomic_energies;
    auto readout_1_weights_view = readout_1_weights;

    Kokkos::parallel_for(
        "rrnlb_readout0",
        num_nodes,
        KOKKOS_LAMBDA (const int i) {
            double e_i = atomic_energies_view(node_types(i));
            for (int k = 0; k < num_channels_local; ++k) {
                e_i += readout_1_weights_view(k) * static_cast<double>(feat0(i, k));
                feat0_adj(i, k) = static_cast<AccumPrecision>(readout_1_weights_view(k));
            }
            node_energies_view(i) += e_i;
        });
    ensure_sr_2d_double(rrnlb_sr_feat1_double, ws_nodes, product1_dim_out);
    auto feat1_double = rrnlb_sr_feat1_double;
    Kokkos::parallel_for(
        "rrnlb_feat1_cast",
        num_nodes * product1_dim_out,
        KOKKOS_LAMBDA (const int ip) {
            const int i = ip / product1_dim_out;
            const int p = ip % product1_dim_out;
            feat1_double(i, p) = static_cast<double>(feat1(i, p));
        });
    ensure_sr_1d_double(rrnlb_sr_readout2_out, ws_nodes);
    ensure_sr_2d_double(rrnlb_sr_readout2_adj, ws_nodes, product1_dim_out);
    auto readout2_out = rrnlb_sr_readout2_out;
    auto readout2_adj = rrnlb_sr_readout2_adj;
    readout_2.evaluate_gradient(feat1_double, readout2_out, readout2_adj);
    Kokkos::parallel_for(
        "rrnlb_readout2_accum",
        num_nodes * product1_dim_out,
        KOKKOS_LAMBDA (const int ip) {
            const int i = ip / product1_dim_out;
            const int p = ip % product1_dim_out;
            if (p == 0) {
                node_energies_view(i) += readout2_out(i);
            }
            feat1_adj(i, p) = static_cast<AccumPrecision>(readout2_adj(i, p));
        });
    auto skip1_adj = feat1_adj;
    ensure_sr_2d(rrnlb_sr_product1_in_adj, ws_nodes, product1_dim_in);
    auto product1_in_adj = rrnlb_sr_product1_in_adj;
    rrnlb_apply_linear_transpose(rrnlb_product_linear_1, num_nodes, feat1_adj, product1_in_adj);

    // Fused M1 adjoint scatter: device-dispatched over (node, part) pairs.
    if constexpr (!std::is_same_v<Precision, AccumPrecision>) {
        if (rrnlb_M1_adj_ap.extent_int(0) < num_nodes
            || rrnlb_M1_adj_ap.extent_int(1) != num_channels_local) {
            Kokkos::realloc(rrnlb_M1_adj_ap, num_nodes, num_channels_local);
        }
        Kokkos::deep_copy(rrnlb_M1_adj_ap, static_cast<AccumPrecision>(0.0));
        auto M1_adj_ap_view = rrnlb_M1_adj_ap;
        Kokkos::parallel_for(
            "rrnlb_product1_adj_to_M1_adj_ap",
            num_nodes * p1_num_parts,
            KOKKOS_LAMBDA (const int ipart) {
                const int i = ipart / p1_num_parts;
                const int pp = ipart % p1_num_parts;
                const int offset = p1_in_offset(pp);
                const int mul = p1_in_mul(pp);
                const int l = p1_in_l(pp);
                const int ir_dim = 2 * l + 1;
                for (int k = 0; k < mul; ++k) {
                    M1_adj_ap_view(i, k) = product1_in_adj(i, offset + k * ir_dim);
                }
            });
        rrnlb_debug("M1 reverse (mixed AP) begin");
        reverse_M1_mixed_rrnlb(num_nodes, node_types, rrnlb_M1_adj_ap, rrnlb_A1_adj_ap);
        rrnlb_debug("M1 reverse (mixed AP) done");
    } else {
        if (M1_adj.extent(0) < num_nodes || M1_adj.extent(1) != num_channels_local) {
            Kokkos::realloc(M1_adj, num_nodes, num_channels_local);
        }
        Kokkos::deep_copy(M1_adj, static_cast<Precision>(0.0));
        auto M1_adj_view = M1_adj;
        Kokkos::parallel_for(
            "rrnlb_product1_adj_to_M1_adj",
            num_nodes * p1_num_parts,
            KOKKOS_LAMBDA (const int ipart) {
                const int i = ipart / p1_num_parts;
                const int pp = ipart % p1_num_parts;
                const int offset = p1_in_offset(pp);
                const int mul = p1_in_mul(pp);
                const int l = p1_in_l(pp);
                const int ir_dim = 2 * l + 1;
                for (int k = 0; k < mul; ++k) {
                    M1_adj_view(i, k) = product1_in_adj(i, offset + k * ir_dim);
                }
            });
        rrnlb_debug("M1 reverse begin");
        reverse_M1(num_nodes, node_types);
        rrnlb_debug("M1 reverse done");
    }

    // Fused A1_adj -> interaction1_adj layout transform.
    const bool interaction1_adj_resized = ensure_sr_2d(
        rrnlb_sr_interaction1_adj, ws_nodes, layer1.linear_2.dim_out);
    auto interaction1_adj = rrnlb_sr_interaction1_adj;
    if (interaction1_adj_resized) {
        Kokkos::deep_copy(
            Kokkos::subview(interaction1_adj, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
            static_cast<AccumPrecision>(0.0));
    }
    const auto& h_l1_out_offset = layer1.linear_2.h_parts_out_offset;
    const auto& h_l1_out_mul = layer1.linear_2.h_parts_out_mul;
    const auto& h_l1_out_l = layer1.linear_2.h_parts_out_l;
    const auto l1_out_offset = layer1.linear_2.parts_out_offset;
    const auto l1_out_mul = layer1.linear_2.parts_out_mul;
    const auto l1_out_l = layer1.linear_2.parts_out_l;
    const int l1_num_parts = static_cast<int>(h_l1_out_offset.size());
    if constexpr (!std::is_same_v<Precision, AccumPrecision>) {
        auto A1_adj_ap_view = rrnlb_A1_adj_ap;
        Kokkos::parallel_for(
            "rrnlb_A1_adj_ap_to_layer1",
            num_nodes * l1_num_parts,
            KOKKOS_LAMBDA (const int ipart) {
                const int i = ipart / l1_num_parts;
                const int pp = ipart % l1_num_parts;
                const int offset = l1_out_offset(pp);
                const int mul = l1_out_mul(pp);
                const int l = l1_out_l(pp);
                const int ir_dim = 2 * l + 1;
                const int lm0 = l * l;
                for (int k = 0; k < mul; ++k) {
                    for (int m = 0; m < ir_dim; ++m) {
                        interaction1_adj(i, offset + k * ir_dim + m) =
                            A1_adj_ap_view(i, lm0 + m, k);
                    }
                }
            });
    } else {
        auto A1_adj_view = A1_adj;
        Kokkos::parallel_for(
            "rrnlb_A1_adj_to_layer1",
            num_nodes * l1_num_parts,
            KOKKOS_LAMBDA (const int ipart) {
                const int i = ipart / l1_num_parts;
                const int pp = ipart % l1_num_parts;
                const int offset = l1_out_offset(pp);
                const int mul = l1_out_mul(pp);
                const int l = l1_out_l(pp);
                const int ir_dim = 2 * l + 1;
                const int lm0 = l * l;
                for (int k = 0; k < mul; ++k) {
                    for (int m = 0; m < ir_dim; ++m) {
                        interaction1_adj(i, offset + k * ir_dim + m) =
                            static_cast<AccumPrecision>(A1_adj_view(i, lm0 + m, k));
                    }
                }
            });
    }

    ensure_sr_2d(rrnlb_sr_feat0_from_layer1_adj, ws_nodes, product0_dim_out);
    auto feat0_from_layer1_adj = rrnlb_sr_feat0_from_layer1_adj;
    Kokkos::deep_copy(
        Kokkos::subview(feat0_from_layer1_adj, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
        static_cast<AccumPrecision>(0.0));
    rrnlb_debug("layer1 reverse begin");
    reverse_rrnlb_interaction_layer(
        1, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, rrnlb_first_neigh_view,
        feat0, cache1, interaction1_adj, skip1_adj, feat0_from_layer1_adj, -1, Kokkos::View<const int*>(),
        rrnlb_total_edges, rrnlb_edge_to_receiver);
    rrnlb_debug("layer1 reverse done");
    Kokkos::parallel_for(
        "rrnlb_accum_feat0_adj",
        num_nodes * product0_dim_out,
        KOKKOS_LAMBDA (const int ip) {
            const int i = ip / product0_dim_out;
            const int p = ip % product0_dim_out;
            feat0_adj(i, p) += feat0_from_layer1_adj(i, p);
        });
    auto skip0_adj = feat0_adj;
    ensure_sr_2d(rrnlb_sr_product0_in_adj, ws_nodes, product0_dim_in);
    auto product0_in_adj = rrnlb_sr_product0_in_adj;
    rrnlb_apply_linear_transpose(rrnlb_product_linear_0, num_nodes, feat0_adj, product0_in_adj);

    // Fused M0 adjoint scatter: device-dispatched over (node, part) pairs.
    if constexpr (!std::is_same_v<Precision, AccumPrecision>) {
        if (rrnlb_M0_adj_ap.extent_int(0) < num_nodes
            || rrnlb_M0_adj_ap.extent_int(1) != num_LM_local
            || rrnlb_M0_adj_ap.extent_int(2) != num_channels_local) {
            Kokkos::realloc(rrnlb_M0_adj_ap, num_nodes, num_LM_local, num_channels_local);
        }
        Kokkos::deep_copy(rrnlb_M0_adj_ap, static_cast<AccumPrecision>(0.0));
        auto M0_adj_ap_view = rrnlb_M0_adj_ap;
        Kokkos::parallel_for(
            "rrnlb_product0_adj_to_M0_adj_ap",
            num_nodes * p0_num_parts,
            KOKKOS_LAMBDA (const int ipart) {
                const int i = ipart / p0_num_parts;
                const int pp = ipart % p0_num_parts;
                const int offset = p0_in_offset(pp);
                const int mul = p0_in_mul(pp);
                const int l = p0_in_l(pp);
                const int ir_dim = 2 * l + 1;
                const int lm0 = l * l;
                for (int k = 0; k < mul; ++k) {
                    for (int m = 0; m < ir_dim; ++m) {
                        M0_adj_ap_view(i, lm0 + m, k) =
                            product0_in_adj(i, offset + k * ir_dim + m);
                    }
                }
            });
        rrnlb_debug("M0 reverse (mixed AP) begin");
        reverse_M0_mixed_rrnlb(num_nodes, node_types, rrnlb_M0_adj_ap, rrnlb_A0_adj_ap);
        rrnlb_debug("M0 reverse (mixed AP) done");
    } else {
        if (M0_adj.extent(0) < num_nodes
            || M0_adj.extent(1) != num_LM_local
            || M0_adj.extent(2) != num_channels_local) {
            Kokkos::realloc(M0_adj, num_nodes, num_LM_local, num_channels_local);
        }
        Kokkos::deep_copy(M0_adj, static_cast<Precision>(0.0));
        auto M0_adj_view = M0_adj;
        Kokkos::parallel_for(
            "rrnlb_product0_adj_to_M0_adj",
            num_nodes * p0_num_parts,
            KOKKOS_LAMBDA (const int ipart) {
                const int i = ipart / p0_num_parts;
                const int pp = ipart % p0_num_parts;
                const int offset = p0_in_offset(pp);
                const int mul = p0_in_mul(pp);
                const int l = p0_in_l(pp);
                const int ir_dim = 2 * l + 1;
                const int lm0 = l * l;
                for (int k = 0; k < mul; ++k) {
                    for (int m = 0; m < ir_dim; ++m) {
                        M0_adj_view(i, lm0 + m, k) = product0_in_adj(i, offset + k * ir_dim + m);
                    }
                }
            });
        rrnlb_debug("M0 reverse begin");
        reverse_M0(num_nodes, node_types);
        rrnlb_debug("M0 reverse done");
    }

    // Fused A0_adj -> interaction0_adj layout transform.
    const bool interaction0_adj_resized = ensure_sr_2d(
        rrnlb_sr_interaction0_adj, ws_nodes, layer0.linear_2.dim_out);
    auto interaction0_adj = rrnlb_sr_interaction0_adj;
    if (interaction0_adj_resized) {
        Kokkos::deep_copy(
            Kokkos::subview(interaction0_adj, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
            static_cast<AccumPrecision>(0.0));
    }
    const auto l0_out_offset = layer0.linear_2.parts_out_offset;
    const auto l0_out_mul = layer0.linear_2.parts_out_mul;
    const auto l0_out_l = layer0.linear_2.parts_out_l;
    const int l0_num_parts = l0_out_offset.extent_int(0);
    if constexpr (!std::is_same_v<Precision, AccumPrecision>) {
        auto A0_adj_ap_view = rrnlb_A0_adj_ap;
        Kokkos::parallel_for(
            "rrnlb_A0_adj_ap_to_layer0",
            num_nodes * l0_num_parts,
            KOKKOS_LAMBDA (const int ipart) {
                const int i = ipart / l0_num_parts;
                const int pp = ipart % l0_num_parts;
                const int offset = l0_out_offset(pp);
                const int mul = l0_out_mul(pp);
                const int l = l0_out_l(pp);
                const int ir_dim = 2 * l + 1;
                const int lm0 = l * l;
                for (int k = 0; k < mul; ++k) {
                    for (int m = 0; m < ir_dim; ++m) {
                        interaction0_adj(i, offset + k * ir_dim + m) =
                            A0_adj_ap_view(i, lm0 + m, k);
                    }
                }
            });
    } else {
        auto A0_adj_view = A0_adj;
        Kokkos::parallel_for(
            "rrnlb_A0_adj_to_layer0",
            num_nodes * l0_num_parts,
            KOKKOS_LAMBDA (const int ipart) {
                const int i = ipart / l0_num_parts;
                const int pp = ipart % l0_num_parts;
                const int offset = l0_out_offset(pp);
                const int mul = l0_out_mul(pp);
                const int l = l0_out_l(pp);
                const int ir_dim = 2 * l + 1;
                const int lm0 = l * l;
                for (int k = 0; k < mul; ++k) {
                    for (int m = 0; m < ir_dim; ++m) {
                        interaction0_adj(i, offset + k * ir_dim + m) =
                            static_cast<AccumPrecision>(A0_adj_view(i, lm0 + m, k));
                    }
                }
            });
    }

    ensure_sr_2d(rrnlb_sr_node_embed_adj, ws_nodes, num_channels_local);
    auto node_embed_adj = rrnlb_sr_node_embed_adj;
    Kokkos::deep_copy(
        Kokkos::subview(node_embed_adj, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
        static_cast<AccumPrecision>(0.0));
    rrnlb_debug("layer0 reverse begin");
    reverse_rrnlb_interaction_layer(
        0, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, rrnlb_first_neigh_view,
        node_embed, cache0, interaction0_adj, skip0_adj, node_embed_adj, -1, Kokkos::View<const int*>(),
        rrnlb_total_edges, rrnlb_edge_to_receiver);
    rrnlb_debug("layer0 reverse done");
    rrnlb_gpu_debug_once = true;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_node_energies_forces(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const int*> first_neigh,
    int total_edges,
    Kokkos::View<const int*> edge_to_receiver)
{
    if (node_energies.size() < num_nodes)
        Kokkos::realloc(node_energies, num_nodes);
    if (node_forces.size() < xyz.size())
        Kokkos::realloc(node_forces, xyz.size());
    Kokkos::deep_copy(node_energies, 0.0);
    Kokkos::deep_copy(node_forces, 0.0);

    if (has_zbl)
        zbl.compute_ZBL(
            num_nodes, node_types, num_neigh, neigh_types,
            atomic_numbers, r, xyz, node_energies, node_forces);

    if (interaction_mode_rrnlb) {
        compute_Y(xyz);
        compute_rrnlb_node_energies_forces(
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            first_neigh,
            total_edges,
            edge_to_receiver);
        return;
    }

    compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_Y(xyz);

    compute_A0(num_nodes, node_types, num_neigh, neigh_types);
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M0(num_nodes, node_types);
    compute_H1(num_nodes);

    compute_Phi1(num_nodes, num_neigh, neigh_indices);
    compute_A1(num_nodes);
    compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M1(num_nodes, node_types);
    compute_H2(num_nodes, node_types);

    compute_readouts(num_nodes, node_types);

    reverse_H2(num_nodes, node_types, false);
    reverse_M1(num_nodes, node_types);
    reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    reverse_A1(num_nodes);
    reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);

    reverse_H1(num_nodes);
    reverse_M0(num_nodes, node_types);
    reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_R0(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (r.size() > R0.extent(0)) {
        Kokkos::realloc(R0, r.size(), (l_max+1)*num_channels);
        Kokkos::realloc(R0_deriv, r.size(), (l_max+1)*num_channels);
    }

    // TODO: shouldn't need all this
    // Build i_list
    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            const int num_neigh_i = num_neigh(i); 
            if (final)
                first_neigh(i) = update;
            update += num_neigh_i;
        });
    Kokkos::fence();
    Kokkos::View<int*> i_list("i_list", r.size());
    Kokkos::parallel_for("ij lists",
        num_nodes,
        KOKKOS_LAMBDA (const int i) {
            int ij = first_neigh(i);
            for (int j=0; j<num_neigh(i); ++j) {
                i_list(ij) = i;
                ij += 1;
            }
        });
    Kokkos::fence();

    const int l_max = this->l_max;
    const int num_channels = this->num_channels;
    const auto num_types = atomic_numbers.size();
    const auto h = R0_spline_h;
    const auto c = R0_spline_coefficients;
    auto R0 = this->R0;
    auto R0_deriv = this->R0_deriv;

    Kokkos::parallel_for(
        "Compute R0",
        Kokkos::TeamPolicy<>(r.size(), Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int ij = team_member.league_rank();
            const int type_i = node_types(i_list(ij));
            const int type_j = neigh_types(ij);
            const int type_ij = type_i*num_types+type_j;
            // compute x, x^2, x^3
            const int n = static_cast<int>(r(ij)/h); // TODO: bounds checking?
            const double x = r(ij) - h*n;
            const double xx = x*x;
            const double xxx = xx*x;
            const double two_x = 2*x;
            const double three_xx = 3*xx;
            // compute function values
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, (l_max+1)*num_channels),
                [&] (const int lk) {
                    const double c0 = c(type_ij,n,0,lk);
                    const double c1 = c(type_ij,n,1,lk); 
                    const double c2 = c(type_ij,n,2,lk); 
                    const double c3 = c(type_ij,n,3,lk); 
                    R0(ij,lk) = c0 + c1*x + c2*xx + c3*xxx;
                    R0_deriv(ij,lk) = c1 + c2*two_x + c3*three_xx;
                });
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_R1(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (r.size() > R1.extent(0)) {
        Kokkos::realloc(R1, r.size(), Phi1_l.size()*num_channels);
        Kokkos::realloc(R1_deriv, r.size(), Phi1_l.size()*num_channels);
    }
    radial_1.evaluate(num_nodes, node_types, num_neigh, neigh_types, r, R1, R1_deriv);
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_Y(Kokkos::View<const double*> xyz) {

#ifndef SYMMETRIX_SPHERICART_CUDA

    const int num = xyz.extent(0) / 3;
    if (Y.extent(0) < num*num_lm) {
        Kokkos::realloc(Y, num*num_lm);
        Kokkos::realloc(Y_grad, 3*num*num_lm);
    }

    const auto num_lm = this->num_lm;
    auto Y = this->Y;
    auto Y_grad = this->Y_grad;

    // TODO: review whether this is strictly necessary
    // shuffle to match e3nn
    if (xyz_shuffled.extent(0) < 3*num)
        Kokkos::realloc(xyz_shuffled, 3*num);
    auto xyz_shuffled = this->xyz_shuffled;
    Kokkos::parallel_for("shuffle_xyz", num, KOKKOS_LAMBDA (int i) {
        xyz_shuffled(3*i) = xyz(3*i+2);
        xyz_shuffled(3*i+1) = xyz(3*i);
        xyz_shuffled(3*i+2) = xyz(3*i+1);
    });
    Kokkos::fence();

    // call sphericart on host
    auto h_xyz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xyz_shuffled);
    auto h_Y = Kokkos::create_mirror_view(Y);
    auto h_Y_grad = Kokkos::create_mirror_view(Y_grad);
    sphericart::SphericalHarmonics<Precision> sphericart(l_max);
    sphericart.compute_array_with_gradients(
        h_xyz.data(), 3*num, h_Y.data(), 3*num*num_lm, h_Y_grad.data(), 3*num*num_lm),
    Kokkos::deep_copy(Y, h_Y);
    Kokkos::deep_copy(Y_grad, h_Y_grad);

    // unshuffle gradient
    if (Y_grad_shuffled.extent(0) < 3*num*num_lm)
        Kokkos::realloc(Y_grad_shuffled, 3*num*num_lm);
    Kokkos::deep_copy(Y_grad_shuffled, Y_grad);
    auto Y_grad_shuffled = this->Y_grad_shuffled;
    Kokkos::parallel_for("unshuffle_Y_grad", num, KOKKOS_LAMBDA (int i) {
        for (int lm=0; lm<num_lm; ++lm) {
            Y_grad(3*i*num_lm+0*num_lm+lm) = Y_grad_shuffled(3*i*num_lm+1*num_lm+lm);
            Y_grad(3*i*num_lm+1*num_lm+lm) = Y_grad_shuffled(3*i*num_lm+2*num_lm+lm);
            Y_grad(3*i*num_lm+2*num_lm+lm) = Y_grad_shuffled(3*i*num_lm+0*num_lm+lm);
        }
    });
    Kokkos::fence();

    // normalize to match e3nn conventions
    Kokkos::parallel_for("normalize_Y", num*num_lm, KOKKOS_LAMBDA (int i) {
        Y(i) *= 2*std::sqrt(M_PI);
    });
    Kokkos::parallel_for("normalize_Y_grad", 3*num*num_lm, KOKKOS_LAMBDA (int i) {
        Y_grad(i) *= 2*std::sqrt(M_PI);
    });
    Kokkos::fence();

#else // SYMMETRIX_SPHERICART_CUDA

    const int num = xyz.extent(0) / 3;
    const int num_lm = (l_max+1)*(l_max+1);
    if (Y.size() < num*num_lm) {
        Kokkos::realloc(Y, num*num_lm);
        Kokkos::realloc(Y_grad, 3*num*num_lm);
    }
    auto Y = this->Y;
    auto Y_grad = this->Y_grad;

    // shuffle to match e3nn
    if (xyz_shuffled.extent(0) < 3*num)
        Kokkos::realloc(xyz_shuffled, 3*num);
    auto xyz_shuffled = this->xyz_shuffled;
    Kokkos::parallel_for("shuffle_xyz", num, KOKKOS_LAMBDA (int i) {
        xyz_shuffled(3*i) = xyz(3*i+2);
        xyz_shuffled(3*i+1) = xyz(3*i);
        xyz_shuffled(3*i+2) = xyz(3*i+1);
    });
    Kokkos::fence();

    // call sphericart
    sphericart::cuda::SphericalHarmonics<Precision> sphericart(l_max);
    sphericart.compute_with_gradients(xyz_shuffled.data(), num, Y.data(), Y_grad.data());

    // unshuffle gradient
    if (Y_grad_shuffled.extent(0) < 3*num*num_lm)
        Kokkos::realloc(Y_grad_shuffled, 3*num*num_lm);
    Kokkos::deep_copy(Y_grad_shuffled, Y_grad);
    auto Y_grad_shuffled = this->Y_grad_shuffled;
    Kokkos::parallel_for("unshuffle_Y_grad", num, KOKKOS_LAMBDA (int i) {
        for (int lm=0; lm<num_lm; ++lm) {
            Y_grad(3*i*num_lm+0*num_lm+lm) = Y_grad_shuffled(3*i*num_lm+1*num_lm+lm);
            Y_grad(3*i*num_lm+1*num_lm+lm) = Y_grad_shuffled(3*i*num_lm+2*num_lm+lm);
            Y_grad(3*i*num_lm+2*num_lm+lm) = Y_grad_shuffled(3*i*num_lm+0*num_lm+lm);
        }
    });
    Kokkos::fence();

    // normalize to match e3nn conventions
    const double normalization_factor = 2 * std::sqrt(M_PI);
    Kokkos::parallel_for("normalize_Y", num*num_lm, KOKKOS_LAMBDA (int i) {
        Y(i) *= normalization_factor;
    });
    Kokkos::parallel_for("normalize_Y_grad", 3*num*num_lm, KOKKOS_LAMBDA (int i) {
        Y_grad(i) *= normalization_factor;
    });

#endif

    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_A0(
    const int num_nodes,
    View<const int*> node_types,
    View<const int*> num_neigh,
    View<const int*> neigh_types)
{
    if (A0.extent(0) != num_nodes)
        Kokkos::realloc(A0, num_nodes, num_lm, num_channels);
    Kokkos::deep_copy(A0, 0.0);

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });
    Kokkos::fence();

    const int num_lm = this->num_lm;
    const int num_channels = this->num_channels;
    const auto R0 = this->R0;
    const auto Y = this->Y;
    auto A0 = this->A0;

    parallel_for("Compute A0",
        TeamPolicy<>(num_nodes*num_lm, Kokkos::AUTO, 32)
             .set_scratch_size(0, PerTeam(num_channels*sizeof(double))),
        KOKKOS_LAMBDA (TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / num_lm;
            const int lm = team_member.league_rank() % num_lm;
            const int l = Kokkos::sqrt(lm);
            for (int j=0; j<num_neigh(i); ++j) {
                const int ij = first_neigh(i) + j;
                const double Y_ij_lm = Y(ij*num_lm+lm);
                parallel_for(
                    TeamVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        A0(i,lm,k) += R0(ij,l*num_channels+k) * Y_ij_lm;
                    });
            }
        });

    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_A0(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    auto A0_adj = this->A0_adj;
    auto num_lm = this->num_lm;
    auto num_channels = this->num_channels;
    auto R0 = this->R0;
    auto R0_deriv = this->R0_deriv;
    auto Y = this->Y;
    auto Y_grad = this->Y_grad;
    auto node_forces = this->node_forces;

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            const int num_neigh_i = num_neigh(i); 
            if (final)
                first_neigh(i) = update;
            update += num_neigh_i;
        });
    Kokkos::fence();

    Kokkos::parallel_for("Reverse A0",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            for (int j=0; j<num_neigh(i); ++j) {
                const int ij = i0 + j;
                const double r_ij = r(ij);
                const double x_ij = xyz(3*ij) / r_ij;
                const double y_ij = xyz(3*ij+1) / r_ij;
                const double z_ij = xyz(3*ij+2) / r_ij;
                const Precision* Y_ij = &Y(ij*num_lm);
                const Precision* Y_grad_ij = &Y_grad(3*ij*num_lm);
                double f_x, f_y, f_z;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team_member, num_lm),
                    [&] (const int lm, double& f_x, double& f_y, double& f_z) {
                        const int l = Kokkos::sqrt(lm);
                        double t1, t2;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team_member, num_channels),
                            [&] (const int k, double& t1, double& t2) {
                                t1 += R0_deriv(ij,l*num_channels+k) * A0_adj(i,lm,k);
                                t2 += R0(ij,l*num_channels+k) * A0_adj(i,lm,k);
                            }, t1, t2);
                        f_x += t1*x_ij*Y_ij[lm] + t2*Y_grad_ij[lm];
                        f_y += t1*y_ij*Y_ij[lm] + t2*Y_grad_ij[num_lm+lm];
                        f_z += t1*z_ij*Y_ij[lm] + t2*Y_grad_ij[2*num_lm+lm];
                    }, f_x, f_y, f_z);
                    Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                        node_forces(3*ij)   -= f_x;
                        node_forces(3*ij+1) -= f_y;
                        node_forces(3*ij+2) -= f_z;
                    });
            }
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (not A0_scaled) return;

    // compute A0 splines
    if (r.size() > A0_spline_values.extent(0)) {
        Kokkos::realloc(A0_spline_values, r.size(), 1);
        Kokkos::realloc(A0_spline_derivs, r.size(), 1);
    }
    A0_splines.evaluate(num_nodes, node_types, num_neigh, neigh_types, r, A0_spline_values, A0_spline_derivs);

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    // perform the scaling
    const auto A0_spline_values = this->A0_spline_values;
    const auto A0_spline_derivs = this->A0_spline_derivs;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto A0 = this->A0;
    Kokkos::parallel_for(
        "MACEKokkos::compute_A0_scaled",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            // compute scale factor
            double A0_scale_factor;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j, double& lsum) {
                    lsum += A0_spline_values(first_neigh(i)+j,0);
                }, A0_scale_factor);
            A0_scale_factor += 1.0;
            team_member.team_barrier();
            // perform the scaling
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A0(i,lmk/num_channels,lmk%num_channels) /= A0_scale_factor;
                });
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    if (not A0_scaled) return;

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    // update the derivatives
    // Warning: Assumes node_forces have been initialized elsewhere
    const auto A0 = this->A0;
    const auto A0_adj = this->A0_adj;
    const auto A0_spline_values = this->A0_spline_values;
    const auto A0_spline_derivs = this->A0_spline_derivs;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto node_forces = this->node_forces;
    Kokkos::parallel_for(
        "MACEKokkos::reverse_A0_scaled",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            // compute scale factor
            double A0_scale_factor;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j, double& lsum) {
                    lsum += A0_spline_values(first_neigh(i)+j,0);
                }, A0_scale_factor);
            A0_scale_factor += 1.0;
            team_member.team_barrier();
            // update dE/dxyz
            double dA0_dot_A0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (const int lmk, double& lsum) {
                    const int lm = lmk / num_channels;
                    const int k = lmk % num_channels;
                    lsum += A0_adj(i,lm,k) * A0(i,lm,k);
                }, dA0_dot_A0);
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j) {
                    const int ij = first_neigh(i) + j;
                    const double f = A0_spline_values(ij,0);
                    const double d = A0_spline_derivs(ij,0);
                    node_forces(3*ij+0) += dA0_dot_A0/A0_scale_factor*d*xyz(3*ij+0)/r(ij);
                    node_forces(3*ij+1) += dA0_dot_A0/A0_scale_factor*d*xyz(3*ij+1)/r(ij);
                    node_forces(3*ij+2) += dA0_dot_A0/A0_scale_factor*d*xyz(3*ij+2)/r(ij);
                });
            // update dE/dA0
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A0_adj(i,lmk/num_channels,lmk%num_channels) /= A0_scale_factor;
                });
        });
    Kokkos::fence();
}

#if 0
template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_M0(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    Kokkos::resize(M0, num_nodes, num_LM, num_channels);

    auto A0 = this->A0;
    auto M0 = this->M0;
    auto M0_monomials = this->M0_monomials;
    auto M0_weights = this->M0_weights;
    auto num_LM = this->num_LM;
    auto num_channels = this->num_channels;

    Kokkos::fence();
    Kokkos::parallel_for(
        "Compute M0",
        Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right>>(
            {0,0,0}, {num_nodes,num_LM,num_channels}),
        KOKKOS_LAMBDA (const int i, const int LM, const int k) {
            M0(i,LM,k) = 0.0;
            for (int u=0; u<M0_monomials(LM).extent(0); ++u) {
                double monomial = A0(i,M0_monomials(LM)(u,0),k);
                for (int v=1; v<M0_monomials(LM).extent(1); ++v) {
                    if (M0_monomials(LM)(u,v) == -1)
                        break;
                    monomial *= A0(i,M0_monomials(LM)(u,v),k);
                }
                M0(i,LM,k) += M0_weights(LM)(node_types(i),k,u) * monomial;
            }
        });
    Kokkos::fence();
}
#endif

//#if 0
template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_M0(int num_nodes, Kokkos::View<const int*> node_types)
{
    if (M0.extent(0) < num_nodes)
        Kokkos::realloc(M0, num_nodes, num_LM, num_channels);
    for (int LM=0; LM<num_LM; ++LM) {
        if (M0_poly_values(LM).extent(0) < num_nodes)
            M0_poly_values(LM) = Kokkos::View<Precision***,Kokkos::LayoutRight>(
                Kokkos::view_alloc(std::string("M0_poly_values_")+std::to_string(LM),Kokkos::WithoutInitializing),
                num_nodes, M0_poly_coeff(LM).extent(1), num_channels);
    }

    const auto A0 = this->A0;
    const auto M0_poly_spec = this->M0_poly_spec;
    const auto M0_poly_coeff = this->M0_poly_coeff;
    const auto M0_poly_values = this->M0_poly_values;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    const auto num_LM = this->num_LM;
    auto M0 = this->M0;

    Kokkos::parallel_for("Compute M0",
        Kokkos::TeamPolicy<>(num_nodes*num_LM, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / num_LM;
            const int LM = team_member.league_rank() % num_LM;
            const int type_i = node_types(i);
            const int num_edges = M0_poly_spec(LM).extent_int(0);
            const auto poly_vals = M0_poly_values(LM);
            const auto poly_coeff = M0_poly_coeff(LM);
            const auto poly_spec = M0_poly_spec(LM);
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, num_channels),
                [&] (const int k) {
                    Precision acc = static_cast<Precision>(0.0);
                    for (int lm = 0; lm < num_lm; ++lm) {
                        const Precision v = A0(i, lm, k);
                        poly_vals(i, lm, k) = v;
                        acc += poly_coeff(type_i, lm, k) * v;
                    }
                    for (int p = 0; p < num_edges; ++p) {
                        const int p0 = poly_spec(p, 0);
                        const int p1 = poly_spec(p, 1);
                        const Precision v = poly_vals(i, p0, k) * poly_vals(i, p1, k);
                        poly_vals(i, num_lm + p, k) = v;
                        acc += poly_coeff(type_i, num_lm + p, k) * v;
                    }
                    M0(i, LM, k) = acc;
                });
        });
}
//#endif

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_M0_from_rrnlb_layer0_out(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> layer0_out)
{
    if (rrnlb_layers_kokkos.empty()) {
        throw std::runtime_error("RRNLB layer0 output requested but no RRNLB layers are available.");
    }
    const auto& linear_out = rrnlb_layers_kokkos[0].linear_2;
    const auto parts_out_offset = linear_out.parts_out_offset;
    const auto parts_out_mul = linear_out.parts_out_mul;
    const auto parts_out_l = linear_out.parts_out_l;
    const int num_parts = parts_out_offset.extent_int(0);

    if (M0.extent(0) < num_nodes)
        Kokkos::realloc(M0, num_nodes, num_LM, num_channels);
    for (int LM = 0; LM < num_LM; ++LM) {
        if (M0_poly_values(LM).extent(0) < num_nodes)
            M0_poly_values(LM) = Kokkos::View<Precision***,Kokkos::LayoutRight>(
                Kokkos::view_alloc(
                    std::string("M0_poly_values_") + std::to_string(LM),
                    Kokkos::WithoutInitializing),
                num_nodes, M0_poly_coeff(LM).extent(1), num_channels);
    }

    const auto M0_poly_spec = this->M0_poly_spec;
    const auto M0_poly_coeff = this->M0_poly_coeff;
    const auto M0_poly_values = this->M0_poly_values;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    const auto num_LM = this->num_LM;
    auto M0 = this->M0;

    Kokkos::parallel_for("Compute M0 from RRNLB layer0 out",
        Kokkos::TeamPolicy<>(num_nodes * num_LM, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / num_LM;
            const int LM = team_member.league_rank() % num_LM;
            const int type_i = node_types(i);
            const int num_edges = M0_poly_spec(LM).extent_int(0);
            const auto poly_vals = M0_poly_values(LM);
            const auto poly_coeff = M0_poly_coeff(LM);
            const auto poly_spec = M0_poly_spec(LM);
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, num_channels),
                [&] (const int k) {
                    Precision acc = static_cast<Precision>(0.0);
                    for (int p = 0; p < num_parts; ++p) {
                        const int mul = parts_out_mul(p);
                        if (k >= mul) continue;
                        const int l = parts_out_l(p);
                        const int ir_dim = 2 * l + 1;
                        const int offset = parts_out_offset(p);
                        const int lm0 = l * l;
                        for (int m = 0; m < ir_dim; ++m) {
                            const int lm = lm0 + m;
                            const Precision v = layer0_out(i, offset + k * ir_dim + m);
                            poly_vals(i, lm, k) = v;
                            acc += poly_coeff(type_i, lm, k) * v;
                        }
                    }
                    for (int p = 0; p < num_edges; ++p) {
                        const int p0 = poly_spec(p, 0);
                        const int p1 = poly_spec(p, 1);
                        const Precision v = poly_vals(i, p0, k) * poly_vals(i, p1, k);
                        poly_vals(i, num_lm + p, k) = v;
                        acc += poly_coeff(type_i, num_lm + p, k) * v;
                    }
                    M0(i, LM, k) = acc;
                });
        });
}

#if 0
template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_M0(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    Kokkos::realloc(A0_adj, A0.extent(0), A0.extent(1), A0.extent(2));
    Kokkos::deep_copy(A0_adj, 0.0);

    // local references to class members accessed in the parallel region
    auto A0 = this->A0;
    auto A0_adj = this->A0_adj;
    auto M0_adj = this->M0_adj;
    auto M0_monomials = this->M0_monomials;
    auto M0_weights = this->M0_weights;
    auto num_channels = this->num_channels;
    auto num_LM = this->num_LM;

    Kokkos::parallel_for(
        "Reverse M0",
        Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Right>>(
            {0,0,0}, {num_nodes,num_LM,num_channels}),
        KOKKOS_LAMBDA (const int i, const int LM, const int k) {
            for (int u=0; u<M0_monomials(LM).extent(0); ++u) {
                for (int v=0; v<M0_monomials(LM).extent(1); ++v) {
                    if (M0_monomials(LM)(u,v) == -1) break;
                    double deriv = M0_adj(i,LM,k);
                    for (int w=0; w<M0_monomials(LM).extent(1); ++w) {
                        if (M0_monomials(LM)(u,w) == -1) break;
                        if (v == w) continue;
                        deriv *= A0(i,M0_monomials(LM)(u,w),k);
                    }
                    Kokkos::atomic_add(
                        &A0_adj(i,M0_monomials(LM)(u,v),k),
                        M0_weights(LM)(node_types(i),k,u)*deriv);
                }
            }
        });
    Kokkos::fence();
}
#endif

//#if 0
template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_M0(int num_nodes, Kokkos::View<const int*> node_types)
{
    if (A0_adj.extent(0) < num_nodes)
        Kokkos::realloc(A0_adj, A0.extent(0), A0.extent(1), A0.extent(2));
    Kokkos::deep_copy(A0_adj, 0.0);
    for (int LM=0; LM<num_LM; ++LM) {
        if (M0_poly_adjoints(LM).extent(0) < num_nodes)
            M0_poly_adjoints(LM) = Kokkos::View<Precision***,Kokkos::LayoutRight>(
                Kokkos::view_alloc(std::string("M0_poly_adjoints_")+std::to_string(LM),Kokkos::WithoutInitializing),
                num_nodes, M0_poly_coeff(LM).extent(1), num_channels);
    }

    // TODO: prune
    const auto M0_adj = this->M0_adj;
    const auto M0_poly_spec = this->M0_poly_spec;
    const auto M0_poly_coeff = this->M0_poly_coeff;
    const auto M0_poly_adjoints = this->M0_poly_adjoints;
    const auto M0_poly_values = this->M0_poly_values;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    const auto num_LM = this->num_LM;
    auto A0_adj = this->A0_adj;

    Kokkos::parallel_for("Reverse M0",
        Kokkos::TeamPolicy<>(num_nodes*num_LM, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / num_LM;
            const int LM = team_member.league_rank() % num_LM;
            // initialize
            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2,Kokkos::Iterate::Right>,Kokkos::TeamPolicy<>::member_type>(
                        team_member, M0_poly_coeff(LM).extent(1), num_channels),
                [&] (const int p, const int k) {
                    M0_poly_adjoints(LM)(i,p,k) = M0_poly_coeff(LM)(node_types(i),p,k);
                });
            team_member.team_barrier();
            // backwards pass
            for (int p=M0_poly_spec(LM).extent(0)-1; p>=0; --p) {
                const int p0 = M0_poly_spec(LM)(p,0);
                const int p1 = M0_poly_spec(LM)(p,1);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        // TODO: use scratch space
                        M0_poly_adjoints(LM)(i,p0,k) += M0_poly_adjoints(LM)(i,num_lm+p,k)*M0_poly_values(LM)(i,p1,k);
                        M0_poly_adjoints(LM)(i,p1,k) += M0_poly_adjoints(LM)(i,num_lm+p,k)*M0_poly_values(LM)(i,p0,k);
                    });
            }
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2,Kokkos::Iterate::Right>,Kokkos::TeamPolicy<>::member_type>(
                        team_member, num_lm, num_channels),
                [&] (const int lm, const int k) {
                    Kokkos::atomic_add(&A0_adj(i,lm,k), M0_poly_adjoints(LM)(i,lm,k) * M0_adj(i,LM,k));
                });
        });
}
//#endif

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_M0_mixed_rrnlb(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    const Kokkos::View<const AccumPrecision***,Kokkos::LayoutRight>& M0_adj_in,
    Kokkos::View<AccumPrecision***,Kokkos::LayoutRight>& A0_adj_out)
{
    if (A0_adj_out.extent_int(0) < num_nodes
        || A0_adj_out.extent_int(1) != num_lm
        || A0_adj_out.extent_int(2) != num_channels) {
        Kokkos::realloc(A0_adj_out, num_nodes, num_lm, num_channels);
    }

    const auto M0_poly_spec = this->M0_poly_spec;
    const auto M0_poly_coeff = this->M0_poly_coeff;
    const auto M0_poly_values = this->M0_poly_values;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    const auto num_LM = this->num_LM;

    const auto m0_impl = rrnlb_mpi_ap_native_m0_impl();
    bool use_atomic_legacy = m0_impl == RRNLBMpiApNativeM0Impl::AtomicLegacy;
    bool use_node_reduce_v2 = m0_impl == RRNLBMpiApNativeM0Impl::NodeReduceV2;
    int max_need_p = 0;
    std::size_t node_reduce_v2_scratch_bytes = 0;
    if (use_node_reduce_v2) {
        for (int LM = 0; LM < num_LM; ++LM) {
            max_need_p = std::max(max_need_p, M0_poly_coeff(LM).extent_int(1));
        }
        node_reduce_v2_scratch_bytes =
            static_cast<std::size_t>(max_need_p)
            * static_cast<std::size_t>(num_channels)
            * sizeof(AccumPrecision);
        constexpr std::size_t kMaxNodeReduceV2ScratchBytes = 48 * 1024;
        if (node_reduce_v2_scratch_bytes > kMaxNodeReduceV2ScratchBytes) {
            use_node_reduce_v2 = false;
            use_atomic_legacy = true;
        }
    }

    if (use_atomic_legacy) {
        auto A0_adj_active = Kokkos::subview(
            A0_adj_out,
            Kokkos::make_pair(0, num_nodes),
            Kokkos::ALL(),
            Kokkos::ALL());
        Kokkos::deep_copy(A0_adj_active, static_cast<AccumPrecision>(0.0));

        if (rrnlb_M0_poly_adjoints_ap.extent_int(0) != num_LM) {
            rrnlb_M0_poly_adjoints_ap =
                Kokkos::View<Kokkos::View<AccumPrecision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
                    Kokkos::view_alloc("rrnlb_M0_poly_adjoints_ap", Kokkos::SequentialHostInit),
                    num_LM);
        }
        for (int LM = 0; LM < num_LM; ++LM) {
            const int need_p = M0_poly_coeff(LM).extent_int(1);
            if (rrnlb_M0_poly_adjoints_ap(LM).extent_int(0) < num_nodes
                || rrnlb_M0_poly_adjoints_ap(LM).extent_int(1) != need_p
                || rrnlb_M0_poly_adjoints_ap(LM).extent_int(2) != num_channels) {
                rrnlb_M0_poly_adjoints_ap(LM) = Kokkos::View<AccumPrecision***,Kokkos::LayoutRight>(
                    Kokkos::view_alloc(
                        std::string("rrnlb_M0_poly_adjoints_ap_") + std::to_string(LM),
                        Kokkos::WithoutInitializing),
                    num_nodes,
                    need_p,
                    num_channels);
            }
        }

        const auto M0_poly_adjoints_ap = this->rrnlb_M0_poly_adjoints_ap;
        Kokkos::parallel_for(
            "Reverse M0 mixed rrnlb (atomic legacy)",
            Kokkos::TeamPolicy<>(num_nodes * num_LM, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank() / num_LM;
                const int LM = team_member.league_rank() % num_LM;
                const int type_i = node_types(i);

                Kokkos::parallel_for(
                    Kokkos::TeamVectorMDRange<
                        Kokkos::Rank<2, Kokkos::Iterate::Right>,
                        Kokkos::TeamPolicy<>::member_type>(
                        team_member, M0_poly_coeff(LM).extent_int(1), num_channels),
                    [&] (const int p, const int k) {
                        M0_poly_adjoints_ap(LM)(i, p, k) =
                            static_cast<AccumPrecision>(M0_poly_coeff(LM)(type_i, p, k));
                    });
                team_member.team_barrier();

                for (int p = M0_poly_spec(LM).extent_int(0) - 1; p >= 0; --p) {
                    const int p0 = M0_poly_spec(LM)(p, 0);
                    const int p1 = M0_poly_spec(LM)(p, 1);
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team_member, num_channels),
                        [&] (const int k) {
                            const AccumPrecision upstream =
                                M0_poly_adjoints_ap(LM)(i, num_lm + p, k);
                            M0_poly_adjoints_ap(LM)(i, p0, k) +=
                                upstream
                                * static_cast<AccumPrecision>(M0_poly_values(LM)(i, p1, k));
                            M0_poly_adjoints_ap(LM)(i, p1, k) +=
                                upstream
                                * static_cast<AccumPrecision>(M0_poly_values(LM)(i, p0, k));
                        });
                }
                team_member.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamVectorMDRange<
                        Kokkos::Rank<2, Kokkos::Iterate::Right>,
                        Kokkos::TeamPolicy<>::member_type>(
                        team_member, num_lm, num_channels),
                    [&] (const int lm, const int k) {
                        Kokkos::atomic_add(
                            &A0_adj_out(i, lm, k),
                            M0_poly_adjoints_ap(LM)(i, lm, k) * M0_adj_in(i, LM, k));
                    });
            });
        return;
    }

    if (use_node_reduce_v2) {
        const int max_need_p_capture = max_need_p;
        const std::size_t scratch_bytes = node_reduce_v2_scratch_bytes;
        Kokkos::parallel_for(
            "Reverse M0 mixed rrnlb (node reduce v2)",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32)
                .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank();
                const int type_i = node_types(i);
                Kokkos::View<
                    AccumPrecision**,
                    Kokkos::LayoutRight,
                    typename Kokkos::TeamPolicy<>::member_type::scratch_memory_space>
                    m0_poly_adj_scratch(
                        team_member.team_scratch(0),
                        max_need_p_capture,
                        num_channels);

                Kokkos::parallel_for(
                    Kokkos::TeamVectorMDRange<
                        Kokkos::Rank<2, Kokkos::Iterate::Right>,
                        Kokkos::TeamPolicy<>::member_type>(
                        team_member, num_lm, num_channels),
                    [&] (const int lm, const int k) {
                        A0_adj_out(i, lm, k) = static_cast<AccumPrecision>(0.0);
                    });
                team_member.team_barrier();

                for (int LM = 0; LM < num_LM; ++LM) {
                    const int need_p = M0_poly_coeff(LM).extent_int(1);
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorMDRange<
                            Kokkos::Rank<2, Kokkos::Iterate::Right>,
                            Kokkos::TeamPolicy<>::member_type>(
                            team_member, need_p, num_channels),
                        [&] (const int p, const int k) {
                            m0_poly_adj_scratch(p, k) =
                                static_cast<AccumPrecision>(M0_poly_coeff(LM)(type_i, p, k));
                        });
                    team_member.team_barrier();

                    for (int p = M0_poly_spec(LM).extent_int(0) - 1; p >= 0; --p) {
                        const int p0 = M0_poly_spec(LM)(p, 0);
                        const int p1 = M0_poly_spec(LM)(p, 1);
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(team_member, num_channels),
                            [&] (const int k) {
                                const AccumPrecision upstream =
                                    m0_poly_adj_scratch(num_lm + p, k);
                                m0_poly_adj_scratch(p0, k) +=
                                    upstream
                                    * static_cast<AccumPrecision>(M0_poly_values(LM)(i, p1, k));
                                m0_poly_adj_scratch(p1, k) +=
                                    upstream
                                    * static_cast<AccumPrecision>(M0_poly_values(LM)(i, p0, k));
                            });
                    }
                    team_member.team_barrier();

                    Kokkos::parallel_for(
                        Kokkos::TeamVectorMDRange<
                            Kokkos::Rank<2, Kokkos::Iterate::Right>,
                            Kokkos::TeamPolicy<>::member_type>(
                            team_member, num_lm, num_channels),
                        [&] (const int lm, const int k) {
                            A0_adj_out(i, lm, k) +=
                                m0_poly_adj_scratch(lm, k) * M0_adj_in(i, LM, k);
                        });
                    team_member.team_barrier();
                }
            });
        return;
    }

    if (rrnlb_M0_poly_adjoints_ap.extent_int(0) != num_LM) {
        rrnlb_M0_poly_adjoints_ap =
            Kokkos::View<Kokkos::View<AccumPrecision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
                Kokkos::view_alloc("rrnlb_M0_poly_adjoints_ap", Kokkos::SequentialHostInit),
                num_LM);
    }
    for (int LM = 0; LM < num_LM; ++LM) {
        const int need_p = M0_poly_coeff(LM).extent_int(1);
        if (rrnlb_M0_poly_adjoints_ap(LM).extent_int(0) < num_nodes
            || rrnlb_M0_poly_adjoints_ap(LM).extent_int(1) != need_p
            || rrnlb_M0_poly_adjoints_ap(LM).extent_int(2) != num_channels) {
            rrnlb_M0_poly_adjoints_ap(LM) = Kokkos::View<AccumPrecision***,Kokkos::LayoutRight>(
                Kokkos::view_alloc(
                    std::string("rrnlb_M0_poly_adjoints_ap_") + std::to_string(LM),
                    Kokkos::WithoutInitializing),
                num_nodes,
                need_p,
                num_channels);
        }
    }

    const auto M0_poly_adjoints_ap = this->rrnlb_M0_poly_adjoints_ap;
    Kokkos::parallel_for(
        "Reverse M0 mixed rrnlb prep (node reduce)",
        Kokkos::TeamPolicy<>(num_nodes * num_LM, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / num_LM;
            const int LM = team_member.league_rank() % num_LM;
            const int type_i = node_types(i);

            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2, Kokkos::Iterate::Right>,
                    Kokkos::TeamPolicy<>::member_type>(
                    team_member, M0_poly_coeff(LM).extent_int(1), num_channels),
                [&] (const int p, const int k) {
                    M0_poly_adjoints_ap(LM)(i, p, k) =
                        static_cast<AccumPrecision>(M0_poly_coeff(LM)(type_i, p, k));
                });
            team_member.team_barrier();

            for (int p = M0_poly_spec(LM).extent_int(0) - 1; p >= 0; --p) {
                const int p0 = M0_poly_spec(LM)(p, 0);
                const int p1 = M0_poly_spec(LM)(p, 1);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        const AccumPrecision upstream =
                            M0_poly_adjoints_ap(LM)(i, num_lm + p, k);
                        M0_poly_adjoints_ap(LM)(i, p0, k) +=
                            upstream
                            * static_cast<AccumPrecision>(M0_poly_values(LM)(i, p1, k));
                        M0_poly_adjoints_ap(LM)(i, p1, k) +=
                            upstream
                            * static_cast<AccumPrecision>(M0_poly_values(LM)(i, p0, k));
                    });
            }
        });

    Kokkos::parallel_for(
        "Reverse M0 mixed rrnlb (node reduce)",
        Kokkos::TeamPolicy<>(num_nodes * num_lm, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / num_lm;
            const int lm = team_member.league_rank() % num_lm;
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, num_channels),
                [&] (const int k) {
                    AccumPrecision accum = static_cast<AccumPrecision>(0.0);
                    for (int LM = 0; LM < num_LM; ++LM) {
                        accum += M0_poly_adjoints_ap(LM)(i, lm, k) * M0_adj_in(i, LM, k);
                    }
                    A0_adj_out(i, lm, k) = accum;
                });
        });
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_H1(
    const int num_nodes)
{
    if (H1.extent(0) < M0.extent(0))
        Kokkos::realloc(H1, M0.extent(0), M0.extent(1), M0.extent(2));

    auto L_max = this->L_max;
    auto H1 = this->H1;
    auto H1_weights = this->H1_weights;
    auto M0 = this->M0;

    Kokkos::parallel_for("Compute H1",
        Kokkos::TeamPolicy<>(num_nodes*(L_max+1), Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (L_max+1);
            const int l = team_member.league_rank() % (L_max+1);
            auto M0_il = Kokkos::subview(M0, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(H1_weights, l, Kokkos::ALL, Kokkos::ALL);
            auto H1_il = Kokkos::subview(H1, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Unblocked>
                ::invoke(team_member, 1.0, M0_il, W_il, 0.0, H1_il);
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_H1(
    const int num_nodes)
{
    if (M0_adj.extent(0) < M0.extent(0))
        Kokkos::realloc(M0_adj, M0.extent(0), M0.extent(1), M0.extent(2));

    auto L_max = this->L_max;
    auto M0_adj = this->M0_adj;
    auto H1_weights = this->H1_weights;
    auto H1_adj = this->H1_adj;

    Kokkos::parallel_for("Reverse H1",
        Kokkos::TeamPolicy<>(num_nodes*(L_max+1), Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (L_max+1);
            const int l = team_member.league_rank() % (L_max+1);
            auto H1_adj_il = Kokkos::subview(H1_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(H1_weights, l, Kokkos::ALL, Kokkos::ALL);
            auto M0_adj_il = Kokkos::subview(M0_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::Transpose,
                                    KokkosBatched::Algo::Gemm::Unblocked>
                ::invoke(team_member, 1.0, H1_adj_il, W_il, 0.0, M0_adj_il);
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_Phi1(
    const int num_nodes,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices)
{
    // Compute Phi1_lelm1lm2 (named Phi1r)
    if (Phi1r.extent(0) < num_nodes)
        Kokkos::realloc(Phi1r, num_nodes, num_lelm1lm2, num_channels);
    if (Phi1.extent(0) < num_nodes)
        Kokkos::realloc(Phi1, num_nodes, num_lme, num_channels);
    Kokkos::deep_copy(Phi1r, 0.0);
    Kokkos::deep_copy(Phi1, 0.0);

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    const auto num_lelm1lm2 = this->num_lelm1lm2;
    const auto Phi1_lm1 = this->Phi1_lm1;
    const auto Phi1_lm2 = this->Phi1_lm2;
    const auto Phi1_lel1l2 = this->Phi1_lel1l2;
    const auto Phi1_lme = this->Phi1_lme;
    const auto Phi1_lelm1lm2 = this->Phi1_lelm1lm2;
    const auto Phi1_clebsch_gordan = this->Phi1_clebsch_gordan;
    const auto R1 = this->R1;
    const auto Y = this->Y;
    const auto H1 = this->H1;
    auto Phi1 = this->Phi1;
    auto Phi1r = this->Phi1r;

#if 0
    Kokkos::parallel_for("Compute Phi1r",
        Kokkos::TeamPolicy<>(num_nodes*num_lelm1lm2, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / num_lelm1lm2;
            const int lelm1lm2 = team_member.league_rank() % num_lelm1lm2;
            const int i0 = first_neigh(i);
            const int lm1 = Phi1_lm1(lelm1lm2);
            const int lm2 = Phi1_lm2(lelm1lm2);
            const int lel1l2 = Phi1_lel1l2(lelm1lm2);
            for (int j=0; j<num_neigh(i); ++j) {
                const int ij = i0 + j;
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        Phi1r(i,lelm1lm2,k) += R1(ij,lel1l2*num_channels+k) * Y(ij*num_lm+lm1) * H1(neigh_indices(ij),lm2,k);
                    });
            }
        });
    Kokkos::fence();
#endif

//#if 0
    Kokkos::parallel_for("Compute Phi1r",
        Kokkos::TeamPolicy<>(num_nodes*num_lelm1lm2, Kokkos::AUTO, 32)
             .set_scratch_size(0, Kokkos::PerTeam(num_channels*sizeof(double))),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / num_lelm1lm2;
            const int lelm1lm2 = team_member.league_rank() % num_lelm1lm2;
            const int i0 = first_neigh(i);
            const int lm1 = Phi1_lm1(lelm1lm2);
            const int lm2 = Phi1_lm2(lelm1lm2);
            const int lel1l2 = Phi1_lel1l2(lelm1lm2);
            // initialize Phi1r_i_lelm1lm2 in scratch space
            auto Phi1r_i_lelm1lm2 = Kokkos::View<double*>(team_member.team_scratch(0), num_channels);
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, num_channels),
                [=] (const int k) {
                    Phi1r_i_lelm1lm2(k) = 0.0;
                });
            team_member.team_barrier();
            // compute Phi1r_i_lelm1lm2
            for (int j=0; j<num_neigh(i); ++j) {
                const int ij = i0 + j;
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        Phi1r_i_lelm1lm2(k) += R1(ij,lel1l2*num_channels+k) * Y(ij*num_lm+lm1) * H1(neigh_indices(ij),lm2,k);
                    });
            }
            team_member.team_barrier();
            // store Phi1r_i_lelm1lm2
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, num_channels),
                [=] (const int k) {
                    Phi1r(i,lelm1lm2,k) = Phi1r_i_lelm1lm2(k);
                });
        });
    Kokkos::fence();
//#endif

    // Compute Phi1 using CG coefficients
    Kokkos::parallel_for("Compute Phi1",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const double C = Phi1_clebsch_gordan(p);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        Phi1(i,Phi1_lme(p),k) += C * Phi1r(i,Phi1_lelm1lm2(p),k);
                    });
            }
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_Phi1(
    const int num_nodes,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    bool zero_dxyz,
    bool zero_H1_adj)
{
    if (dPhi1r.extent(0) < Phi1r.extent(0))
        Kokkos::realloc(dPhi1r, Phi1r.extent(0), Phi1r.extent(1), Phi1r.extent(2)); 
    if (node_forces.size() < xyz.size())
        Kokkos::resize(node_forces, xyz.size());
    if (H1_adj.extent(0) < H1.extent(0))
        Kokkos::resize(H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
    if (zero_dxyz)
        Kokkos::deep_copy(node_forces, 0.0);
    Kokkos::deep_copy(dPhi1r, 0.0);
    if (zero_H1_adj)
        Kokkos::deep_copy(H1_adj, 0.0);

    const auto num_lm = this->num_lm;
    const auto num_channels = this->num_channels;
    const auto num_lelm1lm2 = this->num_lelm1lm2;
    const auto Phi1_lm1 = this->Phi1_lm1;
    const auto Phi1_lm2 = this->Phi1_lm2;
    const auto Phi1_lel1l2 = this->Phi1_lel1l2;
    const auto Phi1_lme = this->Phi1_lme;
    const auto Phi1_lelm1lm2 = this->Phi1_lelm1lm2;
    const auto Phi1_clebsch_gordan = this->Phi1_clebsch_gordan;
    const auto R1 = this->R1;
    const auto R1_deriv = this->R1_deriv;
    const auto Y = this->Y;
    const auto Y_grad = this->Y_grad;
    const auto H1 = this->H1;
    const auto H1_adj = this->H1_adj;
    const auto node_forces = this->node_forces;
    auto dPhi1r = this->dPhi1r;
    auto dPhi1 = this->dPhi1;

    // Compute dE/dPhi1 (named dPhi1)
    Kokkos::parallel_for("Reverse Phi1",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const double C = Phi1_clebsch_gordan(p);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        dPhi1r(i,Phi1_lelm1lm2(p),k) += C * dPhi1(i,Phi1_lme(p),k);
                    });
            }
        });

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            const int num_neigh_i = num_neigh(i); 
            if (final)
                first_neigh(i) = update;
            update += num_neigh_i;
        });
    Kokkos::fence();

    Kokkos::parallel_for("Reverse Phi1r",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            for (int j=0; j<num_neigh(i); ++j) {
                const int ij = i0 + j;
                double f_x, f_y, f_z;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team_member, num_lelm1lm2),
                    [=] (const int lelm1lm2, double& f_x, double& f_y, double& f_z) {
                        const int lm1 = Phi1_lm1(lelm1lm2);
                        const int lm2 = Phi1_lm2(lelm1lm2);
                        const int lel1l2 = Phi1_lel1l2(lelm1lm2);
                        double t1, t2;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team_member, num_channels),
                            [=] (const int k, double& t1, double& t2) {
                                t1 += R1_deriv(ij,lel1l2*num_channels+k) * H1(neigh_indices(ij),lm2,k) * dPhi1r(i,lelm1lm2,k); 
                                t2 += R1(ij,lel1l2*num_channels+k) * H1(neigh_indices(ij),lm2,k) * dPhi1r(i,lelm1lm2,k);
                                Kokkos::atomic_add(
                                    &H1_adj(neigh_indices(ij),lm2,k),
                                    R1(ij,lel1l2*num_channels+k) * Y(ij*num_lm+lm1) * dPhi1r(i,lelm1lm2,k));
                            }, t1, t2);
                        f_x += t1*xyz(3*ij)/r(ij)*Y(ij*num_lm+lm1) + t2*Y_grad(3*ij*num_lm+lm1);
                        f_y += t1*xyz(3*ij+1)/r(ij)*Y(ij*num_lm+lm1) + t2*Y_grad((3*ij+1)*num_lm+lm1);
                        f_z += t1*xyz(3*ij+2)/r(ij)*Y(ij*num_lm+lm1) + t2*Y_grad((3*ij+2)*num_lm+lm1);
                    }, f_x, f_y, f_z);
                team_member.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team_member), [=]() {
                    node_forces(3*ij)   -= f_x;
                    node_forces(3*ij+1) -= f_y;
                    node_forces(3*ij+2) -= f_z;
                });
            }
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_A1(int num_nodes)
{
    // The core matrix multiplication is:
    //         [A1_il]_mk = \sum_(ek') [Phi1_il]_m(ek') [W_il]_(ek')k
    if (A1.extent(0) < num_nodes)
        Kokkos::realloc(A1, num_nodes, num_lm, num_channels);

    const auto l_max = this->l_max;
    const auto num_channels = this->num_channels;
    const auto Phi1_l = this->Phi1_l;
    const auto Phi1 = this->Phi1;
    auto A1_weights = this->A1_weights;
    auto A1 = this->A1;

    Kokkos::parallel_for("Compute A1",
        Kokkos::TeamPolicy<>(num_nodes*(l_max+1), Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (l_max+1);
            const int l = team_member.league_rank() % (l_max+1);
            int lme = 0;
            int num_eta = 0;
            for (int p=0; p<Phi1_l.size(); ++p) {
                const int ll = Phi1_l(p);
                if (ll < l)
                    lme += 2*ll+1;
                if (ll == l)
                    num_eta += 1;
            }
            auto Phi1_il = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                &Phi1(i,lme,0), 2*l+1, num_eta*num_channels);
            auto A1_il = Kokkos::subview(A1, i, Kokkos::make_pair(l*l,l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Blocked>
                ::invoke(team_member, 1.0, Phi1_il, A1_weights(l), 0.0, A1_il);
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_A1(int num_nodes)
{
    // The core matrix multiplication is:
    //         [dE/dPhi1_il]_m(ek) = \sum_k' [dE/dA1_il]_mk' [trans(W_il)]_k'(ek)
    if (dPhi1.extent(0) < num_nodes)
        Kokkos::realloc(dPhi1, num_nodes, num_lme, num_channels);

    const auto l_max = this->l_max;
    const auto num_channels = this->num_channels;
    const auto Phi1_l = this->Phi1_l;
    const auto A1_adj = this->A1_adj;
    const auto A1_weights_trans = this->A1_weights_trans;
    auto dPhi1 = this->dPhi1;

    Kokkos::parallel_for("Reverse A1",
        Kokkos::TeamPolicy<>(num_nodes*(l_max+1), Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (l_max+1);
            const int l = team_member.league_rank() % (l_max+1);
            int lme = 0;
            int num_eta = 0;
            for (int p=0; p<Phi1_l.size(); ++p) {
                const int ll = Phi1_l(p);
                if (ll < l)
                    lme += 2*ll+1;
                if (ll == l)
                    num_eta += 1;
            }
            auto dA1_il = Kokkos::subview(A1_adj, i, Kokkos::make_pair(l*l,l*l+2*l+1), Kokkos::ALL);
            auto dPhi1_il = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                &dPhi1(i,lme,0), 2*l+1, num_eta*num_channels);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Blocked>
                ::invoke(team_member, 1.0, dA1_il, A1_weights_trans(l), 0.0, dPhi1_il);
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (not A1_scaled) return;

    // compute A1 splines
    if (A1_spline_values.extent(0) < r.size()) {
        Kokkos::realloc(A1_spline_values, r.size(), 1);
        Kokkos::realloc(A1_spline_derivs, r.size(), 1);
    }
    A1_splines.evaluate(num_nodes, node_types, num_neigh, neigh_types, r, A1_spline_values, A1_spline_derivs);

    // compute first_neigh
    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    // perform the scaling
    auto A1 = this->A1;
    auto A1_spline_values = this->A1_spline_values;
    auto A1_spline_derivs = this->A1_spline_derivs;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    Kokkos::parallel_for(
        "MACEKokkos::compute_A1_scaled",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            // compute scale factor
            double A1_scale_factor;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j, double& lsum) {
                    lsum += A1_spline_values(i0+j,0);
                }, A1_scale_factor);
            A1_scale_factor += 1.0;
            team_member.team_barrier();
            // perform the scaling
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A1(i,lmk/num_channels,lmk%num_channels) /= A1_scale_factor;
                });
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    if (not A1_scaled) return;

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    // update the derivatives
    // Warning: Assumes node_forces have been initialized elsewhere
    const auto A1 = this->A1;
    const auto A1_adj = this->A1_adj;
    const auto A1_spline_values = this->A1_spline_values;
    const auto A1_spline_derivs = this->A1_spline_derivs;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto node_forces = this->node_forces;
    Kokkos::parallel_for(
        "MACEKokkos::reverse_A1_scaled",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            // scale factor
            double A1_scale_factor;// = 1.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j, double& lsum) {
                    lsum += A1_spline_values(i0+j,0);
                }, A1_scale_factor);
            A1_scale_factor += 1.0;
            team_member.team_barrier();
            // update dE/dxyz
            double dA1_dot_A1;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (const int lmk, double& lsum) {
                    const int lm = lmk / num_channels;
                    const int k = lmk % num_channels;
                    lsum += A1_adj(i,lm,k) * A1(i,lm,k);
                }, dA1_dot_A1);
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j) {
                    const int ij = first_neigh(i) + j;
                    const double f = A1_spline_values(ij,0);
                    const double d = A1_spline_derivs(ij,0);
                    node_forces(3*ij+0) += dA1_dot_A1/A1_scale_factor*d*xyz(3*ij+0)/r(ij);
                    node_forces(3*ij+1) += dA1_dot_A1/A1_scale_factor*d*xyz(3*ij+1)/r(ij);
                    node_forces(3*ij+2) += dA1_dot_A1/A1_scale_factor*d*xyz(3*ij+2)/r(ij);
                });
            // update dE/dA1
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A1_adj(i,lmk/num_channels,lmk%num_channels) /= A1_scale_factor;
                });
        });
    Kokkos::fence();
}

#if 0
template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_M1(int num_nodes, Kokkos::View<const int*> node_types)
{
    Kokkos::realloc(M1, num_nodes, num_channels);

    auto A1 = this->A1;
    auto M1 = this->M1;
    auto M1_monomials = this->M1_monomials;
    auto M1_weights = this->M1_weights;
    auto num_channels = this->num_channels;

    Kokkos::parallel_for(
        "Compute M1",
        Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Right>>(
            {0,0}, {num_nodes,num_channels}),
        KOKKOS_LAMBDA (const int i, const int k) {
            M1(i,k) = 0.0;
            for (int u=0; u<M1_monomials.extent(0); ++u) {
                double monomial = A1(i,M1_monomials(u,0),k);
                for (int v=1; v<M1_monomials.extent(1); ++v) {
                    if (M1_monomials(u,v) == -1)
                        break;
                    monomial *= A1(i,M1_monomials(u,v),k);
                }
                M1(i,k) += M1_weights(node_types(i),k,u) * monomial;
            }
        });
    Kokkos::fence();
}
#endif

//#if 0
template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_M1(int num_nodes, Kokkos::View<const int*> node_types)
{
    if (M1.extent(0) < num_nodes)
        Kokkos::realloc(M1, num_nodes, num_channels);
    if (M1_poly_values.extent(0) < num_nodes)
        Kokkos::realloc(M1_poly_values, num_nodes, num_lm+M1_poly_spec.extent(0), num_channels); 

    const auto A1 = this->A1;
    const auto M1_poly_spec = this->M1_poly_spec;
    const auto M1_poly_coeff = this->M1_poly_coeff;
    const auto M1_poly_values = this->M1_poly_values;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto M1 = this->M1;

    Kokkos::parallel_for("Compute M1",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int type_i = node_types(i);
            const int num_edges = M1_poly_spec.extent_int(0);
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, num_channels),
                [&] (const int k) {
                    Precision acc = static_cast<Precision>(0.0);
                    for (int p = 0; p < num_lm; ++p) {
                        const Precision v = A1(i, p, k);
                        M1_poly_values(i, p, k) = v;
                        acc += M1_poly_coeff(type_i, p, k) * v;
                    }
                    for (int p = 0; p < num_edges; ++p) {
                        const int p0 = M1_poly_spec(p, 0);
                        const int p1 = M1_poly_spec(p, 1);
                        const Precision v = M1_poly_values(i, p0, k) * M1_poly_values(i, p1, k);
                        M1_poly_values(i, num_lm + p, k) = v;
                        acc += M1_poly_coeff(type_i, num_lm + p, k) * v;
                    }
                    M1(i, k) = acc;
                });
        });
}
//#endif

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_M1_from_rrnlb_layer1_out(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> layer1_out)
{
    if (rrnlb_layers_kokkos.size() < 2) {
        throw std::runtime_error("RRNLB layer1 output requested but fewer than two RRNLB layers are available.");
    }
    const auto& linear_out = rrnlb_layers_kokkos[1].linear_2;
    const auto parts_out_offset = linear_out.parts_out_offset;
    const auto parts_out_mul = linear_out.parts_out_mul;
    const auto parts_out_l = linear_out.parts_out_l;
    const int num_parts = parts_out_offset.extent_int(0);

    if (M1.extent(0) < num_nodes)
        Kokkos::realloc(M1, num_nodes, num_channels);
    if (M1_poly_values.extent(0) < num_nodes)
        Kokkos::realloc(M1_poly_values, num_nodes, num_lm + M1_poly_spec.extent(0), num_channels);

    const auto M1_poly_spec = this->M1_poly_spec;
    const auto M1_poly_coeff = this->M1_poly_coeff;
    const auto M1_poly_values = this->M1_poly_values;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto M1 = this->M1;

    Kokkos::parallel_for("Compute M1 from RRNLB layer1 out",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int type_i = node_types(i);
            const int num_edges = M1_poly_spec.extent_int(0);
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team_member, num_channels),
                [&] (const int k) {
                    Precision acc = static_cast<Precision>(0.0);
                    for (int p = 0; p < num_parts; ++p) {
                        const int mul = parts_out_mul(p);
                        if (k >= mul) continue;
                        const int l = parts_out_l(p);
                        const int ir_dim = 2 * l + 1;
                        const int offset = parts_out_offset(p);
                        const int lm0 = l * l;
                        for (int m = 0; m < ir_dim; ++m) {
                            const int lm = lm0 + m;
                            const Precision v = layer1_out(i, offset + k * ir_dim + m);
                            M1_poly_values(i, lm, k) = v;
                            acc += M1_poly_coeff(type_i, lm, k) * v;
                        }
                    }
                    for (int p = 0; p < num_edges; ++p) {
                        const int p0 = M1_poly_spec(p, 0);
                        const int p1 = M1_poly_spec(p, 1);
                        const Precision v = M1_poly_values(i, p0, k) * M1_poly_values(i, p1, k);
                        M1_poly_values(i, num_lm + p, k) = v;
                        acc += M1_poly_coeff(type_i, num_lm + p, k) * v;
                    }
                    M1(i, k) = acc;
                });
        });
}

#if 0
template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_M1(int num_nodes, Kokkos::View<const int*> node_types)
{
    Kokkos::realloc(A1_adj, A1.extent(0), A1.extent(1), A1.extent(2));
    Kokkos::deep_copy(A1_adj, 0.0);

    auto A1 = this->A1;
    auto A1_adj = this->A1_adj;
    auto M1_adj = this->M1_adj;
    auto M1_monomials = this->M1_monomials;
    auto M1_weights = this->M1_weights;
    auto num_channels = this->num_channels;

    Kokkos::parallel_for(
        "Reverse M1",
        Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Right>>(
            {0,0}, {num_nodes,num_channels}),
        KOKKOS_LAMBDA (const int i, const int k) {
            for (int u=0; u<M1_monomials.extent(0); ++u) {
                for (int v=0; v<M1_monomials.extent(1); ++v) {
                    if (M1_monomials(u,v) == -1) break;
                    double deriv = M1_adj(i,k);
                    for (int w=0; w<M1_monomials.extent(1); ++w) {
                        if (M1_monomials(u,w) == -1) break;
                        if (v == w) continue;
                        deriv *= A1(i,M1_monomials(u,w),k);
                    }
                    Kokkos::atomic_add(
                        &A1_adj(i,M1_monomials(u,v),k),
                        M1_weights(node_types(i),k,u)*deriv);
                }
            }
        });
    Kokkos::fence();
}
#endif

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_M1(int num_nodes, Kokkos::View<const int*> node_types)
{
    if (A1_adj.extent(0) < num_nodes)
        Kokkos::realloc(A1_adj, A1.extent(0), A1.extent(1), A1.extent(2));
    Kokkos::deep_copy(A1_adj, 0.0);
    if (M1_poly_adjoints.extent(0) < num_nodes)
        Kokkos::realloc(M1_poly_adjoints, num_nodes, M1_poly_coeff.extent(1), num_channels); 

    // TODO: prune
    const auto A1_adj = this->A1_adj;
    const auto M1_adj = this->M1_adj;
    const auto M1_monomials = this->M1_monomials;
    const auto M1_weights = this->M1_weights;
    const auto M1_poly_spec = this->M1_poly_spec;
    const auto M1_poly_coeff = this->M1_poly_coeff;
    const auto M1_poly_adjoints = this->M1_poly_adjoints;
    const auto M1_poly_values = this->M1_poly_values;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto M1 = this->M1;

    Kokkos::parallel_for("Reverse M1",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            // initialize
            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2,Kokkos::Iterate::Right>,Kokkos::TeamPolicy<>::member_type>(
                        team_member, M1_poly_coeff.extent(1), num_channels),
                [&] (const int p, const int k) {
                    M1_poly_adjoints(i,p,k) = M1_poly_coeff(node_types(i),p,k);
                });
            team_member.team_barrier();
            // backwards pass
            for (int p=M1_poly_spec.extent(0)-1; p>=0; --p) {
                const int p0 = M1_poly_spec(p,0);
                const int p1 = M1_poly_spec(p,1);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        M1_poly_adjoints(i,p0,k) += M1_poly_adjoints(i,num_lm+p,k)*M1_poly_values(i,p1,k);
                        M1_poly_adjoints(i,p1,k) += M1_poly_adjoints(i,num_lm+p,k)*M1_poly_values(i,p0,k);
                    });
            }
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2,Kokkos::Iterate::Right>,Kokkos::TeamPolicy<>::member_type>(
                        team_member, num_lm, num_channels),
                [&] (const int lm, const int k) {
                    A1_adj(i,lm,k) = M1_poly_adjoints(i,lm,k) * M1_adj(i,k);
                });
        });
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_M1_mixed_rrnlb(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    const Kokkos::View<const AccumPrecision**,Kokkos::LayoutRight>& M1_adj_in,
    Kokkos::View<AccumPrecision***,Kokkos::LayoutRight>& A1_adj_out)
{
    if (A1_adj_out.extent_int(0) < num_nodes
        || A1_adj_out.extent_int(1) != num_lm
        || A1_adj_out.extent_int(2) != num_channels) {
        Kokkos::realloc(A1_adj_out, num_nodes, num_lm, num_channels);
    }

    if (rrnlb_M1_poly_adjoints_ap.extent_int(0) < num_nodes
        || rrnlb_M1_poly_adjoints_ap.extent_int(1) != M1_poly_coeff.extent_int(1)
        || rrnlb_M1_poly_adjoints_ap.extent_int(2) != num_channels) {
        Kokkos::realloc(
            rrnlb_M1_poly_adjoints_ap,
            num_nodes,
            M1_poly_coeff.extent_int(1),
            num_channels);
    }

    const auto M1_poly_spec = this->M1_poly_spec;
    const auto M1_poly_coeff = this->M1_poly_coeff;
    const auto M1_poly_values = this->M1_poly_values;
    const auto M1_poly_adjoints_ap = this->rrnlb_M1_poly_adjoints_ap;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;

    Kokkos::parallel_for(
        "Reverse M1 mixed rrnlb",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int type_i = node_types(i);

            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2, Kokkos::Iterate::Right>,
                    Kokkos::TeamPolicy<>::member_type>(
                    team_member, M1_poly_coeff.extent_int(1), num_channels),
                [&] (const int p, const int k) {
                    M1_poly_adjoints_ap(i, p, k) =
                        static_cast<AccumPrecision>(M1_poly_coeff(type_i, p, k));
                });
            team_member.team_barrier();

            for (int p = M1_poly_spec.extent_int(0) - 1; p >= 0; --p) {
                const int p0 = M1_poly_spec(p, 0);
                const int p1 = M1_poly_spec(p, 1);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        const AccumPrecision upstream = M1_poly_adjoints_ap(i, num_lm + p, k);
                        M1_poly_adjoints_ap(i, p0, k) +=
                            upstream * static_cast<AccumPrecision>(M1_poly_values(i, p1, k));
                        M1_poly_adjoints_ap(i, p1, k) +=
                            upstream * static_cast<AccumPrecision>(M1_poly_values(i, p0, k));
                    });
            }
            team_member.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamVectorMDRange<
                    Kokkos::Rank<2, Kokkos::Iterate::Right>,
                    Kokkos::TeamPolicy<>::member_type>(
                    team_member, num_lm, num_channels),
                [&] (const int lm, const int k) {
                    A1_adj_out(i, lm, k) = M1_poly_adjoints_ap(i, lm, k) * M1_adj_in(i, k);
                });
        });
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::compute_H2(int num_nodes, Kokkos::View<const int*> node_types)
{
    if (H2.extent(0) < num_nodes or H2.extent(1) != num_channels)
        Kokkos::realloc(H2, num_nodes, num_channels);
    Kokkos::deep_copy(H2, 0.0);

    auto num_channels = this->num_channels;
    auto H2 = this->H2;
    auto H2_weights_for_H1 = this->H2_weights_for_H1;
    auto H1 = this->H1;
    auto H2_weights_for_M1 = this->H2_weights_for_M1;
    auto M1 = this->M1;

    Kokkos::parallel_for(
        "Compute H2 from H1",
        num_nodes*num_channels,
        KOKKOS_LAMBDA (const int ik) {
            const int i = ik / num_channels;
            const int k = ik % num_channels;
                for (int kp=0; kp<num_channels; ++kp) {
                    H2(i,k) += H2_weights_for_H1(node_types(i),kp*num_channels+k) * H1(i,0,kp);
                }
        });
    Kokkos::fence();
    Kokkos::parallel_for(
        "Compute H2 from M1",
        num_nodes*num_channels,
        KOKKOS_LAMBDA (const int ik) {
            const int i = ik / num_channels;
            const int k = ik % num_channels;
            for (int kp=0; kp<num_channels; ++kp) {
                H2(i,k) += H2_weights_for_M1(kp*num_channels+k) * M1(i,kp);
            }
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::reverse_H2(int num_nodes, Kokkos::View<const int*> node_types, bool zero_H1_adj)
{
    if (H1_adj.extent(0) < H1.extent(0))
        Kokkos::resize(H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
    if (M1_adj.extent(0) < M1.extent(0))
        Kokkos::realloc(M1_adj, M1.extent(0), M1.extent(1));

    if (zero_H1_adj)
        Kokkos::deep_copy(H1_adj, 0.0);
    Kokkos::deep_copy(M1_adj, 0.0);

    auto num_channels = this->num_channels;
    auto H1_adj = this->H1_adj;
    auto M1_adj = this->M1_adj;
    auto H2_adj = this->H2_adj;
    auto H2_weights_for_H1 = this->H2_weights_for_H1;
    auto H2_weights_for_M1 = this->H2_weights_for_M1;

    Kokkos::parallel_for(
        "Reverse H2",
        num_nodes*num_channels,
        KOKKOS_LAMBDA (const int ik) {
            const int i = ik / num_channels;
            const int k = ik % num_channels;
            for (int kp=0; kp<num_channels; ++kp) {
                H1_adj(i,0,k) += H2_weights_for_H1(node_types(i),k*num_channels+kp) * H2_adj(i,kp);
                M1_adj(i,k) +=  H2_weights_for_M1(k*num_channels+kp) * H2_adj(i,kp);
            }
        });
    Kokkos::fence();
}

template <typename Precision, typename AccumPrecision>
double MACEKokkos<Precision, AccumPrecision>::compute_readouts(int num_nodes, const Kokkos::View<const int*> node_types)
{
    if (H1_adj.extent(0) < H1.extent(0))
        Kokkos::realloc(H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
    if (H2_adj.extent(0) < num_nodes or H2_adj.extent(1) != num_channels)
        Kokkos::realloc(H2_adj, num_nodes, num_channels);
    if (readout_2_output.extent(0) < num_nodes)
        Kokkos::realloc(readout_2_output, num_nodes);

    // Warning: Although it doesn't appear necessary to set H1_adj to zero,
    //          it matters when the number of nodes associated with H1 is greater than num_nodes.
    //          There is probably a better way to manage this.
    Kokkos::deep_copy(H1_adj, 0.0);

    auto num_channels = this->num_channels;
    auto node_energies = this->node_energies;
    auto atomic_energies = this->atomic_energies;
    auto H1 = this->H1;
    auto H1_adj = this->H1_adj;
    auto readout_1_weights = this->readout_1_weights;
    
    // atomic energies
    Kokkos::parallel_for("Compute Readouts 1", num_nodes, KOKKOS_LAMBDA (const int i) {
        node_energies(i) += atomic_energies(node_types(i));
    });
    Kokkos::fence();
    // first readout
    Kokkos::parallel_for("Compute Readouts 1", num_nodes, KOKKOS_LAMBDA (const int i) {
        for (int k=0; k<num_channels; ++k) {
            node_energies(i) += readout_1_weights(k) * H1(i,0,k);
            H1_adj(i,0,k) = readout_1_weights(k);
        }
    });
    Kokkos::fence();
    // second readout
    auto H2 = Kokkos::subview(this->H2, make_pair(0,num_nodes), Kokkos::ALL); 
    auto readout_2_output = Kokkos::subview(this->readout_2_output, make_pair(0,num_nodes));
    auto H2_adj = Kokkos::subview(this->H2_adj, make_pair(0,num_nodes), Kokkos::ALL); 
    readout_2.evaluate_gradient(H2, readout_2_output, H2_adj);
    Kokkos::parallel_for("Compute Readouts 2", num_nodes, KOKKOS_LAMBDA (const int i) {
        node_energies(i) += readout_2_output(i);
    });

    double energy;
    Kokkos::parallel_reduce(num_nodes, KOKKOS_LAMBDA (const int& i, double& local_sum) {
        local_sum += node_energies(i);
    }, energy);

    Kokkos::fence();

    return energy;
}

template <typename Precision, typename AccumPrecision>
void MACEKokkos<Precision, AccumPrecision>::load_from_json(std::string filename)
{
    std::ifstream f(filename);
    nlohmann::json file = nlohmann::json::parse(f);
    interaction_mode_rrnlb = false;
    rrnlb_layers_kokkos.clear();
    const std::string interaction_mode = file.value("interaction_mode", "legacy");
    if (interaction_mode != "legacy") {
        if (interaction_mode == "rrnlb") {
            interaction_mode_rrnlb = true;
            MACE rrnlb_host(filename);

            num_elements = rrnlb_host.num_elements;
            num_channels = rrnlb_host.num_channels;
            r_cut = rrnlb_host.r_cut;
            l_max = rrnlb_host.l_max;
            num_lm = rrnlb_host.num_lm;
            L_max = rrnlb_host.L_max;
            num_LM = rrnlb_host.num_LM;
            atomic_numbers = toKokkosView("atomic_numbers", rrnlb_host.atomic_numbers);
            atomic_energies = toKokkosView("atomic_energies", rrnlb_host.atomic_energies);

            has_zbl = rrnlb_host.has_zbl;
            if (has_zbl) {
                zbl = ZBLKokkos(
                    file["zbl_a_exp"].get<double>(),
                    file["zbl_a_prefactor"].get<double>(),
                    file["zbl_c"].get<std::vector<double>>(),
                    file["zbl_covalent_radii"].get<std::vector<double>>(),
                    file["zbl_p"].get<int>());
            }

            auto node_embedding = std::vector<Precision>(
                rrnlb_host.node_embedding_species_values.begin(),
                rrnlb_host.node_embedding_species_values.end());
            set_kokkos_view(
                rrnlb_node_embedding,
                node_embedding,
                num_elements,
                num_channels);

            rrnlb_product_linear_0 = make_rrnlb_linear_kokkos<Precision, AccumPrecision>(
                rrnlb_host.product_linear_0, "rrnlb_product_linear_0");
            rrnlb_product_linear_1 = make_rrnlb_linear_kokkos<Precision, AccumPrecision>(
                rrnlb_host.product_linear_1, "rrnlb_product_linear_1");

            if (!file.contains("interaction_layers") || !file["interaction_layers"].is_array()) {
                throw std::runtime_error("RRNLB JSON missing interaction_layers.");
            }
            const auto& layers_json = file.at("interaction_layers");
            if (layers_json.size() != rrnlb_host.rrnlb_layers.size()) {
                throw std::runtime_error("RRNLB interaction layer count mismatch.");
            }

            rrnlb_layers_kokkos.clear();
            rrnlb_layers_kokkos.reserve(rrnlb_host.rrnlb_layers.size());
            for (int li = 0; li < static_cast<int>(rrnlb_host.rrnlb_layers.size()); ++li) {
                const auto& src = rrnlb_host.rrnlb_layers[li];
                const auto& layer_json = layers_json.at(li);

                RRNLBLayerKokkos layer;
                layer.alpha = src.alpha;
                layer.beta = src.beta;
                layer.avg_num_neighbors = src.avg_num_neighbors;
                layer.tp_weight_numel = src.tp_weight_numel;
                layer.gate_scalar_cst = src.gate_scalar_cst;
                layer.linear_up = make_rrnlb_linear_kokkos<Precision, AccumPrecision>(
                    src.linear_up, "rrnlb_layer" + std::to_string(li) + "_linear_up");
                layer.linear_res = make_rrnlb_linear_kokkos<Precision, AccumPrecision>(
                    src.linear_res, "rrnlb_layer" + std::to_string(li) + "_linear_res");
                layer.linear_1 = make_rrnlb_linear_kokkos<Precision, AccumPrecision>(
                    src.linear_1, "rrnlb_layer" + std::to_string(li) + "_linear_1");
                layer.linear_2 = make_rrnlb_linear_kokkos<Precision, AccumPrecision>(
                    src.linear_2, "rrnlb_layer" + std::to_string(li) + "_linear_2");
                layer.skip_tp = make_rrnlb_linear_kokkos<Precision, AccumPrecision>(
                    src.skip_tp, "rrnlb_layer" + std::to_string(li) + "_skip_tp");

                const int num_target = static_cast<int>(src.target_parts.size());
                layer.num_gate_parts = num_target;
                Kokkos::realloc(layer.target_offset, num_target);
                Kokkos::realloc(layer.target_mul, num_target);
                Kokkos::realloc(layer.target_l, num_target);
                Kokkos::realloc(layer.nonlin_offset, num_target);
                Kokkos::realloc(layer.nonlin_mul, num_target);
                Kokkos::realloc(layer.nonlin_l, num_target);
                auto h_target_offset = Kokkos::create_mirror_view(layer.target_offset);
                auto h_target_mul = Kokkos::create_mirror_view(layer.target_mul);
                auto h_target_l = Kokkos::create_mirror_view(layer.target_l);
                auto h_nonlin_offset = Kokkos::create_mirror_view(layer.nonlin_offset);
                auto h_nonlin_mul = Kokkos::create_mirror_view(layer.nonlin_mul);
                auto h_nonlin_l = Kokkos::create_mirror_view(layer.nonlin_l);
                for (int p = 0; p < num_target; ++p) {
                    h_target_offset(p) = src.target_parts[p].offset;
                    h_target_mul(p) = src.target_parts[p].mul;
                    h_target_l(p) = src.target_parts[p].l;
                    h_nonlin_offset(p) = src.nonlin_parts[p].offset;
                    h_nonlin_mul(p) = src.nonlin_parts[p].mul;
                    h_nonlin_l(p) = src.nonlin_parts[p].l;
                }
                Kokkos::deep_copy(layer.target_offset, h_target_offset);
                Kokkos::deep_copy(layer.target_mul, h_target_mul);
                Kokkos::deep_copy(layer.target_l, h_target_l);
                Kokkos::deep_copy(layer.nonlin_offset, h_nonlin_offset);
                Kokkos::deep_copy(layer.nonlin_mul, h_nonlin_mul);
                Kokkos::deep_copy(layer.nonlin_l, h_nonlin_l);
                layer.gate_gate_cst = toKokkosView(
                    ("rrnlb_layer" + std::to_string(li) + "_gate_gate_cst").c_str(),
                    std::vector<Precision>(src.gate_gate_cst.begin(), src.gate_gate_cst.end()));

                const int num_conv = static_cast<int>(src.conv_instructions.size());
                Kokkos::realloc(layer.conv_in_offset, num_conv);
                Kokkos::realloc(layer.conv_out_offset, num_conv);
                Kokkos::realloc(layer.conv_mul, num_conv);
                Kokkos::realloc(layer.conv_weight_offset, num_conv);
                Kokkos::realloc(layer.conv_in_ir_dim, num_conv);
                Kokkos::realloc(layer.conv_out_ir_dim, num_conv);
                Kokkos::realloc(layer.conv_term_offset, num_conv);
                Kokkos::realloc(layer.conv_term_count, num_conv);
                auto h_conv_in_offset = Kokkos::create_mirror_view(layer.conv_in_offset);
                auto h_conv_out_offset = Kokkos::create_mirror_view(layer.conv_out_offset);
                auto h_conv_mul = Kokkos::create_mirror_view(layer.conv_mul);
                auto h_conv_weight_offset = Kokkos::create_mirror_view(layer.conv_weight_offset);
                auto h_conv_in_ir_dim = Kokkos::create_mirror_view(layer.conv_in_ir_dim);
                auto h_conv_out_ir_dim = Kokkos::create_mirror_view(layer.conv_out_ir_dim);
                auto h_conv_term_offset = Kokkos::create_mirror_view(layer.conv_term_offset);
                auto h_conv_term_count = Kokkos::create_mirror_view(layer.conv_term_count);
                const int h_up_dim = layer.linear_up.dim_out;
                const int conv_out_dim = layer.linear_1.dim_in;
                std::vector<unsigned char> active_in_mask(h_up_dim, static_cast<unsigned char>(0));
                std::vector<unsigned char> active_out_mask(conv_out_dim, static_cast<unsigned char>(0));

                int total_terms = 0;
                int max_mul = 0;
                for (const auto& ins : src.conv_instructions) {
                    total_terms += static_cast<int>(ins.terms.size());
                    if (ins.mul > max_mul) max_mul = ins.mul;
                }
                layer.conv_max_mul = max_mul;
                Kokkos::realloc(layer.conv_term_m_out, total_terms);
                Kokkos::realloc(layer.conv_term_m_in1, total_terms);
                Kokkos::realloc(layer.conv_term_y_lm, total_terms);
                Kokkos::realloc(layer.conv_term_coeff, total_terms);
                auto h_term_m_out = Kokkos::create_mirror_view(layer.conv_term_m_out);
                auto h_term_m_in1 = Kokkos::create_mirror_view(layer.conv_term_m_in1);
                auto h_term_y_lm = Kokkos::create_mirror_view(layer.conv_term_y_lm);
                auto h_term_coeff = Kokkos::create_mirror_view(layer.conv_term_coeff);

                int term_offset = 0;
                for (int q = 0; q < num_conv; ++q) {
                    const auto& ins = src.conv_instructions[q];
                    const auto& in_part = src.edge_parts[ins.i_in1];
                    const auto& out_part = src.linear_1.parts_in[ins.i_out];
                    const int in_offset = in_part.offset;
                    const int in_ir_dim = 2 * in_part.l + 1;
                    h_conv_in_offset(q) = in_offset;
                    h_conv_out_offset(q) = out_part.offset;
                    h_conv_mul(q) = ins.mul;
                    h_conv_weight_offset(q) = ins.weight_offset;
                    h_conv_in_ir_dim(q) = in_ir_dim;
                    h_conv_out_ir_dim(q) = 2 * out_part.l + 1;
                    h_conv_term_offset(q) = term_offset;
                    h_conv_term_count(q) = ins.terms.size();
                    for (int t = 0; t < static_cast<int>(ins.terms.size()); ++t) {
                        h_term_m_out(term_offset + t) = ins.terms[t].m_out;
                        const int m_in1 = ins.terms[t].m_in1;
                        h_term_m_in1(term_offset + t) = m_in1;
                        h_term_y_lm(term_offset + t) = ins.terms[t].y_lm;
                        h_term_coeff(term_offset + t) = static_cast<Precision>(ins.terms[t].coeff);
                        for (int k = 0; k < ins.mul; ++k) {
                            const int in_idx = in_offset + k * in_ir_dim + m_in1;
                            const int out_idx = out_part.offset + k * (2 * out_part.l + 1) + ins.terms[t].m_out;
                            if (in_idx < 0 || in_idx >= h_up_dim) {
                                throw std::runtime_error(
                                    "RRNLB conv instruction index out of bounds for linear_up output.");
                            }
                            if (out_idx < 0 || out_idx >= conv_out_dim) {
                                throw std::runtime_error(
                                    "RRNLB conv instruction index out of bounds for linear_1 input.");
                            }
                            active_in_mask[in_idx] = static_cast<unsigned char>(1);
                            active_out_mask[out_idx] = static_cast<unsigned char>(1);
                        }
                    }
                    term_offset += ins.terms.size();
                }

                std::vector<int> active_in_indices;
                active_in_indices.reserve(h_up_dim);
                for (int p = 0; p < h_up_dim; ++p) {
                    if (active_in_mask[p] != static_cast<unsigned char>(0)) {
                        active_in_indices.push_back(p);
                    }
                }
                if (active_in_indices.empty()) {
                    active_in_indices.resize(h_up_dim);
                    for (int p = 0; p < h_up_dim; ++p) active_in_indices[p] = p;
                }
                layer.conv_active_in_count = static_cast<int>(active_in_indices.size());
                Kokkos::realloc(layer.conv_active_in_indices, layer.conv_active_in_count);
                Kokkos::realloc(layer.conv_active_in_inverse, h_up_dim);
                auto h_conv_active_in_indices = Kokkos::create_mirror_view(layer.conv_active_in_indices);
                auto h_conv_active_in_inverse = Kokkos::create_mirror_view(layer.conv_active_in_inverse);
                for (int p = 0; p < h_up_dim; ++p) h_conv_active_in_inverse(p) = -1;
                for (int p_local = 0; p_local < layer.conv_active_in_count; ++p_local) {
                    const int p = active_in_indices[p_local];
                    h_conv_active_in_indices(p_local) = p;
                    h_conv_active_in_inverse(p) = p_local;
                }

                std::vector<int> active_out_indices;
                active_out_indices.reserve(conv_out_dim);
                for (int p = 0; p < conv_out_dim; ++p) {
                    if (active_out_mask[p] != static_cast<unsigned char>(0)) {
                        active_out_indices.push_back(p);
                    }
                }
                if (active_out_indices.empty()) {
                    active_out_indices.resize(conv_out_dim);
                    for (int p = 0; p < conv_out_dim; ++p) active_out_indices[p] = p;
                }
                layer.conv_active_out_count = static_cast<int>(active_out_indices.size());
                Kokkos::realloc(layer.conv_active_out_indices, layer.conv_active_out_count);
                Kokkos::realloc(layer.conv_active_out_inverse, conv_out_dim);
                auto h_conv_active_out_indices = Kokkos::create_mirror_view(layer.conv_active_out_indices);
                auto h_conv_active_out_inverse = Kokkos::create_mirror_view(layer.conv_active_out_inverse);
                for (int p = 0; p < conv_out_dim; ++p) h_conv_active_out_inverse(p) = -1;
                for (int p_local = 0; p_local < layer.conv_active_out_count; ++p_local) {
                    const int p = active_out_indices[p_local];
                    h_conv_active_out_indices(p_local) = p;
                    h_conv_active_out_inverse(p) = p_local;
                }
                Kokkos::deep_copy(layer.conv_in_offset, h_conv_in_offset);
                Kokkos::deep_copy(layer.conv_out_offset, h_conv_out_offset);
                Kokkos::deep_copy(layer.conv_mul, h_conv_mul);
                Kokkos::deep_copy(layer.conv_weight_offset, h_conv_weight_offset);
                Kokkos::deep_copy(layer.conv_in_ir_dim, h_conv_in_ir_dim);
                Kokkos::deep_copy(layer.conv_out_ir_dim, h_conv_out_ir_dim);
                Kokkos::deep_copy(layer.conv_term_offset, h_conv_term_offset);
                Kokkos::deep_copy(layer.conv_term_count, h_conv_term_count);
                Kokkos::deep_copy(layer.conv_term_m_out, h_term_m_out);
                Kokkos::deep_copy(layer.conv_term_m_in1, h_term_m_in1);
                Kokkos::deep_copy(layer.conv_term_y_lm, h_term_y_lm);
                Kokkos::deep_copy(layer.conv_term_coeff, h_term_coeff);
                Kokkos::deep_copy(layer.conv_active_in_indices, h_conv_active_in_indices);
                Kokkos::deep_copy(layer.conv_active_in_inverse, h_conv_active_in_inverse);
                Kokkos::deep_copy(layer.conv_active_out_indices, h_conv_active_out_indices);
                Kokkos::deep_copy(layer.conv_active_out_inverse, h_conv_active_out_inverse);

                // --- Build flattened convolution work table ---
                {
                    int total_work = 0;
                    for (int q = 0; q < num_conv; ++q) {
                        total_work += static_cast<int>(src.conv_instructions[q].terms.size())
                                    * src.conv_instructions[q].mul;
                    }
                    layer.conv_work_table_size = total_work;

                    Kokkos::realloc(layer.conv_work_out_idx, total_work);
                    Kokkos::realloc(layer.conv_work_in_idx, total_work);
                    Kokkos::realloc(layer.conv_work_w_idx, total_work);
                    Kokkos::realloc(layer.conv_work_y_lm, total_work);
                    Kokkos::realloc(layer.conv_work_coeff, total_work);
                    Kokkos::realloc(layer.conv_work_in_local_idx, total_work);
                    Kokkos::realloc(layer.conv_work_out_slot, total_work);

                    auto h_work_out_idx = Kokkos::create_mirror_view(layer.conv_work_out_idx);
                    auto h_work_in_idx = Kokkos::create_mirror_view(layer.conv_work_in_idx);
                    auto h_work_w_idx = Kokkos::create_mirror_view(layer.conv_work_w_idx);
                    auto h_work_y_lm = Kokkos::create_mirror_view(layer.conv_work_y_lm);
                    auto h_work_coeff = Kokkos::create_mirror_view(layer.conv_work_coeff);
                    auto h_work_in_local_idx = Kokkos::create_mirror_view(layer.conv_work_in_local_idx);
                    auto h_work_out_slot = Kokkos::create_mirror_view(layer.conv_work_out_slot);

                    int w_pos = 0;
                    for (int q = 0; q < num_conv; ++q) {
                        const auto& ins = src.conv_instructions[q];
                        const auto& in_part = src.edge_parts[ins.i_in1];
                        const auto& out_part = src.linear_1.parts_in[ins.i_out];
                        const int in_off = in_part.offset;
                        const int in_ir = 2 * in_part.l + 1;
                        const int out_off = out_part.offset;
                        const int out_ir = 2 * out_part.l + 1;
                        const int w_off = ins.weight_offset;
                        const int mul = ins.mul;

                        for (int t = 0; t < static_cast<int>(ins.terms.size()); ++t) {
                            const int m_out = ins.terms[t].m_out;
                            const int m_in1 = ins.terms[t].m_in1;
                            const int ylm = ins.terms[t].y_lm;
                            const auto coeff = static_cast<Precision>(ins.terms[t].coeff);

                            for (int k = 0; k < mul; ++k) {
                                const int out_idx = out_off + k * out_ir + m_out;
                                const int in_idx = in_off + k * in_ir + m_in1;
                                h_work_out_idx(w_pos) = out_idx;
                                h_work_in_idx(w_pos) = in_idx;
                                h_work_w_idx(w_pos) = w_off + k;
                                h_work_y_lm(w_pos) = ylm;
                                h_work_coeff(w_pos) = coeff;
                                h_work_in_local_idx(w_pos) =
                                    (in_idx < static_cast<int>(h_conv_active_in_inverse.extent(0)))
                                        ? h_conv_active_in_inverse(in_idx) : in_idx;
                                h_work_out_slot(w_pos) =
                                    (out_idx < static_cast<int>(h_conv_active_out_inverse.extent(0)))
                                        ? h_conv_active_out_inverse(out_idx) : -1;
                                w_pos += 1;
                            }
                        }
                    }

                    Kokkos::deep_copy(layer.conv_work_out_idx, h_work_out_idx);
                    Kokkos::deep_copy(layer.conv_work_in_idx, h_work_in_idx);
                    Kokkos::deep_copy(layer.conv_work_w_idx, h_work_w_idx);
                    Kokkos::deep_copy(layer.conv_work_y_lm, h_work_y_lm);
                    Kokkos::deep_copy(layer.conv_work_coeff, h_work_coeff);
                    Kokkos::deep_copy(layer.conv_work_in_local_idx, h_work_in_local_idx);
                    Kokkos::deep_copy(layer.conv_work_out_slot, h_work_out_slot);
                }

                const auto& radial = layer_json.at("radial");
                layer.radial_h = radial.at("spline_h").get<double>();
                auto tp_values = radial.at("tp_weights_values")
                    .get<std::vector<std::vector<std::vector<double>>>>();
                auto tp_derivs = radial.at("tp_weights_derivs")
                    .get<std::vector<std::vector<std::vector<double>>>>();
                if (tp_values.empty() || tp_values[0].empty()) {
                    throw std::runtime_error("RRNLB radial tp spline payload is empty.");
                }
                const int num_pairs = static_cast<int>(tp_values.size());
                const int num_nodes_rad = static_cast<int>(tp_values[0][0].size());
                layer.radial_num_intervals = num_nodes_rad - 1;
                Kokkos::realloc(
                    layer.tp_spline_coeff,
                    num_pairs,
                    layer.radial_num_intervals,
                    4,
                    layer.tp_weight_numel);
                auto h_tp_coeff = Kokkos::create_mirror_view(layer.tp_spline_coeff);
                for (int p = 0; p < num_pairs; ++p) {
                    for (int i = 0; i < layer.radial_num_intervals; ++i) {
                        for (int w = 0; w < layer.tp_weight_numel; ++w) {
                            const double v0 = tp_values[p][w][i];
                            const double v1 = tp_values[p][w][i + 1];
                            const double d0 = tp_derivs[p][w][i];
                            const double d1 = tp_derivs[p][w][i + 1];
                            const double h = layer.radial_h;
                            h_tp_coeff(p, i, 0, w) = static_cast<Precision>(v0);
                            h_tp_coeff(p, i, 1, w) = static_cast<Precision>(d0);
                            h_tp_coeff(p, i, 2, w) = static_cast<Precision>(
                                (-3.0 * v0 - 2.0 * h * d0 + 3.0 * v1 - h * d1) / (h * h));
                            h_tp_coeff(p, i, 3, w) = static_cast<Precision>(
                                (2.0 * v0 + h * d0 - 2.0 * v1 + h * d1) / (h * h * h));
                        }
                    }
                }
                Kokkos::deep_copy(layer.tp_spline_coeff, h_tp_coeff);

                auto dens_values = radial.at("edge_density_values")
                    .get<std::vector<std::vector<double>>>();
                auto dens_derivs = radial.at("edge_density_derivs")
                    .get<std::vector<std::vector<double>>>();
                Kokkos::realloc(
                    layer.density_spline_coeff,
                    num_pairs,
                    layer.radial_num_intervals,
                    4);
                auto h_density_coeff = Kokkos::create_mirror_view(layer.density_spline_coeff);
                for (int p = 0; p < num_pairs; ++p) {
                    for (int i = 0; i < layer.radial_num_intervals; ++i) {
                        const double v0 = dens_values[p][i];
                        const double v1 = dens_values[p][i + 1];
                        const double d0 = dens_derivs[p][i];
                        const double d1 = dens_derivs[p][i + 1];
                        const double h = layer.radial_h;
                        h_density_coeff(p, i, 0) = static_cast<Precision>(v0);
                        h_density_coeff(p, i, 1) = static_cast<Precision>(d0);
                        h_density_coeff(p, i, 2) = static_cast<Precision>(
                            (-3.0 * v0 - 2.0 * h * d0 + 3.0 * v1 - h * d1) / (h * h));
                        h_density_coeff(p, i, 3) = static_cast<Precision>(
                            (2.0 * v0 + h * d0 - 2.0 * v1 + h * d1) / (h * h * h));
                    }
                }
                Kokkos::deep_copy(layer.density_spline_coeff, h_density_coeff);

                // Select scratch vs global staging for fused kernels based on
                // per-layer scratch requirements vs 48 KB shared memory limit.
                {
                    const int fwd_scratch = (layer.linear_1.dim_in + layer.linear_2.dim_in) * sizeof(Precision);
                    const int rev_scratch = layer.linear_1.dim_out * sizeof(AccumPrecision);
                    const int limit = 48 * 1024;
                    layer.fusion_storage_mode_fwd = (fwd_scratch <= limit)
                        ? RRNLBFusionStorageMode::ScratchTiled : RRNLBFusionStorageMode::GlobalStaged;
                    layer.fusion_storage_mode_rev = (rev_scratch <= limit)
                        ? RRNLBFusionStorageMode::ScratchTiled : RRNLBFusionStorageMode::GlobalStaged;
                    layer.fusion_tile_width_fwd = layer.linear_1.dim_in + layer.linear_2.dim_in;
                    layer.fusion_tile_width_rev = layer.linear_1.dim_out;
                }

                rrnlb_layers_kokkos.push_back(std::move(layer));
            }

            // Initialize per-layer precomputed edge radial descriptor buffers.
            rrnlb_edge_tp_values.resize(rrnlb_layers_kokkos.size());
            rrnlb_edge_tp_derivs.resize(rrnlb_layers_kokkos.size());
            rrnlb_edge_density_value.resize(rrnlb_layers_kokkos.size());
            rrnlb_edge_density_deriv.resize(rrnlb_layers_kokkos.size());

            // Polynomial graph coefficients for M0
            M0_poly_spec = Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
                Kokkos::view_alloc("M0_poly_spec", Kokkos::SequentialHostInit), num_LM);
            M0_poly_coeff = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
                Kokkos::view_alloc("M0_poly_coeff", Kokkos::SequentialHostInit), num_LM);
            for (int LM = 0; LM < num_LM; ++LM) {
                const auto& base_poly = rrnlb_host.P0[LM * num_channels];
                M0_poly_spec(LM) = Kokkos::View<int**,Kokkos::LayoutRight>(
                    Kokkos::view_alloc("M0_poly_spec_lm", Kokkos::WithoutInitializing),
                    base_poly.edges.size(),
                    2);
                auto h_spec = Kokkos::create_mirror_view(M0_poly_spec(LM));
                for (int p = 0; p < static_cast<int>(base_poly.edges.size()); ++p) {
                    h_spec(p, 0) = base_poly.edges[p][0];
                    h_spec(p, 1) = base_poly.edges[p][1];
                }
                Kokkos::deep_copy(M0_poly_spec(LM), h_spec);

                M0_poly_coeff(LM) = Kokkos::View<Precision***,Kokkos::LayoutRight>(
                    Kokkos::view_alloc("M0_poly_coeff_lm", Kokkos::WithoutInitializing),
                    num_elements,
                    base_poly.node_coefficients.size(),
                    num_channels);
                auto h_coeff = Kokkos::create_mirror_view(M0_poly_coeff(LM));
                for (int a = 0; a < num_elements; ++a) {
                    for (int k = 0; k < num_channels; ++k) {
                        const auto& poly =
                            rrnlb_host.P0[a * num_LM * num_channels + LM * num_channels + k];
                        for (int p = 0; p < static_cast<int>(poly.node_coefficients.size()); ++p) {
                            h_coeff(a, p, k) = static_cast<Precision>(poly.node_coefficients[p]);
                        }
                    }
                }
                Kokkos::deep_copy(M0_poly_coeff(LM), h_coeff);
            }
            M0_poly_values = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
                Kokkos::view_alloc("M0_poly_values", Kokkos::SequentialHostInit), num_LM);
            M0_poly_adjoints = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
                Kokkos::view_alloc("M0_poly_adjoints", Kokkos::SequentialHostInit), num_LM);

            // Polynomial graph coefficients for M1
            const auto& base_m1 = rrnlb_host.P1[0];
            Kokkos::realloc(M1_poly_spec, base_m1.edges.size(), 2);
            auto h_M1_poly_spec = Kokkos::create_mirror_view(M1_poly_spec);
            for (int p = 0; p < static_cast<int>(base_m1.edges.size()); ++p) {
                h_M1_poly_spec(p, 0) = base_m1.edges[p][0];
                h_M1_poly_spec(p, 1) = base_m1.edges[p][1];
            }
            Kokkos::deep_copy(M1_poly_spec, h_M1_poly_spec);
            Kokkos::realloc(M1_poly_coeff, num_elements, base_m1.node_coefficients.size(), num_channels);
            auto h_M1_poly_coeff = Kokkos::create_mirror_view(M1_poly_coeff);
            for (int a = 0; a < num_elements; ++a) {
                for (int k = 0; k < num_channels; ++k) {
                    const auto& poly = rrnlb_host.P1[a * num_channels + k];
                    for (int p = 0; p < static_cast<int>(poly.node_coefficients.size()); ++p) {
                        h_M1_poly_coeff(a, p, k) = static_cast<Precision>(poly.node_coefficients[p]);
                    }
                }
            }
            Kokkos::deep_copy(M1_poly_coeff, h_M1_poly_coeff);

            readout_1_weights = toKokkosView("readout_1_weights", rrnlb_host.readout_1_weights);
            auto readout_2_weights_1 = file["readout_2_weights_1"].get<std::vector<double>>();
            auto readout_2_weights_2 = file["readout_2_weights_2"].get<std::vector<double>>();
            readout_2 = MultilayerPerceptronKokkos(
                std::vector<int>{num_channels, 16, 1},
                std::vector<std::vector<double>>{readout_2_weights_1, readout_2_weights_2},
                file["readout_2_scale_factor"]);
            return;
        }
        throw std::runtime_error(
            std::string("Unsupported interaction_mode '") + interaction_mode + "'.");
    }
    
    // Basic model information
    num_elements = file["num_elements"];
    num_channels = file["num_channels"];
    r_cut = file["r_cut"];
    l_max = file["l_max"];
    num_lm = (l_max+1)*(l_max+1);
    L_max = file["L_max"];
    num_LM = (L_max+1)*(L_max+1);
    atomic_numbers = toKokkosView("atomic_numbers", file["atomic_numbers"].get<std::vector<int>>());
    atomic_energies = toKokkosView("atomic_energies", file["atomic_energies"].get<std::vector<double>>());

    // ZBL
    has_zbl = file["has_zbl"].get<bool>();
    if (has_zbl)
        zbl = ZBLKokkos(
            file["zbl_a_exp"].get<double>(),
            file["zbl_a_prefactor"].get<double>(),
            file["zbl_c"].get<std::vector<double>>(),
            file["zbl_covalent_radii"].get<std::vector<double>>(),
            file["zbl_p"].get<int>());

    // R0
    const double spl_h = file["radial_spline_h"];
    auto spl_values_0 = file["radial_spline_values_0"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_0 = file["radial_spline_derivs_0"].get<std::vector<std::vector<std::vector<double>>>>();
    auto c = Kokkos::View<Precision****,Kokkos::LayoutRight>(
        "c", atomic_numbers.size()*atomic_numbers.size(), spl_values_0[0][0].size()-1, 4, (l_max+1)*num_channels);
    auto h_c = Kokkos::create_mirror_view(c);
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int b=0; b<atomic_numbers.size(); ++b) {
            const int ab = a*atomic_numbers.size()+b;
            const int ab_unordered = (a <= b)
                ? a*(2*atomic_numbers.size()-a-1)/2 + b
                : b*(2*atomic_numbers.size()-b-1)/2 + a;
            auto spl_values = spl_values_0[ab_unordered];
            auto spl_derivs = spl_derivs_0[ab_unordered];
            for (int i=0; i<spl_values_0[0][0].size()-1; ++i) {
                for (int lk=0; lk<(l_max+1)*num_channels; ++lk) {
                    h_c(ab,i,0,lk) = spl_values[lk][i];
                    h_c(ab,i,1,lk) = spl_derivs[lk][i];
                    h_c(ab,i,2,lk) = (-3*spl_values[lk][i] -2*spl_h*spl_derivs[lk][i]
                                        + 3*spl_values[lk][i+1] - spl_h*spl_derivs[lk][i+1]) / (spl_h*spl_h);
                    h_c(ab,i,3,lk) = (2*spl_values[lk][i] + spl_h*spl_derivs[lk][i]
                                        - 2*spl_values[lk][i+1] + spl_h*spl_derivs[lk][i+1]) / (spl_h*spl_h*spl_h);
                }
            }
            // add H0_weights
            auto H0_weights = file["H0_weights"].get<std::vector<double>>();
            for (int i=0; i<spl_values_0[0][0].size()-1; ++i) {
                for (int lk=0; lk<(l_max+1)*num_channels; ++lk) {
                    const int k = lk % num_channels;
                    h_c(ab,i,0,lk) *= H0_weights[b*num_channels+k];
                    h_c(ab,i,1,lk) *= H0_weights[b*num_channels+k]; 
                    h_c(ab,i,2,lk) *= H0_weights[b*num_channels+k]; 
                    h_c(ab,i,3,lk) *= H0_weights[b*num_channels+k]; 
                }
            }
            // add A0_weights
            auto A0_weights = file["A0_weights"].get<std::vector<std::vector<std::vector<double>>>>();
            for (int i=0; i<spl_values_0[0][0].size()-1; ++i) {
                for (int l=0; l<=l_max; ++l) {
                    auto c0 = std::vector<double>(&h_c(ab,i,0,l*num_channels), &h_c(ab,i,0,l*num_channels)+num_channels);
                    auto c1 = std::vector<double>(&h_c(ab,i,1,l*num_channels), &h_c(ab,i,1,l*num_channels)+num_channels);
                    auto c2 = std::vector<double>(&h_c(ab,i,2,l*num_channels), &h_c(ab,i,2,l*num_channels)+num_channels);
                    auto c3 = std::vector<double>(&h_c(ab,i,3,l*num_channels), &h_c(ab,i,3,l*num_channels)+num_channels);
                    for (int k=0; k<num_channels; ++k) {
                        h_c(ab,i,0,l*num_channels+k) = 0.0;
                        h_c(ab,i,1,l*num_channels+k) = 0.0;
                        h_c(ab,i,2,l*num_channels+k) = 0.0;
                        h_c(ab,i,3,l*num_channels+k) = 0.0;
                        for (int kp=0; kp<num_channels; ++kp) {
                            h_c(ab,i,0,l*num_channels+k) += A0_weights[a][l][kp*num_channels+k]*c0[kp];
                            h_c(ab,i,1,l*num_channels+k) += A0_weights[a][l][kp*num_channels+k]*c1[kp];
                            h_c(ab,i,2,l*num_channels+k) += A0_weights[a][l][kp*num_channels+k]*c2[kp];
                            h_c(ab,i,3,l*num_channels+k) += A0_weights[a][l][kp*num_channels+k]*c3[kp];
                        }
                    }
                }
            }
        }
    }
    Kokkos::deep_copy(c, h_c);
    R0_spline_h = spl_h;
    R0_spline_coefficients = c;

    // R1
    auto spl_values_1 = file["radial_spline_values_1"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_1 = file["radial_spline_derivs_1"].get<std::vector<std::vector<std::vector<double>>>>();
    radial_1 = RadialFunctionSetKokkos<Precision>(spl_h, spl_values_1, spl_derivs_1);

    // A0 scaling
    A0_scaled = file["A0_scaled"].get<bool>();
    if (A0_scaled) {
        const double A0_spline_h = file["A0_spline_h"];
        auto A0_spline_values = std::vector<std::vector<std::vector<double>>>();
        for (auto& values : file["A0_spline_values"].get<std::vector<std::vector<double>>>())
            A0_spline_values.push_back({values});  // adds dimension to reach 3d
        auto A0_spline_derivs = std::vector<std::vector<std::vector<double>>>();
        for (auto& derivs : file["A0_spline_derivs"].get<std::vector<std::vector<double>>>())
            A0_spline_derivs.push_back({derivs});  // adds dimension to reach 3d
        A0_splines = RadialFunctionSetKokkos<double>(A0_spline_h, A0_spline_values, A0_spline_derivs);
    }

    // M0 weights and monomials
    auto M0_weights_file = file["M0_weights"].get<std::map<std::string,std::map<std::string,std::map<std::string,std::vector<double>>>>>();
    auto M0_monomials_file = file["M0_monomials"].get<std::map<std::string,std::vector<std::vector<int>>>>();
    M0_weights = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_weights", Kokkos::SequentialHostInit), num_LM);
    M0_monomials = Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_monomials", Kokkos::SequentialHostInit), num_LM);
    for (int LM=0; LM<num_LM; ++LM) {
        M0_weights(LM) = Kokkos::View<Precision***,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("M0_weights_") + std::to_string(LM), Kokkos::WithoutInitializing),
            atomic_numbers.size(), num_channels, M0_monomials_file[std::to_string(LM)].size());
        auto h_M0_weights_LM = Kokkos::create_mirror_view(M0_weights(LM));
        for (int a=0; a<atomic_numbers.size(); ++a)
            for (int k=0; k<num_channels; ++k)
                for (int w=0; w<M0_monomials_file[std::to_string(LM)].size(); ++w)
                    h_M0_weights_LM(a,k,w) = M0_weights_file[std::to_string(a)][std::to_string(LM)][std::to_string(k)][w];
        Kokkos::deep_copy(M0_weights(LM), h_M0_weights_LM);
        M0_monomials(LM) = Kokkos::View<int**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("M0_monomials_") + std::to_string(LM), Kokkos::WithoutInitializing),
            M0_monomials_file[std::to_string(LM)].size(), 3);// TODO: hardcoded 3
        auto h_M0_monomials_LM = Kokkos::create_mirror_view(M0_monomials(LM));
        Kokkos::deep_copy(h_M0_monomials_LM, -1);
        for (int i=0; i<M0_monomials_file[std::to_string(LM)].size(); ++i) {
            for (int j=0; j<M0_monomials_file[std::to_string(LM)][i].size(); ++j) {
                h_M0_monomials_LM(i,j) = M0_monomials_file[std::to_string(LM)][i][j];
            }
        }
        Kokkos::deep_copy(M0_monomials(LM), h_M0_monomials_LM);
    }

    // M0_poly_spec
    M0_poly_spec = Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_poly_spec",Kokkos::SequentialHostInit), num_LM);
    for (int LM=0; LM<num_LM; ++LM) {
        auto P = MultivariatePolynomial(
            num_lm,
            M0_weights_file[std::to_string(0)][std::to_string(LM)][std::to_string(0)],
            M0_monomials_file[std::to_string(LM)]);
        M0_poly_spec(LM) = Kokkos::View<int**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("M0_poly_spec_")+std::to_string(LM),Kokkos::WithoutInitializing),
            P.edges.size(), 2);
        auto h_M0_poly_spec_LM = Kokkos::create_mirror_view(M0_poly_spec(LM));
        for (int p=0; p<P.edges.size(); ++p) {
            h_M0_poly_spec_LM(p,0) = P.edges[p][0];
            h_M0_poly_spec_LM(p,1) = P.edges[p][1];
        }
        Kokkos::deep_copy(M0_poly_spec(LM), h_M0_poly_spec_LM);
    }
    // M0_poly_coeff
    M0_poly_coeff = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_poly_coeff",Kokkos::SequentialHostInit), num_LM);
    for (int LM=0; LM<num_LM; ++LM) {
        auto P = MultivariatePolynomial(
            num_lm,
            M0_weights_file[std::to_string(0)][std::to_string(LM)][std::to_string(0)],
            M0_monomials_file[std::to_string(LM)]);
        M0_poly_coeff(LM) = Kokkos::View<Precision***,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("M0_poly_coeff_")+std::to_string(LM),Kokkos::WithoutInitializing),
            atomic_numbers.size(), P.node_coefficients.size(), num_channels);
        auto h_M0_poly_coeff_LM = Kokkos::create_mirror_view(M0_poly_coeff(LM));
        for (int a=0; a<atomic_numbers.size(); ++a) {
            for (int k=0; k<num_channels; ++k) {
                auto P = MultivariatePolynomial(
                    num_lm,
                    M0_weights_file[std::to_string(a)][std::to_string(LM)][std::to_string(k)],
                    M0_monomials_file[std::to_string(LM)]);
                for (int p=0; p<P.node_coefficients.size(); ++p) {
                    h_M0_poly_coeff_LM(a,p,k) = P.node_coefficients[p];
                }
            }
        }
        Kokkos::deep_copy(M0_poly_coeff(LM), h_M0_poly_coeff_LM);
    }
    M0_poly_values = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_poly_values",Kokkos::SequentialHostInit), num_LM);
    M0_poly_adjoints = Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_poly_adjoints",Kokkos::SequentialHostInit), num_LM);

    // H1 weights
    set_kokkos_view(
        H1_weights,
        file["H1_weights"].get<std::vector<Precision>>(),
        L_max+1,
        num_channels,
        num_channels);

    // Phi1
    Phi1_l = toKokkosView("Phi1_l", file["Phi1_l"].get<std::vector<int>>());
    Phi1_l1 = toKokkosView("Phi1_l1", file["Phi1_l1"].get<std::vector<int>>());
    Phi1_l2 = toKokkosView("Phi1_l2", file["Phi1_l2"].get<std::vector<int>>());
    Phi1_lme = toKokkosView("Phi1_lme", file["Phi1_lme"].get<std::vector<int>>());
    Phi1_clebsch_gordan = toKokkosView("Phi1_clebsch_gordan", file["Phi1_clebsch_gordan"].get<std::vector<Precision>>());
    Phi1_lelm1lm2 = toKokkosView("Phi1_lelm1lm2", file["Phi1_lelm1lm2"].get<std::vector<int>>());
    num_lme = 0;
    auto h_Phi1_l = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l);
    for (int i=0; i<h_Phi1_l.size(); ++i)
        num_lme += 2*h_Phi1_l(i)+1;
    num_lelm1lm2 = 0;
    auto h_Phi1_l1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l1);
    auto h_Phi1_l2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l2);
    for (int le=0; le<h_Phi1_l.size(); ++le)
        num_lelm1lm2 += (2*h_Phi1_l1(le)+1)*(2*h_Phi1_l2(le)+1);

    // for new approach to Phi1
    std::vector<int> Phi1_lm1, Phi1_lm2, Phi1_lel1l2;
    int lelm1lm2 = 0;
    for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
        const int l1 = h_Phi1_l1(lel1l2);
        const int l2 = h_Phi1_l2(lel1l2);
        for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
            for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                Phi1_lm1.push_back(lm1);
                Phi1_lm2.push_back(lm2);
                Phi1_lel1l2.push_back(lel1l2);
                lelm1lm2 += 1;
            }
        }
    }
    this->Phi1_lm1 = toKokkosView("Phi1_lm1", Phi1_lm1);
    this->Phi1_lm2 = toKokkosView("Phi1_lm2", Phi1_lm2);
    this->Phi1_lel1l2 = toKokkosView("Phi1_lel1l2", Phi1_lel1l2);

    // A1 weights
    auto file_A1_weights = file["A1_weights"].get<std::vector<std::vector<Precision>>>();
    A1_weights = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("A1_weights", Kokkos::SequentialHostInit), l_max+1);
    A1_weights_trans = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("A1_weights_trans", Kokkos::SequentialHostInit), l_max+1);
    for (int l=0; l<=l_max; ++l) {
        int num_eta = 0;
        auto h_Phi1_l = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l);
        for (int i=0; i<h_Phi1_l.size(); ++i)
            num_eta += (h_Phi1_l(i) == l);
        A1_weights(l) = Kokkos::View<Precision**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("A1_weights_") + std::to_string(l), Kokkos::WithoutInitializing),
            num_eta*num_channels, num_channels);
        A1_weights_trans(l) = Kokkos::View<Precision**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("A1_weights_") + std::to_string(l), Kokkos::WithoutInitializing),
            num_channels, num_eta*num_channels);
        auto h_A1_weights_l = Kokkos::create_mirror_view(A1_weights(l));
        auto h_A1_weights_trans_l = Kokkos::create_mirror_view(A1_weights_trans(l));
        for (int i=0; i<num_eta*num_channels; ++i) {
            for (int j=0; j<num_channels; ++j) {
                h_A1_weights_l(i,j) = file_A1_weights[l][i*num_channels+j];
                h_A1_weights_trans_l(j,i) = file_A1_weights[l][i*num_channels+j];
            }
        }
        Kokkos::deep_copy(A1_weights(l), h_A1_weights_l);
        Kokkos::deep_copy(A1_weights_trans(l), h_A1_weights_trans_l);
    }

    // A1 scaling
    A1_scaled = file["A1_scaled"].get<bool>();
    if (A1_scaled) {
        const double A1_spline_h = file["A1_spline_h"];
        auto A1_spline_values = std::vector<std::vector<std::vector<double>>>();
        for (auto& values : file["A1_spline_values"].get<std::vector<std::vector<double>>>())
            A1_spline_values.push_back({values});  // adds dimension to reach 3d
        auto A1_spline_derivs = std::vector<std::vector<std::vector<double>>>();
        for (auto& derivs : file["A1_spline_derivs"].get<std::vector<std::vector<double>>>())
            A1_spline_derivs.push_back({derivs});  // adds dimension to reach 3d
        A1_splines = RadialFunctionSetKokkos<double>(A1_spline_h, A1_spline_values, A1_spline_derivs);
    }

    // M1 weights and monomials
    auto M1_weights = file["M1_weights"].get<std::map<std::string,std::map<std::string,std::vector<double>>>>();
    const int num_terms = M1_weights[std::to_string(0)][std::to_string(0)].size();
    Kokkos::resize(this->M1_weights, atomic_numbers.size(), num_channels, num_terms);
    auto h_M1_weights = Kokkos::create_mirror_view(this->M1_weights);
    for (int a=0; a<atomic_numbers.size(); ++a)
        for (int k=0; k<num_channels; ++k)
            for (int w=0; w<num_terms; ++w)
                h_M1_weights(a,k,w) = M1_weights[std::to_string(a)][std::to_string(k)][w];
    Kokkos::deep_copy(this->M1_weights, h_M1_weights);
    auto M1_monomials = file["M1_monomials"].get<std::vector<std::vector<int>>>();
    Kokkos::resize(this->M1_monomials, num_terms, 3);// TODO: hardcoded 3
    auto h_M1_monomials = Kokkos::create_mirror_view(this->M1_monomials);
    Kokkos::deep_copy(h_M1_monomials, -1);
    for (int i=0; i<num_terms; ++i)
        for (int j=0; j<M1_monomials[i].size(); ++j)
            h_M1_monomials(i,j) = M1_monomials[i][j];
    Kokkos::deep_copy(this->M1_monomials, h_M1_monomials);
    // Begin recursive
    auto P1 = std::vector<MultivariatePolynomial>();
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int k=0; k<num_channels; ++k) {
            P1.push_back(MultivariatePolynomial(
                num_lm,
                M1_weights[std::to_string(a)][std::to_string(k)],
                M1_monomials));
        }
    }
    // M1_poly_spec
    Kokkos::realloc(M1_poly_spec, P1[0].edges.size(), 2);
    auto h_M1_poly_spec = Kokkos::create_mirror_view(M1_poly_spec);
    for (int p=0; p<P1[0].edges.size(); ++p) {
        h_M1_poly_spec(p,0) = P1[0].edges[p][0];
        h_M1_poly_spec(p,1) = P1[0].edges[p][1];
    }
    Kokkos::deep_copy(M1_poly_spec, h_M1_poly_spec);
    // M1_poly_coeff
    Kokkos::realloc(M1_poly_coeff, atomic_numbers.size(), num_lm+P1[0].edges.size(), num_channels);
    auto h_M1_poly_coeff = Kokkos::create_mirror_view(M1_poly_coeff);
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int p=0; p<num_lm+P1[0].edges.size(); ++p) {
            for (int k=0; k<num_channels; ++k) {
                h_M1_poly_coeff(a,p,k) = P1[a*num_channels+k].node_coefficients[p];
            }
        }
    }
    Kokkos::deep_copy(M1_poly_coeff, h_M1_poly_coeff);

    // H2
    auto H2_weights_for_H1_vec = file["H2_weights_for_H1"].get<std::vector<std::vector<double>>>();
    H2_weights_for_H1 = Kokkos::View<double**,Kokkos::LayoutRight>("H2_weights_for_H2", num_elements, num_channels*num_channels);
    auto h_H2_weights_for_H1 = Kokkos::create_mirror_view(H2_weights_for_H1);
    for (int i=0; i<num_elements; ++i) {
        for (int j=0; j<num_channels*num_channels; ++j) {
            h_H2_weights_for_H1(i,j) = H2_weights_for_H1_vec[i][j];
        }
    }
    Kokkos::deep_copy(H2_weights_for_H1, h_H2_weights_for_H1);
    H2_weights_for_M1 = toKokkosView("H2_weights_for_M1", file["H2_weights_for_M1"].get<std::vector<double>>());

    // Readouts
    // WARNING!
    // hardcoded 16
    readout_1_weights = toKokkosView("readout_1_weights", file["readout_1_weights"].get<std::vector<double>>());
    auto readout_2_weights_1 = file["readout_2_weights_1"].get<std::vector<double>>();
    auto readout_2_weights_2 = file["readout_2_weights_2"].get<std::vector<double>>();
    readout_2 = MultilayerPerceptronKokkos(
        std::vector<int>{num_channels, 16, 1},
        std::vector<std::vector<double>>{readout_2_weights_1, readout_2_weights_2},
        file["readout_2_scale_factor"]);
}

template class MACEKokkos<float, float>;
template class MACEKokkos<float, double>;
template class MACEKokkos<double, double>;
