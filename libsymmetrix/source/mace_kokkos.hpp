#include <string>
#include <vector>
#include <span>
#include <memory>

#include "Kokkos_UnorderedMap.hpp"

#include "cubic_spline_kokkos.hpp"
#include "cubic_spline_set_kokkos.hpp"
#include "multilayer_perceptron_kokkos.hpp"
#include "multivariate_polynomial.hpp"//TODO
#include "multivariate_polynomial_kokkos.hpp"
#include "radial_function_set_kokkos.hpp"
#include "zbl_kokkos.hpp"

class MACE;

template <typename Precision, typename AccumPrecision = Precision>
class MACEKokkos {

public:

using TensorPrecision = Precision;
using AccumulationPrecision = AccumPrecision;

MACEKokkos(std::string filename);
~MACEKokkos();

struct RRNLBLinearKokkos {
    int dim_in = 0;
    int dim_out = 0;
    // Host-side irrep metadata cached at load time to avoid repeated
    // device->host mirrors in per-step hot loops.
    std::vector<int> h_parts_in_offset;
    std::vector<int> h_parts_in_mul;
    std::vector<int> h_parts_in_l;
    std::vector<int> h_parts_out_offset;
    std::vector<int> h_parts_out_mul;
    std::vector<int> h_parts_out_l;
    // Host-side instruction metadata for host-launched GEMM backends.
    std::vector<int> h_ins_in_offset;
    std::vector<int> h_ins_out_offset;
    std::vector<int> h_ins_mul_in;
    std::vector<int> h_ins_mul_out;
    std::vector<int> h_ins_ir_dim;
    std::vector<int> h_ins_weight_offset;
    std::vector<Precision> h_ins_path_weight;
    // Host copies of grouped plans for packed GEMM path.
    std::vector<int> h_fwd_group_out_offset;
    std::vector<int> h_fwd_group_mul_out;
    std::vector<int> h_fwd_group_ir_dim;
    std::vector<int> h_fwd_group_first_ins;
    std::vector<int> h_fwd_group_num_ins;
    std::vector<int> h_fwd_group_ins_index;
    std::vector<int> h_fwd_group_ins_k_offset;
    std::vector<int> h_fwd_group_k_total;
    std::vector<int> h_fwd_group_pack_weight_offset;
    std::vector<int> h_rev_group_in_offset;
    std::vector<int> h_rev_group_mul_in;
    std::vector<int> h_rev_group_ir_dim;
    std::vector<int> h_rev_group_first_ins;
    std::vector<int> h_rev_group_num_ins;
    std::vector<int> h_rev_group_ins_index;
    std::vector<int> h_rev_group_ins_n_offset;
    std::vector<int> h_rev_group_n_total;
    std::vector<int> h_rev_group_pack_weight_offset;
    int active_in_count = 0;
    int active_out_count = 0;
    bool fwd_group_pack_valid = true;
    bool rev_group_pack_valid = true;
    bool sparse_output_reset_enabled = false;
    int fwd_group_max_ir_dim = 0;
    int fwd_group_max_k_total = 0;
    int fwd_group_max_mul_out = 0;
    int rev_group_max_ir_dim = 0;
    int rev_group_max_n_total = 0;
    int rev_group_max_mul_in = 0;
    Kokkos::View<int*> parts_in_offset;
    Kokkos::View<int*> parts_in_mul;
    Kokkos::View<int*> parts_in_l;
    Kokkos::View<int*> parts_in_dim;
    Kokkos::View<int*> parts_out_offset;
    Kokkos::View<int*> parts_out_mul;
    Kokkos::View<int*> parts_out_l;
    Kokkos::View<int*> parts_out_dim;
    Kokkos::View<int*> ins_in_offset;
    Kokkos::View<int*> ins_out_offset;
    Kokkos::View<int*> ins_mul_in;
    Kokkos::View<int*> ins_mul_out;
    Kokkos::View<int*> ins_ir_dim;
    Kokkos::View<int*> ins_weight_offset;
    Kokkos::View<Precision*> ins_path_weight;
    Kokkos::View<Precision*> ins_weights;
    // Grouped instruction plans for portable batched Kokkos kernels.
    // Forward groups are keyed by output part, transpose groups by input part.
    Kokkos::View<int*> fwd_group_out_offset;
    Kokkos::View<int*> fwd_group_mul_out;
    Kokkos::View<int*> fwd_group_ir_dim;
    Kokkos::View<int*> fwd_group_first_ins;
    Kokkos::View<int*> fwd_group_num_ins;
    Kokkos::View<int*> fwd_group_ins_index;
    Kokkos::View<int*> rev_group_in_offset;
    Kokkos::View<int*> rev_group_mul_in;
    Kokkos::View<int*> rev_group_ir_dim;
    Kokkos::View<int*> rev_group_first_ins;
    Kokkos::View<int*> rev_group_num_ins;
    Kokkos::View<int*> rev_group_ins_index;
    Kokkos::View<int*> active_in_indices;
    Kokkos::View<int*> active_out_indices;
    // Packed weights for grouped dense GEMM path.
    // Forward: per-group [K_total, N_out] contiguous blocks.
    Kokkos::View<Precision*> fwd_group_pack_weights;
    // Reverse: per-group [N_total, K_in] contiguous blocks.
    Kokkos::View<Precision*> rev_group_pack_weights;
    Kokkos::View<int*> bias_indices;
    Kokkos::View<Precision*> bias_values;
};

struct RRNLBLayerKokkos {
    double alpha = 0.0;
    double beta = 0.0;
    double avg_num_neighbors = 0.0;
    int tp_weight_numel = 0;
    int conv_max_mul = 0;
    int conv_active_in_count = 0;
    int conv_active_out_count = 0;
    double gate_scalar_cst = 1.0;

    RRNLBLinearKokkos linear_up;
    RRNLBLinearKokkos linear_res;
    RRNLBLinearKokkos linear_1;
    RRNLBLinearKokkos linear_2;
    RRNLBLinearKokkos skip_tp;

    Kokkos::View<int*> target_offset;
    Kokkos::View<int*> target_mul;
    Kokkos::View<int*> target_l;
    Kokkos::View<int*> nonlin_offset;
    Kokkos::View<int*> nonlin_mul;
    Kokkos::View<int*> nonlin_l;
    Kokkos::View<Precision*> gate_gate_cst;

    Kokkos::View<int*> conv_in_offset;
    Kokkos::View<int*> conv_out_offset;
    Kokkos::View<int*> conv_mul;
    Kokkos::View<int*> conv_weight_offset;
    Kokkos::View<int*> conv_in_ir_dim;
    Kokkos::View<int*> conv_out_ir_dim;
    Kokkos::View<int*> conv_term_offset;
    Kokkos::View<int*> conv_term_count;
    Kokkos::View<int*> conv_term_m_out;
    Kokkos::View<int*> conv_term_m_in1;
    Kokkos::View<int*> conv_term_y_lm;
    Kokkos::View<Precision*> conv_term_coeff;
    Kokkos::View<int*> conv_active_in_indices;
    Kokkos::View<int*> conv_active_in_inverse;
    Kokkos::View<int*> conv_active_out_indices;
    Kokkos::View<int*> conv_active_out_inverse;

    double radial_h = 0.0;
    int radial_num_intervals = 0;
    Kokkos::View<Precision****,Kokkos::LayoutRight> tp_spline_coeff;
    Kokkos::View<Precision***,Kokkos::LayoutRight> density_spline_coeff;

    // Flattened convolution work table (model-static, built once at load time).
    // Each entry represents one (instruction q, term t, channel k) triple.
    int conv_work_table_size = 0;
    Kokkos::View<int*> conv_work_out_idx;
    Kokkos::View<int*> conv_work_in_idx;
    Kokkos::View<int*> conv_work_w_idx;
    Kokkos::View<int*> conv_work_y_lm;
    Kokkos::View<Precision*> conv_work_coeff;
    Kokkos::View<int*> conv_work_in_local_idx;
    Kokkos::View<int*> conv_work_out_slot;
};

struct RRNLBLayerEpochDispatchCacheKokkos {
    bool valid = false;
    long long epoch_id = -1;
    int sender_nodes = 0;
    int total_edges = 0;
    int total_sender_segments = 0;
    // Forward compact active-index metadata (cached by epoch/model shape).
    Kokkos::View<int*> fwd_active_out_indices;
    Kokkos::View<int*> fwd_active_out_inverse;
    // Reverse compact active-index metadata (cached by epoch/model shape).
    Kokkos::View<int*> rev_active_in_indices;
    Kokkos::View<int*> rev_active_in_inverse;
};

struct RRNLBLayerCacheKokkos {
    Kokkos::View<Precision**,Kokkos::LayoutRight> h_up;
    Kokkos::View<Precision**,Kokkos::LayoutRight> h_up_targets;
    Kokkos::View<Precision**,Kokkos::LayoutRight> x_targets;
    Kokkos::View<Precision**,Kokkos::LayoutRight> h_res;
    Kokkos::View<Precision**,Kokkos::LayoutRight> conv_accum;
    Kokkos::View<Precision*> density;
    Kokkos::View<Precision**,Kokkos::LayoutRight> lin1_raw;
    Kokkos::View<Precision**,Kokkos::LayoutRight> pre_gate;
    Kokkos::View<Precision**,Kokkos::LayoutRight> gated;
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> gated_adj;
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> pre_gate_adj;
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> lin1_raw_adj;
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> h_res_adj;
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> conv_adj;
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> h_up_adj;
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> h_up_adj_targets;
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> skip_input_adj_targets;
    Kokkos::View<AccumPrecision*> density_adj;
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> x_up_adj;
    Kokkos::View<int*> sender_edge_counts;
    Kokkos::View<int*> sender_edge_offsets;
    Kokkos::View<int*> sender_edge_cursor;
    Kokkos::View<int*> sender_edge_indices;
    Kokkos::View<int*> sender_segment_counts;
    Kokkos::View<int*> sender_segment_offsets;
    Kokkos::View<int*> sender_segment_to_sender;
    // High-water recovery: avoid shrink/realloc churn across neighbor/topology
    // fluctuations by retaining capacity and binding active extents.
    int capacity_nodes = 0;
    int capacity_sender_nodes = 0;
    int capacity_edges = 0;
    int active_nodes = 0;
    int active_sender_nodes = 0;
    int active_edges = 0;
    RRNLBLayerEpochDispatchCacheKokkos dispatch_cache;
};

struct RRNLBPhaseCounters {
    double forward_interaction_seconds = 0.0;
    double reverse_interaction_seconds = 0.0;
    double linear_forward_seconds = 0.0;
    double linear_transpose_seconds = 0.0;
    double comm_pack_seconds = 0.0;
    double comm_unpack_seconds = 0.0;
    double workspace_reset_seconds = 0.0;
    long long forward_interaction_calls = 0;
    long long reverse_interaction_calls = 0;
    long long linear_forward_calls = 0;
    long long linear_transpose_calls = 0;
    long long comm_pack_calls = 0;
    long long comm_unpack_calls = 0;
    long long workspace_reset_calls = 0;
};

struct RrnlbCacheStamp {
    long long model_rev = 0;
    long long system_rev = 0;
    long long epoch_rev = -1;
    long long capacity_rev = 0;
};

struct RrnlbCacheStats {
    long long l0_rebuilds = 0;
    long long l1_rebuilds = 0;
    long long l2_rebuilds = 0;
    long long l0_hits = 0;
    long long l1_hits = 0;
    long long l2_hits = 0;
    long long l0_rebuild_us = 0;
    long long l1_rebuild_us = 0;
    long long l2_rebuild_us = 0;
    long long capacity_growths = 0;
};

struct RrnlbModelStaticCacheKokkos {
    bool initialized = false;
    long long revision = 0;
};

struct RrnlbSystemStaticCacheKokkos {
    bool initialized = false;
    long long revision = 0;
    int max_lammps_type_seen = 0;
};

struct RrnlbEpochTopologyCacheKokkos {
    bool initialized = false;
    long long epoch = -1;
    unsigned long long topology_sig_nodes = 0;
    unsigned long long topology_sig_edges = 0;
    int num_nodes = 0;
    int sender_nodes = 0;
    int total_edges = 0;
    int sender_segment_edges = 0;
    int total_sender_segments = 0;
    Kokkos::View<int*> first_neigh;
    Kokkos::View<int*> edge_to_receiver;
    Kokkos::View<int*> edge_pair_index;
    Kokkos::View<int*> sender_edge_counts;
    Kokkos::View<int*> sender_edge_offsets;
    Kokkos::View<int*> sender_edge_cursor;
    Kokkos::View<int*> sender_edge_indices;
    Kokkos::View<int*> sender_segment_counts;
    Kokkos::View<int*> sender_segment_offsets;
    Kokkos::View<int*> sender_segment_to_sender;
};

struct RrnlbScratchCacheKokkos {
    int max_nodes = 0;
    int max_edges = 0;
    int max_sender_nodes = 0;
};

// Basic model information
int num_elements;
int num_channels;
double r_cut;
int l_max, num_lm;
int L_max, num_LM;
bool interaction_mode_rrnlb = false;
Kokkos::View<int*> atomic_numbers;
Kokkos::View<double*> atomic_energies;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_node_embedding;
RRNLBLinearKokkos rrnlb_product_linear_0;
RRNLBLinearKokkos rrnlb_product_linear_1;
std::vector<RRNLBLayerKokkos> rrnlb_layers_kokkos;
Kokkos::View<int*> rrnlb_first_neigh;
Kokkos::View<int*> rrnlb_edge_to_receiver;
Kokkos::View<int*> rrnlb_sender_edge_counts;
Kokkos::View<int*> rrnlb_sender_edge_offsets;
Kokkos::View<int*> rrnlb_sender_edge_cursor;
Kokkos::View<int*> rrnlb_sender_edge_indices;
int rrnlb_total_edges = 0;
RRNLBLayerCacheKokkos rrnlb_cache_0;
RRNLBLayerCacheKokkos rrnlb_cache_1;
// Reused packed linear workspaces to avoid per-call allocations in hot loops.
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_fwd_pack_x_workspace;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_fwd_pack_y_workspace;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_rev_pack_y_workspace;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_rev_pack_x_workspace;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_transpose_y_precision_workspace;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_transpose_x_precision_workspace;
Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_M1_adj_ap;
Kokkos::View<AccumPrecision***,Kokkos::LayoutRight> rrnlb_A1_adj_ap;
Kokkos::View<AccumPrecision***,Kokkos::LayoutRight> rrnlb_M0_adj_ap;
Kokkos::View<AccumPrecision***,Kokkos::LayoutRight> rrnlb_A0_adj_ap;
Kokkos::View<AccumPrecision***,Kokkos::LayoutRight> rrnlb_M1_poly_adjoints_ap;
Kokkos::View<Kokkos::View<AccumPrecision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>
    rrnlb_M0_poly_adjoints_ap;
// Persistent single-rank workspaces to avoid per-step cudaMalloc/cudaFree.
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_sr_node_embed;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_sr_interaction0_out;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_sr_skip0;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_sr_interaction1_out;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_sr_skip1;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_sr_product0_in;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_sr_feat0;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_sr_product1_in;
Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_sr_feat1;
Kokkos::View<double**,Kokkos::LayoutRight> rrnlb_sr_feat1_double;
Kokkos::View<double*> rrnlb_sr_readout2_out;
Kokkos::View<double**,Kokkos::LayoutRight> rrnlb_sr_readout2_adj;
Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_sr_feat0_adj;
Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_sr_feat1_adj;
Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_sr_product1_in_adj;
Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_sr_interaction1_adj;
Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_sr_feat0_from_layer1_adj;
Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_sr_product0_in_adj;
Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_sr_interaction0_adj;
Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_sr_node_embed_adj;
// Precomputed per-edge radial descriptors (rebuilt each step, shared across layers).
// Indexed as [layer_index][edge_index, weight_index].
std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>> rrnlb_edge_tp_values;
std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>> rrnlb_edge_tp_derivs;
std::vector<Kokkos::View<Precision*>> rrnlb_edge_density_value;
std::vector<Kokkos::View<Precision*>> rrnlb_edge_density_deriv;

bool rrnlb_portable_v2 = false;
bool rrnlb_phase_stats_enabled_flag = false;
int rrnlb_phase_stats_every_steps = 0;
RRNLBPhaseCounters rrnlb_phase_counters;
RrnlbCacheStamp rrnlb_cache_stamp;
RrnlbCacheStats rrnlb_cache_stats;
RrnlbModelStaticCacheKokkos rrnlb_model_static_cache;
RrnlbSystemStaticCacheKokkos rrnlb_system_static_cache;
RrnlbEpochTopologyCacheKokkos rrnlb_epoch_topology_cache;
RrnlbScratchCacheKokkos rrnlb_scratch_cache;
long long rrnlb_neighbor_epoch = -1;
long long rrnlb_model_revision = 1;
long long rrnlb_system_revision = 1;

// Node energies and forces
Kokkos::View<double*> node_energies, node_forces;
void compute_node_energies_forces(const int num_nodes,
                                  Kokkos::View<const int*> node_types,
                                  Kokkos::View<const int*> num_neigh,
                                  Kokkos::View<const int*> neigh_indices,
                                  Kokkos::View<const int*> neigh_types,
                                  Kokkos::View<const double*> xyz,
                                  Kokkos::View<const double*> r,
                                  Kokkos::View<const int*> first_neigh = Kokkos::View<const int*>(),
                                  int total_edges = -1,
                                  Kokkos::View<const int*> edge_to_receiver = Kokkos::View<const int*>());
void compute_rrnlb_node_energies_forces(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const int*> first_neigh,
    int total_edges,
    Kokkos::View<const int*> edge_to_receiver);
void rrnlb_apply_linear_forward(
    const RRNLBLinearKokkos& linear,
    const int num_nodes,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> x,
    Kokkos::View<Precision**,Kokkos::LayoutRight> y);
void rrnlb_apply_linear_transpose(
    const RRNLBLinearKokkos& linear,
    const int num_nodes,
    Kokkos::View<const AccumPrecision**,Kokkos::LayoutRight> y_adj,
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> x_adj);
void rrnlb_apply_gate_forward(
    const RRNLBLayerKokkos& layer,
    const int num_nodes,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> x,
    Kokkos::View<Precision**,Kokkos::LayoutRight> y);
void rrnlb_apply_gate_reverse(
    const RRNLBLayerKokkos& layer,
    const int num_nodes,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> x,
    Kokkos::View<const AccumPrecision**,Kokkos::LayoutRight> y_adj,
    Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> x_adj);
void compute_rrnlb_interaction_layer_forward(
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
    int num_sender_nodes = -1,
    Kokkos::View<const int*> target_node_indices = Kokkos::View<const int*>(),
    int total_edges = 0,
    Kokkos::View<const int*> edge_to_receiver = Kokkos::View<const int*>());
void reverse_rrnlb_interaction_layer(
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
    int num_sender_nodes = -1,
    Kokkos::View<const int*> target_node_indices = Kokkos::View<const int*>(),
    int total_edges = 0,
    Kokkos::View<const int*> edge_to_receiver = Kokkos::View<const int*>(),
    Kokkos::View<const int*> sender_edge_offsets = Kokkos::View<const int*>(),
    Kokkos::View<const int*> sender_edge_indices = Kokkos::View<const int*>(),
    Kokkos::View<const int*> sender_segment_offsets = Kokkos::View<const int*>(),
    Kokkos::View<const int*> sender_segment_to_sender = Kokkos::View<const int*>(),
    int total_sender_segments = -1);
void precompute_rrnlb_edge_radial_descriptors(
    const int layer_index,
    const int total_edges,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> edge_to_receiver,
    Kokkos::View<const double*> r);
auto rrnlb_portable_v2_enabled() const -> bool { return rrnlb_portable_v2; }
auto rrnlb_phase_stats_enabled() const -> bool { return rrnlb_phase_stats_enabled_flag; }
auto rrnlb_phase_stats_every() const -> int { return rrnlb_phase_stats_every_steps; }
void rrnlb_record_comm_pack(double seconds);
void rrnlb_record_comm_unpack(double seconds);
void rrnlb_record_workspace_reset(double seconds);
void rrnlb_record_forward_interaction(double seconds);
void rrnlb_record_reverse_interaction(double seconds);
void rrnlb_record_linear_forward(double seconds);
void rrnlb_record_linear_transpose(double seconds);
auto rrnlb_take_phase_counters() -> RRNLBPhaseCounters;
void rrnlb_reset_phase_counters();
void rrnlb_set_neighbor_epoch(long long epoch);
void ensure_rrnlb_model_static_cache();
void ensure_rrnlb_system_static_cache(
    const int num_nodes,
    Kokkos::View<const int*> node_types);
void ensure_rrnlb_epoch_topology_cache(
    const int num_nodes,
    const int total_edges,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> first_neigh_input,
    Kokkos::View<const int*> edge_to_receiver_input,
    bool need_sender_maps,
    int sender_nodes = -1);
void ensure_rrnlb_scratch_capacity(
    int num_nodes,
    int total_edges,
    int sender_nodes);
void ensure_rrnlb_layer_workspace_capacity(
    RRNLBLayerCacheKokkos& cache,
    int num_nodes,
    int sender_nodes,
    int total_edges);
void bind_rrnlb_layer_active_views(
    RRNLBLayerCacheKokkos& cache,
    int num_nodes,
    int sender_nodes,
    int total_edges);
void ensure_rrnlb_layer_dispatch_cache(
    const RRNLBLayerKokkos& layer,
    RRNLBLayerCacheKokkos& cache,
    int sender_nodes,
    int total_edges);
auto rrnlb_epoch_topology_fastpath_enabled() const -> bool;
auto rrnlb_get_cache_stats() const -> RrnlbCacheStats { return rrnlb_cache_stats; }
void rrnlb_reset_cache_stats();

// ZBL
bool has_zbl;
ZBLKokkos zbl;

// R0
double R0_spline_h;
Kokkos::View<const Precision****,Kokkos::LayoutRight> R0_spline_coefficients;
Kokkos::View<Precision**,Kokkos::LayoutRight> R0, R0_deriv;
void compute_R0(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types,
                Kokkos::View<const double*> r);

// R1
RadialFunctionSetKokkos<Precision> radial_1;
Kokkos::View<Precision**,Kokkos::LayoutRight> R1, R1_deriv;
void compute_R1(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types,
                Kokkos::View<const double*> r);

// Spherical harmonics
Kokkos::View<Precision*> xyz_shuffled;
Kokkos::View<Precision*> Y, Y_grad;// TODO: make multidimensional
Kokkos::View<Precision*> Y_grad_shuffled;
void compute_Y(Kokkos::View<const double*> xyz);

// A0
Kokkos::View<Precision***,Kokkos::LayoutRight> A0, A0_adj;
void compute_A0(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types);
void reverse_A0(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types,
                Kokkos::View<const double*> xyz,
                Kokkos::View<const double*> r);

// A0 rescaling
bool A0_scaled;
RadialFunctionSetKokkos<double> A0_splines;
Kokkos::View<double**,Kokkos::LayoutRight> A0_spline_values;
Kokkos::View<double**,Kokkos::LayoutRight> A0_spline_derivs;
void compute_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
void reverse_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r);

// M0
Kokkos::View<Precision***,Kokkos::LayoutRight> M0, M0_adj;
Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_monomials;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_weights;
Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_spec;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_coeff;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_values;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_adjoints;
void compute_M0(const int num_nodes, Kokkos::View<const int*> node_types);
void compute_M0_from_rrnlb_layer0_out(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> layer0_out);
void reverse_M0(const int num_nodes, Kokkos::View<const int*> node_types);
void reverse_M0_mixed_rrnlb(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const AccumPrecision***,Kokkos::LayoutRight> M0_adj_in,
    Kokkos::View<AccumPrecision***,Kokkos::LayoutRight> A0_adj_out);

// H1
Kokkos::View<Precision***,Kokkos::LayoutRight> H1, H1_adj;
Kokkos::View<Precision***,Kokkos::LayoutRight> H1_weights;
void compute_H1(const int num_nodes);
void reverse_H1(const int num_nodes);

// Phi1
int num_lelm1lm2, num_lme;
Kokkos::View<int*> Phi1_l, Phi1_l1, Phi1_l2;
Kokkos::View<int*> Phi1_lme, Phi1_lelm1lm2;
Kokkos::View<Precision*> Phi1_clebsch_gordan;
Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1r, dPhi1r;
Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1, dPhi1;
void compute_Phi1(const int num_nodes, Kokkos::View<const int*> num_neigh, Kokkos::View<const int*> neigh_indices);
void reverse_Phi1(const int num_nodes, Kokkos::View<const int*> num_neigh, Kokkos::View<const int*> neigh_indices, Kokkos::View<const double*> xyz, Kokkos::View<const double*> r, bool zero_dxyz = true, bool zero_H1_adj = true);

// TODO for testing of Phi1 strategies
Kokkos::View<int*> Phi1_lm1, Phi1_lm2, Phi1_lel1l2;

// A1
Kokkos::View<Precision***,Kokkos::LayoutRight> A1, A1_adj;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> A1_weights;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> A1_weights_trans;
void compute_A1(int num_nodes);
void reverse_A1(int num_nodes);

// A1 rescaling
bool A1_scaled;
RadialFunctionSetKokkos<double> A1_splines;
Kokkos::View<double**,Kokkos::LayoutRight> A1_spline_values;
Kokkos::View<double**,Kokkos::LayoutRight> A1_spline_derivs;
void compute_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
void reverse_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r);

// M1
Kokkos::View<Precision**,Kokkos::LayoutRight> M1, M1_adj;
Kokkos::View<int**,Kokkos::LayoutRight> M1_monomials;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_weights;
Kokkos::View<int**,Kokkos::LayoutRight> M1_poly_spec;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_poly_coeff;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_poly_values;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_poly_adjoints;
void compute_M1(int num_nodes, Kokkos::View<const int*> node_types);
void compute_M1_from_rrnlb_layer1_out(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> layer1_out);
void reverse_M1(int num_nodes, Kokkos::View<const int*> node_types);
void reverse_M1_mixed_rrnlb(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const AccumPrecision**,Kokkos::LayoutRight> M1_adj_in,
    Kokkos::View<AccumPrecision***,Kokkos::LayoutRight> A1_adj_out);

// H2
Kokkos::View<double**,Kokkos::LayoutRight> H2, H2_adj;
Kokkos::View<double**,Kokkos::LayoutRight> H2_weights_for_H1;
Kokkos::View<double*> H2_weights_for_M1;
void compute_H2(int num_nodes, Kokkos::View<const int*> node_types);
void reverse_H2(int num_nodes, Kokkos::View<const int*> node_types, bool zero_H1_adj = true);

// Readouts
Kokkos::View<double*> readout_1_weights;
MultilayerPerceptronKokkos readout_2;
Kokkos::View<double*> readout_2_output;
double compute_readouts(int num_nodes, const Kokkos::View<const int*> node_types);

// Initializer
void load_from_json(std::string filename);

};
