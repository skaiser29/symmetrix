#include <memory>
#include <string>
#include <vector>
#include <span>

#include "cubic_spline.hpp"
#include "cubic_spline_set.hpp"
#include "multilayer_perceptron.hpp"
#include "multivariate_polynomial.hpp"
#include "zbl.hpp"

class MACE {

public:

MACE(std::string filename);

struct RRNLBIrrepPart {
    int mul = 0;
    int l = 0;
    int p = 1;
    int offset = 0;
    int dim = 0;
};

struct RRNLBLinearInstruction {
    int i_in = -1;
    int i_out = -1;
    int mul_in = 0;
    int mul_out = 0;
    double path_weight = 1.0;
    std::vector<double> weights;
};

struct RRNLBLinear {
    int dim_in = 0;
    int dim_out = 0;
    std::vector<RRNLBIrrepPart> parts_in;
    std::vector<RRNLBIrrepPart> parts_out;
    std::vector<RRNLBLinearInstruction> instructions;
    std::vector<double> bias;
};

struct RRNLBConvTerm {
    int m_out = 0;
    int m_in1 = 0;
    int y_lm = 0;
    double coeff = 0.0;
};

struct RRNLBConvInstruction {
    int i_in1 = -1;
    int i_in2 = -1;
    int i_out = -1;
    int mul = 0;
    int weight_offset = 0;
    int l_in1 = 0;
    int l_in2 = 0;
    int l_out = 0;
    std::vector<RRNLBConvTerm> terms;
};

struct RRNLBLayer {
    double alpha = 0.0;
    double beta = 0.0;
    double avg_num_neighbors = 0.0;
    int tp_weight_numel = 0;
    double gate_scalar_cst = 1.0;
    std::vector<double> gate_gate_cst;
    std::vector<RRNLBIrrepPart> edge_parts;
    std::vector<RRNLBIrrepPart> target_parts;
    std::vector<RRNLBIrrepPart> nonlin_parts;
    std::vector<RRNLBConvInstruction> conv_instructions;
    RRNLBLinear linear_up;
    RRNLBLinear linear_res;
    RRNLBLinear linear_1;
    RRNLBLinear linear_2;
    RRNLBLinear skip_tp;
    std::vector<std::unique_ptr<CubicSplineSet>> tp_splines;
    std::vector<CubicSpline> density_splines;
};

struct RRNLBLayerCache {
    std::vector<double> h_up;
    std::vector<double> h_res;
    std::vector<double> conv_accum;
    std::vector<double> density;
    std::vector<double> lin1_raw;
    std::vector<double> pre_gate;
    std::vector<double> gated;
    std::vector<double> tp_values;
    std::vector<double> gated_adj;
    std::vector<double> pre_gate_adj;
    std::vector<double> lin1_raw_adj;
    std::vector<double> h_res_adj;
    std::vector<double> conv_adj;
    std::vector<double> h_up_adj;
    std::vector<double> density_adj;
    std::vector<double> tp_derivs;
    std::vector<double> y_adj;
    std::vector<double> x_up_adj_tmp;
};

// Basic model information
int num_elements;
int num_channels;
double r_cut;
int l_max, num_lm;
int L_max, num_LM;
std::vector<int> atomic_numbers;
std::vector<double> atomic_energies;
bool interaction_mode_rrnlb = false;
std::vector<double> node_embedding_species_values;
RRNLBLinear product_linear_0;
RRNLBLinear product_linear_1;
std::vector<RRNLBLayer> rrnlb_layers;
std::vector<double> rrnlb_node_feats_0;
std::vector<double> rrnlb_node_feats_1;
RRNLBLayerCache rrnlb_forward_layer0_cache_ws;
RRNLBLayerCache rrnlb_forward_layer1_cache_ws;
std::vector<double> rrnlb_node_embed_ws;
std::vector<double> rrnlb_interaction0_out_ws;
std::vector<double> rrnlb_skip0_ws;
std::vector<double> rrnlb_interaction1_out_ws;
std::vector<double> rrnlb_skip1_ws;
std::vector<double> rrnlb_feat0_adj_ws;
std::vector<double> rrnlb_feat1_adj_ws;
std::vector<double> rrnlb_skip1_adj_ws;
std::vector<double> rrnlb_skip0_adj_ws;
std::vector<double> rrnlb_interaction1_adj_ws;
std::vector<double> rrnlb_interaction0_adj_ws;
std::vector<double> rrnlb_feat0_from_layer1_adj_ws;
std::vector<double> rrnlb_node_embed_adj_ws;
std::vector<double> rrnlb_mlp_input_ws;

// Node energies and forces
std::vector<double> node_energies, node_forces;
void compute_node_energies_forces(const int num_nodes,
                                  std::span<const int> node_types,
                                  std::span<const int> num_neigh,
                                  std::span<const int> neigh_indices,
                                  std::span<const int> neigh_types,
                                  std::span<const double> xyz,
                                  std::span<const double> r);
void compute_rrnlb_forward(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r);
void apply_rrnlb_linear(
    const RRNLBLinear& linear,
    std::span<const double> x,
    std::span<double> y);
void apply_rrnlb_linear_transpose(
    const RRNLBLinear& linear,
    std::span<const double> y_adj,
    std::span<double> x_adj);
void apply_rrnlb_gate(
    const RRNLBLayer& layer,
    std::span<const double> x,
    std::span<double> y);
void apply_rrnlb_gate_reverse(
    const RRNLBLayer& layer,
    std::span<const double> x,
    std::span<const double> y_adj,
    std::span<double> x_adj);
void compute_rrnlb_interaction_layer_forward(
    const int layer_index,
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> r,
    std::span<const double> node_feats_in,
    std::vector<double>& layer_output,
    std::vector<double>& layer_skip,
    RRNLBLayerCache* cache = nullptr,
    int num_sender_nodes = -1,
    std::span<const int> target_node_indices = {});
void reverse_rrnlb_interaction_layer(
    const int layer_index,
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    std::span<const double> node_feats_in,
    RRNLBLayerCache& cache,
    std::span<const double> layer_output_adj,
    std::span<const double> layer_skip_adj,
    std::span<double> node_feats_in_adj,
    int num_sender_nodes = -1,
    std::span<const int> target_node_indices = {});
void compute_rrnlb_M0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const double> node_feats,
    const std::vector<RRNLBIrrepPart>& parts);
void reverse_rrnlb_M0(
    const int num_nodes,
    std::span<const int> node_types,
    const std::vector<RRNLBIrrepPart>& parts,
    std::span<double> node_feats_adj);
void compute_rrnlb_M1(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const double> node_feats,
    const std::vector<RRNLBIrrepPart>& parts);
void reverse_rrnlb_M1(
    const int num_nodes,
    std::span<const int> node_types,
    const std::vector<RRNLBIrrepPart>& parts,
    std::span<double> node_feats_adj);

// ZBL
bool has_zbl;
ZBL zbl;

// Radial functions
std::vector<std::unique_ptr<CubicSplineSet>> spl_set_0;
std::vector<double> R0, R0_deriv;
void compute_R0(const int num_nodes,
                std::span<const int> node_types,
                std::span<const int> num_neigh,
                std::span<const int> neigh_types,
                std::span<const double> r);

// R1
std::vector<std::unique_ptr<CubicSplineSet>> spl_set_1;
std::vector<double> R1, R1_deriv;
void compute_R1(const int num_nodes,
                std::span<const int> node_types,
                std::span<const int> num_neigh,
                std::span<const int> neigh_types,
                std::span<const double> r);

// Spherical harmonics
std::vector<double> Y, Y_grad;
std::vector<double> xyz_shuffled;
void compute_Y(std::span<const double> xyz);

// H0
std::vector<double> H0_weights;

// A0
std::vector<double> A0, A0_adj;
std::vector<std::vector<std::vector<double>>> A0_weights;
void compute_A0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types);
void reverse_A0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r);

// A0 rescaling
bool A0_scaled;
std::vector<CubicSpline> A0_splines;
void compute_A0_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r);
void reverse_A0_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r);

// M0
std::vector<double> M0, M0_grad, M0_adj;
std::vector<MultivariatePolynomial> P0;
void compute_M0(const int num_nodes, std::span<const int> node_types);
// TODO: node_types here only for consistency with Kokkos
void reverse_M0(const int num_nodes, std::span<const int> node_types);

// H1
std::vector<double> H1, H1_adj;
std::vector<double> H1_weights;
void compute_H1(const int num_nodes);
void reverse_H1(const int num_nodes);

// Phi1
int num_lelm1lm2, num_lme;
std::vector<double> Phi1r, dPhi1r;
std::vector<double> Phi1, dPhi1;
std::vector<int> Phi1_l, Phi1_l1, Phi1_l2;
std::vector<int> Phi1_lme, Phi1_lelm1lm2;
std::vector<double> Phi1_clebsch_gordan;
void compute_Phi1(const int num_nodes, std::span<const int> num_neigh, std::span<const int> neigh_indices);
void reverse_Phi1(const int num_nodes, std::span<const int> num_neigh, std::span<const int> neigh_indices, std::span<const double> xyz, std::span<const double> r, bool zero_dxyz = true, bool zero_H1_adj = true);

// A1
std::vector<double> A1, A1_adj;
std::vector<std::vector<double>> A1_weights;
void compute_A1(const int num_nodes);
void reverse_A1(const int num_nodes);

// A1 rescaling
bool A1_scaled;
std::vector<CubicSpline> A1_splines;
void compute_A1_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r);
void reverse_A1_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    bool zero_dxyz = true);

// M1
std::vector<double> M1, M1_grad, M1_adj;
std::vector<MultivariatePolynomial> P1;
void compute_M1(const int num_nodes, std::span<const int> node_types);
// TODO: node_types here only for consistency with Kokkos
void reverse_M1(const int num_nodes, std::span<const int> node_types);

// H2
std::vector<double> H2, H2_adj;
std::vector<std::vector<double>> H2_weights_for_H1;
std::vector<double> H2_weights_for_M1;
void compute_H2(const int num_nodes, std::span<const int> node_types);
void reverse_H2(const int num_nodes, std::span<const int> node_types, bool zero_H1_adj = true);

// Readouts
std::vector<double> readout_1_weights;
std::unique_ptr<MultilayerPerceptron> readout_2;
void compute_readouts(const int num_nodes, std::span<const int> node_types);

// Initializer
void load_from_json(std::string filename);

};
