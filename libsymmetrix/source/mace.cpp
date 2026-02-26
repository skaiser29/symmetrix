#include <iostream> //TODO
#include <fstream>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <sstream>

#include "nlohmann/json.hpp"
#include "sphericart.hpp"

#include "cblas.hpp"
#include "mace.hpp"

namespace {

enum class RRNLBNonlinearAblationMode {
    Off,
    GateIdentity
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
        if (mode == "off") {
            return RRNLBNonlinearAblationMode::Off;
        }
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

auto parse_rrnlb_irrep_parts(const nlohmann::json& irreps_json)
    -> std::vector<MACE::RRNLBIrrepPart>
{
    std::vector<MACE::RRNLBIrrepPart> parts;
    int offset = 0;
    for (const auto& part : irreps_json.at("parts")) {
        MACE::RRNLBIrrepPart parsed;
        parsed.mul = part.at("mul").get<int>();
        parsed.l = part.at("l").get<int>();
        parsed.p = part.at("p").get<int>();
        parsed.offset = offset;
        parsed.dim = parsed.mul * (2 * parsed.l + 1);
        offset += parsed.dim;
        parts.push_back(parsed);
    }
    return parts;
}

auto parse_rrnlb_linear(const nlohmann::json& linear_json)
    -> MACE::RRNLBLinear
{
    MACE::RRNLBLinear linear;
    linear.parts_in = parse_rrnlb_irrep_parts(linear_json.at("irreps_in"));
    linear.parts_out = parse_rrnlb_irrep_parts(linear_json.at("irreps_out"));
    linear.dim_in = linear_json.at("irreps_in").at("dim").get<int>();
    linear.dim_out = linear_json.at("irreps_out").at("dim").get<int>();
    for (const auto& ins : linear_json.at("instructions")) {
        MACE::RRNLBLinearInstruction parsed;
        parsed.i_in = ins.at("i_in").get<int>();
        parsed.i_out = ins.at("i_out").get<int>();
        auto path_shape = ins.at("path_shape").get<std::vector<int>>();
        if (path_shape.size() != 2) {
            throw std::runtime_error("RRNLB linear instruction path_shape must have length 2.");
        }
        parsed.mul_in = path_shape[0];
        parsed.mul_out = path_shape[1];
        parsed.path_weight = ins.at("path_weight").get<double>();
        parsed.weights = ins.at("weight_values").get<std::vector<double>>();
        const int expected = parsed.mul_in * parsed.mul_out;
        if (static_cast<int>(parsed.weights.size()) != expected) {
            throw std::runtime_error("RRNLB linear instruction has unexpected weight length.");
        }
        linear.instructions.push_back(std::move(parsed));
    }
    if (linear_json.contains("bias_values")) {
        linear.bias = linear_json.at("bias_values").get<std::vector<double>>();
    }
    return linear;
}

inline auto sigmoid(const double x) -> double
{
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

inline auto silu(const double x) -> double
{
    return x * sigmoid(x);
}

inline auto silu_deriv(const double x) -> double
{
    const double s = sigmoid(x);
    return s + x * s * (1.0 - s);
}

} // namespace

MACE::MACE(std::string filename)
{
    load_from_json(filename);
}

void MACE::compute_node_energies_forces(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    // TODO: best to resize these within individual routines
    node_energies.resize(num_nodes);
    std::fill(node_energies.begin(), node_energies.end(), 0.0);
    node_forces.resize(xyz.size());
    std::fill(node_forces.begin(), node_forces.end(), 0.0);

    if (has_zbl)
        zbl.compute_ZBL(
            num_nodes, node_types, num_neigh, neigh_types,
            atomic_numbers, r, xyz, node_energies, node_forces);

    compute_Y(xyz);

    if (interaction_mode_rrnlb) {
        compute_rrnlb_forward(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);
        return;
    }

    compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_A0(num_nodes, node_types, num_neigh, neigh_types);
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M0(num_nodes, node_types);
    compute_H1(num_nodes);

    compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_Phi1(num_nodes, num_neigh, neigh_indices);
    compute_A1(num_nodes);
    compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M1(num_nodes, node_types);
    compute_H2(num_nodes, node_types);

    compute_readouts(num_nodes, node_types);

    reverse_H2(num_nodes, node_types, false);
    reverse_M1(num_nodes, node_types);
    reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r, false);
    reverse_A1(num_nodes);
    reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);

    reverse_H1(num_nodes);
    reverse_M0(num_nodes, node_types);
    reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
}

void MACE::compute_R0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    const int num_spl = spl_set_0[0]->num_splines;
    R0.resize(r.size()*num_spl);
    R0_deriv.resize(R0.size());
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            auto R0_ij = std::span<double>(R0.data()+ij*num_spl,num_spl);
            auto R0_deriv_ij = std::span<double>(R0_deriv.data()+ij*num_spl,num_spl);
            spl_set_0[type_ij]->evaluate_derivs(r[ij], R0_ij, R0_deriv_ij);
            ij += 1;
        }
    }
}

void MACE::compute_R1(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    const int num_spl = spl_set_1[0]->num_splines;
    R1.resize(r.size()*num_spl);
    R1_deriv.resize(R1.size());
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            auto R1_ij = std::span<double>(R1.data()+ij*num_spl,num_spl);
            auto R1_deriv_ij = std::span<double>(R1_deriv.data()+ij*num_spl,num_spl);
            spl_set_1[type_ij]->evaluate_derivs(r[ij], R1_ij, R1_deriv_ij);
            ij += 1;
        }
    }
}

void MACE::compute_Y(
    std::span<const double> xyz)
{
    if (xyz.size() == 0) return;
    const int num = xyz.size()/3;
    Y.resize(num*num_lm);
    Y_grad.resize(3*num*num_lm);
    // shuffle to match e3nn conventions
    xyz_shuffled.resize(3*num);
    for (int i=0; i<num; ++i) {
        xyz_shuffled[3*i]   = xyz[3*i+2];
        xyz_shuffled[3*i+1] = xyz[3*i];
        xyz_shuffled[3*i+2] = xyz[3*i+1];
    }
    sphericart::SphericalHarmonics<double> sphericart(l_max);
    sphericart.compute_with_gradients(xyz_shuffled, Y, Y_grad);
    // normalize to match e3nn conventions
    for (int i=0; i<Y.size(); ++i)
        Y[i] *= 2*std::sqrt(std::numbers::pi);
    for (int i=0; i<Y_grad.size(); ++i)
        Y_grad[i] *= 2*std::sqrt(std::numbers::pi);
    // unshuffle gradient
    auto Y_grad_shuffled = Y_grad;
    for (int i=0; i<num; ++i) {
        for (int lm=0; lm<num_lm; ++lm) {
            Y_grad[3*i*num_lm+0*num_lm+lm] = Y_grad_shuffled[3*i*num_lm+1*num_lm+lm];
            Y_grad[3*i*num_lm+1*num_lm+lm] = Y_grad_shuffled[3*i*num_lm+2*num_lm+lm];
            Y_grad[3*i*num_lm+2*num_lm+lm] = Y_grad_shuffled[3*i*num_lm+0*num_lm+lm];
        }
    }
}

void MACE::compute_A0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types)
{
    A0.resize(num_nodes*num_lm*num_channels);

    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {

        // compute Phi0_i
        auto Phi0_i = std::vector<double>(num_lm*num_channels, 0.0);
        for (int j=0; j<num_neigh[i]; ++j) {
            auto Y_ij = Y.data()+ij*num_lm;
            auto H0_ij = H0_weights.data()+neigh_types[ij]*num_channels;
            for (int l=0; l<=l_max; ++l) {
                auto R0_ij_l = R0.data()+ij*(l_max+1)*num_channels+l*num_channels;
                for (int m=-l; m<=l; ++m) {
                    const int lm = l*l+l+m;
                    const double Y_ij_lm = Y_ij[lm];
                    auto Phi0_i_lm = Phi0_i.data()+lm*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        Phi0_i_lm[k] += R0_ij_l[k] * Y_ij_lm * H0_ij[k];
                    }
                }
            }
            ij += 1;
        }

        // [A0_il]_mk = \sum_k' [Phi0_il]_mk' [W_il]_k'k
        for (int l=0; l<=l_max; ++l) {
            auto Phi0_il = Phi0_i.data()+l*l*num_channels;
            auto A0_il = A0.data()+(i*num_lm+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,                        // const CBLAS_LAYOUT Layout
                CblasNoTrans,                         // const CBLAS_TRANSPOSE transa
                CblasNoTrans,                         // const CBLAS_TRANSPOSE transb
                (2*l+1),                              // const MKL_INT m
                num_channels,                         // const MKL_INT n
                num_channels,                         // const MKL_INT k
                1.0,                                  // const double alpha
                Phi0_il,                              // const double *a
                num_channels,                         // const MKL_INT lda
                A0_weights[node_types[i]][l].data(),  // const double *b
                num_channels,                         // const MKL_INT ldb
                0.0,                                  // const double beta
                A0_il,                                // double *c
                num_channels);                        // const MKL_INT ldc
        }
    }
}

void MACE::reverse_A0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {

        auto Phi0_adj_i = std::vector<double>(num_lm*num_channels);

        // [dE/dPhi0_il]_mk = \sum_k' [dE/dA0_il]_mk' [trans(W_il)]_k'k 
        for (int l=0; l<=l_max; ++l) {
            auto Phi0_adj_il = Phi0_adj_i.data()+l*l*num_channels;
            auto A0_adj_il = A0_adj.data()+(i*num_lm+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,                        // const CBLAS_LAYOUT Layout
                CblasNoTrans,                         // const CBLAS_TRANSPOSE transa
                CblasTrans,                           // const CBLAS_TRANSPOSE transb
                (2*l+1),                              // const MKL_INT m
                num_channels,                         // const MKL_INT n
                num_channels,                         // const MKL_INT k
                1.0,                                  // const double alpha
                A0_adj_il,                            // const double *a
                num_channels,                         // const MKL_INT lda
                A0_weights[node_types[i]][l].data(),  // const double *b
                num_channels,                         // const MKL_INT ldb
                0.0,                                  // const double beta
                Phi0_adj_il,                          // double *c
                num_channels);                        // const MKL_INT ldc
        }

        // Warning: Assumes node_forces have been initialized elsewhere
        for (int j=0; j<num_neigh[i]; ++j) {
            auto xyz_ij = xyz.data()+ij*3;
            auto r_ij = r[ij];
            auto Y_ij = Y.data()+ij*num_lm;
            auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
            auto H0_ij = H0_weights.data()+neigh_types[ij]*num_channels;
            auto node_forces_ij = node_forces.data()+ij*3;
            for (int l=0; l<=l_max; ++l) {
                auto R0_ij_l = R0.data()+ij*(l_max+1)*num_channels+l*num_channels;
                auto R0_deriv_ij_l = R0_deriv.data()+ij*(l_max+1)*num_channels+l*num_channels;
                for (int m=-l; m<=l; ++m) {
                    const int lm = l*l+l+m;
                    const double Y_ij_lm = Y_ij[lm];
                    const double Y_grad_ij_lm_x = Y_grad_ij[lm];
                    const double Y_grad_ij_lm_y = Y_grad_ij[num_lm+lm];
                    const double Y_grad_ij_lm_z = Y_grad_ij[2*num_lm+lm];
                    auto Phi0_adj_i_lm = Phi0_adj_i.data()+lm*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        node_forces_ij[0] += -Phi0_adj_i_lm[k] * (
                            xyz_ij[0]/r_ij * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                            + R0_ij_l[k] * Y_grad_ij_lm_x * H0_ij[k] );
                        node_forces_ij[1] += -Phi0_adj_i_lm[k] * (
                            xyz_ij[1]/r_ij * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                            + R0_ij_l[k] * Y_grad_ij_lm_y * H0_ij[k] );
                        node_forces_ij[2] += -Phi0_adj_i_lm[k] * (
                            xyz_ij[2]/r_ij * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                            + R0_ij_l[k] * Y_grad_ij_lm_z * H0_ij[k]);
                    }
                }
            }
            ij += 1;
        }
    }
}

void MACE::compute_A0_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    if (not A0_scaled) return;
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        double A0_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            A0_scale_factor += A0_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        auto A0_i = A0.data()+i*num_lm*num_channels;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A0_i[lmk] /= A0_scale_factor;
    }
}

void MACE::reverse_A0_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    if (not A0_scaled) return;
    // Warning: Assumes node_forces have been initialized elsewhere
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        auto A0_i = A0.data()+i*num_lm*num_channels;
        auto A0_adj_i = A0_adj.data()+i*num_lm*num_channels;
        // recompute the scale factor
        double A0_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            A0_scale_factor += A0_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        // update dE/dxyz
        double dA0_dot_A0 = 0.0;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            dA0_dot_A0 += A0_adj_i[lmk] * A0_i[lmk];
        ij = ij - num_neigh[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            auto [f,d] = A0_splines[type_ij].evaluate_deriv(r[ij]);
            auto xyz_ij = xyz.data()+ij*3;
            auto node_forces_ij = node_forces.data()+ij*3;
            node_forces_ij[0] += dA0_dot_A0/A0_scale_factor*d*xyz_ij[0]/r[ij];
            node_forces_ij[1] += dA0_dot_A0/A0_scale_factor*d*xyz_ij[1]/r[ij];
            node_forces_ij[2] += dA0_dot_A0/A0_scale_factor*d*xyz_ij[2]/r[ij];
            ij += 1;
        }
        // update dE/dA0
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A0_adj_i[lmk] /= A0_scale_factor;
    }
}

void MACE::compute_M0(
    const int num_nodes,
    std::span<const int> node_types)
{
    M0.resize(num_nodes*num_LM*num_channels);
    M0_grad.resize(num_nodes*num_channels*num_LM*num_lm);
    for (int i=0; i<num_nodes; ++i) {
        auto A0_i = A0.data()+i*num_lm*num_channels;
        auto M0_i = M0.data()+i*num_LM*num_channels;
        auto M0_grad_i = M0_grad.data()+i*num_LM*num_channels*num_lm;
        auto x = std::vector<double>(num_lm);
        int lmk = 0;
        for (int lm=0; lm<num_LM; ++lm) {
            for (int k=0; k<num_channels; ++k) {
                cblas_dcopy(num_lm, A0_i+k, num_channels, x.data(), 1);
                auto [f,g] = P0[node_types[i]*num_LM*num_channels+lmk].evaluate_gradient(x);
                M0_i[lmk] = f;
                cblas_dcopy(num_lm, g.data(), 1, M0_grad_i+lm*num_lm*num_channels+k, num_channels);
                lmk += 1;
            }
        }
    }
}

void MACE::compute_rrnlb_M0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const double> node_feats,
    const std::vector<RRNLBIrrepPart>& parts)
{
    if (static_cast<int>(node_types.size()) != num_nodes) {
        throw std::runtime_error("RRNLB M0 expects node_types sized to num_nodes.");
    }
    int in_row_dim = 0;
    for (const auto& part : parts) {
        in_row_dim = std::max(in_row_dim, part.offset + part.dim);
    }
    if (static_cast<int>(node_feats.size()) != num_nodes * in_row_dim) {
        throw std::runtime_error("RRNLB M0 input feature size mismatch.");
    }

    std::vector<int> in_part_for_l(l_max + 1, -1);
    for (int p = 0; p < static_cast<int>(parts.size()); ++p) {
        const auto& part = parts[p];
        if (part.mul != num_channels) {
            throw std::runtime_error("RRNLB M0 input multiplicity must equal num_channels.");
        }
        if (part.l < 0 || part.l > l_max) {
            throw std::runtime_error("RRNLB M0 input angular index is out of bounds.");
        }
        if (in_part_for_l[part.l] >= 0) {
            throw std::runtime_error("RRNLB M0 input irreps contain duplicate l blocks.");
        }
        in_part_for_l[part.l] = p;
    }

    for (const auto& out_part : product_linear_0.parts_in) {
        if (out_part.mul != num_channels) {
            throw std::runtime_error("RRNLB M0 output multiplicity must equal num_channels.");
        }
        if (out_part.l < 0 || out_part.l > L_max) {
            throw std::runtime_error("RRNLB M0 output angular index is out of bounds.");
        }
        if (out_part.offset < 0 || out_part.offset + out_part.dim > product_linear_0.dim_in) {
            throw std::runtime_error("RRNLB M0 output irrep offset is out of bounds.");
        }
    }
    if (product_linear_0.dim_in != num_LM * num_channels) {
        throw std::runtime_error("RRNLB M0 expects product_linear_0 input to match num_LM*num_channels.");
    }

    M0.assign(num_nodes * product_linear_0.dim_in, 0.0);
    M0_grad.assign(num_nodes * product_linear_0.dim_in * num_lm, 0.0);
    std::vector<double> x(num_lm, 0.0);
    for (int i = 0; i < num_nodes; ++i) {
        const int type_i = node_types[i];
        const auto* feats_i = node_feats.data() + i * in_row_dim;
        auto* m0_i = M0.data() + i * product_linear_0.dim_in;
        auto* m0_grad_i = M0_grad.data() + i * product_linear_0.dim_in * num_lm;
        for (int k = 0; k < num_channels; ++k) {
            std::fill(x.begin(), x.end(), 0.0);
            for (int l = 0; l <= l_max; ++l) {
                const int p = in_part_for_l[l];
                if (p < 0) continue;
                const auto& part = parts[p];
                const int ir_dim = 2 * l + 1;
                const auto* src = feats_i + part.offset + k * ir_dim;
                std::copy(src, src + ir_dim, x.begin() + l * l);
            }
            for (const auto& out_part : product_linear_0.parts_in) {
                const int ir_dim = 2 * out_part.l + 1;
                for (int m = 0; m < ir_dim; ++m) {
                    const int lm = out_part.l * out_part.l + m;
                    auto [f, g] = P0[type_i * num_LM * num_channels + lm * num_channels + k]
                                      .evaluate_gradient(x);
                    const int out_idx = out_part.offset + k * ir_dim + m;
                    m0_i[out_idx] = f;
                    std::copy(g.begin(), g.end(), m0_grad_i + out_idx * num_lm);
                }
            }
        }
    }
}

void MACE::reverse_M0(
    const int num_nodes,
    std::span<const int> node_types)
{
    A0_adj.resize(A0.size());
    std::fill(A0_adj.begin(), A0_adj.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto A0_adj_i = A0_adj.data()+i*num_lm*num_channels;
        auto M0_adj_i = M0_adj.data()+i*num_LM*num_channels;
        auto M0_grad_i = M0_grad.data()+i*num_LM*num_lm*num_channels;
        for (int lm=0; lm<num_lm; ++lm) {
            auto A0_adj_ilm = A0_adj_i + lm*num_channels;
            for (int lmp=0; lmp<num_LM; ++lmp) {
                auto M0_adj_ilmp = M0_adj_i + lmp*num_channels;
                auto M0_grad_ilmplm = M0_grad_i +
                    + lmp*num_lm*num_channels
                    + lm*num_channels;
                for (int k=0; k<num_channels; ++k) {
                    A0_adj_ilm[k] += M0_grad_ilmplm[k] * M0_adj_ilmp[k];
                }
            }
        }
    }
}

void MACE::reverse_rrnlb_M0(
    const int num_nodes,
    std::span<const int> node_types,
    const std::vector<RRNLBIrrepPart>& parts,
    std::span<double> node_feats_adj)
{
    if (static_cast<int>(node_types.size()) != num_nodes) {
        throw std::runtime_error("RRNLB reverse M0 expects node_types sized to num_nodes.");
    }
    int in_row_dim = 0;
    for (const auto& part : parts) {
        in_row_dim = std::max(in_row_dim, part.offset + part.dim);
    }
    if (static_cast<int>(node_feats_adj.size()) != num_nodes * in_row_dim) {
        throw std::runtime_error("RRNLB reverse M0 input adjoint size mismatch.");
    }
    if (static_cast<int>(M0_adj.size()) != num_nodes * product_linear_0.dim_in) {
        throw std::runtime_error("RRNLB reverse M0 expects M0_adj sized to product linear input.");
    }
    if (static_cast<int>(M0_grad.size()) != num_nodes * product_linear_0.dim_in * num_lm) {
        throw std::runtime_error("RRNLB reverse M0 expects M0_grad sized to product linear input.");
    }

    std::vector<int> in_part_for_l(l_max + 1, -1);
    for (int p = 0; p < static_cast<int>(parts.size()); ++p) {
        const auto& part = parts[p];
        if (part.mul != num_channels) {
            throw std::runtime_error("RRNLB reverse M0 input multiplicity must equal num_channels.");
        }
        if (part.l < 0 || part.l > l_max) {
            throw std::runtime_error("RRNLB reverse M0 input angular index is out of bounds.");
        }
        if (in_part_for_l[part.l] >= 0) {
            throw std::runtime_error("RRNLB reverse M0 input irreps contain duplicate l blocks.");
        }
        in_part_for_l[part.l] = p;
    }

    for (const auto& out_part : product_linear_0.parts_in) {
        if (out_part.mul != num_channels) {
            throw std::runtime_error("RRNLB reverse M0 output multiplicity must equal num_channels.");
        }
        if (out_part.l < 0 || out_part.l > L_max) {
            throw std::runtime_error("RRNLB reverse M0 output angular index is out of bounds.");
        }
        if (out_part.offset < 0 || out_part.offset + out_part.dim > product_linear_0.dim_in) {
            throw std::runtime_error("RRNLB reverse M0 output irrep offset is out of bounds.");
        }
    }

    std::fill(node_feats_adj.begin(), node_feats_adj.end(), 0.0);
    std::vector<double> x_adj(num_lm, 0.0);
    for (int i = 0; i < num_nodes; ++i) {
        const auto* m0_adj_i = M0_adj.data() + i * product_linear_0.dim_in;
        const auto* m0_grad_i = M0_grad.data() + i * product_linear_0.dim_in * num_lm;
        auto* feats_adj_i = node_feats_adj.data() + i * in_row_dim;
        for (int k = 0; k < num_channels; ++k) {
            std::fill(x_adj.begin(), x_adj.end(), 0.0);
            for (const auto& out_part : product_linear_0.parts_in) {
                const int ir_dim = 2 * out_part.l + 1;
                for (int m = 0; m < ir_dim; ++m) {
                    const int out_idx = out_part.offset + k * ir_dim + m;
                    const double upstream = m0_adj_i[out_idx];
                    const auto* grad = m0_grad_i + out_idx * num_lm;
                    for (int lm = 0; lm < num_lm; ++lm) {
                        x_adj[lm] += upstream * grad[lm];
                    }
                }
            }
            for (int l = 0; l <= l_max; ++l) {
                const int p = in_part_for_l[l];
                if (p < 0) continue;
                const auto& part = parts[p];
                const int ir_dim = 2 * l + 1;
                auto* dst = feats_adj_i + part.offset + k * ir_dim;
                for (int m = 0; m < ir_dim; ++m) {
                    dst[m] += x_adj[l * l + m];
                }
            }
        }
    }
}

void MACE::compute_H1(
    const int num_nodes)
{
    H1.resize(M0.size());
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto M0_il = M0.data()+(i*num_LM+l*l)*num_channels;
            const auto H1_weights_l = H1_weights.data()+l*num_channels*num_channels;
            auto H1_il = H1.data()+(i*num_LM+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,  // const CBLAS_LAYOUT Layout
                CblasNoTrans,   // const CBLAS_TRANSPOSE transa
                CblasNoTrans,   // const CBLAS_TRANSPOSE transb
                2*l+1,          // const MKL_INT m
                num_channels,   // const MKL_INT n
                num_channels,   // const MKL_INT k
                1.0,            // const double alpha
                M0_il,          // const double *a
                num_channels,   // const MKL_INT lda
                H1_weights_l,   // const double *b
                num_channels,   // const MKL_INT ldb
                0.0,            // const double beta
                H1_il,          // double *c
                num_channels);  // const MKL_INT ldc
        }
    }
}

void MACE::reverse_H1(
    const int num_nodes)
{
    M0_adj.resize(M0.size());
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto H1_adj_il = H1_adj.data()+(i*num_LM+l*l)*num_channels;
            const auto H1_weights_l = H1_weights.data()+l*num_channels*num_channels;
            auto M0_adj_il = M0_adj.data()+(i*num_LM+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,  // const CBLAS_LAYOUT Layout
                CblasNoTrans,   // const CBLAS_TRANSPOSE transa
                CblasTrans,     // const CBLAS_TRANSPOSE transb
                2*l+1,          // const MKL_INT m
                num_channels,   // const MKL_INT n
                num_channels,   // const MKL_INT k
                1.0,            // const double alpha
                H1_adj_il,      // const double *a
                num_channels,   // const MKL_INT lda
                H1_weights_l,   // const double *b
                num_channels,   // const MKL_INT ldb
                0.0,            // const double beta
                M0_adj_il,      // double *c
                num_channels);  // const MKL_INT ldc
        }
    }
}

void MACE::compute_Phi1(
    const int num_nodes,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices)
{
    // Compute Phi1_lelm1lm2 (named Phi1r)
    Phi1r.resize(num_nodes*num_lelm1lm2*num_channels);
    std::fill(Phi1r.begin(), Phi1r.end(), 0.0);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1r_i = Phi1r.data()+i*num_lelm1lm2*num_channels;
        for (int j=0; j<num_neigh[i]; ++j) {
            auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
            auto Y_ij = Y.data()+ij*num_lm;
            auto H1_ij = H1.data()+neigh_indices[ij]*num_LM*num_channels;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    const double Y_ij_lm1 = Y_ij[lm1];
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                        auto Phi1r_i_lelm1lm2 = Phi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            Phi1r_i_lelm1lm2[k] += R1_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k];
                        }
                        lelm1lm2 += 1;
                    }
                }
            }
            ij += 1;
        }
    }
    // Compute Phi1 using CG coefficients
    Phi1.resize(num_nodes*num_lme*num_channels);
    std::fill(Phi1.begin(), Phi1.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1_i = Phi1.data()+i*num_lme*num_channels;
        auto Phi1r_i = Phi1r.data()+i*num_lelm1lm2*num_channels;
        for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
            auto Phi1_i_lme = Phi1_i+Phi1_lme[p]*num_channels;
            const double C = Phi1_clebsch_gordan[p];
            auto Phi1r_i_lelm1lm2 = Phi1r_i+Phi1_lelm1lm2[p]*num_channels;
            for (int k=0; k<num_channels; ++k)
                Phi1_i_lme[k] += C * Phi1r_i_lelm1lm2[k];
        }
    }
}

void MACE::reverse_Phi1(
    const int num_nodes,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const double> xyz,
    std::span<const double> r,
    bool zero_dxyz,
    bool zero_H1_adj)
{
    // Compute dE/dPhi1 (named dPhi1)
    dPhi1r.resize(Phi1r.size());
    std::fill(dPhi1r.begin(), dPhi1r.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto dPhi1r_i = dPhi1r.data()+i*num_lelm1lm2*num_channels;
        auto dPhi1_i = dPhi1.data()+i*num_lme*num_channels;
        for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
            auto dPhi1r_i_lelm1lm2 = dPhi1r_i+Phi1_lelm1lm2[p]*num_channels;
            const double C = Phi1_clebsch_gordan[p];
            auto dPhi1_i_lme = dPhi1_i+Phi1_lme[p]*num_channels;
            for (int k=0; k<num_channels; ++k)
                dPhi1r_i_lelm1lm2[k] += C * dPhi1_i_lme[k];
        }
    }
    // Compute partial forces
    node_forces.resize(xyz.size());
    if (zero_dxyz)
        std::fill(node_forces.begin(), node_forces.end(), 0.0);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto dPhi1r_i = dPhi1r.data()+i*num_lelm1lm2*num_channels;
        for (int j=0; j<num_neigh[i]; ++j) {
            auto node_forces_ij = node_forces.data()+3*ij;
            auto xyz_ij = xyz.data()+3*ij;
            auto r_ij = r[ij];
            auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
            auto R1_deriv_ij = R1_deriv.data()+ij*spl_set_1[0]->num_splines;
            auto Y_ij = Y.data()+ij*num_lm;
            auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
            auto H1_ij = H1.data()+neigh_indices[ij]*num_LM*num_channels;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                auto R1_deriv_ij_lel1l2 = R1_deriv_ij+lel1l2*num_channels;
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    const double Y_ij_lm1 = Y_ij[lm1];
                    const double Y_grad_ij_x_lm1 = Y_grad_ij[0*num_lm+lm1];
                    const double Y_grad_ij_y_lm1 = Y_grad_ij[1*num_lm+lm1];
                    const double Y_grad_ij_z_lm1 = Y_grad_ij[2*num_lm+lm1];
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                        auto dPhi1r_i_lelm1lm2 = dPhi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            node_forces_ij[0] += -dPhi1r_i_lelm1lm2[k] * (
                                xyz_ij[0]/r_ij * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_x_lm1 * H1_ij_lm2[k]);
                            node_forces_ij[1] += -dPhi1r_i_lelm1lm2[k] * (
                                xyz_ij[1]/r_ij * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_y_lm1 * H1_ij_lm2[k]);
                            node_forces_ij[2] += -dPhi1r_i_lelm1lm2[k] * (
                                xyz_ij[2]/r_ij * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_z_lm1 * H1_ij_lm2[k]);
                        }
                        lelm1lm2 += 1;
                    }
                }
            }
            ij += 1;
        }
    }
    // Compute dE/dH1 (named dH1)
    H1_adj.resize(H1.size());
    if (zero_H1_adj)
        std::fill(H1_adj.begin(), H1_adj.end(), 0.0);
    ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto dPhi1r_i = dPhi1r.data()+i*num_lelm1lm2*num_channels;
        for (int j=0; j<num_neigh[i]; ++j) {
            auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
            auto Y_ij = Y.data()+ij*num_lm;
            auto H1_adj_ij = H1_adj.data()+neigh_indices[ij]*num_LM*num_channels;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        auto H1_adj_ij_lm2 = H1_adj_ij+lm2*num_channels;
                        auto dPhi1r_i_lelm1lm2 = dPhi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            H1_adj_ij_lm2[k] += R1_ij_lel1l2[k]*Y_ij[lm1]*dPhi1r_i_lelm1lm2[k];
                        }
                        lelm1lm2 += 1;
                    }
                }
            }
            ij += 1;
        }
    }
}

void MACE::compute_A1(
    const int num_nodes)
{
    // The core matrix multiplication is:
    //         [A1_il]_mk = \sum_k' [Phi1_il]_m(ek') [W_il]_(ek')k
    A1.resize(num_nodes*num_lm*num_channels);
    int num_lme = 0;
    std::vector<int> num_e(l_max+1,0);
    for (auto l : Phi1_l) {
        num_lme += 2*l+1;
        num_e[l] += 1;
    }
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1_il = Phi1.data()+i*num_lme*num_channels;
        auto A1_il = A1.data()+i*num_lm*num_channels;
        for (int l=0; l<=l_max; ++l) {
            cblas_dgemm(
                CblasRowMajor,          // const CBLAS_LAYOUT Layout
                CblasNoTrans,           // const CBLAS_TRANSPOSE transa
                CblasNoTrans,           // const CBLAS_TRANSPOSE transb
                (2*l+1),                // const MKL_INT m
                num_channels,           // const MKL_INT n
                num_e[l]*num_channels,  // const MKL_INT k
                1.0,                    // const double alpha
                Phi1_il,                // const double *a
                num_e[l]*num_channels,  // const MKL_INT lda
                A1_weights[l].data(),   // const double *b
                num_channels,           // const MKL_INT ldb
                0.0,                    // const double beta
                A1_il,                  // double *c
                num_channels);          // const MKL_INT ldc
            Phi1_il += (2*l+1)*num_e[l]*num_channels;
            A1_il += (2*l+1)*num_channels;
        }
    }
}

void MACE::reverse_A1(
    const int num_nodes)
{
    // The core matrix multiplication is:
    //         [dE/dPhi1_il]_m(ek) = \sum_k' [dE/dA1_il]_mk' [trans(W_il)]_k'(ek)
    dPhi1.resize(Phi1.size());
    int num_lme = 0;
    std::vector<int> num_e(l_max+1,0);
    for (auto l : Phi1_l) {
        num_lme += 2*l+1;
        num_e[l] += 1;
    }
    for (int i=0; i<num_nodes; ++i) {
        auto A1_adj_il = A1_adj.data()+i*num_lm*num_channels;
        auto dPhi1_il = dPhi1.data()+i*num_lme*num_channels;
        for (int l=0; l<=l_max; ++l) {
            cblas_dgemm(
                CblasRowMajor,          // const CBLAS_LAYOUT Layout
                CblasNoTrans,           // const CBLAS_TRANSPOSE transa
                CblasTrans,             // const CBLAS_TRANSPOSE transb
                (2*l+1),                // const MKL_INT m
                num_e[l]*num_channels,  // const MKL_INT n
                num_channels,           // const MKL_INT k
                1.0,                    // const double alpha
                A1_adj_il,              // const double *a
                num_channels,           // const MKL_INT lda
                A1_weights[l].data(),   // const double *b
                num_channels,           // const MKL_INT ldb
                0.0,                    // const double beta
                dPhi1_il,            // double *c
                num_e[l]*num_channels); // const MKL_INT ldc
            A1_adj_il += (2*l+1)*num_channels;
            dPhi1_il += (2*l+1)*num_e[l]*num_channels;
        }
    }
}

void MACE::compute_A1_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    if (not A1_scaled) return;
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        double A1_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            A1_scale_factor += A1_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        auto A1_i = A1.data()+i*num_lm*num_channels;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A1_i[lmk] /= A1_scale_factor;
    }
}

void MACE::reverse_A1_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    bool zero_dxyz)
{
    if (not A1_scaled) return;
    node_forces.resize(xyz.size());
    if (zero_dxyz)
        std::fill(node_forces.begin(), node_forces.end(), 0.0);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        auto A1_i = A1.data()+i*num_lm*num_channels;
        auto A1_adj_i = A1_adj.data()+i*num_lm*num_channels;
        // recompute the scale factor
        double A1_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            A1_scale_factor += A1_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        // update dE/dxyz
        double dA1_dot_A1 = 0.0;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            dA1_dot_A1 += A1_adj_i[lmk] * A1_i[lmk];
        ij = ij - num_neigh[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            auto [f,d] = A1_splines[type_ij].evaluate_deriv(r[ij]);
            auto xyz_ij = xyz.data()+ij*3;
            auto node_forces_ij = node_forces.data()+ij*3;
            node_forces_ij[0] += dA1_dot_A1/A1_scale_factor*d*xyz_ij[0]/r[ij];
            node_forces_ij[1] += dA1_dot_A1/A1_scale_factor*d*xyz_ij[1]/r[ij];
            node_forces_ij[2] += dA1_dot_A1/A1_scale_factor*d*xyz_ij[2]/r[ij];
            ij += 1;
        }
        // update dE/dA1
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A1_adj_i[lmk] /= A1_scale_factor;
    }
}

void MACE::compute_M1(
    const int num_nodes,
    std::span<const int> node_types)
{
    M1.resize(num_nodes*num_channels);
    M1_grad.resize(num_nodes*num_channels*num_lm*num_channels);
    for (int i=0; i<num_nodes; ++i) {
        auto A1_i = A1.data()+i*num_lm*num_channels;
        auto M1_i = M1.data()+i*num_channels;
        auto M1_grad_i = M1_grad.data()+i*num_channels*num_lm;
        auto x = std::vector<double>(num_lm);
        for (int k=0; k<num_channels; ++k) {
            cblas_dcopy(num_lm, A1_i+k, num_channels, x.data(), 1);
            auto [f,g] = P1[node_types[i]*num_channels+k].evaluate_gradient(x);
            M1_i[k] = f;
            cblas_dcopy(num_lm, g.data(), 1, M1_grad_i+k, num_channels);
        }
    }
}

void MACE::compute_rrnlb_M1(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const double> node_feats,
    const std::vector<RRNLBIrrepPart>& parts)
{
    if (static_cast<int>(node_types.size()) != num_nodes) {
        throw std::runtime_error("RRNLB M1 expects node_types sized to num_nodes.");
    }
    int in_row_dim = 0;
    for (const auto& part : parts) {
        in_row_dim = std::max(in_row_dim, part.offset + part.dim);
    }
    if (static_cast<int>(node_feats.size()) != num_nodes * in_row_dim) {
        throw std::runtime_error("RRNLB M1 input feature size mismatch.");
    }

    std::vector<int> in_part_for_l(l_max + 1, -1);
    for (int p = 0; p < static_cast<int>(parts.size()); ++p) {
        const auto& part = parts[p];
        if (part.mul != num_channels) {
            throw std::runtime_error("RRNLB M1 input multiplicity must equal num_channels.");
        }
        if (part.l < 0 || part.l > l_max) {
            throw std::runtime_error("RRNLB M1 input angular index is out of bounds.");
        }
        if (in_part_for_l[part.l] >= 0) {
            throw std::runtime_error("RRNLB M1 input irreps contain duplicate l blocks.");
        }
        in_part_for_l[part.l] = p;
    }

    if (product_linear_1.dim_in != num_channels) {
        throw std::runtime_error("RRNLB M1 expects product_linear_1 input to match num_channels.");
    }
    if (product_linear_1.parts_in.size() != 1
        || product_linear_1.parts_in[0].l != 0
        || product_linear_1.parts_in[0].mul != num_channels) {
        throw std::runtime_error("RRNLB M1 expects a single scalar product input irrep block.");
    }
    const auto& out_part = product_linear_1.parts_in[0];
    if (out_part.offset < 0 || out_part.offset + out_part.dim > product_linear_1.dim_in) {
        throw std::runtime_error("RRNLB M1 output irrep offset is out of bounds.");
    }

    M1.assign(num_nodes * product_linear_1.dim_in, 0.0);
    M1_grad.assign(num_nodes * product_linear_1.dim_in * num_lm, 0.0);
    std::vector<double> x(num_lm, 0.0);
    for (int i = 0; i < num_nodes; ++i) {
        const int type_i = node_types[i];
        const auto* feats_i = node_feats.data() + i * in_row_dim;
        auto* m1_i = M1.data() + i * product_linear_1.dim_in;
        auto* m1_grad_i = M1_grad.data() + i * product_linear_1.dim_in * num_lm;
        for (int k = 0; k < num_channels; ++k) {
            std::fill(x.begin(), x.end(), 0.0);
            for (int l = 0; l <= l_max; ++l) {
                const int p = in_part_for_l[l];
                if (p < 0) continue;
                const auto& part = parts[p];
                const int ir_dim = 2 * l + 1;
                const auto* src = feats_i + part.offset + k * ir_dim;
                std::copy(src, src + ir_dim, x.begin() + l * l);
            }
            auto [f, g] = P1[type_i * num_channels + k].evaluate_gradient(x);
            const int out_idx = out_part.offset + k;
            m1_i[out_idx] = f;
            std::copy(g.begin(), g.end(), m1_grad_i + out_idx * num_lm);
        }
    }
}

void MACE::reverse_M1(
    const int num_nodes,
    std::span<const int> node_types)
{
    A1_adj.resize(A1.size());
    for (int i=0; i<num_nodes; ++i) {
        auto M1_adj_i = M1_adj.data() + i*num_channels;
        for (int lm=0; lm<num_lm; ++lm) {
            auto A1_adj_ilm = A1_adj.data() + (i*num_lm+lm)*num_channels;
            auto M1_grad_ilm = M1_grad.begin() + (i*num_lm+lm)*num_channels;
            for (int k=0; k<num_channels; ++k) {
                A1_adj_ilm[k] = M1_grad_ilm[k] * M1_adj_i[k];
            }
        }
    }
}

void MACE::reverse_rrnlb_M1(
    const int num_nodes,
    std::span<const int> node_types,
    const std::vector<RRNLBIrrepPart>& parts,
    std::span<double> node_feats_adj)
{
    if (static_cast<int>(node_types.size()) != num_nodes) {
        throw std::runtime_error("RRNLB reverse M1 expects node_types sized to num_nodes.");
    }
    int in_row_dim = 0;
    for (const auto& part : parts) {
        in_row_dim = std::max(in_row_dim, part.offset + part.dim);
    }
    if (static_cast<int>(node_feats_adj.size()) != num_nodes * in_row_dim) {
        throw std::runtime_error("RRNLB reverse M1 input adjoint size mismatch.");
    }
    if (static_cast<int>(M1_adj.size()) != num_nodes * product_linear_1.dim_in) {
        throw std::runtime_error("RRNLB reverse M1 expects M1_adj sized to product linear input.");
    }
    if (static_cast<int>(M1_grad.size()) != num_nodes * product_linear_1.dim_in * num_lm) {
        throw std::runtime_error("RRNLB reverse M1 expects M1_grad sized to product linear input.");
    }
    if (product_linear_1.parts_in.size() != 1
        || product_linear_1.parts_in[0].l != 0
        || product_linear_1.parts_in[0].mul != num_channels) {
        throw std::runtime_error("RRNLB reverse M1 expects a single scalar product input irrep block.");
    }
    const auto& out_part = product_linear_1.parts_in[0];

    std::vector<int> in_part_for_l(l_max + 1, -1);
    for (int p = 0; p < static_cast<int>(parts.size()); ++p) {
        const auto& part = parts[p];
        if (part.mul != num_channels) {
            throw std::runtime_error("RRNLB reverse M1 input multiplicity must equal num_channels.");
        }
        if (part.l < 0 || part.l > l_max) {
            throw std::runtime_error("RRNLB reverse M1 input angular index is out of bounds.");
        }
        if (in_part_for_l[part.l] >= 0) {
            throw std::runtime_error("RRNLB reverse M1 input irreps contain duplicate l blocks.");
        }
        in_part_for_l[part.l] = p;
    }

    std::fill(node_feats_adj.begin(), node_feats_adj.end(), 0.0);
    std::vector<double> x_adj(num_lm, 0.0);
    for (int i = 0; i < num_nodes; ++i) {
        const auto* m1_adj_i = M1_adj.data() + i * product_linear_1.dim_in;
        const auto* m1_grad_i = M1_grad.data() + i * product_linear_1.dim_in * num_lm;
        auto* feats_adj_i = node_feats_adj.data() + i * in_row_dim;
        for (int k = 0; k < num_channels; ++k) {
            std::fill(x_adj.begin(), x_adj.end(), 0.0);
            const int out_idx = out_part.offset + k;
            const double upstream = m1_adj_i[out_idx];
            const auto* grad = m1_grad_i + out_idx * num_lm;
            for (int lm = 0; lm < num_lm; ++lm) {
                x_adj[lm] = upstream * grad[lm];
            }
            for (int l = 0; l <= l_max; ++l) {
                const int p = in_part_for_l[l];
                if (p < 0) continue;
                const auto& part = parts[p];
                const int ir_dim = 2 * l + 1;
                auto* dst = feats_adj_i + part.offset + k * ir_dim;
                for (int m = 0; m < ir_dim; ++m) {
                    dst[m] += x_adj[l * l + m];
                }
            }
        }
    }
}

void MACE::compute_H2(
    const int num_nodes,
    std::span<const int> node_types)
{
    H2.resize(num_nodes*num_channels);
    for (int i=0; i<num_nodes; ++i) {
        auto H2_i = H2.data()+i*num_channels;
        auto H1_i = H1.data()+i*num_LM*num_channels;
        cblas_dgemv(
            CblasRowMajor,                            // const CBLAS_LAYOUT Layout
            CblasTrans,                               // const CBLAS_TRANSPOSE trans
            num_channels,                             // const MKL_INT m
            num_channels,                             // const MKL_INT n
            1.0,                                      // const double alpha
            H2_weights_for_H1[node_types[i]].data(),  // const double *a
            num_channels,                             // const MKL_INT lda
            H1_i,                                     // const double *x
            1,                                        // const MKL_INT incx
            0.0,                                      // const double beta
            H2_i,                                     // double *y
            1);                                       // const MKL_INT incy
        auto M1_i = M1.data()+i*num_channels;
        cblas_dgemv(
            CblasRowMajor,             // const CBLAS_LAYOUT Layout
            CblasTrans,                // const CBLAS_TRANSPOSE trans
            num_channels,              // const MKL_INT m
            num_channels,              // const MKL_INT n
            1.0,                       // const double alpha
            H2_weights_for_M1.data(),  // const double *a
            num_channels,              // const MKL_INT lda
            M1_i,                      // const double *x
            1,                         // const MKL_INT incx
            1.0,                       // const double beta
            H2_i,                      // double *y
            1);                        // const MKL_INT incy
    }
}

void MACE::reverse_H2(
    const int num_nodes,
    std::span<const int> node_types,
    bool zero_H1_adj)
{
    H1_adj.resize(H1.size());
    M1_adj.resize(M1.size());
    if (zero_H1_adj)
        std::fill(H1_adj.begin(), H1_adj.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto H2_adj_i = H2_adj.data()+i*num_channels;
        auto H1_adj_i = H1_adj.data()+i*num_LM*num_channels;
        cblas_dgemv(
            CblasRowMajor,                            // const CBLAS_LAYOUT Layout
            CblasNoTrans,                             // const CBLAS_TRANSPOSE trans
            num_channels,                             // const MKL_INT m
            num_channels,                             // const MKL_INT n
            1.0,                                      // const double alpha
            H2_weights_for_H1[node_types[i]].data(),  // const double *a
            num_channels,                             // const MKL_INT lda
            H2_adj_i,                                 // const double *x
            1,                                        // const MKL_INT incx
            1.0,                                      // const double beta
            H1_adj_i,                                 // double *y
            1);                                       // const MKL_INT incy
        auto M1_adj_i = M1_adj.data()+i*num_channels;
        cblas_dgemv(
            CblasRowMajor,             // const CBLAS_LAYOUT Layout
            CblasNoTrans,              // const CBLAS_TRANSPOSE trans
            num_channels,              // const MKL_INT m
            num_channels,              // const MKL_INT n
            1.0,                       // const double alpha
            H2_weights_for_M1.data(),  // const double *a
            num_channels,              // const MKL_INT lda
            H2_adj_i,                  // const double *x
            1,                         // const MKL_INT incx
            0.0,                       // const double beta
            M1_adj_i,                  // double *y
            1);                        // const MKL_INT incy
    }
}

void MACE::compute_readouts(
    const int num_nodes,
    std::span<const int> node_types)
{
    node_energies.resize(num_nodes);
    H1_adj.resize(H1.size());
    // Warning: Although it doesn't appear necessary to set H1_adj to zero,
    //          it matters when the number of nodes associated with H1 is greater than num_nodes.
    //          There is probably a better way to manage this.
    std::fill(H1_adj.begin(), H1_adj.end(), 0.0);
    H2_adj.resize(H2.size());
    for (int i=0; i<num_nodes; ++i) {
        // atomic energies
        node_energies[i] += atomic_energies[node_types[i]];
        // first readout
        for (int k=0; k<num_channels; ++k) {
            node_energies[i] += readout_1_weights[k]*H1[i*num_LM*num_channels+k];
            H1_adj[i*num_LM*num_channels+k] = readout_1_weights[k];
        }
        // second readout
        auto x = std::vector<double>(H2.begin()+i*num_channels, H2.begin()+(i+1)*num_channels);
        auto [f, g] = readout_2->evaluate_gradient(x);
        node_energies[i] += f[0];
        for (int k=0; k<num_channels; ++k) {
            H2_adj[i*num_channels+k] = g[k];
        }
    }
}

void MACE::apply_rrnlb_linear(
    const RRNLBLinear& linear,
    std::span<const double> x,
    std::span<double> y)
{
    if (static_cast<int>(x.size()) != linear.dim_in) {
        throw std::runtime_error("RRNLB linear input has unexpected dimension.");
    }
    if (static_cast<int>(y.size()) != linear.dim_out) {
        throw std::runtime_error("RRNLB linear output has unexpected dimension.");
    }
    std::fill(y.begin(), y.end(), 0.0);
    for (const auto& ins : linear.instructions) {
        if (ins.i_in < 0 || ins.i_in >= static_cast<int>(linear.parts_in.size())
            || ins.i_out < 0 || ins.i_out >= static_cast<int>(linear.parts_out.size())) {
            throw std::runtime_error("RRNLB linear instruction has invalid part index.");
        }
        const auto& in_part = linear.parts_in[ins.i_in];
        const auto& out_part = linear.parts_out[ins.i_out];
        if (in_part.l != out_part.l) {
            throw std::runtime_error("RRNLB linear instruction maps incompatible angular channels.");
        }
        if (ins.mul_in != in_part.mul || ins.mul_out != out_part.mul) {
            throw std::runtime_error("RRNLB linear instruction has unexpected multiplicity.");
        }
        const int ir_dim = 2 * in_part.l + 1;
        cblas_dgemm(
            CblasRowMajor,                  // const CBLAS_LAYOUT Layout
            CblasTrans,                     // const CBLAS_TRANSPOSE transa
            CblasNoTrans,                   // const CBLAS_TRANSPOSE transb
            ins.mul_out,                    // const MKL_INT m
            ir_dim,                         // const MKL_INT n
            ins.mul_in,                     // const MKL_INT k
            ins.path_weight,                // const double alpha
            ins.weights.data(),             // const double *a
            ins.mul_out,                    // const MKL_INT lda
            x.data() + in_part.offset,      // const double *b
            ir_dim,                         // const MKL_INT ldb
            1.0,                            // const double beta
            y.data() + out_part.offset,     // double *c
            ir_dim);                        // const MKL_INT ldc
    }
    if (!linear.bias.empty()) {
        int bias_offset = 0;
        for (const auto& out_part : linear.parts_out) {
            if (out_part.l != 0) continue;
            if (bias_offset + out_part.mul > static_cast<int>(linear.bias.size())) {
                throw std::runtime_error("RRNLB linear bias has unexpected size.");
            }
            for (int k = 0; k < out_part.mul; ++k) {
                y[out_part.offset + k] += linear.bias[bias_offset + k];
            }
            bias_offset += out_part.mul;
        }
        if (bias_offset != static_cast<int>(linear.bias.size())) {
            throw std::runtime_error("RRNLB linear bias has trailing values.");
        }
    }
}

void MACE::apply_rrnlb_linear_transpose(
    const RRNLBLinear& linear,
    std::span<const double> y_adj,
    std::span<double> x_adj)
{
    if (static_cast<int>(y_adj.size()) != linear.dim_out) {
        throw std::runtime_error("RRNLB linear adjoint output has unexpected dimension.");
    }
    if (static_cast<int>(x_adj.size()) != linear.dim_in) {
        throw std::runtime_error("RRNLB linear adjoint input has unexpected dimension.");
    }
    std::fill(x_adj.begin(), x_adj.end(), 0.0);
    for (const auto& ins : linear.instructions) {
        if (ins.i_in < 0 || ins.i_in >= static_cast<int>(linear.parts_in.size())
            || ins.i_out < 0 || ins.i_out >= static_cast<int>(linear.parts_out.size())) {
            throw std::runtime_error("RRNLB linear instruction has invalid part index.");
        }
        const auto& in_part = linear.parts_in[ins.i_in];
        const auto& out_part = linear.parts_out[ins.i_out];
        if (in_part.l != out_part.l) {
            throw std::runtime_error("RRNLB linear instruction maps incompatible angular channels.");
        }
        if (ins.mul_in != in_part.mul || ins.mul_out != out_part.mul) {
            throw std::runtime_error("RRNLB linear instruction has unexpected multiplicity.");
        }
        const int ir_dim = 2 * in_part.l + 1;
        cblas_dgemm(
            CblasRowMajor,                    // const CBLAS_LAYOUT Layout
            CblasNoTrans,                     // const CBLAS_TRANSPOSE transa
            CblasNoTrans,                     // const CBLAS_TRANSPOSE transb
            ins.mul_in,                       // const MKL_INT m
            ir_dim,                           // const MKL_INT n
            ins.mul_out,                      // const MKL_INT k
            ins.path_weight,                  // const double alpha
            ins.weights.data(),               // const double *a
            ins.mul_out,                      // const MKL_INT lda
            y_adj.data() + out_part.offset,   // const double *b
            ir_dim,                           // const MKL_INT ldb
            1.0,                              // const double beta
            x_adj.data() + in_part.offset,    // double *c
            ir_dim);                          // const MKL_INT ldc
    }
}

void MACE::apply_rrnlb_gate(
    const RRNLBLayer& layer,
    std::span<const double> x,
    std::span<double> y)
{
    if (layer.nonlin_parts.empty() || layer.target_parts.empty()) {
        throw std::runtime_error("RRNLB gate metadata is missing.");
    }
    const bool gate_identity_mode =
        rrnlb_nonlinear_ablation_mode() == RRNLBNonlinearAblationMode::GateIdentity;
    std::fill(y.begin(), y.end(), 0.0);
    const auto& nonlin_scalars = layer.nonlin_parts[0];
    const auto& out_scalars = layer.target_parts[0];
    if (nonlin_scalars.l != 0 || out_scalars.l != 0) {
        throw std::runtime_error("RRNLB gate scalar metadata is invalid.");
    }
    if (out_scalars.mul > nonlin_scalars.mul) {
        throw std::runtime_error("RRNLB gate has inconsistent scalar multiplicities.");
    }
    for (int k = 0; k < out_scalars.mul; ++k) {
        const double v = x[nonlin_scalars.offset + k];
        y[out_scalars.offset + k] = gate_identity_mode
            ? layer.gate_scalar_cst * v
            : layer.gate_scalar_cst * silu(v);
    }

    int gate_offset = nonlin_scalars.offset + out_scalars.mul;
    int gate_idx = 0;
    if (layer.nonlin_parts.size() != layer.target_parts.size()) {
        throw std::runtime_error("RRNLB gate irreps metadata size mismatch.");
    }
    for (size_t p = 1; p < layer.target_parts.size(); ++p) {
        const auto& out_part = layer.target_parts[p];
        const auto& in_part = layer.nonlin_parts[p];
        if (in_part.l != out_part.l || in_part.mul != out_part.mul) {
            throw std::runtime_error("RRNLB gate non-scalar metadata is inconsistent.");
        }
        const double cst = gate_idx < static_cast<int>(layer.gate_gate_cst.size())
            ? layer.gate_gate_cst[gate_idx]
            : 1.0;
        const int ir_dim = 2 * out_part.l + 1;
        for (int k = 0; k < out_part.mul; ++k) {
            const double g = gate_identity_mode ? cst : cst * sigmoid(x[gate_offset + k]);
            for (int m = 0; m < ir_dim; ++m) {
                y[out_part.offset + k * ir_dim + m] =
                    g * x[in_part.offset + k * ir_dim + m];
            }
        }
        gate_offset += out_part.mul;
        gate_idx += 1;
    }
}

void MACE::apply_rrnlb_gate_reverse(
    const RRNLBLayer& layer,
    std::span<const double> x,
    std::span<const double> y_adj,
    std::span<double> x_adj)
{
    if (static_cast<int>(x.size()) != layer.linear_1.dim_out) {
        throw std::runtime_error("RRNLB gate reverse input has unexpected dimension.");
    }
    if (static_cast<int>(y_adj.size()) != layer.linear_2.dim_in) {
        throw std::runtime_error("RRNLB gate reverse output adjoint has unexpected dimension.");
    }
    if (static_cast<int>(x_adj.size()) != layer.linear_1.dim_out) {
        throw std::runtime_error("RRNLB gate reverse input adjoint has unexpected dimension.");
    }
    if (layer.nonlin_parts.empty() || layer.target_parts.empty()) {
        throw std::runtime_error("RRNLB gate reverse metadata is missing.");
    }
    const bool gate_identity_mode =
        rrnlb_nonlinear_ablation_mode() == RRNLBNonlinearAblationMode::GateIdentity;

    std::fill(x_adj.begin(), x_adj.end(), 0.0);

    const auto& nonlin_scalars = layer.nonlin_parts[0];
    const auto& out_scalars = layer.target_parts[0];
    if (nonlin_scalars.l != 0 || out_scalars.l != 0) {
        throw std::runtime_error("RRNLB gate reverse scalar metadata is invalid.");
    }
    if (out_scalars.mul > nonlin_scalars.mul) {
        throw std::runtime_error("RRNLB gate reverse has inconsistent scalar multiplicities.");
    }
    for (int k = 0; k < out_scalars.mul; ++k) {
        x_adj[nonlin_scalars.offset + k] += y_adj[out_scalars.offset + k]
            * layer.gate_scalar_cst
            * (gate_identity_mode ? 1.0 : silu_deriv(x[nonlin_scalars.offset + k]));
    }

    int gate_offset = nonlin_scalars.offset + out_scalars.mul;
    int gate_idx = 0;
    if (layer.nonlin_parts.size() != layer.target_parts.size()) {
        throw std::runtime_error("RRNLB gate reverse irreps metadata size mismatch.");
    }
    for (size_t p = 1; p < layer.target_parts.size(); ++p) {
        const auto& out_part = layer.target_parts[p];
        const auto& in_part = layer.nonlin_parts[p];
        if (in_part.l != out_part.l || in_part.mul != out_part.mul) {
            throw std::runtime_error("RRNLB gate reverse non-scalar metadata is inconsistent.");
        }
        const double cst = gate_idx < static_cast<int>(layer.gate_gate_cst.size())
            ? layer.gate_gate_cst[gate_idx]
            : 1.0;
        const int ir_dim = 2 * out_part.l + 1;
        for (int k = 0; k < out_part.mul; ++k) {
            const double gate_x = x[gate_offset + k];
            const double sig = sigmoid(gate_x);
            const double g = gate_identity_mode ? cst : cst * sig;
            double d_g = 0.0;
            for (int m = 0; m < ir_dim; ++m) {
                const int out_idx = out_part.offset + k * ir_dim + m;
                const int in_idx = in_part.offset + k * ir_dim + m;
                x_adj[in_idx] += y_adj[out_idx] * g;
                if (!gate_identity_mode) {
                    d_g += y_adj[out_idx] * x[in_idx];
                }
            }
            if (!gate_identity_mode) {
                x_adj[gate_offset + k] += d_g * cst * sig * (1.0 - sig);
            }
        }
        gate_offset += out_part.mul;
        gate_idx += 1;
    }
}

void MACE::compute_rrnlb_interaction_layer_forward(
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
    RRNLBLayerCache* cache,
    int num_sender_nodes,
    std::span<const int> target_node_indices)
{
    auto& layer = rrnlb_layers[layer_index];
    const int in_dim = layer.linear_up.dim_in;
    const int sender_nodes = (num_sender_nodes > 0) ? num_sender_nodes : num_nodes;
    if (static_cast<int>(node_feats_in.size()) != sender_nodes * in_dim) {
        throw std::runtime_error("RRNLB layer input has unexpected size.");
    }
    if (!target_node_indices.empty() && static_cast<int>(target_node_indices.size()) != num_nodes) {
        throw std::runtime_error("RRNLB layer target index map has unexpected size.");
    }

    auto target_sender = [&](const int i) -> int {
        if (target_node_indices.empty()) return i;
        return target_node_indices[i];
    };

    std::vector<double> h_up_local;
    std::vector<double> h_res_local;
    std::vector<double> conv_accum_local;
    std::vector<double> density_local;
    std::vector<double> lin1_raw_local;
    std::vector<double> pre_gate_local;
    std::vector<double> gated_local;
    std::vector<double> tp_values_local;
    auto& h_up = (cache != nullptr) ? cache->h_up : h_up_local;
    auto& h_res = (cache != nullptr) ? cache->h_res : h_res_local;
    auto& conv_accum = (cache != nullptr) ? cache->conv_accum : conv_accum_local;
    auto& density = (cache != nullptr) ? cache->density : density_local;
    auto& lin1_raw = (cache != nullptr) ? cache->lin1_raw : lin1_raw_local;
    auto& pre_gate = (cache != nullptr) ? cache->pre_gate : pre_gate_local;
    auto& gated = (cache != nullptr) ? cache->gated : gated_local;
    auto& tp_values = (cache != nullptr) ? cache->tp_values : tp_values_local;

    h_up.resize(static_cast<size_t>(sender_nodes) * layer.linear_up.dim_out);
    h_res.resize(static_cast<size_t>(num_nodes) * layer.linear_res.dim_out);
    layer_skip.resize(num_nodes * layer.skip_tp.dim_out);
    for (int s = 0; s < sender_nodes; ++s) {
        auto x_s = std::span<const double>(node_feats_in.data() + s * in_dim, in_dim);
        auto up_s = std::span<double>(h_up.data() + static_cast<size_t>(s) * layer.linear_up.dim_out, layer.linear_up.dim_out);
        apply_rrnlb_linear(layer.linear_up, x_s, up_s);
    }
    for (int i = 0; i < num_nodes; ++i) {
        const int sender_i = target_sender(i);
        if (sender_i < 0 || sender_i >= sender_nodes) {
            throw std::runtime_error("RRNLB layer target index is out of bounds.");
        }
        auto x_i = std::span<const double>(node_feats_in.data() + sender_i * in_dim, in_dim);
        auto up_i = std::span<const double>(h_up.data() + static_cast<size_t>(sender_i) * layer.linear_up.dim_out, layer.linear_up.dim_out);
        auto res_i = std::span<double>(h_res.data() + static_cast<size_t>(i) * layer.linear_res.dim_out, layer.linear_res.dim_out);
        auto skip_i = std::span<double>(layer_skip.data() + i * layer.skip_tp.dim_out, layer.skip_tp.dim_out);
        apply_rrnlb_linear(layer.linear_res, up_i, res_i);
        apply_rrnlb_linear(layer.skip_tp, x_i, skip_i);
    }

    conv_accum.resize(static_cast<size_t>(num_nodes) * layer.linear_1.dim_in);
    density.resize(num_nodes);
    tp_values.resize(layer.tp_weight_numel);
    std::fill(conv_accum.begin(), conv_accum.end(), 0.0);
    std::fill(density.begin(), density.end(), 0.0);
    std::fill(tp_values.begin(), tp_values.end(), 0.0);

    int ij = 0;
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_neigh[i]; ++j) {
            const int sender = neigh_indices[ij];
            if (sender < 0 || sender >= sender_nodes) {
                throw std::runtime_error("RRNLB layer sender index is out of bounds.");
            }
            const int type_src = neigh_types[ij];
            const int type_tgt = node_types[i];
            const int pair_index = type_src * num_elements + type_tgt;
            if (pair_index < 0 || pair_index >= static_cast<int>(layer.tp_splines.size())) {
                throw std::runtime_error("RRNLB pair index is out of bounds.");
            }
            layer.tp_splines[pair_index]->evaluate(r[ij], tp_values);
            density[i] += layer.density_splines[pair_index].evaluate(r[ij]);
            const auto Y_ij = Y.data() + ij * num_lm;

            auto conv_i = conv_accum.data() + static_cast<size_t>(i) * layer.linear_1.dim_in;
            auto up_sender = h_up.data() + static_cast<size_t>(sender) * layer.linear_up.dim_out;
            for (const auto& ins : layer.conv_instructions) {
                const auto& in_part = layer.edge_parts[ins.i_in1];
                const auto& out_part = layer.linear_1.parts_in[ins.i_out];
                auto conv_i_out = conv_i + out_part.offset;
                auto up_sender_in = up_sender + in_part.offset;
                const int in_ir_dim = 2 * in_part.l + 1;
                const int out_ir_dim = 2 * out_part.l + 1;
                for (const auto& term : ins.terms) {
                    const double c = term.coeff * Y_ij[term.y_lm];
                    for (int k = 0; k < ins.mul; ++k) {
                        conv_i_out[k * out_ir_dim + term.m_out] += c
                            * tp_values[ins.weight_offset + k]
                            * up_sender_in[k * in_ir_dim + term.m_in1];
                    }
                }
            }
            ij += 1;
        }
    }

    lin1_raw.resize(static_cast<size_t>(num_nodes) * layer.linear_1.dim_out);
    pre_gate.resize(static_cast<size_t>(num_nodes) * layer.linear_1.dim_out);
    gated.resize(static_cast<size_t>(num_nodes) * layer.linear_2.dim_in);
    layer_output.resize(num_nodes * layer.linear_2.dim_out);
    for (int i = 0; i < num_nodes; ++i) {
        auto conv_i = std::span<const double>(
            conv_accum.data() + static_cast<size_t>(i) * layer.linear_1.dim_in, layer.linear_1.dim_in);
        auto lin1_raw_i = std::span<double>(
            lin1_raw.data() + static_cast<size_t>(i) * layer.linear_1.dim_out, layer.linear_1.dim_out);
        auto pre_gate_i = std::span<double>(
            pre_gate.data() + static_cast<size_t>(i) * layer.linear_1.dim_out, layer.linear_1.dim_out);
        auto res_i = std::span<const double>(
            h_res.data() + static_cast<size_t>(i) * layer.linear_res.dim_out, layer.linear_res.dim_out);
        auto gated_i = std::span<double>(
            gated.data() + static_cast<size_t>(i) * layer.linear_2.dim_in, layer.linear_2.dim_in);
        auto out_i = std::span<double>(
            layer_output.data() + static_cast<size_t>(i) * layer.linear_2.dim_out, layer.linear_2.dim_out);

        apply_rrnlb_linear(layer.linear_1, conv_i, lin1_raw_i);
        const double denom = density[i] * layer.beta + layer.alpha;
        for (int p = 0; p < layer.linear_1.dim_out; ++p) {
            pre_gate_i[p] = lin1_raw_i[p] / denom + res_i[p];
        }
        apply_rrnlb_gate(layer, pre_gate_i, gated_i);
        apply_rrnlb_linear(layer.linear_2, gated_i, out_i);
    }

    // When cache is provided, forward intermediates remain in cache-owned
    // vectors and are reused across timesteps.
}

void MACE::reverse_rrnlb_interaction_layer(
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
    int num_sender_nodes,
    std::span<const int> target_node_indices)
{
    auto& layer = rrnlb_layers[layer_index];
    const int in_dim = layer.linear_up.dim_in;
    const int sender_nodes = (num_sender_nodes > 0) ? num_sender_nodes : num_nodes;
    if (static_cast<int>(node_feats_in.size()) != sender_nodes * in_dim) {
        throw std::runtime_error("RRNLB reverse layer input has unexpected size.");
    }
    if (static_cast<int>(node_feats_in_adj.size()) != sender_nodes * in_dim) {
        throw std::runtime_error("RRNLB reverse layer input adjoint has unexpected size.");
    }
    if (!target_node_indices.empty() && static_cast<int>(target_node_indices.size()) != num_nodes) {
        throw std::runtime_error("RRNLB reverse layer target index map has unexpected size.");
    }
    if (static_cast<int>(layer_output_adj.size()) != num_nodes * layer.linear_2.dim_out) {
        throw std::runtime_error("RRNLB reverse layer output adjoint has unexpected size.");
    }
    if (static_cast<int>(layer_skip_adj.size()) != num_nodes * layer.skip_tp.dim_out) {
        throw std::runtime_error("RRNLB reverse layer skip adjoint has unexpected size.");
    }
    if (static_cast<int>(cache.h_up.size()) != sender_nodes * layer.linear_up.dim_out
        || static_cast<int>(cache.density.size()) != num_nodes
        || static_cast<int>(cache.lin1_raw.size()) != num_nodes * layer.linear_1.dim_out
        || static_cast<int>(cache.pre_gate.size()) != num_nodes * layer.linear_1.dim_out) {
        throw std::runtime_error("RRNLB reverse layer cache has unexpected size.");
    }

    auto target_sender = [&](const int i) -> int {
        if (target_node_indices.empty()) return i;
        return target_node_indices[i];
    };

    std::fill(node_feats_in_adj.begin(), node_feats_in_adj.end(), 0.0);

    auto& gated_adj = cache.gated_adj;
    auto& pre_gate_adj = cache.pre_gate_adj;
    auto& lin1_raw_adj = cache.lin1_raw_adj;
    auto& h_res_adj = cache.h_res_adj;
    auto& conv_adj = cache.conv_adj;
    auto& h_up_adj = cache.h_up_adj;
    auto& density_adj = cache.density_adj;
    gated_adj.assign(static_cast<size_t>(num_nodes) * layer.linear_2.dim_in, 0.0);
    pre_gate_adj.assign(static_cast<size_t>(num_nodes) * layer.linear_1.dim_out, 0.0);
    lin1_raw_adj.assign(static_cast<size_t>(num_nodes) * layer.linear_1.dim_out, 0.0);
    h_res_adj.assign(static_cast<size_t>(num_nodes) * layer.linear_res.dim_out, 0.0);
    conv_adj.assign(static_cast<size_t>(num_nodes) * layer.linear_1.dim_in, 0.0);
    h_up_adj.assign(static_cast<size_t>(sender_nodes) * layer.linear_up.dim_out, 0.0);
    density_adj.assign(num_nodes, 0.0);

    for (int i = 0; i < num_nodes; ++i) {
        auto out_adj_i = std::span<const double>(
            layer_output_adj.data() + i * layer.linear_2.dim_out, layer.linear_2.dim_out);
        auto gated_adj_i = std::span<double>(
            gated_adj.data() + i * layer.linear_2.dim_in, layer.linear_2.dim_in);
        apply_rrnlb_linear_transpose(layer.linear_2, out_adj_i, gated_adj_i);

        auto pre_gate_i = std::span<const double>(
            cache.pre_gate.data() + i * layer.linear_1.dim_out, layer.linear_1.dim_out);
        auto pre_gate_adj_i = std::span<double>(
            pre_gate_adj.data() + i * layer.linear_1.dim_out, layer.linear_1.dim_out);
        apply_rrnlb_gate_reverse(layer, pre_gate_i, gated_adj_i, pre_gate_adj_i);

        const auto lin1_raw_i = std::span<const double>(
            cache.lin1_raw.data() + i * layer.linear_1.dim_out, layer.linear_1.dim_out);
        auto lin1_raw_adj_i = std::span<double>(
            lin1_raw_adj.data() + i * layer.linear_1.dim_out, layer.linear_1.dim_out);
        auto h_res_adj_i = std::span<double>(
            h_res_adj.data() + i * layer.linear_res.dim_out, layer.linear_res.dim_out);
        const double denom = cache.density[i] * layer.beta + layer.alpha;
        for (int p = 0; p < layer.linear_1.dim_out; ++p) {
            lin1_raw_adj_i[p] += pre_gate_adj_i[p] / denom;
            h_res_adj_i[p] += pre_gate_adj_i[p];
            density_adj[i] +=
                pre_gate_adj_i[p] * (-layer.beta * lin1_raw_i[p] / (denom * denom));
        }
    }

    for (int i = 0; i < num_nodes; ++i) {
        const int sender_i = target_sender(i);
        if (sender_i < 0 || sender_i >= sender_nodes) {
            throw std::runtime_error("RRNLB reverse layer target index is out of bounds.");
        }
        auto h_res_adj_i = std::span<const double>(
            h_res_adj.data() + i * layer.linear_res.dim_out, layer.linear_res.dim_out);
        auto h_up_adj_i = std::span<double>(
            h_up_adj.data() + sender_i * layer.linear_up.dim_out, layer.linear_up.dim_out);
        apply_rrnlb_linear_transpose(layer.linear_res, h_res_adj_i, h_up_adj_i);
    }

    for (int i = 0; i < num_nodes; ++i) {
        auto lin1_raw_adj_i = std::span<const double>(
            lin1_raw_adj.data() + i * layer.linear_1.dim_out, layer.linear_1.dim_out);
        auto conv_adj_i = std::span<double>(
            conv_adj.data() + i * layer.linear_1.dim_in, layer.linear_1.dim_in);
        apply_rrnlb_linear_transpose(layer.linear_1, lin1_raw_adj_i, conv_adj_i);
    }

    auto& tp_values = cache.tp_values;
    auto& tp_derivs = cache.tp_derivs;
    auto& y_adj = cache.y_adj;
    tp_values.assign(layer.tp_weight_numel, 0.0);
    tp_derivs.assign(layer.tp_weight_numel, 0.0);
    y_adj.assign(num_lm, 0.0);
    int ij = 0;
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_neigh[i]; ++j) {
            const int sender = neigh_indices[ij];
            if (sender < 0 || sender >= sender_nodes) {
                throw std::runtime_error("RRNLB reverse layer sender index is out of bounds.");
            }
            const int type_src = neigh_types[ij];
            const int type_tgt = node_types[i];
            const int pair_index = type_src * num_elements + type_tgt;
            if (pair_index < 0 || pair_index >= static_cast<int>(layer.tp_splines.size())) {
                throw std::runtime_error("RRNLB reverse pair index is out of bounds.");
            }
            layer.tp_splines[pair_index]->evaluate_derivs(r[ij], tp_values, tp_derivs);
            auto [edge_density_value, edge_density_deriv] =
                layer.density_splines[pair_index].evaluate_deriv(r[ij]);
            (void)edge_density_value;
            std::fill(y_adj.begin(), y_adj.end(), 0.0);
            double dE_dr = density_adj[i] * edge_density_deriv;

            const auto* y_ij = Y.data() + ij * num_lm;
            auto* conv_adj_i = conv_adj.data() + i * layer.linear_1.dim_in;
            auto* h_up_sender_adj = h_up_adj.data() + sender * layer.linear_up.dim_out;
            const auto* h_up_sender = cache.h_up.data() + sender * layer.linear_up.dim_out;
            for (const auto& ins : layer.conv_instructions) {
                const auto& in_part = layer.edge_parts[ins.i_in1];
                const auto& out_part = layer.linear_1.parts_in[ins.i_out];
                auto* conv_adj_i_out = conv_adj_i + out_part.offset;
                auto* h_up_sender_adj_in = h_up_sender_adj + in_part.offset;
                const auto* h_up_sender_in = h_up_sender + in_part.offset;
                const int in_ir_dim = 2 * in_part.l + 1;
                const int out_ir_dim = 2 * out_part.l + 1;
                for (const auto& term : ins.terms) {
                    const int y_lm = term.y_lm;
                    const double y_lm_value = y_ij[y_lm];
                    for (int k = 0; k < ins.mul; ++k) {
                        const int out_idx = k * out_ir_dim + term.m_out;
                        const int in_idx = k * in_ir_dim + term.m_in1;
                        const int w_idx = ins.weight_offset + k;
                        const double upstream = conv_adj_i_out[out_idx];
                        const double tp_val = tp_values[w_idx];
                        const double up_val = h_up_sender_in[in_idx];
                        const double coeff = term.coeff;
                        const double coeff_upstream = upstream * coeff;
                        h_up_sender_adj_in[in_idx] += coeff_upstream * y_lm_value * tp_val;
                        y_adj[y_lm] += coeff_upstream * tp_val * up_val;
                        dE_dr += coeff_upstream * y_lm_value * up_val * tp_derivs[w_idx];
                    }
                }
            }

            const auto* xyz_ij = xyz.data() + 3 * ij;
            auto* node_forces_ij = node_forces.data() + 3 * ij;
            const auto* y_grad_ij = Y_grad.data() + ij * 3 * num_lm;
            double dE_dxyz_x = dE_dr * xyz_ij[0] / r[ij];
            double dE_dxyz_y = dE_dr * xyz_ij[1] / r[ij];
            double dE_dxyz_z = dE_dr * xyz_ij[2] / r[ij];
            for (int lm = 0; lm < num_lm; ++lm) {
                dE_dxyz_x += y_adj[lm] * y_grad_ij[lm];
                dE_dxyz_y += y_adj[lm] * y_grad_ij[num_lm + lm];
                dE_dxyz_z += y_adj[lm] * y_grad_ij[2 * num_lm + lm];
            }
            node_forces_ij[0] += -dE_dxyz_x;
            node_forces_ij[1] += -dE_dxyz_y;
            node_forces_ij[2] += -dE_dxyz_z;
            ij += 1;
        }
    }

    for (int i = 0; i < num_nodes; ++i) {
        const int sender_i = target_sender(i);
        if (sender_i < 0 || sender_i >= sender_nodes) {
            throw std::runtime_error("RRNLB reverse layer target index is out of bounds.");
        }
        auto skip_adj_i = std::span<const double>(
            layer_skip_adj.data() + i * layer.skip_tp.dim_out, layer.skip_tp.dim_out);
        auto x_adj_i = std::span<double>(
            node_feats_in_adj.data() + sender_i * in_dim, in_dim);
        apply_rrnlb_linear_transpose(layer.skip_tp, skip_adj_i, x_adj_i);
    }

    auto& x_up_adj_tmp = cache.x_up_adj_tmp;
    x_up_adj_tmp.resize(in_dim);
    for (int i = 0; i < sender_nodes; ++i) {
        auto h_up_adj_i = std::span<const double>(
            h_up_adj.data() + i * layer.linear_up.dim_out, layer.linear_up.dim_out);
        auto x_up_adj_i_span = std::span<double>(x_up_adj_tmp.data(), in_dim);
        apply_rrnlb_linear_transpose(layer.linear_up, h_up_adj_i, x_up_adj_i_span);
        auto x_adj_i = std::span<double>(
            node_feats_in_adj.data() + i * in_dim, in_dim);
        for (int p = 0; p < in_dim; ++p) {
            x_adj_i[p] += x_up_adj_tmp[p];
        }
    }
}

void MACE::compute_rrnlb_forward(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    if (rrnlb_layers.size() != 2) {
        throw std::runtime_error("RRNLB forward currently expects exactly 2 interaction layers.");
    }
    if (static_cast<int>(node_embedding_species_values.size()) != num_elements * num_channels) {
        throw std::runtime_error("RRNLB node embedding table has unexpected size.");
    }
    if (product_linear_0.dim_in != num_LM * num_channels) {
        throw std::runtime_error("RRNLB product linear 0 input dimension mismatch.");
    }
    if (product_linear_1.dim_in != num_channels) {
        throw std::runtime_error("RRNLB product linear 1 input dimension mismatch.");
    }
    if (product_linear_0.dim_out != rrnlb_layers[0].skip_tp.dim_out) {
        throw std::runtime_error("RRNLB product 0/skip0 dimensions are inconsistent.");
    }
    if (product_linear_1.dim_out != rrnlb_layers[1].skip_tp.dim_out) {
        throw std::runtime_error("RRNLB product 1/skip1 dimensions are inconsistent.");
    }
    if (static_cast<int>(readout_1_weights.size()) != num_channels) {
        throw std::runtime_error("RRNLB readout_1_weights has unexpected length.");
    }
    if (num_nodes == 0) {
        return;
    }

    rrnlb_node_embed_ws.resize(static_cast<size_t>(num_nodes) * num_channels);
    auto& node_embed = rrnlb_node_embed_ws;
    for (int i = 0; i < num_nodes; ++i) {
        const int t = node_types[i];
        const auto* src = node_embedding_species_values.data() + t * num_channels;
        std::copy(src, src + num_channels, node_embed.begin() + static_cast<size_t>(i) * num_channels);
    }

    auto& layer0_cache = rrnlb_forward_layer0_cache_ws;
    auto& layer1_cache = rrnlb_forward_layer1_cache_ws;

    auto& interaction0_out = rrnlb_interaction0_out_ws;
    auto& skip0 = rrnlb_skip0_ws;
    compute_rrnlb_interaction_layer_forward(
        0,
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        r,
        node_embed,
        interaction0_out,
        skip0,
        &layer0_cache);
    if (static_cast<int>(interaction0_out.size()) != num_nodes * rrnlb_layers[0].linear_2.dim_out) {
        throw std::runtime_error("RRNLB layer0 interaction output has unexpected size.");
    }
    compute_rrnlb_M0(
        num_nodes,
        node_types,
        interaction0_out,
        rrnlb_layers[0].linear_2.parts_out);

    rrnlb_node_feats_0.resize(num_nodes * product_linear_0.dim_out);
    for (int i = 0; i < num_nodes; ++i) {
        auto m0_i = std::span<const double>(M0.data() + i * product_linear_0.dim_in, product_linear_0.dim_in);
        auto feat_i = std::span<double>(
            rrnlb_node_feats_0.data() + i * product_linear_0.dim_out, product_linear_0.dim_out);
        apply_rrnlb_linear(product_linear_0, m0_i, feat_i);
        auto sc_i = skip0.data() + i * product_linear_0.dim_out;
        for (int j = 0; j < product_linear_0.dim_out; ++j) {
            feat_i[j] += sc_i[j];
        }
    }

    auto& interaction1_out = rrnlb_interaction1_out_ws;
    auto& skip1 = rrnlb_skip1_ws;
    compute_rrnlb_interaction_layer_forward(
        1,
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        r,
        rrnlb_node_feats_0,
        interaction1_out,
        skip1,
        &layer1_cache);
    if (static_cast<int>(interaction1_out.size()) != num_nodes * rrnlb_layers[1].linear_2.dim_out) {
        throw std::runtime_error("RRNLB layer1 interaction output has unexpected size.");
    }
    compute_rrnlb_M1(
        num_nodes,
        node_types,
        interaction1_out,
        rrnlb_layers[1].linear_2.parts_out);

    rrnlb_node_feats_1.resize(num_nodes * product_linear_1.dim_out);
    for (int i = 0; i < num_nodes; ++i) {
        auto m1_i = std::span<const double>(M1.data() + i * product_linear_1.dim_in, product_linear_1.dim_in);
        auto feat_i = std::span<double>(
            rrnlb_node_feats_1.data() + i * product_linear_1.dim_out, product_linear_1.dim_out);
        apply_rrnlb_linear(product_linear_1, m1_i, feat_i);
        auto sc_i = skip1.data() + i * product_linear_1.dim_out;
        for (int j = 0; j < product_linear_1.dim_out; ++j) {
            feat_i[j] += sc_i[j];
        }
    }

    rrnlb_feat0_adj_ws.assign(static_cast<size_t>(num_nodes) * product_linear_0.dim_out, 0.0);
    rrnlb_feat1_adj_ws.assign(static_cast<size_t>(num_nodes) * product_linear_1.dim_out, 0.0);
    auto& feat0_adj = rrnlb_feat0_adj_ws;
    auto& feat1_adj = rrnlb_feat1_adj_ws;
    rrnlb_mlp_input_ws.resize(product_linear_1.dim_out);
    auto& mlp_x = rrnlb_mlp_input_ws;
    for (int i = 0; i < num_nodes; ++i) {
        node_energies[i] += atomic_energies[node_types[i]];
        auto feat0_i = rrnlb_node_feats_0.data() + static_cast<size_t>(i) * product_linear_0.dim_out;
        for (int k = 0; k < num_channels; ++k) {
            node_energies[i] += readout_1_weights[k] * feat0_i[k];
            feat0_adj[static_cast<size_t>(i) * product_linear_0.dim_out + k] += readout_1_weights[k];
        }
        auto feat1_i = rrnlb_node_feats_1.data() + static_cast<size_t>(i) * product_linear_1.dim_out;
        std::copy(feat1_i, feat1_i + product_linear_1.dim_out, mlp_x.begin());
        auto [f, g] = readout_2->evaluate_gradient(mlp_x);
        node_energies[i] += f[0];
        for (int k = 0; k < product_linear_1.dim_out; ++k) {
            feat1_adj[static_cast<size_t>(i) * product_linear_1.dim_out + k] += g[k];
        }
    }

    rrnlb_skip1_adj_ws = feat1_adj;
    auto& skip1_adj = rrnlb_skip1_adj_ws;
    M1_adj.assign(num_nodes * product_linear_1.dim_in, 0.0);
    for (int i = 0; i < num_nodes; ++i) {
        auto feat1_adj_i = std::span<const double>(
            feat1_adj.data() + static_cast<size_t>(i) * product_linear_1.dim_out, product_linear_1.dim_out);
        auto m1_adj_i = std::span<double>(M1_adj.data() + static_cast<size_t>(i) * product_linear_1.dim_in, product_linear_1.dim_in);
        apply_rrnlb_linear_transpose(product_linear_1, feat1_adj_i, m1_adj_i);
    }
    rrnlb_interaction1_adj_ws.assign(
        static_cast<size_t>(num_nodes) * rrnlb_layers[1].linear_2.dim_out, 0.0);
    auto& interaction1_adj = rrnlb_interaction1_adj_ws;
    reverse_rrnlb_M1(
        num_nodes,
        node_types,
        rrnlb_layers[1].linear_2.parts_out,
        interaction1_adj);
    rrnlb_feat0_from_layer1_adj_ws.assign(rrnlb_node_feats_0.size(), 0.0);
    auto& feat0_from_layer1_adj = rrnlb_feat0_from_layer1_adj_ws;
    reverse_rrnlb_interaction_layer(
        1,
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz,
        r,
        rrnlb_node_feats_0,
        layer1_cache,
        interaction1_adj,
        skip1_adj,
        feat0_from_layer1_adj);
    for (int i = 0; i < static_cast<int>(feat0_adj.size()); ++i) {
        feat0_adj[i] += feat0_from_layer1_adj[i];
    }

    rrnlb_skip0_adj_ws = feat0_adj;
    auto& skip0_adj = rrnlb_skip0_adj_ws;
    M0_adj.assign(num_nodes * product_linear_0.dim_in, 0.0);
    for (int i = 0; i < num_nodes; ++i) {
        auto feat0_adj_i = std::span<const double>(
            feat0_adj.data() + static_cast<size_t>(i) * product_linear_0.dim_out, product_linear_0.dim_out);
        auto m0_adj_i = std::span<double>(M0_adj.data() + static_cast<size_t>(i) * product_linear_0.dim_in, product_linear_0.dim_in);
        apply_rrnlb_linear_transpose(product_linear_0, feat0_adj_i, m0_adj_i);
    }
    rrnlb_interaction0_adj_ws.assign(
        static_cast<size_t>(num_nodes) * rrnlb_layers[0].linear_2.dim_out, 0.0);
    auto& interaction0_adj = rrnlb_interaction0_adj_ws;
    reverse_rrnlb_M0(
        num_nodes,
        node_types,
        rrnlb_layers[0].linear_2.parts_out,
        interaction0_adj);
    rrnlb_node_embed_adj_ws.assign(node_embed.size(), 0.0);
    auto& node_embed_adj = rrnlb_node_embed_adj_ws;
    reverse_rrnlb_interaction_layer(
        0,
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz,
        r,
        node_embed,
        layer0_cache,
        interaction0_adj,
        skip0_adj,
        node_embed_adj);
}

void MACE::load_from_json(
    const std::string filename)
{
    std::ifstream f(filename);
    nlohmann::json file = nlohmann::json::parse(f);
    const std::string interaction_mode = file.value("interaction_mode", "legacy");
    interaction_mode_rrnlb = false;
    if (interaction_mode == "rrnlb") {
        interaction_mode_rrnlb = true;

        // Basic model information
        num_elements = file["num_elements"];
        num_channels = file["num_channels"];
        r_cut = file["r_cut"];
        l_max = file["l_max"];
        num_lm = (l_max + 1) * (l_max + 1);
        L_max = file["L_max"];
        num_LM = (L_max + 1) * (L_max + 1);
        atomic_numbers = file["atomic_numbers"].get<std::vector<int>>();
        atomic_energies = file["atomic_energies"].get<std::vector<double>>();

        // ZBL
        has_zbl = file["has_zbl"].get<bool>();
        if (has_zbl) {
            zbl = ZBL(
                file["zbl_a_exp"].get<double>(),
                file["zbl_a_prefactor"].get<double>(),
                file["zbl_c"].get<std::vector<double>>(),
                file["zbl_covalent_radii"].get<std::vector<double>>(),
                file["zbl_p"].get<int>());
        }

        if (!file.contains("node_embedding_species_values")) {
            throw std::runtime_error(
                "RRNLB JSON is missing 'node_embedding_species_values'.");
        }
        if (file["node_embedding_species_values"].size() != atomic_numbers.size()) {
            throw std::runtime_error(
                "RRNLB node embedding table size does not match number of elements.");
        }
        node_embedding_species_values.clear();
        for (const auto& v : file["node_embedding_species_values"]) {
            auto row = v.get<std::vector<double>>();
            if (static_cast<int>(row.size()) != num_channels) {
                throw std::runtime_error(
                    "RRNLB node embedding row has unexpected size.");
            }
            node_embedding_species_values.insert(
                node_embedding_species_values.end(), row.begin(), row.end());
        }

        if (!file.contains("product_linears")) {
            throw std::runtime_error("RRNLB JSON is missing 'product_linears'.");
        }
        product_linear_0 = parse_rrnlb_linear(file["product_linears"]["layer0"]);
        product_linear_1 = parse_rrnlb_linear(file["product_linears"]["layer1"]);

        if (!file.contains("interaction_layers") || !file["interaction_layers"].is_array()) {
            throw std::runtime_error(
                "RRNLB JSON is missing required 'interaction_layers' array.");
        }
        const auto& interaction_layers = file["interaction_layers"];
        if (interaction_layers.size() != 2) {
            throw std::runtime_error(
                "RRNLB JSON currently expects exactly 2 interaction layers.");
        }
        rrnlb_layers.clear();
        rrnlb_layers.reserve(interaction_layers.size());
        for (const auto& layer_json : interaction_layers) {
            RRNLBLayer layer;
            layer.alpha = layer_json.at("alpha").get<double>();
            layer.beta = layer_json.at("beta").get<double>();
            layer.avg_num_neighbors = layer_json.at("avg_num_neighbors").get<double>();
            layer.tp_weight_numel = layer_json.at("conv_tp").at("weight_numel").get<int>();

            layer.linear_up = parse_rrnlb_linear(layer_json.at("linears").at("linear_up"));
            layer.linear_res = parse_rrnlb_linear(layer_json.at("linears").at("linear_res"));
            layer.linear_1 = parse_rrnlb_linear(layer_json.at("linears").at("linear_1"));
            layer.linear_2 = parse_rrnlb_linear(layer_json.at("linears").at("linear_2"));
            layer.skip_tp = parse_rrnlb_linear(layer_json.at("linears").at("skip_tp"));
            layer.edge_parts = layer.linear_up.parts_out;
            layer.target_parts = layer.linear_2.parts_in;
            layer.nonlin_parts = layer.linear_1.parts_out;

            if (layer_json.at("gate").contains("scalar_activation_details")) {
                auto scalar_details = layer_json.at("gate").at("scalar_activation_details");
                if (!scalar_details.empty()) {
                    layer.gate_scalar_cst = scalar_details.at(0).at("cst").get<double>();
                }
            }
            if (layer_json.at("gate").contains("gate_activation_details")) {
                for (const auto& gate_act : layer_json.at("gate").at("gate_activation_details")) {
                    layer.gate_gate_cst.push_back(gate_act.at("cst").get<double>());
                }
            }

            if (!layer_json.at("conv_tp").contains("cg_maps")) {
                throw std::runtime_error("RRNLB JSON is missing conv_tp cg_maps.");
            }
            for (const auto& cg_map : layer_json.at("conv_tp").at("cg_maps")) {
                RRNLBConvInstruction ins;
                ins.i_in1 = cg_map.at("i_in1").get<int>();
                ins.i_in2 = cg_map.at("i_in2").get<int>();
                ins.i_out = cg_map.at("i_out").get<int>();
                ins.mul = cg_map.at("mul").get<int>();
                ins.weight_offset = cg_map.at("weight_offset").get<int>();
                ins.l_in1 = cg_map.at("l_in1").get<int>();
                ins.l_in2 = cg_map.at("l_in2").get<int>();
                ins.l_out = cg_map.at("l_out").get<int>();
                for (const auto& term_json : cg_map.at("terms")) {
                    RRNLBConvTerm term;
                    term.m_out = term_json.at("m_out").get<int>();
                    term.m_in1 = term_json.at("m_in1").get<int>();
                    term.y_lm = term_json.at("y_lm").get<int>();
                    term.coeff = term_json.at("coeff").get<double>();
                    ins.terms.push_back(term);
                }
                if (ins.weight_offset < 0
                    || ins.weight_offset + ins.mul > layer.tp_weight_numel) {
                    throw std::runtime_error("RRNLB conv instruction has invalid weight range.");
                }
                layer.conv_instructions.push_back(std::move(ins));
            }

            const auto& radial = layer_json.at("radial");
            const double spline_h = radial.at("spline_h").get<double>();
            auto tp_values = radial.at("tp_weights_values").get<std::vector<std::vector<std::vector<double>>>>();
            auto tp_derivs = radial.at("tp_weights_derivs").get<std::vector<std::vector<std::vector<double>>>>();
            if (tp_values.size() != tp_derivs.size()) {
                throw std::runtime_error("RRNLB radial tp spline values/derivs mismatch.");
            }
            for (size_t p = 0; p < tp_values.size(); ++p) {
                layer.tp_splines.push_back(
                    std::make_unique<CubicSplineSet>(spline_h, tp_values[p], tp_derivs[p]));
            }
            auto dens_values = radial.at("edge_density_values").get<std::vector<std::vector<double>>>();
            auto dens_derivs = radial.at("edge_density_derivs").get<std::vector<std::vector<double>>>();
            if (dens_values.size() != dens_derivs.size()) {
                throw std::runtime_error("RRNLB radial density spline values/derivs mismatch.");
            }
            for (size_t p = 0; p < dens_values.size(); ++p) {
                layer.density_splines.emplace_back(spline_h, dens_values[p], dens_derivs[p]);
            }
            if (layer.tp_splines.size() != static_cast<size_t>(num_elements * num_elements)
                || layer.density_splines.size() != static_cast<size_t>(num_elements * num_elements)) {
                throw std::runtime_error(
                    "RRNLB radial tables must contain ordered pairs for every element pair.");
            }
            rrnlb_layers.push_back(std::move(layer));
        }

        // M0
        auto M0_weights = file["M0_weights"].get<std::map<std::string,std::map<std::string,std::map<std::string,std::vector<double>>>>>();
        auto M0_monomials = file["M0_monomials"].get<std::map<std::string,std::vector<std::vector<int>>>>();
        P0 = std::vector<MultivariatePolynomial>();
        for (int a = 0; a < atomic_numbers.size(); ++a) {
            for (int lm = 0; lm < num_LM; ++lm) {
                for (int k = 0; k < num_channels; ++k) {
                    P0.push_back(MultivariatePolynomial(
                        num_lm,
                        M0_weights[std::to_string(a)][std::to_string(lm)][std::to_string(k)],
                        M0_monomials[std::to_string(lm)]));
                }
            }
        }

        // M1
        auto M1_weights = file["M1_weights"].get<std::map<std::string,std::map<std::string,std::vector<double>>>>();
        auto M1_monomials = file["M1_monomials"].get<std::vector<std::vector<int>>>();
        P1 = std::vector<MultivariatePolynomial>();
        for (int a = 0; a < atomic_numbers.size(); ++a) {
            for (int k = 0; k < num_channels; ++k) {
                P1.push_back(MultivariatePolynomial(
                    num_lm,
                    M1_weights[std::to_string(a)][std::to_string(k)],
                    M1_monomials));
            }
        }

        // Readouts
        readout_1_weights = file["readout_1_weights"].get<std::vector<double>>();
        auto readout_2_weights_1 = file["readout_2_weights_1"].get<std::vector<double>>();
        auto readout_2_weights_2 = file["readout_2_weights_2"].get<std::vector<double>>();
        readout_2 = std::make_unique<MultilayerPerceptron>(
            std::vector<int>{num_channels, 16, 1},
            std::vector<std::vector<double>>{readout_2_weights_1, readout_2_weights_2},
            file["readout_2_scale_factor"]);
        return;
    }
    if (interaction_mode != "legacy") {
        throw std::runtime_error(
            std::string("Unsupported interaction_mode '") + interaction_mode + "'.");
    }
    interaction_mode_rrnlb = false;

    // Basic model information
    num_elements = file["num_elements"];
    num_channels = file["num_channels"];
    r_cut = file["r_cut"];
    l_max = file["l_max"];
    num_lm = (l_max+1)*(l_max+1);
    L_max = file["L_max"];
    num_LM = (L_max+1)*(L_max+1);
    atomic_numbers = file["atomic_numbers"].get<std::vector<int>>();
    atomic_energies = file["atomic_energies"].get<std::vector<double>>();

    // ZBL
    has_zbl = file["has_zbl"].get<bool>();
    if (has_zbl)
        zbl = ZBL(
            file["zbl_a_exp"].get<double>(),
            file["zbl_a_prefactor"].get<double>(),
            file["zbl_c"].get<std::vector<double>>(),
            file["zbl_covalent_radii"].get<std::vector<double>>(),
            file["zbl_p"].get<int>());

    // Radial splines
    const double spl_h = file["radial_spline_h"];
    auto spl_values_0 = file["radial_spline_values_0"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_0 = file["radial_spline_derivs_0"].get<std::vector<std::vector<std::vector<double>>>>();
    for (int i=0; i<spl_values_0.size(); ++i)
        spl_set_0.push_back(std::make_unique<CubicSplineSet>(spl_h, spl_values_0[i], spl_derivs_0[i]));
    auto spl_values_1 = file["radial_spline_values_1"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_1 = file["radial_spline_derivs_1"].get<std::vector<std::vector<std::vector<double>>>>();
    for (int i=0; i<spl_values_1.size(); ++i)
        spl_set_1.push_back(std::make_unique<CubicSplineSet>(spl_h, spl_values_1[i], spl_derivs_1[i]));

    // H0
    H0_weights = file["H0_weights"].get<std::vector<double>>();

    // A0
    A0_weights = file["A0_weights"].get<std::vector<std::vector<std::vector<double>>>>();

    // A0 scaling
    A0_scaled = file["A0_scaled"].get<bool>();
    if (A0_scaled) {
        const double A0_spline_h = file["A0_spline_h"];
        auto A0_spline_values = file["A0_spline_values"].get<std::vector<std::vector<double>>>();
        auto A0_spline_derivs = file["A0_spline_derivs"].get<std::vector<std::vector<double>>>();
        for (int i=0; i<A0_spline_values.size(); ++i)
            A0_splines.push_back(CubicSpline(A0_spline_h, A0_spline_values[i], A0_spline_derivs[i]));
    }

    // M0
    auto M0_weights = file["M0_weights"].get<std::map<std::string,std::map<std::string,std::map<std::string,std::vector<double>>>>>();
    auto M0_monomials = file["M0_monomials"].get<std::map<std::string,std::vector<std::vector<int>>>>();
    P0 = std::vector<MultivariatePolynomial>();
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int lm=0; lm<num_LM; ++lm) {
            for (int k=0; k<num_channels; ++k) {
                P0.push_back(MultivariatePolynomial(
                    num_lm,
                    M0_weights[std::to_string(a)][std::to_string(lm)][std::to_string(k)],
                    M0_monomials[std::to_string(lm)]));
            }
        }
    }

    // H1
    H1_weights = file["H1_weights"].get<std::vector<double>>();

    // Phi1
    Phi1_l = file["Phi1_l"].get<std::vector<int>>();
    Phi1_l1 = file["Phi1_l1"].get<std::vector<int>>();
    Phi1_l2 = file["Phi1_l2"].get<std::vector<int>>();
    Phi1_lme = file["Phi1_lme"].get<std::vector<int>>();
    Phi1_clebsch_gordan = file["Phi1_clebsch_gordan"].get<std::vector<double>>();
    Phi1_lelm1lm2 = file["Phi1_lelm1lm2"].get<std::vector<int>>();
    num_lme = 0;
    for (auto l : Phi1_l)
        num_lme += 2*l+1;
    num_lelm1lm2 = 0;
    for (int le=0; le<Phi1_l.size(); ++le)
        num_lelm1lm2 += (2*Phi1_l1[le]+1)*(2*Phi1_l2[le]+1);

    // A1
    A1_weights = file["A1_weights"].get<std::vector<std::vector<double>>>();

    // A1 scaling
    A1_scaled = file["A1_scaled"].get<bool>();
    if (A1_scaled) {
        const double A1_spline_h = file["A1_spline_h"];
        auto A1_spline_values = file["A1_spline_values"].get<std::vector<std::vector<double>>>();
        auto A1_spline_derivs = file["A1_spline_derivs"].get<std::vector<std::vector<double>>>();
        for (int i=0; i<A1_spline_values.size(); ++i)
            A1_splines.push_back(CubicSpline(A1_spline_h, A1_spline_values[i], A1_spline_derivs[i]));
    }

    // M1
    auto M1_weights = file["M1_weights"].get<std::map<std::string,std::map<std::string,std::vector<double>>>>();
    auto M1_monomials = file["M1_monomials"].get<std::vector<std::vector<int>>>();
    P1 = std::vector<MultivariatePolynomial>();
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int k=0; k<num_channels; ++k) {
            P1.push_back(MultivariatePolynomial(
                num_lm,
                M1_weights[std::to_string(a)][std::to_string(k)],
                M1_monomials));
        }
    }

    // H2
    H2_weights_for_H1 = file["H2_weights_for_H1"].get<std::vector<std::vector<double>>>();
    H2_weights_for_M1 = file["H2_weights_for_M1"].get<std::vector<double>>();

    // Readouts
    // TODO! hardcoded 16
    readout_1_weights = file["readout_1_weights"].get<std::vector<double>>();
    auto readout_2_weights_1 = file["readout_2_weights_1"].get<std::vector<double>>();
    auto readout_2_weights_2 = file["readout_2_weights_2"].get<std::vector<double>>();
    readout_2 = std::make_unique<MultilayerPerceptron>(
        std::vector<int>{num_channels, 16, 1},
        std::vector<std::vector<double>>{readout_2_weights_1, readout_2_weights_2},
        file["readout_2_scale_factor"]);
}
