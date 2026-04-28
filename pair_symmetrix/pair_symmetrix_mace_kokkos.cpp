/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Contributing author: Chuck Witt

#include "pair_symmetrix_mace_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "kokkos_base.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "update.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

namespace {

auto resolve_rrnlb_phase_csv_path(std::string path, const int rank, const int nprocs)
    -> std::string
{
  std::size_t pos = 0;
  bool replaced_rank = false;
  while ((pos = path.find("%r", pos)) != std::string::npos) {
    path.replace(pos, 2, std::to_string(rank));
    pos += 1;
    replaced_rank = true;
  }
  if (!replaced_rank && nprocs > 1) {
    path += ".rank" + std::to_string(rank);
  }
  return path;
}

auto rrnlb_env_flag(const char* name, const bool default_value) -> bool
{
  const char* env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') return default_value;
  if (env[0] == '0') return false;
  if (env[0] == 'f' || env[0] == 'F') return false;
  if (env[0] == 'n' || env[0] == 'N') return false;
  return true;
}

auto rrnlb_debug_net_force_enabled() -> bool
{
  static const bool enabled = rrnlb_env_flag("SYMMETRIX_RRNLB_DEBUG_NET_FORCE", false);
  return enabled;
}

}  // namespace

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::PairSymmetrixMACEKokkos(LAMMPS *lmp)
  : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  comm_forward = 0;  // possibly changed below
  comm_reverse = 0;  // possibly changed below

  kokkosable = 1;
  reverse_comm_device = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  //host_flag = (execution_space == Host);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::~PairSymmetrixMACEKokkos()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memoryKK->destroy_kokkos(k_eatom,eatom);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::compute(int eflag, int vflag)
{
  if (mode == "no_domain_decomposition") {
    compute_no_domain_decomposition(eflag, vflag);
  } else if (mode == "mpi_message_passing") {
    compute_mpi_message_passing(eflag, vflag);
  } else if (mode == "no_mpi_message_passing") {
    compute_no_mpi_message_passing(eflag, vflag);
  }
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; ++i)
    for (int j=i; j<atom->ntypes+1; ++j)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::settings(int narg, char **arg)
{
  if (narg == 0) {
    mode = (comm->nprocs == 1) ? "no_domain_decomposition" : "mpi_message_passing";
  } else if (narg == 1) {
    mode = std::string(arg[0]);
    if (mode != "no_domain_decomposition" and mode != "mpi_message_passing" and mode != "no_mpi_message_passing")
      error->all(FLERR, "The command \'pair_style symmetrix/mace/kk {}\' is invalid", mode);
  } else {
    error->all(FLERR, "Too many pair_style arguments for symmetrix/mace/kk");
  }

  if (mode == "no_domain_decomposition" and comm->nprocs != 1)
    error->all(FLERR, "Cannot use no_domain_decomposition with multiple MPI processes");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  utils::logmesg(lmp, "Loading MACEKokkos model from \'{}\' ... ", arg[2]);
  mace = std::make_unique<MACEKokkos<Precision, AccumPrecision>>(arg[2]);
  utils::logmesg(lmp, "success\n");
  rrnlb_neighbor_epoch_id = -1;
  rrnlb_phase_step_counter = 0;
  rrnlb_phase_csv_header_written = false;
  rrnlb_phase_csv_path.clear();
  if (const char *csv_env = std::getenv("SYMMETRIX_RRNLB_PHASE_CSV");
      csv_env != nullptr && csv_env[0] != '\0') {
    rrnlb_phase_csv_path = resolve_rrnlb_phase_csv_path(csv_env, comm->me, comm->nprocs);
  }
  mace->rrnlb_set_phase_stats_enabled(!rrnlb_phase_csv_path.empty());

  // extract atomic numbers from pair_coeff
  // We need one mapping entry per LAMMPS atom type, not per model element.
  const int num_lammps_types = atom->ntypes;
  if (narg - 3 != num_lammps_types)
    error->all(FLERR, "Incorrect number of element mappings for symmetrix/mace/kk pair_coeff");

  mace_types = Kokkos::View<int*>("mace_types", num_lammps_types);
  auto h_mace_types = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mace_types);
  auto h_mace_atomic_numbers = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mace->atomic_numbers);
  for (int i=0; i<num_lammps_types; ++i)
    h_mace_types(i) = -1;
  for (int i=3; i<narg; ++i) {
    // find atomic number for element in arg[i]
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    // find mace index corresponding to this element
    int mace_index = -1;
    for (int j=0; j<mace->atomic_numbers.size(); ++j)
        if (h_mace_atomic_numbers(j) == atomic_number)
            mace_index = j;
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to MACEKokkos type {}\n",
                   i-2, arg[i], mace_index);
    h_mace_types(i-3) = mace_index;
  }
  Kokkos::deep_copy(mace_types, h_mace_types);

  // set message size
  if (mode == "mpi_message_passing") {
    if (mace->interaction_mode_rrnlb) {
      comm_forward = mace->rrnlb_product_linear_0.dim_out;
      comm_reverse = mace->rrnlb_product_linear_0.dim_out;
    } else {
      comm_forward = mace->num_LM*mace->num_channels;
      comm_reverse = mace->num_LM*mace->num_channels;
    }
  } else {
    comm_forward = 0;
    comm_reverse = 0;
  }

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
double PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return mace->r_cut;
}

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::rrnlb_maybe_emit_phase_stats(
  const char *mode_tag)
{
  if (!mace || !mace->interaction_mode_rrnlb) return;
  if (rrnlb_phase_csv_path.empty()) {
    mace->rrnlb_reset_phase_counters();
    return;
  }

  const auto counters = mace->rrnlb_take_phase_counters();
  rrnlb_phase_step_counter += 1;

  std::ofstream csv(rrnlb_phase_csv_path, std::ios::app);
  if (!csv.good()) return;
  if (!rrnlb_phase_csv_header_written) {
    csv << "step_idx,timestep,rank,mode"
        << ",forward_interaction_seconds,reverse_interaction_seconds"
        << ",linear_forward_seconds,linear_transpose_seconds"
        << ",comm_pack_seconds,comm_unpack_seconds,workspace_reset_seconds"
        << ",reverse_layer0_seconds,reverse_layer1_seconds"
        << ",reverse_h_up_zero_seconds,reverse_linear2_transpose_seconds"
        << ",reverse_gate_normalize_seconds,reverse_linear1_transpose_seconds"
        << ",reverse_linear_res_transpose_seconds,reverse_h_up_scatter_seconds"
        << ",reverse_conv_seconds,reverse_skip_transpose_seconds"
        << ",reverse_skip_scatter_seconds,reverse_linear_up_transpose_seconds"
        << ",reverse_input_adj_accum_seconds"
        << ",reverse_product0_transpose_seconds,reverse_product1_transpose_seconds"
        << ",reverse_m0_mixed_seconds,reverse_m1_mixed_seconds"
        << ",reverse_mpi_comm_seconds"
        << ",forward_interaction_calls,reverse_interaction_calls"
        << ",linear_forward_calls,linear_transpose_calls"
        << ",comm_pack_calls,comm_unpack_calls,workspace_reset_calls"
        << ",reverse_layer0_calls,reverse_layer1_calls"
        << ",reverse_h_up_zero_calls,reverse_linear2_transpose_calls"
        << ",reverse_gate_normalize_calls,reverse_linear1_transpose_calls"
        << ",reverse_linear_res_transpose_calls,reverse_h_up_scatter_calls"
        << ",reverse_conv_calls,reverse_skip_transpose_calls"
        << ",reverse_skip_scatter_calls,reverse_linear_up_transpose_calls"
        << ",reverse_input_adj_accum_calls"
        << ",reverse_product0_transpose_calls,reverse_product1_transpose_calls"
        << ",reverse_m0_mixed_calls,reverse_m1_mixed_calls"
        << ",reverse_mpi_comm_calls"
        << ",fused_forward_global_staged_calls,fused_reverse_global_staged_calls"
        << ",fused_forward_scratch_tiled_calls,fused_reverse_scratch_tiled_calls"
        << ",forward_adaptive_mode_auto_calls"
        << ",forward_adaptive_mode_force_fused_calls"
        << ",forward_adaptive_mode_force_split_calls"
        << ",forward_full_fused_calls,forward_split_calls"
        << ",forward_split_conv_stage_calls,forward_split_norm_gate_stage_calls\n";
    rrnlb_phase_csv_header_written = true;
  }

  csv << std::setprecision(17)
      << rrnlb_phase_step_counter << ','
      << update->ntimestep << ','
      << comm->me << ','
      << mode_tag << ','
      << counters.forward_interaction_seconds << ','
      << counters.reverse_interaction_seconds << ','
      << counters.linear_forward_seconds << ','
      << counters.linear_transpose_seconds << ','
      << counters.comm_pack_seconds << ','
      << counters.comm_unpack_seconds << ','
      << counters.workspace_reset_seconds << ','
      << counters.reverse_layer0_seconds << ','
      << counters.reverse_layer1_seconds << ','
      << counters.reverse_h_up_zero_seconds << ','
      << counters.reverse_linear2_transpose_seconds << ','
      << counters.reverse_gate_normalize_seconds << ','
      << counters.reverse_linear1_transpose_seconds << ','
      << counters.reverse_linear_res_transpose_seconds << ','
      << counters.reverse_h_up_scatter_seconds << ','
      << counters.reverse_conv_seconds << ','
      << counters.reverse_skip_transpose_seconds << ','
      << counters.reverse_skip_scatter_seconds << ','
      << counters.reverse_linear_up_transpose_seconds << ','
      << counters.reverse_input_adj_accum_seconds << ','
      << counters.reverse_product0_transpose_seconds << ','
      << counters.reverse_product1_transpose_seconds << ','
      << counters.reverse_m0_mixed_seconds << ','
      << counters.reverse_m1_mixed_seconds << ','
      << counters.reverse_mpi_comm_seconds << ','
      << counters.forward_interaction_calls << ','
      << counters.reverse_interaction_calls << ','
      << counters.linear_forward_calls << ','
      << counters.linear_transpose_calls << ','
      << counters.comm_pack_calls << ','
      << counters.comm_unpack_calls << ','
      << counters.workspace_reset_calls << ','
      << counters.reverse_layer0_calls << ','
      << counters.reverse_layer1_calls << ','
      << counters.reverse_h_up_zero_calls << ','
      << counters.reverse_linear2_transpose_calls << ','
      << counters.reverse_gate_normalize_calls << ','
      << counters.reverse_linear1_transpose_calls << ','
      << counters.reverse_linear_res_transpose_calls << ','
      << counters.reverse_h_up_scatter_calls << ','
      << counters.reverse_conv_calls << ','
      << counters.reverse_skip_transpose_calls << ','
      << counters.reverse_skip_scatter_calls << ','
      << counters.reverse_linear_up_transpose_calls << ','
      << counters.reverse_input_adj_accum_calls << ','
      << counters.reverse_product0_transpose_calls << ','
      << counters.reverse_product1_transpose_calls << ','
      << counters.reverse_m0_mixed_calls << ','
      << counters.reverse_m1_mixed_calls << ','
      << counters.reverse_mpi_comm_calls << ','
      << counters.fused_forward_global_staged_calls << ','
      << counters.fused_reverse_global_staged_calls << ','
      << counters.fused_forward_scratch_tiled_calls << ','
      << counters.fused_reverse_scratch_tiled_calls << ','
      << counters.forward_adaptive_mode_auto_calls << ','
      << counters.forward_adaptive_mode_force_fused_calls << ','
      << counters.forward_adaptive_mode_force_split_calls << ','
      << counters.forward_full_fused_calls << ','
      << counters.forward_split_calls << ','
      << counters.forward_split_conv_stage_calls << ','
      << counters.forward_split_norm_gate_stage_calls << '\n';
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::init_style()
{
  if (atom->map_user == atom->MAP_NONE) error->all(FLERR, "symmetrix/mace/kk requires \'atom_modify map [yes|array|hash]\'");
  if (force->newton_pair == 0) error->all(FLERR, "symmetrix/mace/kk requires newton pair on");

  if (mode == "no_domain_decomposition" or mode == "mpi_message_passing") {
    auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
    request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                             !std::is_same_v<DeviceType,LMPDeviceType>);
    request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  } else {
    // enforce the communication cutoff is more than twice the model cutoff
    const double comm_cutoff = comm->get_comm_cutoff();
    if (comm->get_comm_cutoff() < (2*mace->r_cut + neighbor->skin)) {
      std::string cutoff_val = std::to_string((2.0 * mace->r_cut) + neighbor->skin);
      char *args[2];
      args[0] = (char *)"cutoff";
      args[1] = const_cast<char *>(cutoff_val.c_str());
      comm->modify_params(2, args);
      if (comm->me == 0) error->warning(FLERR, "symmetrix/mace/kk is setting the communication cutoff to {}", cutoff_val);
    }
    auto request = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
    request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                             !std::is_same_v<DeviceType,LMPDeviceType>);
    request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
int PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  if (mace->interaction_mode_rrnlb && mode == "mpi_message_passing") {
    const int width = mace->rrnlb_product_linear_0.dim_out;
    auto h_feat0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rrnlb_feat0);
    for (int ii=0; ii<n; ++ii) {
      const int i = list[ii];
      for (int k=0; k<width; ++k) {
        buf[ii*width+k] = h_feat0(i,k);
      }
    }
    return n*width;
  }

  auto h_H1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1);
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<mace->num_channels; ++k) {
        buf[ii*mace->num_LM*mace->num_channels+LM*mace->num_channels+k] = h_H1(i,LM,k);
      }
    }
  }
  return n*mace->num_LM*mace->num_channels;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
int PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d &buf, int /*pbc_flag*/, int * /*pbc*/)
{
  const auto d_sendlist = k_sendlist.view<DeviceType>();
  auto d_buf = buf.view<DeviceType>();
  if (mace->interaction_mode_rrnlb && mode == "mpi_message_passing") {
    Kokkos::Timer comm_timer;
    const auto feat0 = rrnlb_feat0;
    const int width = mace->rrnlb_product_linear_0.dim_out;
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::pack_forward_comm_kokkos_rrnlb",
      Kokkos::RangePolicy<DeviceType>(0, n * width),
      KOKKOS_LAMBDA (const int iw) {
        const int ii = iw / width;
        const int k = iw % width;
        const int i = d_sendlist(ii);
        d_buf(ii * width + k) = feat0(i, k);
      });
    Kokkos::fence();
    mace->rrnlb_record_comm_pack(comm_timer.seconds());
    return n * width;
  } else {
    const auto H1 = this->H1;
    const auto num_channels = mace->num_channels;
    const auto num_LM = mace->num_LM;
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::pack_forward_comm_kokkos",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,num_channels}),
      KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
        const int i = d_sendlist(ii);
        d_buf(ii*num_LM*num_channels+LM*num_channels+k) = H1(i,LM,k);
      });
    Kokkos::fence();
    return n*num_LM*num_channels;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::unpack_forward_comm(int n, int first, double *buf)
{
  if (mace->interaction_mode_rrnlb && mode == "mpi_message_passing") {
    const int width = mace->rrnlb_product_linear_0.dim_out;
    auto h_feat0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rrnlb_feat0);
    for (int i=0; i<n; ++i) {
      for (int k=0; k<width; ++k) {
        h_feat0(first+i,k) = static_cast<Precision>(buf[i*width+k]);
      }
    }
    Kokkos::deep_copy(rrnlb_feat0, h_feat0);
    return;
  }

  auto h_H1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1);
  for (int i=0; i<n; ++i) {
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<mace->num_channels; ++k) {
        h_H1((first+i),LM,k) = buf[i*mace->num_LM*mace->num_channels+LM*mace->num_channels+k];
      }
    }
  }
  Kokkos::deep_copy(H1, h_H1);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::unpack_forward_comm_kokkos(int n, int first, DAT::tdual_double_1d &buf)
{
  const auto d_buf = buf.view<DeviceType>();
  if (mace->interaction_mode_rrnlb && mode == "mpi_message_passing") {
    Kokkos::Timer comm_timer;
    auto feat0 = rrnlb_feat0;
    const int width = mace->rrnlb_product_linear_0.dim_out;
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::unpack_forward_comm_kokkos_rrnlb",
      Kokkos::RangePolicy<DeviceType>(0, n * width),
      KOKKOS_LAMBDA (const int iw) {
        const int i = iw / width;
        const int k = iw % width;
        feat0(first + i, k) = d_buf(i * width + k);
      });
    Kokkos::fence();
    mace->rrnlb_record_comm_unpack(comm_timer.seconds());
  } else {
    auto H1 = this->H1;
    const auto num_channels = mace->num_channels;
    const auto num_LM = mace->num_LM;
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::unpack_forward_comm_kokkos",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,num_channels}),
      KOKKOS_LAMBDA (const int i, const int LM, const int k) {
        H1((first+i),LM,k) = d_buf(i*num_LM*num_channels+LM*num_channels+k);
      });
  }
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
int PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::pack_reverse_comm(int n, int first, double *buf)
{
  if (mace->interaction_mode_rrnlb && mode == "mpi_message_passing") {
    const int width = mace->rrnlb_product_linear_0.dim_out;
    auto h_feat0_adj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rrnlb_feat0_adj);
    for (int i=0; i<n; ++i) {
      for (int k=0; k<width; ++k) {
        buf[i*width+k] = h_feat0_adj(first+i,k);
      }
    }
    return n*width;
  }

  // TODO: for some reason this does not work as expected, causing problems
  //       for GPU simulations called with -pk kokkos comm/pair/reverse no
  auto h_H1_adj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1_adj);
  for (int i=0; i<n; ++i) {
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<mace->num_channels; ++k) {
        buf[i*mace->num_LM*mace->num_channels+LM*mace->num_channels+k] = h_H1_adj((first+i),LM,k);
      }
    }
  }
  return n*mace->num_LM*mace->num_channels;
}

/* ---------------------------------------------------------------------- */
template<class DeviceType, typename Precision, typename AccumPrecision>
int PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::pack_reverse_comm_kokkos(
    int n, int first, DAT::tdual_double_1d &buf)
{
  auto d_buf = buf.view<DeviceType>();
  if (mace->interaction_mode_rrnlb && mode == "mpi_message_passing") {
    Kokkos::Timer comm_timer;
    const auto feat0_adj = rrnlb_feat0_adj;
    const int width = mace->rrnlb_product_linear_0.dim_out;
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::pack_reverse_comm_kokkos_rrnlb",
      Kokkos::RangePolicy<DeviceType>(0, n * width),
      KOKKOS_LAMBDA (const int iw) {
        const int i = iw / width;
        const int k = iw % width;
        d_buf(i * width + k) = feat0_adj(first + i, k);
      });
    Kokkos::fence();
    mace->rrnlb_record_comm_pack(comm_timer.seconds());
    return n * width;
  } else {
    const auto H1_adj = this->H1_adj;
    const auto num_channels = mace->num_channels;
    const auto num_LM = mace->num_LM;
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::pack_reverse_comm_kokkos",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,num_channels}),
      KOKKOS_LAMBDA (const int i, const int LM, const int k) {
        d_buf(i*num_LM*num_channels+LM*num_channels+k) = H1_adj((first+i),LM,k);
      });
    Kokkos::fence();
    return n*num_LM*num_channels;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::unpack_reverse_comm(int n, int *list, double *buf)
{
  if (mace->interaction_mode_rrnlb && mode == "mpi_message_passing") {
    const int width = mace->rrnlb_product_linear_0.dim_out;
    auto h_feat0_adj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rrnlb_feat0_adj);
    for (int ii=0; ii<n; ++ii) {
      const int i = list[ii];
      for (int k=0; k<width; ++k) {
        h_feat0_adj(i,k) += static_cast<AccumPrecision>(buf[ii*width+k]);
      }
    }
    Kokkos::deep_copy(rrnlb_feat0_adj, h_feat0_adj);
    return;
  }

  auto h_H1_adj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1_adj);
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<mace->num_channels; ++k) {
        h_H1_adj(i,LM,k) += buf[ii*mace->num_LM*mace->num_channels+LM*mace->num_channels+k];
      }
    }
  }
  Kokkos::deep_copy(H1_adj, h_H1_adj);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::unpack_reverse_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d& buf)
{
  const auto d_sendlist = k_sendlist.view<DeviceType>();
  const auto d_buf = buf.view<DeviceType>();
  if (mace->interaction_mode_rrnlb && mode == "mpi_message_passing") {
    Kokkos::Timer comm_timer;
    auto feat0_adj = rrnlb_feat0_adj;
    auto feat0_adj_nodes = rrnlb_feat0_adj_nodes_comm;
    const auto atom_to_node = rrnlb_atom_to_node_ws;
    const bool compact_unpack = rrnlb_compact_reverse_unpack_active;
    const int nlocal = atom->nlocal;
    const int width = mace->rrnlb_product_linear_0.dim_out;
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::unpack_reverse_comm_kokkos_rrnlb",
      Kokkos::RangePolicy<DeviceType>(0, n * width),
      KOKKOS_LAMBDA (const int iw) {
        const int ii = iw / width;
        const int k = iw % width;
        const int i = d_sendlist(ii);
        const auto value = static_cast<AccumPrecision>(d_buf(ii * width + k));
        if (compact_unpack && i < nlocal) {
          const int node = atom_to_node(i);
          if (node >= 0) {
            Kokkos::atomic_add(&feat0_adj_nodes(node, k), value);
          }
        } else {
          Kokkos::atomic_add(&feat0_adj(i, k), value);
        }
      });
    Kokkos::fence();
    mace->rrnlb_record_comm_unpack(comm_timer.seconds());
  } else {
    auto H1_adj = this->H1_adj;
    const auto num_LM = mace->num_LM;
    const auto num_channels = mace->num_channels;
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::unpack_reverse_comm_kokkos",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,num_channels}),
      KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
        const int i = d_sendlist(ii);
        Kokkos::atomic_add(
          &H1_adj(i,LM,k),
          d_buf(ii*num_LM*num_channels+LM*num_channels+k));
      });
  }
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::compute_no_domain_decomposition(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom && k_eatom.view<DeviceType>().extent(0)<maxeatom) {
     memoryKK->destroy_kokkos(k_eatom,eatom);
     memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
  }

  const double r_cut_squared = mace->r_cut*mace->r_cut;

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto tag = atomKK->k_tag.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();

  auto map_style = atom->map_style;
  auto k_map_array = atomKK->k_map_array;
  auto k_map_hash = atomKK->k_map_hash;
  k_map_array.template sync<DeviceType>();

  // node_indices, node_types, and num_neigh
  const int num_nodes = k_list->inum;
  if (mace->interaction_mode_rrnlb) {
    if (rrnlb_neighbor_epoch_id < 0 || this->neighbor->ago == 0) rrnlb_neighbor_epoch_id += 1;
    mace->rrnlb_set_neighbor_epoch(rrnlb_neighbor_epoch_id);
  }
  if (node_indices.size() < num_nodes) Kokkos::realloc(node_indices, num_nodes);
  if (node_types.size() < num_nodes) Kokkos::realloc(node_types, num_nodes);
  if (num_neigh.size() < num_nodes) Kokkos::realloc(num_neigh, num_nodes);
  Kokkos::deep_copy(num_neigh, 0);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0,num_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0,num_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0,num_nodes));
  auto mace_types = this->mace_types;
  Kokkos::parallel_for("PairSymmetrixMACEKokkos::set_node_based_views",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      node_indices(ii) = i;
      node_types(ii) = mace_types(type(i)-1);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
        [&] (const int jj, int& num_neigh_ii) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x_i;
          const double dy = x(j,1) - y_i;
          const double dz = x(j,2) - z_i;
          const double r_squared = dx*dx + dy*dy + dz*dz;
          if (r_squared < r_cut_squared) {
            num_neigh_ii += 1;
          }
        }, num_neigh(ii));
    });

  // count edges
  int num_edges;
  Kokkos::parallel_reduce("PairSymmetrixMACEKokkos::count_edges",
    num_nodes,
    KOKKOS_LAMBDA (const int ii, int& num_edges) {
      num_edges += num_neigh(ii);
    }, num_edges);

  // first neighbor
  if (first_neigh.size() < num_nodes) Kokkos::realloc(first_neigh, num_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0,num_nodes));
  Kokkos::parallel_scan("PairSymmetrixMACEKokkos::populate_first_neigh",
      num_nodes,
      KOKKOS_LAMBDA (const int ii, int& first_neigh_ii, const bool final) {
          if (final) first_neigh(ii) = first_neigh_ii;
          first_neigh_ii += num_neigh(ii);
      });
  int rrnlb_total_edges_arg = -1;
  Kokkos::View<const int*> rrnlb_edge_to_receiver_arg;
  if (mace->interaction_mode_rrnlb) {
    rrnlb_total_edges_arg = num_edges;
    if (rrnlb_edge_to_receiver.size() < num_edges) {
      Kokkos::realloc(rrnlb_edge_to_receiver, num_edges);
    }
    auto edge_to_receiver =
      Kokkos::subview(this->rrnlb_edge_to_receiver, Kokkos::make_pair(0, num_edges));
    Kokkos::parallel_for(
      "PairSymmetrixMACEKokkos::populate_edge_to_receiver",
      Kokkos::RangePolicy<DeviceType>(0, num_nodes),
      KOKKOS_LAMBDA (const int ii) {
        const int ij0 = first_neigh(ii);
        const int n = num_neigh(ii);
        for (int jj = 0; jj < n; ++jj) {
          edge_to_receiver(ij0 + jj) = ii;
        }
      });
    rrnlb_edge_to_receiver_arg = edge_to_receiver;
  }

  // neigh_indices, neigh_types, xyz, and r
  if (neigh_indices.size() < num_edges) Kokkos::realloc(neigh_indices, num_edges);
  if (neigh_types.size() < num_edges) Kokkos::realloc(neigh_types, num_edges);
  if (xyz.size() < 3*num_edges) Kokkos::realloc(xyz, 3*num_edges);
  if (r.size() < num_edges) Kokkos::realloc(r, num_edges);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,num_edges));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,num_edges));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0,3*num_edges));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0,num_edges));
  // TODO: better parallelization?
  Kokkos::parallel_for("PairSymmetrixMACEKokkos::set_edge_based_views",
    num_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = d_ilist(ii);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      int ij = first_neigh(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        const int j = (d_neighbors(i,jj) & NEIGHMASK);
        const int j_local = AtomKokkos::map_kokkos<DeviceType>(tag(j),map_style,k_map_array,k_map_hash);
        const double dx = x(j,0) - x_i;
        const double dy = x(j,1) - y_i;
        const double dz = x(j,2) - z_i;
        const double r_squared = dx*dx + dy*dy + dz*dz;
        if (r_squared < r_cut_squared) {
          neigh_indices(ij) = j_local;
          neigh_types(ij) = mace_types(type(j)-1);
          xyz(3*ij) = dx;
          xyz(3*ij+1) = dy;
          xyz(3*ij+2) = dz;
          r(ij) = std::sqrt(r_squared);
          ij += 1;
        }
      }
  });

  mace->compute_node_energies_forces(
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r,
    first_neigh, rrnlb_total_edges_arg, rrnlb_edge_to_receiver_arg);

  if (eflag_global) {
    auto node_energies = mace->node_energies;
    double energy;
    Kokkos::parallel_reduce("PairSymmetrixMACEKokkos::energy_reduction",
      num_nodes,
      KOKKOS_LAMBDA (const int i, double& energy) {
        energy += node_energies(i);
      }, energy);
    eng_vdwl += energy;
  }

  if (eflag_atom) {
    auto d_eatom = k_eatom.template view<DeviceType>();
    auto node_energies = mace->node_energies;
    Kokkos::parallel_for("PairSymmetrixMACEKokkos::extract_atomic_energies", num_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(ii) += node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
  }

  auto mace_node_forces = mace->node_forces;
  Kokkos::parallel_for("PairSymmetrixMACEKokkos::force_reduction",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = node_indices(ii);
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
        [&] (const int jj) {
          const int ij = first_neigh(ii) + jj;
          const int j = neigh_indices(ij);
          const double fx = mace_node_forces(3*ij);
          const double fy = mace_node_forces(3*ij+1);
          const double fz = mace_node_forces(3*ij+2);
          Kokkos::atomic_add(&f(j,0), fx);
          Kokkos::atomic_add(&f(j,1), fy);
          Kokkos::atomic_add(&f(j,2), fz);
          Kokkos::atomic_add(&f(i,0), -fx);
          Kokkos::atomic_add(&f(i,1), -fy);
          Kokkos::atomic_add(&f(i,2), -fz);
        });
    });

  if (rrnlb_debug_net_force_enabled()) {
    double local_fx = 0.0;
    double local_fy = 0.0;
    double local_fz = 0.0;
    Kokkos::parallel_reduce(
      "rrnlb_debug_net_force_fx_no_domain",
      Kokkos::RangePolicy<DeviceType>(0, atom->nlocal),
      KOKKOS_LAMBDA (const int i, double& sum) {
        sum += f(i, 0);
      },
      local_fx);
    Kokkos::parallel_reduce(
      "rrnlb_debug_net_force_fy_no_domain",
      Kokkos::RangePolicy<DeviceType>(0, atom->nlocal),
      KOKKOS_LAMBDA (const int i, double& sum) {
        sum += f(i, 1);
      },
      local_fy);
    Kokkos::parallel_reduce(
      "rrnlb_debug_net_force_fz_no_domain",
      Kokkos::RangePolicy<DeviceType>(0, atom->nlocal),
      KOKKOS_LAMBDA (const int i, double& sum) {
        sum += f(i, 2);
      },
      local_fz);
    double net_force_local[3] = {local_fx, local_fy, local_fz};
    double net_force_global[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(net_force_local, net_force_global, 3, MPI_DOUBLE, MPI_SUM, world);
    if (comm->me == 0) {
      const double net_norm = std::sqrt(
        net_force_global[0] * net_force_global[0]
        + net_force_global[1] * net_force_global[1]
        + net_force_global[2] * net_force_global[2]);
      utils::logmesg(
        lmp,
        "[RRNLB debug net force] mode=no_domain_decomposition step={} fx={:.12e} fy={:.12e} "
        "fz={:.12e} norm={:.12e}\n",
        update->ntimestep,
        net_force_global[0],
        net_force_global[1],
        net_force_global[2],
        net_norm);
    }
  }

  if (vflag_global) {
    Kokkos::View<double*,Kokkos::LayoutRight> v("v", 6);
    Kokkos::deep_copy(v, 0.0);
    Kokkos::parallel_for("PairSymmetrixMACEKokkos::virial_reduction",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
          [&] (const int jj) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = mace_node_forces(3*ij);
            const double f_y = mace_node_forces(3*ij+1);
            const double f_z = mace_node_forces(3*ij+2);
            Kokkos::atomic_add(&v(0), x*f_x);
            Kokkos::atomic_add(&v(1), y*f_y);
            Kokkos::atomic_add(&v(2), z*f_z);
            Kokkos::atomic_add(&v(3), 0.5*(x*f_y + y*f_x));
            Kokkos::atomic_add(&v(4), 0.5*(x*f_z + z*f_x));
            Kokkos::atomic_add(&v(5), 0.5*(y*f_z + z*f_y));
          });
      });
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    virial[0] += h_v(0);
    virial[1] += h_v(1);
    virial[2] += h_v(2);
    virial[3] += h_v(3);
    virial[4] += h_v(4);
    virial[5] += h_v(5);
  }

  rrnlb_maybe_emit_phase_stats("no_domain_decomposition");

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/kk.");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::compute_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom && k_eatom.view<DeviceType>().extent(0)<maxeatom) {
     memoryKK->destroy_kokkos(k_eatom,eatom);
     memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
  }

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();

  // node_indices, node_types, and num_neigh
  const double r_cut_squared = mace->r_cut*mace->r_cut;
  const int num_nodes = k_list->inum;
  if (mace->interaction_mode_rrnlb) {
    if (rrnlb_neighbor_epoch_id < 0 || this->neighbor->ago == 0) rrnlb_neighbor_epoch_id += 1;
    mace->rrnlb_set_neighbor_epoch(rrnlb_neighbor_epoch_id);
  }
  const bool rrnlb_phase_timing = mace->rrnlb_phase_stats_enabled();
  auto run_rrnlb_phase_stage = [&] (auto&& body, auto&& recorder) {
    if (rrnlb_phase_timing) {
      Kokkos::fence();
      Kokkos::Timer timer;
      body();
      Kokkos::fence();
      recorder(timer.seconds());
    } else {
      body();
    }
  };
  if (node_indices.size() < num_nodes) Kokkos::realloc(node_indices, num_nodes);
  if (node_types.size() < num_nodes) Kokkos::realloc(node_types, num_nodes);
  if (num_neigh.size() < num_nodes) Kokkos::realloc(num_neigh, num_nodes);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0,num_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0,num_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0,num_nodes));
  auto mace_types = this->mace_types;
  Kokkos::deep_copy(num_neigh, 0);
  Kokkos::parallel_for("Set Node-Based Views",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      node_indices(ii) = i;
      node_types(ii) = mace_types(type(i)-1);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
        [&] (const int jj, int& num_neigh_ii) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x_i;
          const double dy = x(j,1) - y_i;
          const double dz = x(j,2) - z_i;
          const double r_squared = dx*dx + dy*dy + dz*dz;
          if (r_squared < r_cut_squared) {
            num_neigh_ii += 1;
          }
        }, num_neigh(ii));
    });

  // count edges
  int num_edges;
  Kokkos::parallel_reduce("Count Neighbors",
    num_nodes,
    KOKKOS_LAMBDA (const int ii, int& num_edges) {
      num_edges += num_neigh(ii);
    }, num_edges);

  // High-water edge-parallel policy is default-on for pair/MPI RRNLB path.
  // SYMMETRIX_RRNLB_PAIR_EDGE_PARALLEL acts as the shared default for
  // forward+reverse, and each direction can still be overridden explicitly.
  static const bool rrnlb_pair_edge_parallel_default =
    rrnlb_env_flag("SYMMETRIX_RRNLB_PAIR_EDGE_PARALLEL", true);
  static const bool rrnlb_pair_edge_parallel_fwd =
    rrnlb_env_flag(
      "SYMMETRIX_RRNLB_PAIR_EDGE_PARALLEL_FWD",
      rrnlb_pair_edge_parallel_default);
  static const bool rrnlb_pair_edge_parallel_rev =
    rrnlb_env_flag(
      "SYMMETRIX_RRNLB_PAIR_EDGE_PARALLEL_REV",
      rrnlb_pair_edge_parallel_default);
  static const bool rrnlb_compact_reverse_unpack =
    rrnlb_env_flag("SYMMETRIX_RRNLB_COMPACT_REVERSE_UNPACK", true);
  const bool edge_parallel_active = rrnlb_pair_edge_parallel_fwd || rrnlb_pair_edge_parallel_rev;
  const bool force_refresh_topology = edge_parallel_active;
  const int rrnlb_total_edges_fwd = rrnlb_pair_edge_parallel_fwd ? num_edges : 0;
  const int rrnlb_total_edges_rev = rrnlb_pair_edge_parallel_rev ? num_edges : 0;
  const int rrnlb_sender_nodes = atom->nlocal + atom->nghost;
  Kokkos::View<const int*> rrnlb_edge_to_receiver_arg;
  Kokkos::View<const int*> rrnlb_sender_edge_offsets_arg;
  Kokkos::View<const int*> rrnlb_sender_edge_indices_arg;
  Kokkos::View<const int*> rrnlb_sender_segment_offsets_arg;
  Kokkos::View<const int*> rrnlb_sender_segment_to_sender_arg;
  int rrnlb_total_sender_segments_arg = -1;

  // first neighbor
  if (first_neigh.size() < num_nodes) Kokkos::realloc(first_neigh, num_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0,num_nodes));
  Kokkos::parallel_scan("Set First Neighbor",
    num_nodes,
    KOKKOS_LAMBDA (const int ii, int& first_neigh_ii, const bool final) {
        if (final) first_neigh(ii) = first_neigh_ii;
        first_neigh_ii += num_neigh(ii);
    });
  // neigh_indices, neigh_types, xyz, and r
  if (neigh_indices.size() < num_edges) Kokkos::realloc(neigh_indices, num_edges);
  if (neigh_types.size() < num_edges) Kokkos::realloc(neigh_types, num_edges);
  if (xyz.size() < 3*num_edges) Kokkos::realloc(xyz, 3*num_edges);
  if (r.size() < num_edges) Kokkos::realloc(r, num_edges);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,num_edges));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,num_edges));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0,3*num_edges));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0,num_edges));
  Kokkos::parallel_for("Set Edge-Based Views",
    num_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = d_ilist(ii);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      int ij = first_neigh(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        const int j = (d_neighbors(i,jj) & NEIGHMASK);
        const double dx = x(j,0) - x_i;
        const double dy = x(j,1) - y_i;
        const double dz = x(j,2) - z_i;
        const double r_squared = dx*dx + dy*dy + dz*dz;
        if (r_squared < r_cut_squared) {
          neigh_indices(ij) = j;
          neigh_types(ij) = mace_types(type(j)-1);
          xyz(3*ij) = dx;
          xyz(3*ij+1) = dy;
          xyz(3*ij+2) = dz;
          r(ij) = std::sqrt(r_squared);
          ij += 1;
        }
      }
    });

  if (mace->interaction_mode_rrnlb) {
    mace->ensure_rrnlb_model_static_cache();
    mace->ensure_rrnlb_system_static_cache(num_nodes, node_types);
    mace->ensure_rrnlb_epoch_topology_cache(
      num_nodes,
      num_edges,
      num_neigh,
      neigh_indices,
      first_neigh,
      Kokkos::View<const int*>(),
      rrnlb_pair_edge_parallel_rev,
      rrnlb_sender_nodes,
      force_refresh_topology);
    if ((rrnlb_pair_edge_parallel_fwd || rrnlb_pair_edge_parallel_rev) && num_edges > 0) {
      rrnlb_edge_to_receiver_arg = Kokkos::subview(
        mace->rrnlb_epoch_topology_cache.edge_to_receiver,
        Kokkos::make_pair(0, num_edges));
    }
    if (rrnlb_pair_edge_parallel_rev && num_edges > 0) {
      rrnlb_sender_edge_offsets_arg = Kokkos::subview(
        mace->rrnlb_epoch_topology_cache.sender_edge_offsets,
        Kokkos::make_pair(0, rrnlb_sender_nodes + 1));
      rrnlb_sender_edge_indices_arg = Kokkos::subview(
        mace->rrnlb_epoch_topology_cache.sender_edge_indices,
        Kokkos::make_pair(0, num_edges));
      rrnlb_sender_segment_offsets_arg = Kokkos::subview(
        mace->rrnlb_epoch_topology_cache.sender_segment_offsets,
        Kokkos::make_pair(0, rrnlb_sender_nodes + 1));
      rrnlb_total_sender_segments_arg = mace->rrnlb_epoch_topology_cache.total_sender_segments;
      if (rrnlb_total_sender_segments_arg > 0) {
        rrnlb_sender_segment_to_sender_arg = Kokkos::subview(
          mace->rrnlb_epoch_topology_cache.sender_segment_to_sender,
          Kokkos::make_pair(0, rrnlb_total_sender_segments_arg));
      }
    }
  }

  if (mace->node_energies.size() < num_nodes) Kokkos::realloc(mace->node_energies, num_nodes);
  if (mace->node_forces.size() < 3*num_edges) Kokkos::realloc(mace->node_forces, 3*num_edges);
  Kokkos::deep_copy(
    Kokkos::subview(mace->node_energies, Kokkos::make_pair(0, num_nodes)),
    0.0);
  Kokkos::deep_copy(
    Kokkos::subview(mace->node_forces, Kokkos::make_pair(0, 3*num_edges)),
    0.0);

  if (mace->has_zbl)
    mace->zbl.compute_ZBL(
      num_nodes, node_types, num_neigh, neigh_types,
      mace->atomic_numbers, r, xyz, mace->node_energies, mace->node_forces);

  mace->compute_Y(xyz);

  if (mace->interaction_mode_rrnlb) {
    if (mace->rrnlb_layers_kokkos.size() != 2) {
      error->all(FLERR, "RRNLB mpi_message_passing currently expects exactly two interaction layers.");
    }

    const int sender_nodes = atom->nlocal + atom->nghost;
    const int feat0_dim = mace->rrnlb_product_linear_0.dim_out;
    const int feat1_dim = mace->rrnlb_product_linear_1.dim_out;
    const int product0_dim_in = mace->rrnlb_product_linear_0.dim_in;
    const int product1_dim_in = mace->rrnlb_product_linear_1.dim_in;
    const int num_channels = mace->num_channels;
    const int num_lm = mace->num_lm;
    const int num_LM = mace->num_LM;
    mace->ensure_rrnlb_scratch_capacity(num_nodes, num_edges, sender_nodes);
    const int ws_nodes = std::max(mace->rrnlb_scratch_cache.max_nodes, num_nodes);
    const int ws_sender_nodes = std::max(mace->rrnlb_scratch_cache.max_sender_nodes, sender_nodes);

    auto ensure_ws_2d = [](auto& view, const int d0, const int d1) -> bool {
      const bool need_resize =
        view.extent(0) < static_cast<std::size_t>(d0)
        || view.extent(1) < static_cast<std::size_t>(d1);
      if (need_resize) {
        Kokkos::realloc(view, d0, d1);
      }
      return need_resize;
    };
    auto ensure_ws_1d = [](auto& view, const int d0) {
      if (view.extent(0) < static_cast<std::size_t>(d0)) {
        Kokkos::realloc(view, d0);
      }
    };
    auto ensure_exact_1d = [](auto& view, const int d0) -> bool {
      const bool need_resize = view.extent(0) != static_cast<std::size_t>(d0);
      if (need_resize) Kokkos::realloc(view, d0);
      return need_resize;
    };

    ensure_ws_2d(rrnlb_sender_embed_ws, ws_sender_nodes, num_channels);
    auto sender_embed = rrnlb_sender_embed_ws;
    const auto rrnlb_node_embedding = mace->rrnlb_node_embedding;
    Kokkos::parallel_for(
      "RRNLB sender embedding",
      Kokkos::RangePolicy<DeviceType>(0, sender_nodes * num_channels),
      KOKKOS_LAMBDA (const int ik) {
        const int i = ik / num_channels;
        const int k = ik % num_channels;
        const int t = mace_types(type(i) - 1);
        sender_embed(i, k) = rrnlb_node_embedding(t, k);
      });
    const auto& layer0 = mace->rrnlb_layers_kokkos[0];
    const auto& layer1 = mace->rrnlb_layers_kokkos[1];

    ensure_ws_2d(rrnlb_interaction0_out_ws, ws_nodes, layer0.linear_2.dim_out);
    ensure_ws_2d(rrnlb_skip0_ws, ws_nodes, layer0.skip_tp.dim_out);
    auto interaction0_out = rrnlb_interaction0_out_ws;
    auto skip0 = rrnlb_skip0_ws;
    auto& cache0 = rrnlb_cache0;
    mace->compute_rrnlb_interaction_layer_forward(
      0, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r, first_neigh,
      sender_embed, interaction0_out, skip0, cache0, sender_nodes, node_indices,
      rrnlb_total_edges_fwd, rrnlb_edge_to_receiver_arg);

    const bool a0_resized =
      mace->A0.extent(0) < num_nodes
      || mace->A0.extent(1) < num_lm
      || mace->A0.extent(2) < num_channels;
    if (a0_resized) {
      Kokkos::realloc(mace->A0, num_nodes, num_lm, num_channels);
    }
    const auto l0_out_offset = layer0.linear_2.parts_out_offset;
    const auto l0_out_mul = layer0.linear_2.parts_out_mul;
    const auto l0_out_l = layer0.linear_2.parts_out_l;
    const auto& h_l0_out_offset = layer0.linear_2.h_parts_out_offset;
    const auto& h_l0_out_mul = layer0.linear_2.h_parts_out_mul;
    const auto& h_l0_out_l = layer0.linear_2.h_parts_out_l;
    const int l0_num_parts = l0_out_offset.extent_int(0);
    mace->compute_M0_from_rrnlb_layer0_out(num_nodes, node_types, interaction0_out);
    const bool product0_resized = ensure_ws_2d(rrnlb_product0_in_ws, ws_nodes, product0_dim_in);
    auto product0_in = rrnlb_product0_in_ws;
    if (product0_resized) {
      Kokkos::deep_copy(
        Kokkos::subview(product0_in, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
        static_cast<Precision>(0.0));
    }
    auto M0 = mace->M0;
    const auto& h_p0_in_offset = mace->rrnlb_product_linear_0.h_parts_in_offset;
    const auto& h_p0_in_mul = mace->rrnlb_product_linear_0.h_parts_in_mul;
    const auto& h_p0_in_l = mace->rrnlb_product_linear_0.h_parts_in_l;
    const auto p0_in_offset = mace->rrnlb_product_linear_0.parts_in_offset;
    const auto p0_in_mul = mace->rrnlb_product_linear_0.parts_in_mul;
    const auto p0_in_l = mace->rrnlb_product_linear_0.parts_in_l;
    const int p0_num_parts = p0_in_offset.extent_int(0);
    Kokkos::parallel_for(
      "rrnlb_M0_to_product0_mpi",
      Kokkos::RangePolicy<DeviceType>(0, num_nodes * p0_num_parts),
      KOKKOS_LAMBDA (const int ipart) {
        const int i = ipart / p0_num_parts;
        const int p = ipart % p0_num_parts;
        const int offset = p0_in_offset(p);
        const int mul = p0_in_mul(p);
        const int l = p0_in_l(p);
        const int ir_dim = 2 * l + 1;
        const int lm0 = l * l;
        for (int k = 0; k < mul; ++k) {
          for (int m = 0; m < ir_dim; ++m) {
            product0_in(i, offset + k * ir_dim + m) = M0(i, lm0 + m, k);
          }
        }
      });
    ensure_ws_2d(rrnlb_feat0_local_ws, ws_nodes, feat0_dim);
    auto feat0_local = rrnlb_feat0_local_ws;
    mace->rrnlb_apply_linear_forward(
      mace->rrnlb_product_linear_0, num_nodes, product0_in, feat0_local);
    Kokkos::parallel_for(
      "rrnlb_add_skip0_mpi",
      Kokkos::RangePolicy<DeviceType>(0, num_nodes * feat0_dim),
      KOKKOS_LAMBDA (const int ip) {
        const int i = ip / feat0_dim;
        const int p = ip % feat0_dim;
        feat0_local(i, p) += skip0(i, p);
      });

    if (rrnlb_feat0.extent(0) < static_cast<std::size_t>(sender_nodes)
        || rrnlb_feat0.extent(1) != static_cast<std::size_t>(feat0_dim)) {
      Kokkos::realloc(rrnlb_feat0, sender_nodes, feat0_dim);
    }
    const auto node_indices_view = node_indices;
    auto feat0_all = rrnlb_feat0;
    Kokkos::parallel_for(
      "rrnlb_scatter_feat0_for_comm",
      Kokkos::RangePolicy<DeviceType>(0, num_nodes * feat0_dim),
      KOKKOS_LAMBDA (const int ip) {
        const int ii = ip / feat0_dim;
        const int p = ip % feat0_dim;
        const int i = node_indices_view(ii);
        feat0_all(i, p) = feat0_local(ii, p);
      });
    // Explicitly order producer kernel before MPI comm pack path.
    Kokkos::fence();
    comm->forward_comm(this);
    // Ensure unpack completed before layer-1 consumes rrnlb_feat0.
    Kokkos::fence();

    ensure_ws_2d(rrnlb_interaction1_out_ws, ws_nodes, layer1.linear_2.dim_out);
    ensure_ws_2d(rrnlb_skip1_ws, ws_nodes, layer1.skip_tp.dim_out);
    auto interaction1_out = rrnlb_interaction1_out_ws;
    auto skip1 = rrnlb_skip1_ws;
    auto& cache1 = rrnlb_cache1;
    mace->compute_rrnlb_interaction_layer_forward(
      1, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r, first_neigh,
      rrnlb_feat0, interaction1_out, skip1, cache1, sender_nodes, node_indices,
      rrnlb_total_edges_fwd, rrnlb_edge_to_receiver_arg);

    const bool a1_resized =
      mace->A1.extent(0) < num_nodes
      || mace->A1.extent(1) < num_lm
      || mace->A1.extent(2) < num_channels;
    if (a1_resized) {
      Kokkos::realloc(mace->A1, num_nodes, num_lm, num_channels);
    }
    const auto l1_out_offset = layer1.linear_2.parts_out_offset;
    const auto l1_out_mul = layer1.linear_2.parts_out_mul;
    const auto l1_out_l = layer1.linear_2.parts_out_l;
    const auto& h_l1_out_offset = layer1.linear_2.h_parts_out_offset;
    const auto& h_l1_out_mul = layer1.linear_2.h_parts_out_mul;
    const auto& h_l1_out_l = layer1.linear_2.h_parts_out_l;
    const int l1_num_parts = l1_out_offset.extent_int(0);
    mace->compute_M1_from_rrnlb_layer1_out(num_nodes, node_types, interaction1_out);
    const bool product1_resized = ensure_ws_2d(rrnlb_product1_in_ws, ws_nodes, product1_dim_in);
    auto product1_in = rrnlb_product1_in_ws;
    if (product1_resized) {
      Kokkos::deep_copy(
        Kokkos::subview(product1_in, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
        static_cast<Precision>(0.0));
    }
    auto M1 = mace->M1;
    const auto& h_p1_in_offset = mace->rrnlb_product_linear_1.h_parts_in_offset;
    const auto& h_p1_in_mul = mace->rrnlb_product_linear_1.h_parts_in_mul;
    const auto& h_p1_in_l = mace->rrnlb_product_linear_1.h_parts_in_l;
    for (std::size_t p = 0; p < h_p1_in_l.size(); ++p) {
      if (h_p1_in_l[p] != 0) {
        error->all(FLERR, "RRNLB mpi_message_passing currently expects scalar-only product_linear_1 input.");
      }
    }
    const auto p1_in_offset = mace->rrnlb_product_linear_1.parts_in_offset;
    const auto p1_in_mul = mace->rrnlb_product_linear_1.parts_in_mul;
    const auto p1_in_l = mace->rrnlb_product_linear_1.parts_in_l;
    const int p1_num_parts = p1_in_offset.extent_int(0);
    int ap_map_p1_size = 0;
    int ap_map_l1_size = 0;
    int ap_map_p0_size = 0;
    int ap_map_l0_size = 0;
    if constexpr (!std::is_same_v<Precision, AccumPrecision>) {
      for (int p = 0; p < p1_num_parts; ++p) {
        ap_map_p1_size += h_p1_in_mul[p];
      }
      bool rebuild_p1 =
        ensure_exact_1d(rrnlb_ap_map_p1_src_col, ap_map_p1_size)
        | ensure_exact_1d(rrnlb_ap_map_p1_dst_k, ap_map_p1_size);
      if (rebuild_p1 && ap_map_p1_size > 0) {
        auto h_src = Kokkos::create_mirror_view(rrnlb_ap_map_p1_src_col);
        auto h_dst_k = Kokkos::create_mirror_view(rrnlb_ap_map_p1_dst_k);
        int e = 0;
        for (int p = 0; p < p1_num_parts; ++p) {
          const int offset = h_p1_in_offset[p];
          const int mul = h_p1_in_mul[p];
          const int ir_dim = 2 * h_p1_in_l[p] + 1;
          for (int k = 0; k < mul; ++k) {
            h_src(e) = offset + k * ir_dim;
            h_dst_k(e) = k;
            ++e;
          }
        }
        Kokkos::deep_copy(rrnlb_ap_map_p1_src_col, h_src);
        Kokkos::deep_copy(rrnlb_ap_map_p1_dst_k, h_dst_k);
      }

      for (int p = 0; p < l1_num_parts; ++p) {
        ap_map_l1_size += h_l1_out_mul[p] * (2 * h_l1_out_l[p] + 1);
      }
      bool rebuild_l1 =
        ensure_exact_1d(rrnlb_ap_map_l1_dst_col, ap_map_l1_size)
        | ensure_exact_1d(rrnlb_ap_map_l1_src_lm, ap_map_l1_size)
        | ensure_exact_1d(rrnlb_ap_map_l1_src_k, ap_map_l1_size);
      if (rebuild_l1 && ap_map_l1_size > 0) {
        auto h_dst = Kokkos::create_mirror_view(rrnlb_ap_map_l1_dst_col);
        auto h_src_lm = Kokkos::create_mirror_view(rrnlb_ap_map_l1_src_lm);
        auto h_src_k = Kokkos::create_mirror_view(rrnlb_ap_map_l1_src_k);
        int e = 0;
        for (int p = 0; p < l1_num_parts; ++p) {
          const int offset = h_l1_out_offset[p];
          const int mul = h_l1_out_mul[p];
          const int l = h_l1_out_l[p];
          const int ir_dim = 2 * l + 1;
          const int lm0 = l * l;
          for (int k = 0; k < mul; ++k) {
            for (int m = 0; m < ir_dim; ++m) {
              h_dst(e) = offset + k * ir_dim + m;
              h_src_lm(e) = lm0 + m;
              h_src_k(e) = k;
              ++e;
            }
          }
        }
        Kokkos::deep_copy(rrnlb_ap_map_l1_dst_col, h_dst);
        Kokkos::deep_copy(rrnlb_ap_map_l1_src_lm, h_src_lm);
        Kokkos::deep_copy(rrnlb_ap_map_l1_src_k, h_src_k);
      }

      for (int p = 0; p < p0_num_parts; ++p) {
        ap_map_p0_size += h_p0_in_mul[p] * (2 * h_p0_in_l[p] + 1);
      }
      bool rebuild_p0 =
        ensure_exact_1d(rrnlb_ap_map_p0_src_col, ap_map_p0_size)
        | ensure_exact_1d(rrnlb_ap_map_p0_dst_lm, ap_map_p0_size)
        | ensure_exact_1d(rrnlb_ap_map_p0_dst_k, ap_map_p0_size);
      if (rebuild_p0 && ap_map_p0_size > 0) {
        auto h_src = Kokkos::create_mirror_view(rrnlb_ap_map_p0_src_col);
        auto h_dst_lm = Kokkos::create_mirror_view(rrnlb_ap_map_p0_dst_lm);
        auto h_dst_k = Kokkos::create_mirror_view(rrnlb_ap_map_p0_dst_k);
        int e = 0;
        for (int p = 0; p < p0_num_parts; ++p) {
          const int offset = h_p0_in_offset[p];
          const int mul = h_p0_in_mul[p];
          const int l = h_p0_in_l[p];
          const int ir_dim = 2 * l + 1;
          const int lm0 = l * l;
          for (int k = 0; k < mul; ++k) {
            for (int m = 0; m < ir_dim; ++m) {
              h_src(e) = offset + k * ir_dim + m;
              h_dst_lm(e) = lm0 + m;
              h_dst_k(e) = k;
              ++e;
            }
          }
        }
        Kokkos::deep_copy(rrnlb_ap_map_p0_src_col, h_src);
        Kokkos::deep_copy(rrnlb_ap_map_p0_dst_lm, h_dst_lm);
        Kokkos::deep_copy(rrnlb_ap_map_p0_dst_k, h_dst_k);
      }

      for (int p = 0; p < l0_num_parts; ++p) {
        ap_map_l0_size += h_l0_out_mul[p] * (2 * h_l0_out_l[p] + 1);
      }
      bool rebuild_l0 =
        ensure_exact_1d(rrnlb_ap_map_l0_dst_col, ap_map_l0_size)
        | ensure_exact_1d(rrnlb_ap_map_l0_src_lm, ap_map_l0_size)
        | ensure_exact_1d(rrnlb_ap_map_l0_src_k, ap_map_l0_size);
      if (rebuild_l0 && ap_map_l0_size > 0) {
        auto h_dst = Kokkos::create_mirror_view(rrnlb_ap_map_l0_dst_col);
        auto h_src_lm = Kokkos::create_mirror_view(rrnlb_ap_map_l0_src_lm);
        auto h_src_k = Kokkos::create_mirror_view(rrnlb_ap_map_l0_src_k);
        int e = 0;
        for (int p = 0; p < l0_num_parts; ++p) {
          const int offset = h_l0_out_offset[p];
          const int mul = h_l0_out_mul[p];
          const int l = h_l0_out_l[p];
          const int ir_dim = 2 * l + 1;
          const int lm0 = l * l;
          for (int k = 0; k < mul; ++k) {
            for (int m = 0; m < ir_dim; ++m) {
              h_dst(e) = offset + k * ir_dim + m;
              h_src_lm(e) = lm0 + m;
              h_src_k(e) = k;
              ++e;
            }
          }
        }
        Kokkos::deep_copy(rrnlb_ap_map_l0_dst_col, h_dst);
        Kokkos::deep_copy(rrnlb_ap_map_l0_src_lm, h_src_lm);
        Kokkos::deep_copy(rrnlb_ap_map_l0_src_k, h_src_k);
      }
    }
    Kokkos::parallel_for(
      "rrnlb_M1_to_product1_mpi",
      Kokkos::RangePolicy<DeviceType>(0, num_nodes * p1_num_parts),
      KOKKOS_LAMBDA (const int ipart) {
        const int i = ipart / p1_num_parts;
        const int p = ipart % p1_num_parts;
        const int offset = p1_in_offset(p);
        const int mul = p1_in_mul(p);
        const int l = p1_in_l(p);
        const int ir_dim = 2 * l + 1;
        for (int k = 0; k < mul; ++k) {
          product1_in(i, offset + k * ir_dim) = M1(i, k);
        }
      });

    ensure_ws_2d(rrnlb_feat1_ws, ws_nodes, feat1_dim);
    auto feat1 = rrnlb_feat1_ws;
    mace->rrnlb_apply_linear_forward(
      mace->rrnlb_product_linear_1, num_nodes, product1_in, feat1);
    Kokkos::parallel_for(
      "rrnlb_add_skip1_mpi",
      Kokkos::RangePolicy<DeviceType>(0, num_nodes * feat1_dim),
      KOKKOS_LAMBDA (const int ip) {
        const int i = ip / feat1_dim;
        const int p = ip % feat1_dim;
        feat1(i, p) += skip1(i, p);
      });

    ensure_ws_2d(rrnlb_feat0_adj_local_ws, ws_nodes, feat0_dim);
    ensure_ws_2d(rrnlb_feat1_adj_ws, ws_nodes, feat1_dim);
    auto feat0_adj_local = rrnlb_feat0_adj_local_ws;
    auto feat1_adj = rrnlb_feat1_adj_ws;
    Kokkos::deep_copy(
      Kokkos::subview(feat0_adj_local, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
      static_cast<AccumPrecision>(0.0));
    Kokkos::deep_copy(
      Kokkos::subview(feat1_adj, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
      static_cast<AccumPrecision>(0.0));
    auto node_energies_view = mace->node_energies;
    const auto atomic_energies = mace->atomic_energies;
    const auto readout_1_weights = mace->readout_1_weights;
    const auto feat0_comm = rrnlb_feat0;
    Kokkos::parallel_for(
      "rrnlb_readout1_mpi",
      Kokkos::RangePolicy<DeviceType>(0, num_nodes),
      KOKKOS_LAMBDA (const int ii) {
        const int i = node_indices_view(ii);
        const int t = node_types(ii);
        double e_i = atomic_energies(t);
        for (int k = 0; k < num_channels; ++k) {
          e_i += readout_1_weights(k) * static_cast<double>(feat0_comm(i, k));
          feat0_adj_local(ii, k) = static_cast<AccumPrecision>(readout_1_weights(k));
        }
        node_energies_view(ii) += e_i;
      });

    ensure_ws_2d(rrnlb_feat1_double_ws, ws_nodes, feat1_dim);
    auto feat1_double = rrnlb_feat1_double_ws;
    Kokkos::parallel_for(
      "rrnlb_feat1_cast_mpi",
      Kokkos::RangePolicy<DeviceType>(0, num_nodes * feat1_dim),
      KOKKOS_LAMBDA (const int ip) {
        const int i = ip / feat1_dim;
        const int p = ip % feat1_dim;
        feat1_double(i, p) = static_cast<double>(feat1(i, p));
      });
    ensure_ws_1d(rrnlb_readout2_out_ws, ws_nodes);
    ensure_ws_2d(rrnlb_readout2_adj_ws, ws_nodes, feat1_dim);
    auto readout2_out = rrnlb_readout2_out_ws;
    auto readout2_adj = rrnlb_readout2_adj_ws;
    mace->readout_2.evaluate_gradient(feat1_double, readout2_out, readout2_adj);
    Kokkos::parallel_for(
      "rrnlb_readout2_accum_mpi",
      Kokkos::RangePolicy<DeviceType>(0, num_nodes * feat1_dim),
      KOKKOS_LAMBDA (const int ip) {
        const int i = ip / feat1_dim;
        const int p = ip % feat1_dim;
        if (p == 0) node_energies_view(i) += readout2_out(i);
        feat1_adj(i, p) = static_cast<AccumPrecision>(readout2_adj(i, p));
      });

    auto skip1_adj = feat1_adj;
    ensure_ws_2d(rrnlb_product1_in_adj_ws, ws_nodes, product1_dim_in);
    auto product1_in_adj = rrnlb_product1_in_adj_ws;
    run_rrnlb_phase_stage([&] {
      mace->rrnlb_apply_linear_transpose(
        mace->rrnlb_product_linear_1, num_nodes, feat1_adj, product1_in_adj);
    }, [&] (double seconds) {
      mace->rrnlb_record_reverse_product1_transpose(seconds);
    });

    if constexpr (!std::is_same_v<Precision, AccumPrecision>) {
      const bool m1_adj_ap_resized =
        mace->rrnlb_M1_adj_ap.extent(0) < num_nodes
        || mace->rrnlb_M1_adj_ap.extent(1) < num_channels;
      if (m1_adj_ap_resized) {
        Kokkos::realloc(mace->rrnlb_M1_adj_ap, num_nodes, num_channels);
      }
      auto M1_adj_ap = mace->rrnlb_M1_adj_ap;
      Kokkos::deep_copy(
        Kokkos::subview(M1_adj_ap, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
        static_cast<AccumPrecision>(0.0));
      const auto map_src_col = rrnlb_ap_map_p1_src_col;
      const auto map_dst_k = rrnlb_ap_map_p1_dst_k;
      Kokkos::parallel_for(
        "rrnlb_product1_adj_to_M1_adj_ap_mpi_fast",
        Kokkos::RangePolicy<DeviceType>(0, num_nodes * ap_map_p1_size),
        KOKKOS_LAMBDA (const int idx) {
          const int i = idx / ap_map_p1_size;
          const int e = idx % ap_map_p1_size;
          Kokkos::atomic_add(
            &M1_adj_ap(i, map_dst_k(e)),
            static_cast<AccumPrecision>(product1_in_adj(i, map_src_col(e))));
        });
      run_rrnlb_phase_stage([&] {
        mace->reverse_M1_mixed_rrnlb(
          num_nodes, node_types, mace->rrnlb_M1_adj_ap, mace->rrnlb_A1_adj_ap);
      }, [&] (double seconds) {
        mace->rrnlb_record_reverse_m1_mixed(seconds);
      });
    } else {
      const bool m1_adj_resized =
        mace->M1_adj.extent(0) < num_nodes
        || mace->M1_adj.extent(1) < num_channels;
      if (m1_adj_resized) {
        Kokkos::realloc(mace->M1_adj, num_nodes, num_channels);
      }
      auto M1_adj = mace->M1_adj;
      Kokkos::parallel_for(
        "rrnlb_product1_adj_to_M1_adj_mpi",
        Kokkos::RangePolicy<DeviceType>(0, num_nodes * p1_num_parts),
        KOKKOS_LAMBDA (const int ipart) {
          const int i = ipart / p1_num_parts;
          const int p = ipart % p1_num_parts;
          const int offset = p1_in_offset(p);
          const int mul = p1_in_mul(p);
          const int l = p1_in_l(p);
          const int ir_dim = 2 * l + 1;
          for (int k = 0; k < mul; ++k) {
            M1_adj(i, k) = product1_in_adj(i, offset + k * ir_dim);
          }
        });
      run_rrnlb_phase_stage([&] {
        mace->reverse_M1(num_nodes, node_types);
      }, [&] (double seconds) {
        mace->rrnlb_record_reverse_m1_mixed(seconds);
      });
    }

    const bool interaction1_adj_resized =
      ensure_ws_2d(rrnlb_interaction1_adj_ws, ws_nodes, layer1.linear_2.dim_out);
    auto interaction1_adj = rrnlb_interaction1_adj_ws;
    if constexpr (!std::is_same_v<Precision, AccumPrecision>) {
      Kokkos::deep_copy(
        Kokkos::subview(interaction1_adj, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
        static_cast<AccumPrecision>(0.0));
      auto A1_adj_ap = mace->rrnlb_A1_adj_ap;
      const auto map_dst_col = rrnlb_ap_map_l1_dst_col;
      const auto map_src_lm = rrnlb_ap_map_l1_src_lm;
      const auto map_src_k = rrnlb_ap_map_l1_src_k;
      Kokkos::parallel_for(
        "rrnlb_A1_adj_ap_to_layer1_mpi_fast",
        Kokkos::RangePolicy<DeviceType>(0, num_nodes * ap_map_l1_size),
        KOKKOS_LAMBDA (const int idx) {
          const int i = idx / ap_map_l1_size;
          const int e = idx % ap_map_l1_size;
          Kokkos::atomic_add(
            &interaction1_adj(i, map_dst_col(e)),
            A1_adj_ap(i, map_src_lm(e), map_src_k(e)));
        });
    } else {
      if (interaction1_adj_resized) {
        Kokkos::deep_copy(
          Kokkos::subview(interaction1_adj, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
          static_cast<AccumPrecision>(0.0));
      }
      auto A1_adj = mace->A1_adj;
      Kokkos::parallel_for(
        "rrnlb_A1_adj_to_layer1_mpi",
        Kokkos::RangePolicy<DeviceType>(0, num_nodes * l1_num_parts),
        KOKKOS_LAMBDA (const int ipart) {
          const int i = ipart / l1_num_parts;
          const int p = ipart % l1_num_parts;
          const int offset = l1_out_offset(p);
          const int mul = l1_out_mul(p);
          const int l = l1_out_l(p);
          const int ir_dim = 2 * l + 1;
          const int lm0 = l * l;
          for (int k = 0; k < mul; ++k) {
            for (int m = 0; m < ir_dim; ++m) {
              interaction1_adj(i, offset + k * ir_dim + m) = A1_adj(i, lm0 + m, k);
            }
          }
        });
    }

    ensure_ws_2d(rrnlb_feat0_from_layer1_adj_ws, ws_sender_nodes, feat0_dim);
    auto feat0_from_layer1_adj = rrnlb_feat0_from_layer1_adj_ws;
    Kokkos::deep_copy(
      Kokkos::subview(feat0_from_layer1_adj, Kokkos::make_pair(0, sender_nodes), Kokkos::ALL),
      static_cast<AccumPrecision>(0.0));
    mace->reverse_rrnlb_interaction_layer(
      1, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, first_neigh,
      rrnlb_feat0, cache1, interaction1_adj, skip1_adj, feat0_from_layer1_adj,
      sender_nodes, node_indices, rrnlb_total_edges_rev, rrnlb_edge_to_receiver_arg,
      rrnlb_sender_edge_offsets_arg, rrnlb_sender_edge_indices_arg,
      rrnlb_sender_segment_offsets_arg, rrnlb_sender_segment_to_sender_arg,
      rrnlb_total_sender_segments_arg);

    rrnlb_feat0_adj = feat0_from_layer1_adj;
    auto feat0_adj_all = rrnlb_feat0_adj;
    ensure_ws_2d(rrnlb_feat0_adj_nodes_ws, ws_nodes, feat0_dim);
    auto feat0_adj_nodes = rrnlb_feat0_adj_nodes_ws;
    Kokkos::parallel_for(
      "rrnlb_init_feat0_adj_nodes_mpi",
      Kokkos::RangePolicy<DeviceType>(0, num_nodes * feat0_dim),
      KOKKOS_LAMBDA (const int ip) {
        const int ii = ip / feat0_dim;
        const int p = ip % feat0_dim;
        const int i = node_indices_view(ii);
        const auto adj = feat0_adj_all(i, p) + feat0_adj_local(ii, p);
        feat0_adj_all(i, p) = adj;
        feat0_adj_nodes(ii, p) = adj;
      });
    const bool use_compact_reverse_unpack =
      rrnlb_compact_reverse_unpack && reverse_comm_device != 0;
    if (use_compact_reverse_unpack) {
      ensure_ws_1d(rrnlb_atom_to_node_ws, sender_nodes);
      auto atom_to_node = rrnlb_atom_to_node_ws;
      Kokkos::deep_copy(
        Kokkos::subview(atom_to_node, Kokkos::make_pair(0, sender_nodes)),
        -1);
      Kokkos::parallel_for(
        "rrnlb_atom_to_node_mpi",
        Kokkos::RangePolicy<DeviceType>(0, num_nodes),
        KOKKOS_LAMBDA (const int ii) {
          atom_to_node(node_indices_view(ii)) = ii;
        });
      rrnlb_feat0_adj_nodes_comm = feat0_adj_nodes;
      rrnlb_compact_reverse_unpack_active = true;
    }
    // Explicitly order adjoint accumulation before MPI reverse comm pack.
    run_rrnlb_phase_stage([&] {
      Kokkos::fence();
      comm->reverse_comm(this);
      // Ensure reverse comm unpack completed before node-index gather.
      Kokkos::fence();
    }, [&] (double seconds) {
      mace->rrnlb_record_reverse_mpi_comm(seconds);
    });
    rrnlb_compact_reverse_unpack_active = false;

    if (!use_compact_reverse_unpack) {
      const auto feat0_adj_comm = rrnlb_feat0_adj;
      Kokkos::parallel_for(
        "rrnlb_gather_feat0_adj_mpi",
        Kokkos::RangePolicy<DeviceType>(0, num_nodes * feat0_dim),
        KOKKOS_LAMBDA (const int ip) {
          const int ii = ip / feat0_dim;
          const int p = ip % feat0_dim;
          const int i = node_indices_view(ii);
          feat0_adj_nodes(ii, p) = feat0_adj_comm(i, p);
        });
    }

    auto skip0_adj = feat0_adj_nodes;
    ensure_ws_2d(rrnlb_product0_in_adj_ws, ws_nodes, product0_dim_in);
    auto product0_in_adj = rrnlb_product0_in_adj_ws;
    run_rrnlb_phase_stage([&] {
      mace->rrnlb_apply_linear_transpose(
        mace->rrnlb_product_linear_0, num_nodes, feat0_adj_nodes, product0_in_adj);
    }, [&] (double seconds) {
      mace->rrnlb_record_reverse_product0_transpose(seconds);
    });

    if constexpr (!std::is_same_v<Precision, AccumPrecision>) {
      const bool m0_adj_ap_resized =
        mace->rrnlb_M0_adj_ap.extent(0) < num_nodes
        || mace->rrnlb_M0_adj_ap.extent(1) < num_LM
        || mace->rrnlb_M0_adj_ap.extent(2) < num_channels;
      if (m0_adj_ap_resized) {
        Kokkos::realloc(mace->rrnlb_M0_adj_ap, num_nodes, num_LM, num_channels);
      }
      auto M0_adj_ap = mace->rrnlb_M0_adj_ap;
      Kokkos::deep_copy(
        Kokkos::subview(M0_adj_ap, Kokkos::make_pair(0, num_nodes), Kokkos::ALL, Kokkos::ALL),
        static_cast<AccumPrecision>(0.0));
      const auto map_src_col = rrnlb_ap_map_p0_src_col;
      const auto map_dst_lm = rrnlb_ap_map_p0_dst_lm;
      const auto map_dst_k = rrnlb_ap_map_p0_dst_k;
      Kokkos::parallel_for(
        "rrnlb_product0_adj_to_M0_adj_ap_mpi_fast",
        Kokkos::RangePolicy<DeviceType>(0, num_nodes * ap_map_p0_size),
        KOKKOS_LAMBDA (const int idx) {
          const int i = idx / ap_map_p0_size;
          const int e = idx % ap_map_p0_size;
          Kokkos::atomic_add(
            &M0_adj_ap(i, map_dst_lm(e), map_dst_k(e)),
            static_cast<AccumPrecision>(product0_in_adj(i, map_src_col(e))));
        });
      run_rrnlb_phase_stage([&] {
        mace->reverse_M0_mixed_rrnlb(
          num_nodes, node_types, mace->rrnlb_M0_adj_ap, mace->rrnlb_A0_adj_ap);
      }, [&] (double seconds) {
        mace->rrnlb_record_reverse_m0_mixed(seconds);
      });
    } else {
      const bool m0_adj_resized =
        mace->M0_adj.extent(0) < num_nodes
        || mace->M0_adj.extent(1) < num_LM
        || mace->M0_adj.extent(2) < num_channels;
      if (m0_adj_resized) {
        Kokkos::realloc(mace->M0_adj, num_nodes, num_LM, num_channels);
      }
      auto M0_adj = mace->M0_adj;
      Kokkos::parallel_for(
        "rrnlb_product0_adj_to_M0_adj_mpi",
        Kokkos::RangePolicy<DeviceType>(0, num_nodes * p0_num_parts),
        KOKKOS_LAMBDA (const int ipart) {
          const int i = ipart / p0_num_parts;
          const int p = ipart % p0_num_parts;
          const int offset = p0_in_offset(p);
          const int mul = p0_in_mul(p);
          const int l = p0_in_l(p);
          const int ir_dim = 2 * l + 1;
          const int lm0 = l * l;
          for (int k = 0; k < mul; ++k) {
            for (int m = 0; m < ir_dim; ++m) {
              M0_adj(i, lm0 + m, k) = product0_in_adj(i, offset + k * ir_dim + m);
            }
          }
        });
      run_rrnlb_phase_stage([&] {
        mace->reverse_M0(num_nodes, node_types);
      }, [&] (double seconds) {
        mace->rrnlb_record_reverse_m0_mixed(seconds);
      });
    }

    const bool interaction0_adj_resized =
      ensure_ws_2d(rrnlb_interaction0_adj_ws, ws_nodes, layer0.linear_2.dim_out);
    auto interaction0_adj = rrnlb_interaction0_adj_ws;
    if constexpr (!std::is_same_v<Precision, AccumPrecision>) {
      Kokkos::deep_copy(
        Kokkos::subview(interaction0_adj, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
        static_cast<AccumPrecision>(0.0));
      auto A0_adj_ap = mace->rrnlb_A0_adj_ap;
      const auto map_dst_col = rrnlb_ap_map_l0_dst_col;
      const auto map_src_lm = rrnlb_ap_map_l0_src_lm;
      const auto map_src_k = rrnlb_ap_map_l0_src_k;
      Kokkos::parallel_for(
        "rrnlb_A0_adj_ap_to_layer0_mpi_fast",
        Kokkos::RangePolicy<DeviceType>(0, num_nodes * ap_map_l0_size),
        KOKKOS_LAMBDA (const int idx) {
          const int i = idx / ap_map_l0_size;
          const int e = idx % ap_map_l0_size;
          Kokkos::atomic_add(
            &interaction0_adj(i, map_dst_col(e)),
            A0_adj_ap(i, map_src_lm(e), map_src_k(e)));
        });
    } else {
      if (interaction0_adj_resized) {
        Kokkos::deep_copy(
          Kokkos::subview(interaction0_adj, Kokkos::make_pair(0, num_nodes), Kokkos::ALL),
          static_cast<AccumPrecision>(0.0));
      }
      auto A0_adj = mace->A0_adj;
      Kokkos::parallel_for(
        "rrnlb_A0_adj_to_layer0_mpi",
        Kokkos::RangePolicy<DeviceType>(0, num_nodes * l0_num_parts),
        KOKKOS_LAMBDA (const int ipart) {
          const int i = ipart / l0_num_parts;
          const int p = ipart % l0_num_parts;
          const int offset = l0_out_offset(p);
          const int mul = l0_out_mul(p);
          const int l = l0_out_l(p);
          const int ir_dim = 2 * l + 1;
          const int lm0 = l * l;
          for (int k = 0; k < mul; ++k) {
            for (int m = 0; m < ir_dim; ++m) {
              interaction0_adj(i, offset + k * ir_dim + m) = A0_adj(i, lm0 + m, k);
            }
          }
        });
    }

    ensure_ws_2d(rrnlb_sender_embed_adj_ws, ws_sender_nodes, num_channels);
    auto sender_embed_adj = rrnlb_sender_embed_adj_ws;
    Kokkos::deep_copy(
      Kokkos::subview(sender_embed_adj, Kokkos::make_pair(0, sender_nodes), Kokkos::ALL),
      static_cast<AccumPrecision>(0.0));
    mace->reverse_rrnlb_interaction_layer(
      0, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, first_neigh,
      sender_embed, cache0, interaction0_adj, skip0_adj, sender_embed_adj,
      sender_nodes, node_indices, rrnlb_total_edges_rev, rrnlb_edge_to_receiver_arg,
      rrnlb_sender_edge_offsets_arg, rrnlb_sender_edge_indices_arg,
      rrnlb_sender_segment_offsets_arg, rrnlb_sender_segment_to_sender_arg,
      rrnlb_total_sender_segments_arg);
  } else {
    mace->compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_A0(num_nodes, node_types, num_neigh, neigh_types);
    mace->compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_M0(num_nodes, node_types);
    mace->compute_H1(num_nodes);

    // sort H1 contributions by i (rather than ii)
    if (H1.extent(0) < k_list->inum+atom->nghost)
      Kokkos::realloc(H1, (k_list->inum+atom->nghost), mace->num_LM, mace->num_channels);
    auto num_LM = mace->num_LM;
    auto num_channels = mace->num_channels;
    auto mace_H1 = mace->H1;
    auto H1 = this->H1;
    Kokkos::parallel_for("Sort H1",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {num_nodes,num_LM,num_channels}),
      KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
        const int i = d_ilist(ii);
        H1(i,LM,k) = mace_H1(ii,LM,k);
      });
    Kokkos::fence();
    comm->forward_comm(this);
    Kokkos::fence();
    mace->H1 = H1;

    mace->compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_Phi1(num_nodes, num_neigh, neigh_indices);
    mace->compute_A1(num_nodes);
    mace->compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_M1(num_nodes, node_types);
    mace->compute_H2(num_nodes, node_types);

    mace->compute_readouts(num_nodes, node_types);

    mace->reverse_H2(num_nodes, node_types, false);
    mace->reverse_M1(num_nodes, node_types);
    mace->reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    mace->reverse_A1(num_nodes);
    mace->reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);

    H1_adj = mace->H1_adj;
    Kokkos::fence();
    comm->reverse_comm(this);
    Kokkos::fence();

    mace->reverse_H1(num_nodes);
    mace->reverse_M0(num_nodes, node_types);
    mace->reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    mace->reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  }

  if (eflag_global) {
    auto node_energies = mace->node_energies;
    double energy;
    Kokkos::parallel_reduce("Energy Reduction",
      num_nodes,
      KOKKOS_LAMBDA (const int i, double& energy) {
        energy += node_energies(i);
      }, energy);
    eng_vdwl += energy;
  }

  if (eflag_atom) {
    auto d_eatom = k_eatom.template view<DeviceType>();
    auto node_energies = mace->node_energies;
    Kokkos::parallel_for("Extract Atomic Energies", num_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(ii) += node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
  }

  auto mace_node_forces = mace->node_forces;
  Kokkos::parallel_for("Force Reduction",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = node_indices(ii);
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
        [&] (const int jj) {
          const int ij = first_neigh(ii) + jj;
          const int j = neigh_indices(ij);
          const double fx = mace_node_forces(3*ij);
          const double fy = mace_node_forces(3*ij+1);
          const double fz = mace_node_forces(3*ij+2);
          Kokkos::atomic_add(&f(j,0), fx);
          Kokkos::atomic_add(&f(j,1), fy);
          Kokkos::atomic_add(&f(j,2), fz);
          Kokkos::atomic_add(&f(i,0), -fx);
          Kokkos::atomic_add(&f(i,1), -fy);
          Kokkos::atomic_add(&f(i,2), -fz);
        });
    });

  if (rrnlb_debug_net_force_enabled()) {
    double local_fx = 0.0;
    double local_fy = 0.0;
    double local_fz = 0.0;
    Kokkos::parallel_reduce(
      "rrnlb_debug_net_force_fx_mpi",
      Kokkos::RangePolicy<DeviceType>(0, atom->nlocal),
      KOKKOS_LAMBDA (const int i, double& sum) {
        sum += f(i, 0);
      },
      local_fx);
    Kokkos::parallel_reduce(
      "rrnlb_debug_net_force_fy_mpi",
      Kokkos::RangePolicy<DeviceType>(0, atom->nlocal),
      KOKKOS_LAMBDA (const int i, double& sum) {
        sum += f(i, 1);
      },
      local_fy);
    Kokkos::parallel_reduce(
      "rrnlb_debug_net_force_fz_mpi",
      Kokkos::RangePolicy<DeviceType>(0, atom->nlocal),
      KOKKOS_LAMBDA (const int i, double& sum) {
        sum += f(i, 2);
      },
      local_fz);
    double net_force_local[3] = {local_fx, local_fy, local_fz};
    double net_force_global[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(net_force_local, net_force_global, 3, MPI_DOUBLE, MPI_SUM, world);
    if (comm->me == 0) {
      const double net_norm = std::sqrt(
        net_force_global[0] * net_force_global[0]
        + net_force_global[1] * net_force_global[1]
        + net_force_global[2] * net_force_global[2]);
      utils::logmesg(
        lmp,
        "[RRNLB debug net force] mode=mpi_message_passing step={} fx={:.12e} fy={:.12e} "
        "fz={:.12e} norm={:.12e}\n",
        update->ntimestep,
        net_force_global[0],
        net_force_global[1],
        net_force_global[2],
        net_norm);
    }
  }

  if (vflag_global) {
    Kokkos::View<double*,Kokkos::LayoutRight> v("v", 6);
    Kokkos::deep_copy(v, 0.0);
    Kokkos::parallel_for("Virial Reduction",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
          [&] (const int jj) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = mace_node_forces(3*ij);
            const double f_y = mace_node_forces(3*ij+1);
            const double f_z = mace_node_forces(3*ij+2);
            Kokkos::atomic_add(&v(0), x*f_x);
            Kokkos::atomic_add(&v(1), y*f_y);
            Kokkos::atomic_add(&v(2), z*f_z);
            Kokkos::atomic_add(&v(3), 0.5*(x*f_y + y*f_x));
            Kokkos::atomic_add(&v(4), 0.5*(x*f_z + z*f_x));
            Kokkos::atomic_add(&v(5), 0.5*(y*f_z + z*f_y));
          });
      });
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    virial[0] += h_v(0);
    virial[1] += h_v(1);
    virial[2] += h_v(2);
    virial[3] += h_v(3);
    virial[4] += h_v(4);
    virial[5] += h_v(5);
  }

  rrnlb_maybe_emit_phase_stats("mpi_message_passing");

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/kk.");
}


/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision, typename AccumPrecision>
void PairSymmetrixMACEKokkos<DeviceType, Precision, AccumPrecision>::compute_no_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom && k_eatom.view<DeviceType>().extent(0)<maxeatom) {
     memoryKK->destroy_kokkos(k_eatom,eatom);
     memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
  }

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();

  const double r_cut_squared = mace->r_cut*mace->r_cut;
  const bool interaction_mode_rrnlb = mace->interaction_mode_rrnlb;

  // locate ghosts within r_cut of locals
  auto is_local = Kokkos::Bitset(atom->nlocal+atom->nghost);
  Kokkos::parallel_for("fill is_local",
    list->inum,
    KOKKOS_LAMBDA (const int ii) {
      const int i = d_ilist(ii);
      is_local.set(i);
    });
  Kokkos::fence();
  auto is_ghost = Kokkos::Bitset(atom->nlocal+atom->nghost);
  Kokkos::parallel_for("fill is_ghost",
    Kokkos::TeamPolicy<>(list->inum, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
        [&] (const int jj) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x_i;
          const double dy = x(j,1) - y_i;
          const double dz = x(j,2) - z_i;
          const double r_squared = dx*dx + dy*dy + dz*dz;
          if (r_squared<r_cut_squared and not is_local.test(j))
            is_ghost.set(j);
        });
    });
  Kokkos::fence();

  // RRNLB requires a 2-hop node set for 2 interaction layers.
  if (interaction_mode_rrnlb) {
    auto is_ghost_level1 = Kokkos::Bitset(atom->nlocal+atom->nghost);
    Kokkos::parallel_for(
      "copy is_ghost to is_ghost_level1",
      is_ghost.size(),
      KOKKOS_LAMBDA (const int i) {
        if (is_ghost.test(i)) is_ghost_level1.set(i);
      });
    Kokkos::fence();
    Kokkos::parallel_for("extend is_ghost to second shell",
      Kokkos::TeamPolicy<>(atom->nlocal+atom->nghost, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int i = team_member.league_rank();
        if (!is_ghost_level1.test(i)) return;
        const double x_i = x(i,0);
        const double y_i = x(i,1);
        const double z_i = x(i,2);
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
          [&] (const int jj) {
            const int j = (d_neighbors(i,jj) & NEIGHMASK);
            const double dx = x(j,0) - x_i;
            const double dy = x(j,1) - y_i;
            const double dz = x(j,2) - z_i;
            const double r_squared = dx*dx + dy*dy + dz*dz;
            if (r_squared < r_cut_squared and not is_local.test(j))
              is_ghost.set(j);
          });
      });
    Kokkos::fence();
  }

  // set num_local_nodes and num_ghost_nodes
  const int num_local_nodes = list->inum;
  const int num_ghost_nodes = is_ghost.count();

  // collect indices of ghosts within r_cut of locals
  auto ghost_indices = Kokkos::View<int*>("ghost_indices", num_ghost_nodes);
  Kokkos::parallel_scan("populate ghost_indices",
    is_ghost.size(),
    KOKKOS_LAMBDA(int i, int& update, const bool final) {
    if (final && is_ghost.test(i))
      ghost_indices(update) = i;
    update += is_ghost.test(i);
  });
  Kokkos::fence();

  // populate node_indices, node_types, and num_neigh
  if (node_indices.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(node_indices, num_local_nodes+num_ghost_nodes);
  if (node_types.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(node_types, num_local_nodes+num_ghost_nodes);
  if (num_neigh.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(num_neigh, num_local_nodes+num_ghost_nodes);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  Kokkos::deep_copy(num_neigh, 0);
  auto mace_types = this->mace_types;
  Kokkos::parallel_for("populate node-based views",
    Kokkos::TeamPolicy<>(num_local_nodes+num_ghost_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = (ii<num_local_nodes) ? d_ilist(ii) : ghost_indices(ii-num_local_nodes);
      node_indices(ii) = i;
      node_types(ii) = mace_types(type(i)-1);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
        [&] (const int jj, int& num_neigh_ii) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x_i;
          const double dy = x(j,1) - y_i;
          const double dz = x(j,2) - z_i;
          const double r_squared = dx*dx + dy*dy + dz*dz;
          if (r_squared < r_cut_squared &&
              (!interaction_mode_rrnlb || is_local.test(j) || is_ghost.test(j)))
            num_neigh_ii += 1;
        }, num_neigh(ii));
    });
  Kokkos::fence();

  // count edges
  int num_local_edges;
  Kokkos::parallel_reduce("count local edges",
    num_local_nodes,
    KOKKOS_LAMBDA (const int ii, int& num_local_edges) {
      num_local_edges += num_neigh(ii);
    }, num_local_edges);
  int num_ghost_edges;
  Kokkos::parallel_reduce("count ghost edges",
    Kokkos::RangePolicy<>(num_local_nodes, num_local_nodes+num_ghost_nodes),
    KOKKOS_LAMBDA (const int ii, int& num_ghost_edges) {
      num_ghost_edges += num_neigh(ii);
    }, num_ghost_edges);
  Kokkos::fence();

  // first neighbor
  if (first_neigh.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(first_neigh, num_local_nodes+num_ghost_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  Kokkos::parallel_scan("populate first neighbor",
    num_local_nodes+num_ghost_nodes,
    KOKKOS_LAMBDA (const int ii, int& first_neigh_ii, const bool final) {
      if (final) first_neigh(ii) = first_neigh_ii;
      first_neigh_ii += num_neigh(ii);
    });
  Kokkos::fence();

  // populate neigh_indices, neigh_types, xyz, and r
  if (neigh_indices.size() < num_local_edges+num_ghost_edges)
    Kokkos::realloc(neigh_indices, num_local_edges+num_ghost_edges);
  if (neigh_types.size() < num_local_edges+num_ghost_edges)
    Kokkos::realloc(neigh_types, num_local_edges+num_ghost_edges);
  if (xyz.size() < 3*(num_local_edges+num_ghost_edges))
    Kokkos::realloc(xyz, 3*(num_local_edges+num_ghost_edges));
  if (r.size() < num_local_edges+num_ghost_edges)
    Kokkos::realloc(r, num_local_edges+num_ghost_edges);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,num_local_edges+num_ghost_edges));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,num_local_edges+num_ghost_edges));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0,3*(num_local_edges+num_ghost_edges)));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0,num_local_edges+num_ghost_edges));
  Kokkos::parallel_for("populate edge-based views",
    num_local_nodes+num_ghost_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = node_indices(ii);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      int ij = first_neigh(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        const int j = (d_neighbors(i,jj) & NEIGHMASK);
        const double dx = x(j,0) - x_i;
        const double dy = x(j,1) - y_i;
        const double dz = x(j,2) - z_i;
        const double r_squared = dx*dx + dy*dy + dz*dz;
        if (r_squared < r_cut_squared &&
            (!interaction_mode_rrnlb || is_local.test(j) || is_ghost.test(j))) {
          neigh_indices(ij) = j;
          neigh_types(ij) = mace_types(type(j)-1);
          xyz(3*ij) = dx;
          xyz(3*ij+1) = dy;
          xyz(3*ij+2) = dz;
          r(ij) = std::sqrt(r_squared);
          ij += 1;
        }
      }
  });

  // RRNLB needs ii-mapping for all edges; legacy path only uses local-edge segment.
  const int num_neigh_ii_indices =
      mace->interaction_mode_rrnlb ? num_local_edges+num_ghost_edges : num_local_edges;
  if (neigh_ii_indices.size() < num_neigh_ii_indices)
    Kokkos::realloc(neigh_ii_indices, num_neigh_ii_indices);
  auto neigh_ii_indices =
      Kokkos::subview(this->neigh_ii_indices, Kokkos::make_pair(0,num_neigh_ii_indices));
  Kokkos::parallel_for("populate neigh_ii_indices",
    num_neigh_ii_indices,
    KOKKOS_LAMBDA (const int ij) {
      const int j = neigh_indices(ij);
      for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ++ii) {
        if (node_indices(ii) == j) {
          neigh_ii_indices(ij) = ii;
          break;
        }
      }
    });
  Kokkos::fence();

  // ----- begin mace evaluation -----

  if (mace->interaction_mode_rrnlb) {
    if (rrnlb_neighbor_epoch_id < 0 || this->neighbor->ago == 0) rrnlb_neighbor_epoch_id += 1;
    mace->rrnlb_set_neighbor_epoch(rrnlb_neighbor_epoch_id);
    const int total_num_nodes = num_local_nodes + num_ghost_nodes;
    const int total_num_edges = num_local_edges + num_ghost_edges;
    mace->ensure_rrnlb_model_static_cache();
    mace->ensure_rrnlb_system_static_cache(total_num_nodes, node_types);
    mace->compute_node_energies_forces(
        total_num_nodes,
        node_types,
        num_neigh,
        neigh_ii_indices,
        neigh_types,
        xyz,
        r,
        first_neigh,
        total_num_edges);
  } else {
    if (mace->node_energies.size() < num_local_nodes)
      Kokkos::realloc(mace->node_energies, num_local_nodes);
    Kokkos::deep_copy(
      Kokkos::subview(mace->node_energies, Kokkos::make_pair(0, num_local_nodes)),
      0.0);
    if (mace->node_forces.size() < 3*(num_local_edges+num_ghost_edges))
      Kokkos::realloc(mace->node_forces, 3*(num_local_edges+num_ghost_edges));
    Kokkos::deep_copy(
      Kokkos::subview(
        mace->node_forces,
        Kokkos::make_pair(0, 3 * (num_local_edges + num_ghost_edges))),
      0.0);

    if (mace->has_zbl)
      mace->zbl.compute_ZBL(
        num_local_nodes, node_types, num_neigh, neigh_types,
        mace->atomic_numbers, r, xyz, mace->node_energies, mace->node_forces);

    mace->compute_Y(xyz);

    mace->compute_R0(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_A0(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types);
    mace->compute_A0_scaled(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_M0(num_local_nodes+num_ghost_nodes, node_types);
    mace->compute_H1(num_local_nodes+num_ghost_nodes);

    mace->compute_R1(num_local_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_Phi1(num_local_nodes, num_neigh, neigh_ii_indices);
    mace->compute_A1(num_local_nodes);
    mace->compute_A1_scaled(num_local_nodes, node_types, num_neigh, neigh_types, r);
    mace->compute_M1(num_local_nodes, node_types);
    mace->compute_H2(num_local_nodes, node_types);

    mace->compute_readouts(num_local_nodes, node_types);
    
    mace->reverse_H2(num_local_nodes, node_types, false);
    mace->reverse_M1(num_local_nodes, node_types);
    mace->reverse_A1_scaled(num_local_nodes, node_types, num_neigh, neigh_types, xyz, r);
    mace->reverse_A1(num_local_nodes);
    mace->reverse_Phi1(num_local_nodes, num_neigh, neigh_ii_indices, xyz, r, false, false);

    mace->reverse_H1(num_local_nodes+num_ghost_nodes);
    mace->reverse_M0(num_local_nodes+num_ghost_nodes, node_types);
    mace->reverse_A0_scaled(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, xyz, r);
    mace->reverse_A0(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, xyz, r);
  }

  // ----- end mace evaluation -----

  if (eflag_global) {
    auto node_energies = mace->node_energies;
    double energy;
    Kokkos::parallel_reduce("energy reduction", num_local_nodes, KOKKOS_LAMBDA (const int i, double& energy) {
        energy += node_energies(i);
      }, energy);
    eng_vdwl += energy;
  }

  if (eflag_atom) {
    auto d_eatom = k_eatom.template view<DeviceType>();
    auto node_energies = mace->node_energies;
    Kokkos::parallel_for("extract atomic energies", num_local_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(ii) += node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
  }

  auto mace_node_forces = mace->node_forces;
  Kokkos::parallel_for("force reduction",
    Kokkos::TeamPolicy<>(num_local_nodes+num_ghost_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = node_indices(ii);
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
        [&] (const int jj) {
          const int ij = first_neigh(ii) + jj;
          const int j = neigh_indices(ij);
          const double fx = mace_node_forces(3*ij);
          const double fy = mace_node_forces(3*ij+1);
          const double fz = mace_node_forces(3*ij+2);
          Kokkos::atomic_add(&f(j,0), fx);
          Kokkos::atomic_add(&f(j,1), fy);
          Kokkos::atomic_add(&f(j,2), fz);
          Kokkos::atomic_add(&f(i,0), -fx);
          Kokkos::atomic_add(&f(i,1), -fy);
          Kokkos::atomic_add(&f(i,2), -fz);
        });
    });

  if (rrnlb_debug_net_force_enabled()) {
    double local_fx = 0.0;
    double local_fy = 0.0;
    double local_fz = 0.0;
    Kokkos::parallel_reduce(
      "rrnlb_debug_net_force_fx_no_mpi",
      Kokkos::RangePolicy<DeviceType>(0, atom->nlocal),
      KOKKOS_LAMBDA (const int i, double& sum) {
        sum += f(i, 0);
      },
      local_fx);
    Kokkos::parallel_reduce(
      "rrnlb_debug_net_force_fy_no_mpi",
      Kokkos::RangePolicy<DeviceType>(0, atom->nlocal),
      KOKKOS_LAMBDA (const int i, double& sum) {
        sum += f(i, 1);
      },
      local_fy);
    Kokkos::parallel_reduce(
      "rrnlb_debug_net_force_fz_no_mpi",
      Kokkos::RangePolicy<DeviceType>(0, atom->nlocal),
      KOKKOS_LAMBDA (const int i, double& sum) {
        sum += f(i, 2);
      },
      local_fz);
    double net_force_local[3] = {local_fx, local_fy, local_fz};
    double net_force_global[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(net_force_local, net_force_global, 3, MPI_DOUBLE, MPI_SUM, world);
    if (comm->me == 0) {
      const double net_norm = std::sqrt(
        net_force_global[0] * net_force_global[0]
        + net_force_global[1] * net_force_global[1]
        + net_force_global[2] * net_force_global[2]);
      utils::logmesg(
        lmp,
        "[RRNLB debug net force] mode=no_mpi_message_passing step={} fx={:.12e} fy={:.12e} "
        "fz={:.12e} norm={:.12e}\n",
        update->ntimestep,
        net_force_global[0],
        net_force_global[1],
        net_force_global[2],
        net_norm);
    }
  }

  if (vflag_global) {
    Kokkos::View<double*,Kokkos::LayoutRight> v("v", 6);
    Kokkos::deep_copy(v, 0.0);
    Kokkos::parallel_for("virial reduction",
      Kokkos::TeamPolicy<>(num_local_nodes+num_ghost_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
          [&] (const int jj) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = mace_node_forces(3*ij);
            const double f_y = mace_node_forces(3*ij+1);
            const double f_z = mace_node_forces(3*ij+2);
            Kokkos::atomic_add(&v(0), x*f_x);
            Kokkos::atomic_add(&v(1), y*f_y);
            Kokkos::atomic_add(&v(2), z*f_z);
            Kokkos::atomic_add(&v(3), 0.5*(x*f_y + y*f_x));
            Kokkos::atomic_add(&v(4), 0.5*(x*f_z + z*f_x));
            Kokkos::atomic_add(&v(5), 0.5*(y*f_z + z*f_y));
          });
      });
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    virial[0] += h_v(0);
    virial[1] += h_v(1);
    virial[2] += h_v(2);
    virial[3] += h_v(3);
    virial[4] += h_v(4);
    virial[5] += h_v(5);
  }

  rrnlb_maybe_emit_phase_stats("no_mpi_message_passing");

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/kk.");
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairSymmetrixMACEKokkos<LMPDeviceType,double,double>;
template class PairSymmetrixMACEKokkos<LMPDeviceType,float,float>;
template class PairSymmetrixMACEKokkos<LMPDeviceType,float,double>;
#ifdef LMP_KOKKOS_GPU
template class PairSymmetrixMACEKokkos<LMPHostType,double,double>;
template class PairSymmetrixMACEKokkos<LMPHostType,float,float>;
template class PairSymmetrixMACEKokkos<LMPHostType,float,double>;
#endif
}
