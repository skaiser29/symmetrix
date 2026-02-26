/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
#define PairSymmetrixMACEKokkosDeviceDouble PairSymmetrixMACEKokkos<LMPDeviceType,double,double>
#define PairSymmetrixMACEKokkosHostDouble PairSymmetrixMACEKokkos<LMPHostType,double,double>
#define PairSymmetrixMACEKokkosDeviceFloat PairSymmetrixMACEKokkos<LMPDeviceType,float,float>
#define PairSymmetrixMACEKokkosHostFloat PairSymmetrixMACEKokkos<LMPHostType,float,float>
#define PairSymmetrixMACEKokkosDeviceMixed PairSymmetrixMACEKokkos<LMPDeviceType,float,double>
#define PairSymmetrixMACEKokkosHostMixed PairSymmetrixMACEKokkos<LMPHostType,float,double>

PairStyle(symmetrix/mace/kk,PairSymmetrixMACEKokkosDeviceDouble);
PairStyle(symmetrix/mace/kk/device,PairSymmetrixMACEKokkosDeviceDouble);
PairStyle(symmetrix/mace/kk/host,PairSymmetrixMACEKokkosHostDouble);
PairStyle(symmetrix/mace/float32/kk,PairSymmetrixMACEKokkosDeviceFloat);
PairStyle(symmetrix/mace/float32/kk/device,PairSymmetrixMACEKokkosDeviceFloat);
PairStyle(symmetrix/mace/float32/kk/host,PairSymmetrixMACEKokkosHostFloat);
PairStyle(symmetrix/mace/mixed/kk,PairSymmetrixMACEKokkosDeviceMixed);
PairStyle(symmetrix/mace/mixed/kk/device,PairSymmetrixMACEKokkosDeviceMixed);
PairStyle(symmetrix/mace/mixed/kk/host,PairSymmetrixMACEKokkosHostMixed);

#undef PairSymmetrixMACEKokkosDeviceDouble
#undef PairSymmetrixMACEKokkosHostDouble
#undef PairSymmetrixMACEKokkosDeviceFloat
#undef PairSymmetrixMACEKokkosHostFloat
#undef PairSymmetrixMACEKokkosDeviceMixed
#undef PairSymmetrixMACEKokkosHostMixed
// clang-format on
#else

#ifndef LMP_PAIR_SYMMETRIX_MACE_KOKKOS_H
#define LMP_PAIR_SYMMETRIX_MACE_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"

#include "mace_kokkos.hpp"

namespace LAMMPS_NS {

template<class DeviceType, typename Precision = double, typename AccumPrecision = Precision>
class PairSymmetrixMACEKokkos : public Pair, public KokkosBase {

 public:
  PairSymmetrixMACEKokkos(class LAMMPS *);
  ~PairSymmetrixMACEKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d&, int, int*) override;
  void unpack_forward_comm(int, int, double *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  int pack_reverse_comm(int, int, double *) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  void unpack_reverse_comm(int, int *, double *) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d&) override;
  void compute_no_domain_decomposition(int, int);
  void compute_mpi_message_passing(int, int);
 void compute_no_mpi_message_passing(int, int);

 protected:
  std::string mode;
  std::unique_ptr<MACEKokkos<Precision, AccumPrecision>> mace;
  Kokkos::View<int*> mace_types;
  Kokkos::View<Precision***,Kokkos::LayoutRight> H1, H1_adj;
  Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_feat0;
  Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_feat0_adj;
  typename MACEKokkos<Precision, AccumPrecision>::RRNLBLayerCacheKokkos rrnlb_cache0, rrnlb_cache1;
  Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_sender_embed_ws;
  Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_interaction0_out_ws;
  Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_skip0_ws;
  Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_product0_in_ws;
  Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_feat0_local_ws;
  Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_interaction1_out_ws;
  Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_skip1_ws;
  Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_product1_in_ws;
  Kokkos::View<Precision**,Kokkos::LayoutRight> rrnlb_feat1_ws;
  Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_feat0_adj_local_ws;
  Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_feat1_adj_ws;
  Kokkos::View<double**,Kokkos::LayoutRight> rrnlb_feat1_double_ws;
  Kokkos::View<double*> rrnlb_readout2_out_ws;
  Kokkos::View<double**,Kokkos::LayoutRight> rrnlb_readout2_adj_ws;
  Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_product1_in_adj_ws;
  Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_interaction1_adj_ws;
  Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_feat0_from_layer1_adj_ws;
  Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_feat0_adj_nodes_ws;
  Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_product0_in_adj_ws;
  Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_interaction0_adj_ws;
  Kokkos::View<AccumPrecision**,Kokkos::LayoutRight> rrnlb_sender_embed_adj_ws;

  // neighbor list variables
  Kokkos::View<int*> node_indices;
  Kokkos::View<int*> node_types;
  Kokkos::View<int*> num_neigh;
  Kokkos::View<int*> first_neigh;
  Kokkos::View<int*> neigh_types;
  Kokkos::View<int*> neigh_indices;
  Kokkos::View<int*> neigh_ii_indices;
  Kokkos::View<double*> xyz;
  Kokkos::View<double*> r;
  Kokkos::View<int*> rrnlb_edge_to_receiver;
  long long rrnlb_neighbor_epoch_id = -1;
  long long rrnlb_phase_step_counter = 0;
  bool rrnlb_phase_csv_header_written = false;
  std::string rrnlb_phase_csv_path;

  const std::array<std::string,118> periodic_table =
    { "H", "He",
     "Li", "Be",                                                              "B",  "C",  "N",  "O",  "F", "Ne",
     "Na", "Mg",                                                             "Al", "Si",  "P",  "S", "Cl", "Ar",
     "K",  "Ca", "Sc", "Ti",  "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
     "Rb", "Sr",  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",  "I", "Xe",
     "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                       "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
     "Fr", "Ra", "Ac", "Th", "Pa",  "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                       "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

  virtual void allocate();
  void rrnlb_maybe_emit_phase_stats(const char *mode_tag);

 private:
  DAT::ttransform_kkacc_1d k_eatom;

};
}    // namespace LAMMPS_NS

#endif
#endif
