"""ASE Calculator for symmetrix implementation of equivariant graph neural
network library

This file was written and publicly released by Dr. Noam Bernstein as part of his
work for the U. S. Government, and is not subject to copyright.
"""
import json
import logging
import atexit
import gc
import threading
import weakref
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np

try:
    from matscipy.neighbours import neighbour_list as neighbor_list
except:
    logging.warning("Symmetrix using slow ase.neighborlist.neighbor_list")
    from ase.neighborlist import neighbor_list

from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

_KOKKOS_CALC_REGISTRY = weakref.WeakSet()
_KOKKOS_ATEXIT_REGISTERED = False
_KOKKOS_REGISTRY_LOCK = threading.Lock()


def _release_kokkos_evaluator(calc):
    # Ensure Kokkos-backed C++ state is torn down before runtime finalize.
    if hasattr(calc, "evaluator"):
        calc.evaluator = None
    if hasattr(calc, "_symmetrix"):
        calc._symmetrix = None


def _track_kokkos_calculator(calc):
    try:
        _KOKKOS_CALC_REGISTRY.add(calc)
    except TypeError:
        # Best-effort tracking; object may not be weakref-able in unusual subclasses.
        pass


def _register_kokkos_atexit(symmetrix_ext):
    global _KOKKOS_ATEXIT_REGISTERED
    with _KOKKOS_REGISTRY_LOCK:
        if _KOKKOS_ATEXIT_REGISTERED:
            return

        def _finalize_kokkos_runtime():
            try:
                live_calcs = list(_KOKKOS_CALC_REGISTRY)
            except Exception:
                live_calcs = []
            for calc in live_calcs:
                try:
                    _release_kokkos_evaluator(calc)
                except Exception:
                    pass
            try:
                _KOKKOS_CALC_REGISTRY.clear()
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass
            try:
                if symmetrix_ext._kokkos_is_initialized():
                    symmetrix_ext._finalize_kokkos()
            except Exception as exc:
                logging.debug(f"Kokkos finalization at exit failed: {exc}")

        atexit.register(_finalize_kokkos_runtime)
        _KOKKOS_ATEXIT_REGISTERED = True


class Symmetrix(Calculator):
    """ASE Calculator using symmetrix library to evaluate equivariant graph neural network 
    potential energy functions

    Parameters
    ----------
    model_file: str
        JSON-format model file used for potential energy

    Notes
    -----
    Wraps symmetrix library from https://github.com/wcwitt/symmetrix via python interface at https://pypi.org/project/symmetrix/
    """
    implemented_properties = ['energy', 'free_energy', 'energies', 'forces', 'stress']


    def __init__(self, model_file, dtype="float64", use_kokkos=True, **kwargs):
        Calculator.__init__(self, **kwargs)
        if dtype not in ["float32", "float64"]:
            raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes are 'float64' and 'float32'.")
        self.use_kokkos = use_kokkos
        self._torch_calc = None
        self._torch_backend = False
        self._symmetrix = None
        self.evaluator = None

        runtime_backend_data = self._load_runtime_backend_data(model_file)
        if runtime_backend_data and runtime_backend_data.get("runtime_backend") == "torch_mace":
            self._init_torch_backend(
                runtime_backend_data.get("model_path", model_file),
                runtime_backend_data.get("head", kwargs.get("head")),
                dtype,
                kwargs,
            )
            return

        try:
            from . import symmetrix as symmetrix_ext
            self._symmetrix = symmetrix_ext
        except (ModuleNotFoundError, ImportError):
            symmetrix_ext = None

        if symmetrix_ext is not None:
            if use_kokkos and not hasattr(symmetrix_ext, "MACEKokkos"):
                raise RuntimeError("Symmetrix was built without Kokkos support.")
            if self.use_kokkos:
                _register_kokkos_atexit(symmetrix_ext)
                if not symmetrix_ext._kokkos_is_initialized():
                    symmetrix_ext._init_kokkos()
                MACE = symmetrix_ext.MACEKokkos if dtype == "float64" else symmetrix_ext.MACEKokkosFloat
            else:
                if dtype == "float32":
                    raise ValueError(f"dtype '{dtype}' requires `use_kokkos = True`")
                MACE = symmetrix_ext.MACE
            try:
                self.evaluator = MACE(str(model_file))
            except RuntimeError: # expecting json.exception.parse_error.101
                # import this here so that torch/mace support isn't needed if file is already symmetrix json
                from .extract_mace_data import extract_mace_data
                kwargs_extract = {k: v for k, v in kwargs.items()
                    if k in ['species',
                             'head',
                             'num_spline_points']}
                logging.warning(f"Converting model from pytorch model to symmetrix dict with {kwargs_extract}")
                data = extract_mace_data(model_file, **kwargs_extract)
                if data.get("runtime_backend") == "torch_mace":
                    self._init_torch_backend(
                        data.get("model_path", model_file),
                        data.get("head", kwargs_extract.get("head")),
                        dtype,
                        kwargs,
                    )
                    return
                with NamedTemporaryFile("w") as fout:
                    logging.warning(f"Converting via NamedTemporaryFile {fout.name}")
                    fout.write(json.dumps(data))
                    self.evaluator = MACE(fout.name)
            self.cutoff = self.evaluator.r_cut
            if self.use_kokkos and not self._torch_backend:
                _track_kokkos_calculator(self)
            return

        from .extract_mace_data import extract_mace_data
        kwargs_extract = {k: v for k, v in kwargs.items()
            if k in ['species',
                     'head',
                     'num_spline_points']}
        data = extract_mace_data(model_file, **kwargs_extract)
        if data.get("runtime_backend") == "torch_mace":
            self._init_torch_backend(
                data.get("model_path", model_file),
                data.get("head", kwargs_extract.get("head")),
                dtype,
                kwargs,
            )
            return
        raise RuntimeError(
            "Symmetrix extension module is not available, and this model does not "
            "support torch fallback. Build/install the compiled symmetrix module."
        )

    @staticmethod
    def _load_runtime_backend_data(model_file):
        path = Path(str(model_file))
        if path.suffix.lower() != ".json" or not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(data, dict) and data.get("runtime_backend"):
            return data
        return None

    def _init_torch_backend(self, model_file, head, dtype, kwargs):
        import torch
        from mace.calculators.mace import MACECalculator

        if self.use_kokkos:
            logging.warning(
                "Using torch MACE runtime fallback on CPU. "
                "Kokkos acceleration is unavailable for this backend."
            )
        loaded = torch.load(str(model_file), map_location="cpu", weights_only=False)
        models = None
        head_name = head
        if isinstance(loaded, list):
            models = loaded
        elif isinstance(loaded, torch.nn.Module):
            models = [loaded]
        elif hasattr(loaded, "model"):
            wrapper = loaded.model
            if isinstance(wrapper, torch.nn.Module):
                if hasattr(wrapper, "model") and isinstance(wrapper.model, torch.nn.Module):
                    models = [wrapper.model]
                    if head_name is None and hasattr(wrapper, "head") and hasattr(wrapper.model, "heads"):
                        try:
                            head_idx = int(wrapper.head.detach().cpu().item())
                            if 0 <= head_idx < len(wrapper.model.heads):
                                head_name = wrapper.model.heads[head_idx]
                        except Exception:
                            pass
                else:
                    models = [wrapper]

        self._torch_backend = True
        self.cutoff = float("nan")
        if models is not None:
            self._torch_calc = MACECalculator(
                models=models,
                head=head_name,
                device=kwargs.get("device", "cpu"),
                default_dtype=dtype,
            )
        else:
            self._torch_calc = MACECalculator(
                model_path=str(model_file),
                head=head_name,
                device=kwargs.get("device", "cpu"),
                default_dtype=dtype,
            )

    def __del__(self):
        try:
            if getattr(self, "use_kokkos", False) and not getattr(self, "_torch_backend", False):
                _release_kokkos_evaluator(self)
        except Exception:
            pass

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if self._torch_backend:
            atoms_ref = self.atoms.copy()
            atoms_ref.calc = self._torch_calc
            energy = float(atoms_ref.get_potential_energy())
            try:
                energies = np.asarray(atoms_ref.get_potential_energies())
            except PropertyNotImplementedError:
                energies = np.full(len(atoms_ref), energy / max(len(atoms_ref), 1))
            forces = np.asarray(atoms_ref.get_forces())
            self.results['energy'] = self.results['free_energy'] = energy
            self.results['energies'] = energies
            self.results['forces'] = forces
            try:
                self.results['stress'] = np.asarray(atoms_ref.get_stress())
            except PropertyNotImplementedError:
                pass
            return

        ase_atomic_numbers = self.atoms.get_atomic_numbers().tolist()
        mace_atomic_numbers = self.evaluator.atomic_numbers
        i_list, j_list, r, xyz = neighbor_list('ijdD', self.atoms, self.cutoff)
        num_nodes = np.max(i_list) + 1
        node_types = [mace_atomic_numbers.index(ase_atomic_numbers[i]) for i in range(num_nodes)]
        num_neigh = np.bincount(j_list, minlength=num_nodes)
        neigh_types = [mace_atomic_numbers.index(ase_atomic_numbers[j]) for j in j_list]
        self.evaluator.compute_node_energies_forces(
            num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)

        self.results['energy'] = self.results['free_energy'] = np.sum(self.evaluator.node_energies)
        self.results['energies'] = np.asarray(self.evaluator.node_energies)

        pair_forces = np.asarray(self.evaluator.node_forces).reshape((-1, 3))
        pair_forces = pair_forces[:len(i_list), :]  # currently, `evaluator.node_forces` is a container
                                                    # which can grow larger than the actual number of pairs

        # atom forces from pair_forces
        N_atoms = len(self.atoms)
        atom_forces = np.zeros((N_atoms, 3))
        atom_forces[:, 0] = np.bincount(j_list, weights=pair_forces[:, 0], minlength=N_atoms) - np.bincount(i_list, weights=pair_forces[:, 0], minlength=N_atoms)
        atom_forces[:, 1] = np.bincount(j_list, weights=pair_forces[:, 1], minlength=N_atoms) - np.bincount(i_list, weights=pair_forces[:, 1], minlength=N_atoms)
        atom_forces[:, 2] = np.bincount(j_list, weights=pair_forces[:, 2], minlength=N_atoms) - np.bincount(i_list, weights=pair_forces[:, 2], minlength=N_atoms)

        self.results['forces'] = atom_forces

        # stress from pair_forces
        self.results['stress'] = full_3x3_to_voigt_6_stress((-pair_forces.T @ xyz)  / self.atoms.get_volume())
