import torch
torch.serialization.add_safe_globals([slice])

import os
import logging
import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from e3nn.o3 import Irrep, Irreps, Linear, wigner_3j
from mace.modules.radial import ZBLBasis
from mace.tools.cg import U_matrix_real
from mace.tools.scripts_utils import remove_pt_head, extract_config_mace_model

from ase.data import chemical_symbols


def _remove_pt_head_robust(model, head_to_keep=None):
    """remove_pt_head variant that tolerates scalar TorchScript constants in state_dict.

    Some serialized wrappers (e.g. LAMMPS_MLIAP_MACE) carry scalar graph constants under
    `readouts.*` keys. Upstream `remove_pt_head` indexes `param.shape[0]` for all readout
    entries and fails on those 0-d tensors.
    """
    try:
        return remove_pt_head(model, head_to_keep)
    except IndexError as exc:
        if "tuple index out of range" not in str(exc):
            raise

    if not hasattr(model, "heads") or len(model.heads) <= 1:
        raise ValueError("Model must be a multihead model with more than one head")

    if head_to_keep is None:
        try:
            head_idx = next(i for i, h in enumerate(model.heads) if h != "pt_head")
        except StopIteration as e:
            raise ValueError("No non-PT head found in model") from e
    else:
        try:
            head_idx = model.heads.index(head_to_keep)
        except ValueError as e:
            raise ValueError(f"Head {head_to_keep} not found in model") from e

    model_config = extract_config_mace_model(model)
    model_config["heads"] = [model.heads[head_idx]]
    model_config["atomic_energies"] = (
        model.atomic_energies_fn.atomic_energies[head_idx]
        .unsqueeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    model_config["atomic_inter_scale"] = model.scale_shift.scale[head_idx].item()
    model_config["atomic_inter_shift"] = model.scale_shift.shift[head_idx].item()
    mlp_count_irreps = model_config["MLP_irreps"].count((0, 1))

    new_model = model.__class__(**model_config)
    state_dict = model.state_dict()
    new_state_dict = {}
    num_heads = len(model.heads)

    for name, param in state_dict.items():
        # Wrapped checkpoints can store 1-D readout parameters with a leading
        # singleton axis, e.g. [1, N] instead of [N]. Normalize to canonical
        # shape so head slicing logic remains consistent with upstream.
        if (
            param.ndim > 1
            and param.shape[0] == 1
            and ("readouts" in name or "embedding_readout.linear" in name)
        ):
            param = param.reshape(-1)

        if "atomic_energies" in name:
            new_state_dict[name] = param[head_idx:head_idx + 1]
            continue
        if "scale" in name or "shift" in name:
            new_state_dict[name] = param[head_idx:head_idx + 1]
            continue
        if "embedding_readout.linear" in name:
            new_state_dict[name] = param.reshape(-1, num_heads)[:, head_idx].flatten()
            continue
        if "readouts" not in name:
            new_state_dict[name] = param
            continue

        # TorchScript constants in wrapped checkpoints can be scalar tensors.
        # Keep them unchanged and let strict=False loading ignore extras.
        if param.ndim == 0:
            new_state_dict[name] = param
            continue

        channels_per_head = param.shape[0] // num_heads
        start_idx = head_idx * channels_per_head
        end_idx = start_idx + channels_per_head
        if "linear_2.weight" in name:
            end_idx = start_idx + channels_per_head // 2

        if "linear.weight" in name:
            new_state_dict[name] = param.reshape(-1, num_heads)[:, head_idx].flatten()
        elif "linear_1.weight" in name:
            new_state_dict[name] = param.reshape(-1, num_heads, mlp_count_irreps)[
                :, head_idx, :
            ].flatten()
        elif "linear_1.bias" in name:
            if param.shape == torch.Size([0]):
                continue
            new_state_dict[name] = param.reshape(num_heads, mlp_count_irreps)[
                head_idx, :
            ].flatten()
        elif "linear_mid.weight" in name:
            new_state_dict[name] = param.reshape(
                num_heads,
                mlp_count_irreps,
                num_heads,
                mlp_count_irreps,
            )[head_idx, :, head_idx, :].flatten() / (num_heads ** 0.5)
        elif "linear_mid.bias" in name:
            if param.shape == torch.Size([0]):
                continue
            new_state_dict[name] = param.reshape(num_heads, mlp_count_irreps)[
                head_idx, :
            ].flatten()
        elif "linear_2.weight" in name:
            new_state_dict[name] = param.reshape(num_heads, -1, num_heads)[
                head_idx, :, head_idx
            ].flatten() / (num_heads ** 0.5)
        elif "linear_2.bias" in name:
            if param.shape == torch.Size([0]):
                continue
            new_state_dict[name] = param[head_idx].flatten()
        else:
            new_state_dict[name] = param[start_idx:end_idx]

    # Some wrapped checkpoints store vectors with a leading singleton axis
    # (e.g., [1, N] instead of [N]). Align all provided tensors to the target
    # state dict shapes to keep robust loading behavior.
    target_state = new_model.state_dict()
    aligned_state_dict = {}
    for name, param in new_state_dict.items():
        target_param = target_state.get(name, None)
        if target_param is None:
            continue
        aligned_param = param
        if aligned_param.shape != target_param.shape:
            if aligned_param.numel() != target_param.numel():
                raise RuntimeError(
                    f"Failed to align parameter '{name}' from shape "
                    f"{tuple(aligned_param.shape)} to {tuple(target_param.shape)}"
                )
            aligned_param = aligned_param.reshape(target_param.shape)
        if aligned_param.dtype != target_param.dtype:
            aligned_param = aligned_param.to(dtype=target_param.dtype)
        aligned_state_dict[name] = aligned_param

    new_model.load_state_dict(aligned_state_dict, strict=False)
    return new_model

def extract_mace_data(model, species, head=None, num_spline_points=256):
    """Extract data from pytorch model file into structure that can be
    written as symmetrix JSON data file

    Parameters
    ----------
    model: str / Path
        path to pytorch model file
    species: list(int / str)
        list of atomic numbers or chemical symbols to extract
    head: str, default None
        head to keep, if multihead model, default same as mace.tools.scripts_utils.remove_pt_head
    num_spline_points: int, default 256
        number of spline points to approximate various functions

    Returns
    -------
    output_data: dict with symmetrix model data
    """
    model_path = os.fspath(model)
    loaded = torch.load(
        model_path,
        map_location=torch.device('cpu'),
        weights_only=False
    )
    default_head = head
    if isinstance(loaded, list):
        if len(loaded) == 0 or not isinstance(loaded[0], torch.nn.Module):
            raise RuntimeError(f"Unsupported list model container from {model_path}")
        model = loaded[0]
    elif isinstance(loaded, torch.nn.Module):
        model = loaded
    elif hasattr(loaded, "model"):
        wrapper = loaded.model
        if isinstance(wrapper, torch.nn.Module) and hasattr(wrapper, "model") and isinstance(wrapper.model, torch.nn.Module):
            model = wrapper.model
            if default_head is None and hasattr(wrapper, "head") and hasattr(model, "heads"):
                try:
                    head_idx = int(wrapper.head.detach().cpu().item())
                    if 0 <= head_idx < len(model.heads):
                        default_head = model.heads[head_idx]
                except Exception:
                    pass
        elif isinstance(wrapper, torch.nn.Module):
            model = wrapper
        else:
            raise RuntimeError(f"Unsupported wrapped model type from {model_path}: {type(wrapper)}")
    else:
        raise RuntimeError(f"Unsupported model object from {model_path}: {type(loaded)}")

    # LAMMPS MLIAP wrapper .pt files can store TorchScript-fused product weights
    # with a state-dict layout that cannot be faithfully converted by the native
    # Symmetrix extractor. If a sibling source .model exists, prefer that.
    if isinstance(model, torch.nn.Module):
        state_keys = model.state_dict().keys()
        has_fused_product_state = any(
            key.startswith("products.0.symmetric_contractions.weight")
            for key in state_keys
        )
        if has_fused_product_state:
            suffix = "-mliap_lammps.pt"
            candidate_path = None
            if model_path.endswith(suffix):
                candidate_path = model_path[:-len(suffix)]
            if candidate_path is not None and os.path.exists(candidate_path):
                logging.warning(
                    "Detected fused MLIAP wrapper state layout. "
                    f"Loading source model '{candidate_path}' for extraction."
                )
                loaded_src = torch.load(
                    candidate_path,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
                if isinstance(loaded_src, list):
                    if len(loaded_src) == 0 or not isinstance(loaded_src[0], torch.nn.Module):
                        raise RuntimeError(
                            f"Unsupported list model container from {candidate_path}"
                        )
                    model = loaded_src[0]
                elif isinstance(loaded_src, torch.nn.Module):
                    model = loaded_src
                else:
                    raise RuntimeError(
                        f"Unsupported source model object from {candidate_path}: {type(loaded_src)}"
                    )
                model_path = candidate_path
            else:
                raise RuntimeError(
                    "Cannot extract native Symmetrix data from fused LAMMPS MLIAP "
                    f"wrapper '{model_path}' without a source .model file. "
                    "Please pass the original .model path."
                )
    model = model.to(torch.float64)

    if species is None:
        species = []

    # extract atomic numbers
    atomic_numbers = []
    for sp in species:
        try:
            Z = int(sp)
        except ValueError:
            try:
                Z = chemical_symbols.index(sp)
            except ValueError as exc:
                raise ValueError("Failed to parse {sp} as atomic number or chemical species") from exc
        atomic_numbers.append(Z)

    # ensure that splines goes smoothly to 0 at outer cutoff
    spline_bc_type = ("not-a-knot", "clamped")

    ### ----- EXTRACT SINGLE HEAD -----

    head_extraction_failed = False
    if hasattr(model, 'heads') and len(model.heads) != 1:
        torch.set_default_dtype(next(model.parameters()).dtype)
        try:
            model = _remove_pt_head_robust(model, default_head)
            head = default_head
        except Exception as exc:
            head_extraction_failed = True
            head = default_head
            logging.warning(
                "Failed to strip to a single head during extraction. "
                f"Error: {type(exc).__name__}: {exc}"
            )

    ### ----- CHECK FOR COMPATIBILITY -----

    if len(model.interactions) != 2:
        raise RuntimeError("Currently, symmetrix only supports two-layer MACE models.")

    from mace.modules.blocks import (
        RealAgnosticInteractionBlock,
        RealAgnosticDensityInteractionBlock,
        RealAgnosticResidualInteractionBlock,
        RealAgnosticDensityResidualInteractionBlock,
        RealAgnosticResidualNonLinearInteractionBlock,
    )
    interaction_0_supported = (
        isinstance(model.interactions[0], RealAgnosticInteractionBlock)
        or isinstance(model.interactions[0], RealAgnosticDensityInteractionBlock)
        or isinstance(model.interactions[0], RealAgnosticResidualNonLinearInteractionBlock)
    )
    if not interaction_0_supported:
        raise RuntimeError(
            "Currently, symmetrix only supports MACE models whose first interaction is "
            "RealAgnosticInteractionBlock, RealAgnosticDensityInteractionBlock, or "
            "RealAgnosticResidualNonLinearInteractionBlock.")

    interaction_1_supported = (
        isinstance(model.interactions[1], RealAgnosticResidualInteractionBlock)
        or isinstance(model.interactions[1], RealAgnosticDensityResidualInteractionBlock)
        or isinstance(model.interactions[1], RealAgnosticResidualNonLinearInteractionBlock)
    )
    if not interaction_1_supported:
        raise RuntimeError(
            "Currently, symmetrix only supports MACE models whose second interaction is "
            "RealAgnosticResidualInteractionBlock, RealAgnosticDensityResidualInteractionBlock, or "
            "RealAgnosticResidualNonLinearInteractionBlock.")

    has_residual_nonlinear = (
        isinstance(model.interactions[0], RealAgnosticResidualNonLinearInteractionBlock)
        or isinstance(model.interactions[1], RealAgnosticResidualNonLinearInteractionBlock)
    )
    all_residual_nonlinear = (
        isinstance(model.interactions[0], RealAgnosticResidualNonLinearInteractionBlock)
        and isinstance(model.interactions[1], RealAgnosticResidualNonLinearInteractionBlock)
    )

    if head_extraction_failed:
        raise RuntimeError(
            "Failed to reduce multihead model to a single head for native Symmetrix extraction."
        )

    if (model.spherical_harmonics._lmax != 3):
        raise RuntimeError("Currently, symmetrix only supports MACE models with l_max=3.")

    ### ----- HELPER FUNCTION -----

    def linear_simplify(linear):
        simplified = Linear(Irreps(linear.irreps_in).simplify(),
                            Irreps(linear.irreps_out).simplify())
        simplified.weight = linear.weight
        simplified.bias = linear.bias
        return simplified

    def evaluate_radial_features_and_cutoff(r_values, model_i, model_j):
        radial_out = model.radial_embedding(
            torch.tensor(r_values, dtype=torch.get_default_dtype()).unsqueeze(-1),
            torch.eye(len(model.atomic_numbers), dtype=torch.get_default_dtype()),
            torch.tensor([[model_i], [model_j]], dtype=torch.int64),
            model.atomic_numbers,
        )
        if isinstance(radial_out, tuple):
            radial_features, cutoff = radial_out
            if cutoff is not None:
                cutoff = cutoff.reshape(-1, 1)
            return radial_features, cutoff
        return radial_out, None

    def _safe_int(value, default=None):
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _safe_float(value, default=None):
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _instruction_get(ins, attr_name, tuple_index, default=None):
        value = getattr(ins, attr_name, None)
        if value is not None:
            return value
        try:
            return ins[tuple_index]
        except Exception:
            return default

    def _serialize_irreps(irreps):
        irreps_obj = Irreps(irreps)
        return {
            "str": str(irreps_obj),
            "dim": int(irreps_obj.dim),
            "num_irreps": int(irreps_obj.num_irreps),
            "lmax": int(irreps_obj.lmax),
            "parts": [
                {
                    "mul": int(mul),
                    "l": int(ir.l),
                    "p": int(ir.p),
                    "ir": str(ir),
                }
                for mul, ir in irreps_obj
            ],
        }

    def _serialize_linear(linear):
        layer = {
            "irreps_in": _serialize_irreps(linear.irreps_in),
            "irreps_out": _serialize_irreps(linear.irreps_out),
            "instructions": [],
        }
        for idx, ins in enumerate(linear.instructions):
            w = linear.weight_view_for_instruction(idx).numpy(force=True)
            path_shape = _instruction_get(ins, "path_shape", 2, tuple(w.shape))
            path_shape = [int(v) for v in path_shape]
            layer["instructions"].append(
                {
                    "index": int(idx),
                    "i_in": _safe_int(_instruction_get(ins, "i_in", 0), -1),
                    "i_out": _safe_int(_instruction_get(ins, "i_out", 1), -1),
                    "path_shape": path_shape,
                    "path_weight": _safe_float(_instruction_get(ins, "path_weight", 3), 1.0),
                    "weight_numel": int(w.size),
                    "weight_values": w.reshape(-1).tolist(),
                }
            )
        bias = getattr(linear, "bias", None)
        if bias is not None:
            layer["bias_values"] = bias.numpy(force=True).reshape(-1).tolist()
        return layer

    def _serialize_conv_tp_instructions(conv_tp):
        serialized = []
        for idx, ins in enumerate(conv_tp.instructions):
            path_shape = _instruction_get(ins, "path_shape", 6, ())
            path_shape = [int(v) for v in path_shape]
            serialized.append(
                {
                    "index": int(idx),
                    "i_in1": _safe_int(_instruction_get(ins, "i_in1", 0), -1),
                    "i_in2": _safe_int(_instruction_get(ins, "i_in2", 1), -1),
                    "i_out": _safe_int(_instruction_get(ins, "i_out", 2), -1),
                    "connection_mode": str(_instruction_get(ins, "connection_mode", 3, "")),
                    "has_weight": bool(_instruction_get(ins, "has_weight", 4, True)),
                    "path_weight": _safe_float(_instruction_get(ins, "path_weight", 5), 1.0),
                    "path_shape": path_shape,
                }
            )
        return serialized

    def _irreps_offsets(irreps):
        offsets = []
        offset = 0
        for mul, ir in Irreps(irreps):
            dim = int(mul) * (2 * int(ir.l) + 1)
            offsets.append(
                {
                    "offset": int(offset),
                    "dim": int(dim),
                    "mul": int(mul),
                    "l": int(ir.l),
                    "p": int(ir.p),
                }
            )
            offset += dim
        return offsets

    def _serialize_conv_tp_cg_maps(conv_tp):
        irreps_in1_offsets = _irreps_offsets(conv_tp.irreps_in1)
        irreps_in2_offsets = _irreps_offsets(conv_tp.irreps_in2)
        irreps_out_offsets = _irreps_offsets(conv_tp.irreps_out)
        dim_in1 = int(Irreps(conv_tp.irreps_in1).dim)
        dim_in2 = int(Irreps(conv_tp.irreps_in2).dim)
        dim_out = int(Irreps(conv_tp.irreps_out).dim)
        weight_numel = int(conv_tp.weight_numel)

        x = torch.zeros((1, dim_in1), dtype=torch.get_default_dtype())
        y = torch.zeros((1, dim_in2), dtype=torch.get_default_dtype())
        w = torch.zeros((1, weight_numel), dtype=torch.get_default_dtype())

        maps = []
        weight_offset = 0
        for idx, ins in enumerate(conv_tp.instructions):
            i_in1 = _safe_int(_instruction_get(ins, "i_in1", 0), -1)
            i_in2 = _safe_int(_instruction_get(ins, "i_in2", 1), -1)
            i_out = _safe_int(_instruction_get(ins, "i_out", 2), -1)
            in1_meta = irreps_in1_offsets[i_in1]
            in2_meta = irreps_in2_offsets[i_in2]
            out_meta = irreps_out_offsets[i_out]
            mul = in1_meta["mul"]
            if mul <= 0:
                raise RuntimeError(f"Invalid conv instruction multiplicity for instruction {idx}.")
            terms = []
            for m_in2 in range(2 * in2_meta["l"] + 1):
                # e3nn flatten order is (mul, m), so m-index is the fast axis.
                y_index = in2_meta["offset"] + m_in2
                y_lm = in2_meta["l"] * (in2_meta["l"] + 1) + (m_in2 - in2_meta["l"])
                for m_in1 in range(2 * in1_meta["l"] + 1):
                    # Use multiplicity channel 0 to identify the CG map terms.
                    x_index = in1_meta["offset"] + m_in1
                    x.zero_()
                    y.zero_()
                    w.zero_()
                    x[0, x_index] = 1.0
                    y[0, y_index] = 1.0
                    w[0, weight_offset] = 1.0
                    out = conv_tp(x, y, w).numpy(force=True).reshape(-1)
                    for m_out in range(2 * out_meta["l"] + 1):
                        out_index = out_meta["offset"] + m_out
                        coeff = float(out[out_index])
                        if abs(coeff) > 1e-12:
                            terms.append(
                                {
                                    "m_out": int(m_out),
                                    "m_in1": int(m_in1),
                                    "y_lm": int(y_lm),
                                    "coeff": coeff,
                                }
                            )
            maps.append(
                {
                    "instruction_index": int(idx),
                    "i_in1": int(i_in1),
                    "i_in2": int(i_in2),
                    "i_out": int(i_out),
                    "mul": int(mul),
                    "weight_offset": int(weight_offset),
                    "l_in1": int(in1_meta["l"]),
                    "l_in2": int(in2_meta["l"]),
                    "l_out": int(out_meta["l"]),
                    "terms": terms,
                }
            )
            weight_offset += mul
        if weight_offset != weight_numel:
            raise RuntimeError(
                f"Unexpected conv_tp weight layout: consumed {weight_offset} of {weight_numel} entries."
            )
        return maps

    def _serialize_activation_list(acts):
        if acts is None:
            return []
        if hasattr(acts, "acts"):
            acts_iter = acts.acts
        elif isinstance(acts, (list, tuple)):
            acts_iter = acts
        else:
            acts_iter = [acts]
        names = []
        for act in acts_iter:
            if hasattr(act, "__name__"):
                names.append(act.__name__)
            else:
                names.append(str(act))
        return names

    def _serialize_activation_details(acts):
        if acts is None:
            return []
        if hasattr(acts, "acts"):
            acts_iter = acts.acts
        elif isinstance(acts, (list, tuple)):
            acts_iter = acts
        else:
            acts_iter = [acts]
        details = []
        for act in acts_iter:
            name = None
            base = getattr(act, "f", None)
            if base is not None and hasattr(base, "__name__"):
                name = base.__name__
            elif hasattr(act, "__name__"):
                name = act.__name__
            else:
                name = str(act)
            details.append(
                {
                    "name": str(name),
                    "repr": str(act),
                    "cst": float(getattr(act, "cst", 1.0)),
                }
            )
        return details

    def _build_augmented_rrnlb_edge_features(interaction, r_values, model_i, model_j):
        edge_feats, cutoff = evaluate_radial_features_and_cutoff(r_values, model_i, model_j)
        node_attrs_eye = torch.eye(len(model.atomic_numbers), dtype=torch.get_default_dtype())
        node_attrs = torch.stack((node_attrs_eye[model_i], node_attrs_eye[model_j]), dim=0)
        source_embedding = interaction.source_embedding(node_attrs)[0:1, :]
        target_embedding = interaction.target_embedding(node_attrs)[1:2, :]
        source_embedding = source_embedding.repeat(len(r_values), 1)
        target_embedding = target_embedding.repeat(len(r_values), 1)
        augmented = torch.cat([edge_feats, source_embedding, target_embedding], dim=-1)
        return augmented, cutoff

    def _extract_rrnlb_phase0(output_dict, atomic_numbers_subset):
        output_dict["interaction_mode"] = "rrnlb"
        output_dict["interaction_schema_version"] = 2
        output_dict["head"] = head
        output_dict["source_model_path"] = os.path.abspath(model_path)
        output_dict["node_embedding"] = _serialize_linear(model.node_embedding.linear)
        output_dict["node_embedding_species_values"] = []
        node_attrs_eye = torch.eye(len(model.atomic_numbers), dtype=torch.get_default_dtype())
        for a_i in atomic_numbers_subset:
            model_i = model.atomic_numbers.tolist().index(a_i)
            emb_i = model.node_embedding(node_attrs_eye[model_i:model_i + 1, :])
            output_dict["node_embedding_species_values"].append(
                emb_i.numpy(force=True).reshape(-1).tolist()
            )
        output_dict["product_linears"] = {
            "layer0": _serialize_linear(model.products[0].linear),
            "layer1": _serialize_linear(model.products[1].linear),
        }

        pair_types_ordered = []
        for a_i in atomic_numbers_subset:
            for a_j in atomic_numbers_subset:
                pair_types_ordered.append([int(a_i), int(a_j)])
        output_dict["pair_types_ordered"] = pair_types_ordered

        r_values, h = np.linspace(1e-12, r_cut, num_spline_points, retstep=True)

        interaction_layers = []
        for layer_idx, interaction in enumerate(model.interactions):
            if not isinstance(interaction, RealAgnosticResidualNonLinearInteractionBlock):
                raise RuntimeError(
                    "Phase 0 RRNLB extractor currently requires both interactions to be "
                    "RealAgnosticResidualNonLinearInteractionBlock."
                )

            layer_data = {
                "layer_index": int(layer_idx),
                "interaction_class": type(interaction).__name__,
                "node_feats_irreps": _serialize_irreps(interaction.node_feats_irreps),
                "edge_irreps": _serialize_irreps(interaction.edge_irreps),
                "target_irreps": _serialize_irreps(interaction.target_irreps),
                "hidden_irreps": _serialize_irreps(interaction.hidden_irreps),
                "irreps_nonlin": _serialize_irreps(interaction.irreps_nonlin),
                "edge_feats_irreps": _serialize_irreps(interaction.edge_feats_irreps),
                "edge_attrs_irreps": _serialize_irreps(interaction.edge_attrs_irreps),
                "conv_tp": {
                    "weight_numel": int(interaction.conv_tp.weight_numel),
                    "irreps_in1": _serialize_irreps(interaction.conv_tp.irreps_in1),
                    "irreps_in2": _serialize_irreps(interaction.conv_tp.irreps_in2),
                    "irreps_out": _serialize_irreps(interaction.conv_tp.irreps_out),
                    "instructions": _serialize_conv_tp_instructions(interaction.conv_tp),
                    "cg_maps": _serialize_conv_tp_cg_maps(interaction.conv_tp),
                },
                "alpha": float(interaction.alpha.detach().cpu().item()),
                "beta": float(interaction.beta.detach().cpu().item()),
                "avg_num_neighbors": float(interaction.avg_num_neighbors),
                "radial_mlp_layers": [int(v) for v in interaction.radial_MLP],
                "linears": {
                    "source_embedding": _serialize_linear(interaction.source_embedding),
                    "target_embedding": _serialize_linear(interaction.target_embedding),
                    "linear_up": _serialize_linear(interaction.linear_up),
                    "linear_res": _serialize_linear(interaction.linear_res),
                    "linear_1": _serialize_linear(interaction.linear_1),
                    "linear_2": _serialize_linear(interaction.linear_2),
                    "skip_tp": _serialize_linear(interaction.skip_tp),
                },
            }

            gate = interaction.equivariant_nonlin
            layer_data["gate"] = {
                "irreps_scalars": _serialize_irreps(gate.irreps_scalars),
                "irreps_gates": _serialize_irreps(gate.irreps_gates),
                "irreps_gated": _serialize_irreps(gate.irreps_gated),
                "irreps_in": _serialize_irreps(gate.irreps_in),
                "irreps_out": _serialize_irreps(gate.irreps_out),
                "scalar_acts": _serialize_activation_list(gate.act_scalars),
                "gate_acts": _serialize_activation_list(gate.act_gates),
                "scalar_activation_details": _serialize_activation_details(gate.act_scalars),
                "gate_activation_details": _serialize_activation_details(gate.act_gates),
            }

            tp_weight_spline_values = []
            tp_weight_spline_derivs = []
            edge_density_spline_values = []
            edge_density_spline_derivs = []
            for a_i in atomic_numbers_subset:
                for a_j in atomic_numbers_subset:
                    model_i = model.atomic_numbers.tolist().index(a_i)
                    model_j = model.atomic_numbers.tolist().index(a_j)
                    augmented_edge_feats, cutoff = _build_augmented_rrnlb_edge_features(
                        interaction, r_values, model_i, model_j
                    )

                    tp_weights = interaction.conv_tp_weights(augmented_edge_feats)
                    edge_density = torch.tanh(interaction.density_fn(augmented_edge_feats) ** 2)
                    if cutoff is not None:
                        tp_weights = tp_weights * cutoff
                        edge_density = edge_density * cutoff

                    tp_weights_np = tp_weights.numpy(force=True)
                    spl_tp = [
                        CubicSpline(r_values, tp_weights_np[:, k], bc_type=spline_bc_type)
                        for k in range(tp_weights_np.shape[1])
                    ]
                    tp_weight_spline_values.append([spl(r_values).tolist() for spl in spl_tp])
                    tp_weight_spline_derivs.append([spl.derivative()(r_values).tolist() for spl in spl_tp])

                    edge_density_np = edge_density.numpy(force=True).reshape(-1)
                    spl_density = CubicSpline(r_values, edge_density_np, bc_type=spline_bc_type)
                    edge_density_spline_values.append(spl_density(r_values).tolist())
                    edge_density_spline_derivs.append(spl_density.derivative()(r_values).tolist())

            layer_data["radial"] = {
                "spline_h": float(h),
                "tp_weights_values": tp_weight_spline_values,
                "tp_weights_derivs": tp_weight_spline_derivs,
                "edge_density_values": edge_density_spline_values,
                "edge_density_derivs": edge_density_spline_derivs,
            }
            interaction_layers.append(layer_data)

        output_dict["interaction_layers"] = interaction_layers

        # Product 0 polynomial payload (same extraction concept as legacy path).
        correlation_0 = model.products[0].symmetric_contractions.contractions[0].correlation

        def _u_sparse(irrep_out, irreps_in, corr_in_max):
            U = [[]]
            for corr in range(1, corr_in_max + 1):
                try:
                    U_matrix = U_matrix_real(irreps_in, [irrep_out], corr, use_cueq_cg=False)[1]
                except TypeError:
                    U_matrix = U_matrix_real(irreps_in, [irrep_out], corr)[1]
                if irrep_out.l == 0:
                    U_matrix = U_matrix.unsqueeze(0)
                num_eta = U_matrix.shape[-1]
                U_matrix = U_matrix.flatten()
                U_sparse_corr = [[{} for _ in range(num_eta)] for _ in range(2 * irrep_out.l + 1)]
                j = 0
                for m in range(2 * irrep_out.l + 1):
                    for lm_list in itertools.product(range((l_max + 1) ** 2), repeat=corr):
                        for eta in range(num_eta):
                            if abs(U_matrix[j]) > 1e-12:
                                lm_tuple_sorted = tuple(sorted(lm_list))
                                if lm_tuple_sorted not in U_sparse_corr[m][eta]:
                                    U_sparse_corr[m][eta][lm_tuple_sorted] = 0.0
                                U_sparse_corr[m][eta][lm_tuple_sorted] += U_matrix[j].item()
                            j += 1
                U.append(U_sparse_corr)
            return U

        irreps_in_0 = [ir[1] for ir in model.products[0].symmetric_contractions.irreps_in]
        irreps_out_0 = [ir[1] for ir in model.products[0].symmetric_contractions.irreps_out]

        def _species_axis_index(weights_tensor, model_species_index):
            if weights_tensor.shape[0] == 1:
                return 0
            return model_species_index

        C0 = {}
        M0 = {}
        for i, a in enumerate(atomic_numbers_subset):
            model_i = model.atomic_numbers.tolist().index(a)
            C0[i] = {}
            for l, irrep_out in enumerate(irreps_out_0):
                U = _u_sparse(irrep_out, irreps_in_0, correlation_0)
                W = [[]]
                W.append(model.products[0].symmetric_contractions.contractions[l].weights[1].numpy(force=True))
                W.append(model.products[0].symmetric_contractions.contractions[l].weights[0].numpy(force=True))
                W.append(model.products[0].symmetric_contractions.contractions[l].weights_max.numpy(force=True))
                for m in range(-l, l + 1):
                    lm = l * (l + 1) + m
                    C0[i][lm] = {}
                    for k in range(num_channels):
                        P_lmk = {}
                        for corr in range(1, correlation_0 + 1):
                            for eta in range(len(U[corr][l + m])):
                                for key, value in U[corr][l + m][eta].items():
                                    if key not in P_lmk:
                                        P_lmk[key] = 0.0
                                    species_i = _species_axis_index(W[corr], model_i)
                                    P_lmk[key] += float(W[corr][species_i, eta, k]) * value
                        C0[i][lm][k] = list(P_lmk.values())
                        M0[lm] = [list(key) for key in P_lmk.keys()]
        output_dict["M0_weights"] = C0
        output_dict["M0_monomials"] = M0

        # Product 1 polynomial payload.
        correlation_1 = model.products[1].symmetric_contractions.contractions[0].correlation
        irreps_in_1 = Irreps("0e + 1o + 2e + 3o")
        irreps_out_1 = Irreps("0e")
        U = [[]]
        for corr in range(1, correlation_1 + 1):
            try:
                U_matrix = U_matrix_real(irreps_in_1, irreps_out_1, corr, use_cueq_cg=False)[1]
            except TypeError:
                U_matrix = U_matrix_real(irreps_in_1, irreps_out_1, corr)[1]
            num_nu = U_matrix.shape[-1]
            U_matrix = U_matrix.flatten()
            U_sparse = [{} for _ in range(num_nu)]
            j = 0
            for lm_list in itertools.product(range((l_max + 1) ** 2), repeat=corr):
                for nu in range(num_nu):
                    if abs(U_matrix[j]) > 1e-12:
                        lm_tuple_sorted = tuple(sorted(lm_list))
                        if lm_tuple_sorted not in U_sparse[nu]:
                            U_sparse[nu][lm_tuple_sorted] = 0.0
                        U_sparse[nu][lm_tuple_sorted] += U_matrix[j].item()
                    j += 1
            U.append(U_sparse)
        W = [[]]
        W.append(model.products[1].symmetric_contractions.contractions[0].weights[1].numpy(force=True))
        W.append(model.products[1].symmetric_contractions.contractions[0].weights[0].numpy(force=True))
        W.append(model.products[1].symmetric_contractions.contractions[0].weights_max.numpy(force=True))
        C1 = {}
        M1 = []
        for i, a in enumerate(atomic_numbers_subset):
            model_i = model.atomic_numbers.tolist().index(a)
            C1[i] = {}
            for k in range(num_channels):
                P_ik = {}
                for corr in range(1, correlation_1 + 1):
                    for nu in range(len(U[corr])):
                        for key, value in U[corr][nu].items():
                            if key not in P_ik:
                                P_ik[key] = 0.0
                            species_i = _species_axis_index(W[corr], model_i)
                            P_ik[key] += float(W[corr][species_i, nu, k]) * value
                C1[i][k] = list(P_ik.values())
                M1 = [list(key) for key in P_ik.keys()]
        output_dict["M1_weights"] = C1
        output_dict["M1_monomials"] = M1

        # Readout payload for phase 1 forward path.
        readout_1_weights = model.readouts[0].linear.weight.numpy(force=True) / np.sqrt(num_channels)
        output_dict["readout_1_weights"] = (readout_1_weights * model.scale_shift.scale.item()).tolist()
        output_dict["readout_2_weights_1"] = (
            torch.reshape(model.readouts[1].linear_1.weight, (num_channels, 16)).T.numpy(force=True).flatten()
            / np.sqrt(num_channels)
        ).tolist()
        output_dict["readout_2_weights_2"] = (
            model.readouts[1].linear_2.weight.numpy(force=True).flatten()
            / np.sqrt(16)
            * model.scale_shift.scale.item()
        ).tolist()
        output_dict["readout_2_scale_factor"] = model.readouts[1].non_linearity.acts[0].cst
        return output_dict

    ### ----- BASIC MODEL INFO -----

    num_channels = model.node_embedding.linear.irreps_out.count("0e")
    r_cut = model.r_max.item()
    l_max = model.spherical_harmonics._lmax
    L_max =  model.products[0].linear.irreps_out.lmax
    output = {}
    output['interaction_mode'] = "legacy"
    output['num_channels'] = num_channels
    output['r_cut'] = r_cut
    output['l_max'] = l_max
    output['L_max'] = L_max

    ### ----- ATOMIC NUMBERS AND ENERGIES -----

    atomic_numbers = sorted(atomic_numbers)
    if len(atomic_numbers) == 0:
        atomic_numbers = sorted(model.atomic_numbers.tolist())
        logging.warning(f"No atomic_numbers, including all: {atomic_numbers}")
    atomic_energies = [
        torch.atleast_1d(model.atomic_energies_fn.atomic_energies.squeeze())[model.atomic_numbers.tolist().index(a)].item()
            + model.scale_shift.shift.item()
        for a in atomic_numbers]
    output['atomic_numbers'] = atomic_numbers
    output['num_elements'] = len(atomic_numbers)
    output['atomic_energies'] = atomic_energies

    ### --- ZBL ---

    if hasattr(model, "pair_repulsion") and model.pair_repulsion:
        if not isinstance(model.pair_repulsion_fn, ZBLBasis):
            raise Exception("Only ZBL pair_repulsion is supported.")
        output['has_zbl'] = True
        zbl = model.pair_repulsion_fn
        output['zbl_a_exp'] = zbl.a_exp.item()
        output['zbl_a_prefactor'] = zbl.a_prefactor.item()
        output['zbl_c'] = (model.scale_shift.scale.item() * zbl.c.numpy(force=True)).tolist()
        output['zbl_covalent_radii'] = zbl.covalent_radii.numpy(force=True).tolist()
        output['zbl_p'] = zbl.p.item()
    else:
        output['has_zbl'] = False

    if has_residual_nonlinear:
        if not all_residual_nonlinear:
            raise RuntimeError(
                "Phase 0 RRNLB extraction does not yet support mixed interaction families; "
                "both interaction layers must be RealAgnosticResidualNonLinearInteractionBlock."
            )
        logging.warning(
            "Detected RealAgnosticResidualNonLinearInteractionBlock. "
            "Extracting Phase 0 RRNLB schema data."
        )
        return _extract_rrnlb_phase0(output, atomic_numbers)

    ### ----- RADIAL SPLINES -----

    logging.info("R0+R1")
    r,h = np.linspace(1e-12, r_cut, num_spline_points, retstep=True)
    spline_values_0 = []
    spline_derivatives_0 = []
    spline_values_1 = []
    spline_derivatives_1 = []
    for a_i in atomic_numbers:
        for a_j in atomic_numbers:
            if a_j < a_i:
                continue
            model_i = model.atomic_numbers.tolist().index(a_i)
            model_j = model.atomic_numbers.tolist().index(a_j)
            bessels, cutoff = evaluate_radial_features_and_cutoff(r, model_i, model_j)
            # radial basis for interaction 0
            R = model.interactions[0].conv_tp_weights(bessels)
            if cutoff is not None:
                R = R * cutoff
            R = R.numpy(force=True)
            spl_0 = [CubicSpline(r, R[:,k], bc_type=spline_bc_type) for k in range(R.shape[1])]
            spline_values_0.append([spl(r).tolist() for spl in spl_0])
            spline_derivatives_0.append([spl.derivative()(r).tolist() for spl in spl_0])
            # radial basis for interaction 1
            R = model.interactions[1].conv_tp_weights(bessels)
            if cutoff is not None:
                R = R * cutoff
            R = R.numpy(force=True)
            spl_1 = [CubicSpline(r, R[:,k], bc_type=spline_bc_type) for k in range(R.shape[1])]
            spline_values_1.append([spl(r).tolist() for spl in spl_1])
            spline_derivatives_1.append([spl.derivative()(r).tolist() for spl in spl_1])

    output['radial_spline_h'] = float(h)
    output['radial_spline_values_0'] = spline_values_0
    output['radial_spline_derivs_0'] = spline_derivatives_0
    output['radial_spline_values_1'] = spline_values_1
    output['radial_spline_derivs_1'] = spline_derivatives_1

    ### ----- H0 -----

    logging.info("H0")
    H0_weights = (
        np.reshape(model.node_embedding.linear.weight.numpy(force=True),
                   [len(model.atomic_numbers),num_channels]) / np.sqrt(len(model.atomic_numbers))
        @
        np.reshape(model.interactions[0].linear_up.weight.numpy(force=True),
                   [num_channels,num_channels]) / np.sqrt(num_channels)
        )
    indices = [model.atomic_numbers.tolist().index(a) for a in atomic_numbers]
    H0_weights = H0_weights[indices,:]
    output['H0_weights'] = H0_weights.flatten().tolist()

    ### ----- Phi0 -----

    logging.info("Phi0")

    ### ----- A0 -----

    logging.info("A0")
    A0_scaled = True if ("Density" in type(model.interactions[0]).__name__) else False
    output['A0_scaled'] = A0_scaled
    if A0_scaled:
        r,h = np.linspace(1e-12, r_cut, num_spline_points, retstep=True)
        A0_spline_values = []
        A0_spline_derivs = []
        for a_i in atomic_numbers:
            for a_j in atomic_numbers:
                if a_j < a_i:
                    continue
                model_i = model.atomic_numbers.tolist().index(a_i)
                model_j = model.atomic_numbers.tolist().index(a_j)
                bessels, cutoff = evaluate_radial_features_and_cutoff(r, model_i, model_j)
                R = torch.tanh(model.interactions[0].density_fn(bessels)**2)
                if cutoff is not None:
                    R = R * cutoff
                R = R.numpy(force=True)
                spl = CubicSpline(r, R[:,0], bc_type=spline_bc_type)
                A0_spline_values.append(spl(r).tolist())
                A0_spline_derivs.append(spl.derivative()(r).tolist())
        output['A0_spline_h'] = float(h)
        output['A0_spline_values'] = A0_spline_values
        output['A0_spline_derivs'] = A0_spline_derivs
    A0_weights = []
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        A0_weights.append([])
        for l,_,w in model.interactions[0].skip_tp.weight_views(yield_instruction=True):
            w_linear = model.interactions[0].linear.weight_view_for_instruction(l).numpy(force=True) / np.sqrt(num_channels)
            if not A0_scaled:
                w_linear /= model.interactions[0].avg_num_neighbors
            fused = w_linear @ w[:,model_i,:].numpy(force=True) / np.sqrt(len(model.atomic_numbers)*num_channels)
            A0_weights[i].append(fused.flatten().tolist())
    output["A0_weights"] = A0_weights

    #### ----- M0 -----

    logging.info("M0")
    correlation = model.products[0].symmetric_contractions.contractions[0].correlation
    ### Computes U_{lm\eta, l1m1 l2m2 ...}
    #   * `irrep_out` is essentially l_out
    #   * `irreps_in` essentially provides l_in_max
    #   * `corr_in_max` is the max correlation order
    def U_sparse(irrep_out, irreps_in, corr_in_max):
        U = [[]]  # list of lists because U[0] should be empty
        for corr in range(1,corr_in_max+1):
            # get U matrix for this correlation order
            try:
                U_matrix = U_matrix_real(irreps_in, [irrep_out], corr, use_cueq_cg=False)[1]
            except TypeError:
                U_matrix = U_matrix_real(irreps_in, [irrep_out], corr)[1]
            if irrep_out.l == 0:  # makes U_matrix.shape consistent with l>0 cases
                U_matrix = U_matrix.unsqueeze(0)
            num_eta = U_matrix.shape[-1]
            U_matrix = U_matrix.flatten()
            # extract sparse U for this correlation order
            U_sparse_corr = [[{} for _ in range(num_eta)] for _ in range(2*irrep_out.l+1)]
            j = 0
            for m in range(2*irrep_out.l+1):
                for lm_list in itertools.product(range((l_max+1)**2), repeat=corr):
                    for eta in range(num_eta):
                        if abs(U_matrix[j]) > 1e-12:
                            lm_tuple_sorted = tuple(sorted(lm_list))
                            if lm_tuple_sorted not in U_sparse_corr[m][eta].keys():
                                U_sparse_corr[m][eta][lm_tuple_sorted] = 0.0
                            U_sparse_corr[m][eta][lm_tuple_sorted] += U_matrix[j].item()
                        j += 1
            U.append(U_sparse_corr)
        return U
    irreps_in = [ir[1] for ir in model.products[0].symmetric_contractions.irreps_in]
    irreps_out = [ir[1] for ir in model.products[0].symmetric_contractions.irreps_out]
    C = {}
    M = {}
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        C[i] = {}
        for l, irrep_out in enumerate(irreps_out):
            # extract U in sparse format
            U = U_sparse(irrep_out, irreps_in, correlation)
            # extract weights from model
            # warning: slightly odd order of the contractions weights due to reverse countdown
            W = [[]]  # list of lists because W[0] should be empty
            W.append(model.products[0].symmetric_contractions.contractions[l].weights[1].numpy(force=True))
            W.append(model.products[0].symmetric_contractions.contractions[l].weights[0].numpy(force=True))
            W.append(model.products[0].symmetric_contractions.contractions[l].weights_max.numpy(force=True))
            # combine U and W into polynomial-like terms for recursive evaluator
            for m in range(-l,l+1):
                lm = l*(l+1)+m
                C[i][lm] = {}
                for k in range(num_channels):
                    P_lmk = {}
                    for corr in range(1,correlation+1):
                        for eta in range(len(U[corr][l+m])):
                            for key,value in U[corr][l+m][eta].items():
                                if key not in P_lmk.keys():
                                    P_lmk[key] = 0.0
                                P_lmk[key] += float(W[corr][model_i,eta,k]) * value
                    C[i][lm][k] = list(P_lmk.values())
                    M[lm] = [list(key) for key in P_lmk.keys()]
    output['M0_weights'] = C
    output['M0_monomials'] = M

    ### ----- H1 -----

    logging.info("H1")
    H1_weights = np.zeros([L_max+1, num_channels, num_channels])
    weights_0 = np.reshape(
        model.products[0].linear.weight.numpy(force=True),
        [L_max+1,num_channels,num_channels]) / np.sqrt(num_channels)
    weights_1 = np.reshape(
        model.interactions[1].linear_up.weight.numpy(force=True),
        [L_max+1,num_channels,num_channels]) / np.sqrt(num_channels)
    for l in range(L_max+1):
        H1_weights[l,:,:] = weights_0[l,:,:] @ weights_1[l,:,:]
    output["H1_weights"] = H1_weights.flatten().tolist()

    ### ----- Phi1 -----

    logging.info("Phi1")
    Phi1_l = [ir[1].l for ir in model.interactions[1].conv_tp.irreps_out]
    Phi1_l1 = [ins.i_in2 for ins in model.interactions[1].conv_tp.instructions]
    Phi1_l2 = [ins.i_in1 for ins in model.interactions[1].conv_tp.instructions]
    Phi1_clebsch_gordan = []
    Phi1_lme = []
    Phi1_lelm1lm2 = []
    num_lm1 = (l_max+1)**2
    num_lm2 = (L_max+1)**2
    def compute_lem(le, l, m):
        lem = 0
        for j in range(le):
            lem += 2*Phi1_l[j]+1
        return lem+l+m
    def compute_lme(le, l, m):
        e = le - int(sum(np.array(Phi1_l)<l))
        num_e = [int(sum(np.array(Phi1_l)==ll)) for ll in range(l+1)]
        lme = 0
        for ll in range(l):
            lme += (2*ll+1)*num_e[ll]
        return lme + (l+m)*num_e[l]+e
    def compute_lelm1lm2(le,l1,m1,l2,m2):
        lelm1lm2 = 0
        for j in range(le):
            l1,l2 = (Phi1_l1[j], Phi1_l2[j])
            lelm1lm2 += (2*l1+1)*(2*l2+1)
        l, l1, l2 = (Phi1_l[le], Phi1_l1[le], Phi1_l2[le])
        return lelm1lm2 + (l1+m1)*(2*l2+1) + l2+m2
    tp = model.interactions[1].conv_tp
    for l1 in range(l_max+1):
        for m1 in range(-l1,l1+1):
            lm1 = l1*l1+l1+m1
            for l2 in range(L_max+1):
                for m2 in range(-l2,l2+1):
                    R = torch.ones([1,len(tp.instructions)*num_channels],dtype=torch.double)
                    Y = torch.zeros([1,num_lm1], dtype=torch.double)
                    Y[0,lm1] = 1.0
                    H = torch.zeros([1,num_lm2*num_channels],dtype=torch.double)
                    H[0,sum([2*p+1 for p in range(l2)])*num_channels+l2+m2] = 1.0
                    Phi = tp(H, Y, R)
                    # extract Phi values for k=0
                    Phi_0 = []
                    for le in range(len(tp.instructions)):
                        Phi_0_start = sum([2*Phi1_l[p]+1 for p in range(le)])*num_channels
                        for p in range(2*Phi1_l[le]+1):
                            Phi_0.append(Phi[0,Phi_0_start+p].item())
                    for le in range(len(tp.instructions)):
                        l = Phi1_l[le]
                        for m in range(-l,l+1):
                            lem = compute_lem(le,l,m)
                            if np.abs(Phi_0[lem]) > 1e-12:
                                Phi1_lme.append(compute_lme(le,l,m))
                                Phi1_clebsch_gordan.append(Phi_0[lem])
                                Phi1_lelm1lm2.append(compute_lelm1lm2(le,l1,m1,l2,m2))
    output["Phi1_l"] = Phi1_l
    output["Phi1_l1"] = Phi1_l1
    output["Phi1_l2"] = Phi1_l2
    output["Phi1_lme"] = Phi1_lme
    output["Phi1_clebsch_gordan"] = Phi1_clebsch_gordan
    output["Phi1_lelm1lm2"] = Phi1_lelm1lm2

    ### ----- A1 -----

    logging.info("A1")
    A1_scaled = True if ("Density" in type(model.interactions[1]).__name__) else False
    output['A1_scaled'] = A1_scaled
    if A1_scaled:
        r,h = np.linspace(1e-12, r_cut, num_spline_points, retstep=True)
        A1_spline_values = []
        A1_spline_derivs = []
        for a_i in atomic_numbers:
            for a_j in atomic_numbers:
                if a_j < a_i:
                    continue
                model_i = model.atomic_numbers.tolist().index(a_i)
                model_j = model.atomic_numbers.tolist().index(a_j)
                bessels, cutoff = evaluate_radial_features_and_cutoff(r, model_i, model_j)
                R = torch.tanh(model.interactions[1].density_fn(bessels)**2)
                if cutoff is not None:
                    R = R * cutoff
                R = R.numpy(force=True)
                spl = CubicSpline(r, R[:,0], bc_type=spline_bc_type)
                A1_spline_values.append(spl(r).tolist())
                A1_spline_derivs.append(spl.derivative()(r).tolist())
        output['A1_spline_h'] = float(h)
        output['A1_spline_values'] = A1_spline_values
        output['A1_spline_derivs'] = A1_spline_derivs
    A1_weights = []
    num_eta = [sum([l==ll for ll in Phi1_l]) for l in range(l_max+1)]
    A1_linear = linear_simplify(model.interactions[1].linear)
    for l in range(l_max+1):
        w_linear = A1_linear.weight_view_for_instruction(l).numpy(force=True) / np.sqrt(num_eta[l]*num_channels)
        if not A1_scaled:
            w_linear /= model.interactions[1].avg_num_neighbors
        w_linear = np.reshape(w_linear, (num_eta[l], num_channels, num_channels))
        A1_weights.append(w_linear.flatten().tolist())
    output["A1_weights"] = A1_weights

    ### ----- M1 -----

    logging.info("M1")
    # TODO: generalize
    correlation = model.products[1].symmetric_contractions.contractions[0].correlation
    irreps_in = Irreps("0e + 1o + 2e + 3o")
    irreps_out = Irreps("0e")
    # extract U in sparse format
    U = [[]]
    for corr in range(1,4):
        # get U matrix for this correlation order
        try:
            U_matrix = U_matrix_real(irreps_in, irreps_out, corr, use_cueq_cg=False)[1]
        except TypeError:
            U_matrix = U_matrix_real(irreps_in, irreps_out, corr)[1]
        num_nu = U_matrix.shape[-1]
        U_matrix = U_matrix.flatten()
        # extract sparse U for this correlation order
        U_sparse = [{} for _ in range(num_nu)]
        j = 0
        for lm_list in itertools.product(range((l_max+1)**2), repeat=corr):
            for nu in range(num_nu):
                if abs(U_matrix[j]) > 1e-12:
                    lm_tuple_sorted = tuple(sorted(lm_list))
                    if lm_tuple_sorted not in U_sparse[nu].keys():
                        U_sparse[nu][lm_tuple_sorted] = 0.0
                    U_sparse[nu][lm_tuple_sorted] += U_matrix[j].item()
                j += 1
        U.append(U_sparse)
    # extract weights from model
    # warning: slightly odd order of the contractions weights due to reverse countdown
    W = [[]]
    W.append(model.products[1].symmetric_contractions.contractions[0].weights[1].numpy(force=True))
    W.append(model.products[1].symmetric_contractions.contractions[0].weights[0].numpy(force=True))
    W.append(model.products[1].symmetric_contractions.contractions[0].weights_max.numpy(force=True))
    # combine U and W into polynomial-like terms for recursive evaluator
    C = {}
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        C[i] = {}
        for k in range(num_channels):
            P_ik = {}
            for corr in range(1,4):
                for nu in range(len(U[corr])):
                    for key,value in U[corr][nu].items():
                        if key not in P_ik.keys():
                            P_ik[key] = 0.0
                        P_ik[key] += float(W[corr][model_i,nu,k]) * value
            C[i][k] = list(P_ik.values())
            M = [list(key) for key in P_ik.keys()]
    output['M1_weights'] = C
    output['M1_monomials'] = M

    ### ----- H2 -----

    logging.info("H2")
    weights_to_fuse = model.interactions[1].linear_up.weight_view_for_instruction(0).numpy(force=True) / np.sqrt(num_channels)
    weights_to_fuse_rank = np.linalg.matrix_rank(weights_to_fuse)
    if weights_to_fuse_rank < num_channels:
        raise RuntimeError('ERROR: fusing weights have too low rank {weights_to_fuse_rank} < {num_channels}')
    # H2 weights for H1
    H2_weights_for_H1 = []
    for i, a in enumerate(atomic_numbers):
        model_i = model.atomic_numbers.tolist().index(a)
        w = model.interactions[1].skip_tp.weight_view_for_instruction(0)[:,model_i,:].numpy(force=True)
        H2_weights_for_H1.append(w / np.sqrt(len(model.atomic_numbers)*num_channels))
        H2_weights_for_H1[i] = np.linalg.inv(weights_to_fuse) @ H2_weights_for_H1[i]
        H2_weights_for_H1[i] = H2_weights_for_H1[i].flatten().tolist()
    output["H2_weights_for_H1"] = H2_weights_for_H1
    # H2 weights for M1
    output["H2_weights_for_M1"] = (
        model.products[1].linear.weight.numpy(force=True) / np.sqrt(num_channels)).tolist()

    ### ----- READOUTS -----

    # linear readout
    weights_to_fuse = np.reshape(
        model.interactions[1].linear_up.weight.numpy(force=True) / np.sqrt(num_channels),
        [L_max+1,num_channels,num_channels])
    readout_1_weights = model.readouts[0].linear.weight.numpy(force=True) / np.sqrt(num_channels)
    readout_1_weights = np.linalg.inv(weights_to_fuse[0,:,:]) @ readout_1_weights
    output['readout_1_weights'] = (readout_1_weights * model.scale_shift.scale.item()).tolist()

    # nonlinear readout
    #output["mlp_hidden_layers"] = 16 // TODO
    output["readout_2_weights_1"] = (
            torch.reshape(
                model.readouts[1].linear_1.weight,
                (num_channels,16)
            ).T.numpy(force=True).flatten() / np.sqrt(num_channels)
        ).tolist()
    output["readout_2_weights_2"] = (
        model.readouts[1].linear_2.weight.numpy(force=True).flatten() / np.sqrt(16)
        * model.scale_shift.scale.item()).tolist()
    output["readout_2_scale_factor"] = model.readouts[1].non_linearity.acts[0].cst

    return output
