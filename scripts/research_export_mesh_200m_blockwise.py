#!/usr/bin/env python3
"""Blockwise export for the 200 m run; never materializes all ranks/times."""

from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]
sys.path.insert(0, str(ROOT / "src"))

from gragra.adapters.specfem import export as ex, gll, provenance, schema, specfem_bin, writer

SOLVER = Path("/home/washimi/work/specfem3d-rayleigh-init-20260907")
CASE = SOLVER / "research_cases/rayleigh_b5_200m_12km_8rank_dt_half"
LOCAL = CASE / "OUTPUT_FILES/DATABASES_MPI_200M_DT_HALF"
OUT = Path("/tmp/gragra-rayleigh-b5-200m-12km-8rank-dt-half.h5")
NGLL_SHAPE = (5, 5, 5)
NGLL = 125


def repair_metadata(path: Path) -> None:
    with h5py.File(path, "r+") as h5:
        meta = h5["metadata"]
        for key, value in list(meta.attrs.items()):
            if isinstance(value, str):
                del meta.attrs[key]
                meta.attrs.create(key, np.asarray(value, dtype=f"S{max(1, len(value))}"))
        values = {key: meta[key][()].decode() if isinstance(meta[key][()], bytes)
                  else str(meta[key][()]) for key in list(meta.keys())}
        for key in list(meta.keys()):
            del meta[key]
            meta.create_dataset(key, data=values[key], dtype=h5py.string_dtype(
                encoding="utf-8", length=max(1, len(values[key]))))


def main() -> None:
    par = provenance.parse_par_file(CASE / "DATA/Par_file")
    values = par["values"]
    ex._check_profile_inputs(values, "time_sampled")
    all_ranks = ex._read_all_ranks(LOCAL, 8, NGLL)
    ex._check_anchor_gll_bridge(all_ranks.corners8, all_ranks.coordinates, NGLL_SHAPE)
    coords = all_ranks.coordinates.copy()
    coords[:, ex._usual_hex_nodes_flat_indices(NGLL_SHAPE), :] = all_ranks.corners8
    qvol = gll.elements_quadrature_volume(all_ranks.corners8, NGLL_SHAPE)
    is_pml = np.zeros(len(all_ranks.element_id), dtype=bool)
    manifest, mesh_hash = provenance.build_file_manifest(CASE, ["DATA/Par_file"])
    identity = {
        "build_kind": "source",
        "specfem_git_commit": "7ddf59ed6a5394b90a8e1a1e7a945023f0c1285f",
        "source_tree_clean": False,
        "source_tree_hash": "sha256:c0e4f764abdfe0ce582ba0c633f4e483d76ddd421816838821af4a006ae0889a",
        "build_configuration": {},
    }
    cadence = {"output_interval_steps": 1, "f_min_analysis": 2.4,
               "f_max_analysis": 2.5, "minimum_cycles": 1,
               "sampling_oversampling_factor": 4.0,
               "antialias_filter_applied": False}
    scalars = {
        "solver_kind": "specfem3d_cartesian", "solver_version": "research-7ddf59ed",
        "nproc": 8, "float_precision": "single", "density_source": "database_material",
        "dt_s": 0.0025, "nstep": 480,
        "element_id_method": schema.ADOPTED_ELEMENT_ID_METHOD, "f_resolved_estimate": 4.8,
        "field_units": {"coordinates": "m", "displacement": "m"},
        "suppress_utm_projection": True, "endianness": all_ranks.endianness,
        "record_marker_bytes": all_ranks.record_marker_bytes, "integer_kind": all_ranks.integer_kind,
        "material_id_source": "database_element_record", "domain_id_source": "database_material",
        "is_pml_source": "exporter_mvp_no_cpml", "displacement_layout": "per_rank",
    }
    runtime = provenance.build_runtime_manifest([
        provenance.collect_exporter_identity(exporter_code_ref="science-worktree-aa21658")])
    prov = provenance.assemble_provenance(
        output_sampling_kind="time_sampled", par_file=par, mesh_input_hash=mesh_hash,
        mesh_input_manifest=manifest, build_identity=identity, runtime_manifest=runtime,
        pml_mask_hash=provenance.pml_mask_hash(is_pml), t0=-10.0,
        t0_source="12km 200m dt-half output_solver.txt",
        time_step_origin="zero_based_normalized", native_step_origin=1,
        cadence=cadence, cadence_provenance="confirmed", **scalars)
    n_elem = len(all_ranks.element_id)
    time_step = np.arange(1, 480, 2, dtype=np.int64)
    with h5py.File(OUT, "w") as h5:
        def put(path, data, extra):
            h5.create_dataset(path, data=data, chunks=(min(4096, n_elem), *extra),
                              compression="gzip", compression_opts=4)
        put(schema.PATH_MESH_COORDINATES, coords, (NGLL, 3))
        put(schema.PATH_MESH_QUADRATURE_VOLUME, qvol, (NGLL,))
        for path, data in ((schema.PATH_MESH_ELEMENT_ID, all_ranks.element_id),
                           (schema.PATH_MESH_PARTITION_ID, all_ranks.partition_id),
                           (schema.PATH_MESH_DOMAIN_ID, all_ranks.domain_id),
                           (schema.PATH_MESH_MATERIAL_ID, all_ranks.material_id),
                           (schema.PATH_MESH_IS_PML, is_pml),
                           (schema.PATH_MATERIAL_DENSITY, all_ranks.density)):
            put(path, data, () if data.ndim == 1 else (NGLL,))
        dset = h5.create_dataset(schema.PATH_FIELD_DISPLACEMENT,
                                 shape=(240, n_elem, NGLL, 3), dtype=np.float64,
                                 chunks=(1, min(1024, n_elem), NGLL, 3),
                                 compression="gzip", compression_opts=4)
        for rank in range(8):
            ids = all_ranks.element_id_by_rank[rank]
            local, its, _ = specfem_bin.read_rank_displacement_snapshots(
                LOCAL, rank, nspec_rank=len(ids), ngll=NGLL)
            if its != list(range(1, 481)):
                raise ValueError(f"rank {rank}: unexpected snapshot steps")
            dset[:, ids, :, :] = local[1::2]
        h5.create_dataset(schema.PATH_TIME_STEP, data=time_step)
        h5.create_dataset(schema.PATH_TIME, data=time_step.astype(np.float64) * 0.0025 - (-10.0))
        meta = h5.create_group(schema.METADATA_GROUP)
        writer._write_provenance(meta, {**prov, "n_elements_total": n_elem,
                                         "ngll": NGLL, "ngll_shape": np.asarray(NGLL_SHAPE),
                                         "element_id_min": 0, "element_id_max": n_elem - 1,
                                         "total_quadrature_volume_before_exclusion": float(qvol.sum()),
                                         "total_quadrature_volume_after_exclusion": float(qvol.sum()),
                                         "total_mass_before_exclusion": float((all_ranks.density*qvol).sum()),
                                         "total_mass_after_exclusion": float((all_ranks.density*qvol).sum())}, h5py)
    repair_metadata(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
