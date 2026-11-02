"""Type stubs for strux_rs: Zero-dependency Rust + CUDA engine for structural biology and AMS ensembles."""

from typing import Optional, Sequence, Tuple, List
import numpy as np
from numpy.typing import NDArray

class Msa:
    """Multiple Sequence Alignment container supporting zero-copy token tensors and AF-Cluster sub-sampling."""

    sequences: List[str]
    deletion_matrix: List[List[int]]
    descriptions: List[str]
    num_seqs: int
    seq_len: int

    def __init__(
        self,
        sequences: Sequence[str],
        deletion_matrix: Sequence[Sequence[int]],
        descriptions: Sequence[str],
    ) -> None: ...
    def tokens_np(self) -> NDArray[np.uint8]:
        """Zero-copy [num_seqs, seq_len] uint8 OpenFold/AlphaFold residue token tensor."""
        ...
    def deletion_matrix_np(self) -> NDArray[np.int32]:
        """Contiguous [num_seqs, seq_len] int32 deletion count matrix."""
        ...
    def cluster(self, min_identity: float = 0.7) -> Tuple[NDArray[np.uintp], NDArray[np.uintp]]:
        """Cluster sequences by pairwise non-gap sequence identity (Rayon multi-threaded).

        Returns:
            (cluster_labels, representative_indices)
        """
        ...
    def get_cluster(self, cluster_id: int, labels: Sequence[int]) -> "Msa":
        """Extract a sub-alignment (`Msa`) for a specific `cluster_id` (always retaining the query at index 0)."""
        ...

__version__: str

def parse_pdb(
    path: str,
    ca_only: bool = False,
    atom_indices: Optional[Sequence[int]] = None,
) -> NDArray[np.float32]:
    """Parse a single or multi-model `.pdb` file into a `[num_frames, num_atoms, 3]` float32 array (Å)."""
    ...

def parse_cif(
    path: str,
    ca_only: bool = False,
    strip_hydrogens: bool = True,
    use_auth_chain: bool = True,
    atom_indices: Optional[Sequence[int]] = None,
) -> NDArray[np.float32]:
    """Parse an `mmCIF` (`.cif`) file (RCSB PDB, AlphaFold 3, Boltz-2, Chai-1) into `[num_frames, num_atoms, 3]` float32 (Å)."""
    ...

def parse_cif_with_plddt(
    path: str,
    ca_only: bool = False,
    strip_hydrogens: bool = True,
    use_auth_chain: bool = True,
    atom_indices: Optional[Sequence[int]] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Parse an `mmCIF` (`.cif`) file and return `(coords, plddt_or_bfactor)` with shapes `([F, N, 3], [F, N])`."""
    ...

def parse_cif_full(
    path: str,
    ca_only: bool = False,
    strip_hydrogens: bool = True,
    use_auth_chain: bool = True,
    atom_indices: Optional[Sequence[int]] = None,
) -> Tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    List[str],
    NDArray[np.int32],
    List[str],
    List[str],
]:
    """Parse an `mmCIF` (`.cif`) file and return `(coords, plddt, chain_ids, res_ids, res_names, atom_names)`."""
    ...

def parse_dcd(
    path: str,
    atom_indices: Optional[Sequence[int]] = None,
) -> NDArray[np.float32]:
    """Memory-mapped CHARMM/NAMD/OpenMM `.dcd` binary trajectory reader returning `[num_frames, num_atoms, 3]` float32 (Å)."""
    ...

def parse_xtc(
    path: str,
    atom_indices: Optional[Sequence[int]] = None,
) -> NDArray[np.float32]:
    """Pure-Rust 16T Rayon GROMACS `.xtc` XDR 3D trajectory reader returning `[num_frames, num_atoms, 3]` float32 (Å)."""
    ...

def parse_a3m(a3m_string: str) -> Msa:
    """Parse an in-memory `.a3m` alignment string into an `Msa` object."""
    ...

def parse_a3m_file(path: str) -> Msa:
    """Memory-mapped `.a3m` file parser returning an `Msa` object."""
    ...

def parse_stockholm(stockholm_string: str) -> Msa:
    """Parse an in-memory `.sto` (Stockholm) alignment string into an `Msa` object."""
    ...

def parse_stockholm_file(path: str) -> Msa:
    """Memory-mapped `.sto` (Stockholm) file parser returning an `Msa` object."""
    ...

def cluster_msa(
    sequences: Sequence[str],
    min_identity: float = 0.7,
) -> Tuple[NDArray[np.uintp], NDArray[np.uintp]]:
    """Rayon-parallel greedy sequence clustering (AF-Cluster style) returning `(labels, representative_indices)`."""
    ...

def calculate_rg(coords: NDArray[np.float32]) -> float:
    """Calculate the radius of gyration (Å) for an `[N, 3]` coordinate array."""
    ...

def calculate_rmsd_raw(
    coords1: NDArray[np.float32],
    coords2: NDArray[np.float32],
) -> float:
    """Calculate unaligned root-mean-square deviation (Å) between two `[N, 3]` arrays."""
    ...

def calculate_rmsd_kabsch(
    coords1: NDArray[np.float32],
    coords2: NDArray[np.float32],
) -> float:
    """Calculate optimal Kabsch-superimposed RMSD (Å) between two `[N, 3]` arrays."""
    ...

def calculate_lddt(
    coords_ref: NDArray[np.float32],
    coords_target: NDArray[np.float32],
    cutoff: float = 15.0,
) -> Tuple[float, NDArray[np.float32]]:
    """Local Distance Difference Test (lDDT) with thresholds (0.5, 1.0, 2.0, 4.0 Å) and inclusion radius `cutoff`.

    The global score pools every scored distance pair, matching AlphaFold `lddt(per_residue=False)`
    and OpenStructure. No stereochemistry checks are applied.

    Returns:
        (global_lddt, per_residue_lddt)
    """
    ...

def calculate_pose_and_sc_rmsd(
    ref_target_ca: NDArray[np.float32],
    ref_binder_ca: NDArray[np.float32],
    mob_target_ca: NDArray[np.float32],
    mob_binder_ca: NDArray[np.float32],
) -> Tuple[float, float]:
    """Compute monomer self-consistency RMSD (`sc_rmsd`) and target-anchored `pose_rmsd` (Å)."""
    ...

def batch_pose_and_sc_rmsd(
    ref_target_ca: NDArray[np.float32],
    ref_binder_ca: NDArray[np.float32],
    mob_targets_ca: NDArray[np.float32],
    mob_binders_ca: NDArray[np.float32],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Rayon-parallel batch calculation of `(sc_rmsds, pose_rmsds)` across `M` co-folded designs."""
    ...

def analyze_interface_physics(
    coords: NDArray[np.float32],
    atom_names: Sequence[str],
    res_names: Sequence[str],
    chain_ids: Sequence[str],
    res_ids: Sequence[int],
    target_chain: str = "A",
    binder_chain: str = "B",
) -> Tuple[int, int, int, int, NDArray[np.uintp]]:
    """Compute `(interface_contacts, polar_contacts, salt_bridges, buns_count, buns_indices)` in a single pass.

    All terms are distance heuristics: contacts are heavy-atom pairs within 4.5 Å; polar contacts are
    cross-chain N/O pairs within 3.5 Å (no angle or donor/acceptor check); a polar atom within 5 Å of the
    other chain is counted as buried and unsatisfied if no other N/O lies within 3.9 Å. This is not
    Rosetta `BuriedUnsatHbonds`.
    """
    ...

def calculate_sasa(
    coords: NDArray[np.float32],
    radii: Optional[NDArray[np.float32]] = None,
    probe_radius: float = 1.4,
    n_points: int = 96,
) -> Tuple[float, NDArray[np.float32]]:
    """Rayon-parallel 96-point Golden-Spiral Shrake-Rupley SASA returning `(total_sasa, per_atom_sasa)` (Å²)."""
    ...

def calculate_bsa(
    target_coords: NDArray[np.float32],
    binder_coords: NDArray[np.float32],
    probe_radius: float = 1.4,
    n_points: int = 96,
    target_radii: Optional[NDArray[np.float32]] = None,
    binder_radii: Optional[NDArray[np.float32]] = None,
) -> Tuple[float, float, float, float]:
    """Compute buried surface area returning `(bsa_total, sasa_target, sasa_binder, sasa_complex)` (Å²).

    Without radii every atom gets 1.70 Å, which overestimates BSA by a few percent relative to
    element radii. Pass per-atom van der Waals radii for results comparable to FreeSASA.
    """
    ...

def calculate_ph_switch_ddg(
    coords: NDArray[np.float32],
    atom_names: Sequence[str],
    res_names: Sequence[str],
    chain_ids: Sequence[str],
    target_chain: str = "A",
    binder_chain: str = "B",
    ph_low: float = 6.0,
    ph_high: float = 7.4,
    ionic_strength: float = 0.15,
    dielectric: float = 20.0,
) -> Tuple[float, float, float]:
    """Screened Debye-Hückel interfacial pH-switch electrostatics returning `(ddg_ph_kcal, dg_low_kcal, dg_high_kcal)`."""
    ...

def anm_modes(
    ca_coords: NDArray[np.float32],
    cutoff: float = 15.0,
    gamma: float = 1.0,
    n_modes: int = 6,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Anisotropic Network Model (ANM) 3N×3N Hessian normal mode engine returning `(eigenvalues, eigenvectors, bfactors)`."""
    ...

def screen_sequences_biophysics(
    sequences: Sequence[str],
) -> dict[str, NDArray]:
    """Rayon-parallel sequence developability screener returning dict with pI, GRAVY, net_charge_ph74, CamSol, APRs, liabilities, and HQNST fraction."""
    ...

def sliding_kmer_novelty(
    queries: Sequence[str],
    reference_db: Sequence[str],
    k: int = 15,
) -> Tuple[NDArray[np.float32], NDArray[np.uintp]]:
    """Rayon-parallel sliding k-mer novelty screener returning `(max_kmer_identities, min_hamming_distances)`."""
    ...

def cpu_trajectory_rmsd(
    trajectory: NDArray[np.float32],
    reference: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Rayon multi-threaded 1-vs-T Kabsch RMSD returning a `[T]` float32 array."""
    ...

def cpu_pairwise_rmsd(
    trajectory: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Rayon multi-threaded all-to-all `[T, T]` Kabsch pairwise RMSD matrix."""
    ...

def calculate_rmsf(trajectory: NDArray[np.float32]) -> List[float]:
    """Calculate per-atom Root Mean Square Fluctuation (Å) across a `[T, N, 3]` trajectory."""
    ...

def find_interface_contacts(
    target: NDArray[np.float32],
    binder: NDArray[np.float32],
    cutoff: float,
    box_dims: Optional[Tuple[float, float, float]] = None,
) -> List[Tuple[int, int]]:
    """Spatial hash grid O(N) interface contact finder with optional periodic boundary conditions."""
    ...

def trajectory_pca(
    coords: NDArray[np.float32],
    n_components: int = 2,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Rayon-parallel Cartesian trajectory PCA returning `(projections, explained_variance_ratio, components)`."""
    ...

def markov_populations(
    state_sequence: Sequence[int],
    num_states: int,
    lag_time: int = 1,
    temperature: float = 300.0,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Estimate maximum-likelihood Markov transition matrix, stationary populations pi_i, and relative free energy dG_i (kcal/mol).

    Returns:
        (populations, delta_g_kcal_mol, transition_matrix)
    """
    ...

def cuda_is_available() -> bool:
    """Return True if the CUDA driver and GPU trajectory engine are available."""
    ...

def cuda_pairwise_rmsd(trajectory: NDArray[np.float32]) -> NDArray[np.float32]:
    """GPU-accelerated (Theobald QCP) all-to-all `[T, T]` pairwise RMSD matrix."""
    ...

def cuda_cluster_daura(
    trajectory: NDArray[np.float32],
    cutoff: float,
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """GPU-accelerated Daura conformational clustering returning `(cluster_labels, centroid_indices)`."""
    ...

def cuda_contact_map_bitmask(
    trajectory: NDArray[np.float32],
    cutoff: float,
) -> NDArray[np.uint64]:
    """GPU bit-packed contact map across a `[T, N, 3]` trajectory."""
    ...

def cuda_contact_frequency(
    coords: NDArray[np.float32],
    cutoff: Optional[float] = 4.5,
    device_id: Optional[int] = 0,
) -> NDArray[np.float32]:
    """GPU-accelerated `[N, N]` contact frequency matrix across a `[T, N, 3]` trajectory."""
    ...
