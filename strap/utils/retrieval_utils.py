import random
from dataclasses import dataclass
import typing as tp

import h5py
import numba as nb
import numpy as np
import ot
import torch
from geomloss import SamplesLoss

from strap.utils.file_utils import DatasetConfig, get_demo_grp


@dataclass
class RetrievalArgs:
    """
    task_dataset: The `DatasetConfig` of the task dataset to use as the retrieval query.
    offline_dataset: The `DatasetConfig` of the dataset you want to retrieve from.
    output_path: The path to the output file.
    model_key: The key of the encoder to use for retrieval.
    image_keys: The keys of the images to use for retrieval. If it is multiple images, it will average the embeddings.
    num_demos: The number of demos to sample from the task dataset.
    frame_stack: How much to pad the sequence start by for models with frame stacks. In our libero experiments, we set this to 5, and then disable the robomimic padding.
    action_chunk: How much to pad the sequence end by for models with action chunking. In our libero experiments, we set this to 5.
    top_k: How many segments to retrieve.
    task_dataset_filter: A regex pattern or list of patterns to filter the demos in the task dataset.
    offline_dataset_filter: A regex pattern or list of patterns to filter the demos in the offline dataset.
    min_subtraj_len: The minimum length of a subtrajectory to create during slicing.
    verbose: Whether to print verbose logging statements.
    retrieval_seed: The seed to use for retrieval. Defaults to 42.
    """

    task_dataset: DatasetConfig
    offline_dataset: DatasetConfig
    output_path: str
    model_key: str
    image_keys: tp.Union[str, tp.List[str]]
    num_demos: int
    frame_stack: int
    action_chunk: int
    top_k: int
    task_dataset_filter: tp.Union[tp.List[str], str, None] = None
    offline_dataset_filter: tp.Union[tp.List[str], str, None] = None
    min_subtraj_len: int = 20
    verbose: bool = True
    retrieval_seed: int = 42

    def __post_init__(self):
        if isinstance(self.image_keys, str):
            self.image_keys = [self.image_keys]

        # this code will filter the task dataset files by the passed filters. If the filters are not in the file name,
        # they will be removed from the dataset.
        if self.task_dataset_filter is not None:
            if isinstance(self.task_dataset_filter, str):
                self.task_dataset_filter = [self.task_dataset_filter]
            self.task_dataset.filter_(self.task_dataset_filter)

        if self.offline_dataset_filter is not None:
            if isinstance(self.offline_dataset_filter, str):
                self.offline_dataset_filter = [self.offline_dataset_filter]
            self.offline_dataset.filter_(self.offline_dataset_filter)


@dataclass
class TrajectoryEmbedding:
    embedding: np.ndarray
    eef_poses: tp.Union[np.ndarray, None]
    file_path: str
    file_traj_key: str
    file_model_key: str
    file_img_keys: tp.List[str]

    def __len__(self):
        return len(self.embedding)


@dataclass
class RetrievalResult:
    matches: tp.Any


def load_embeddings_into_memory(args: RetrievalArgs) -> tp.List[TrajectoryEmbedding]:
    results: tp.List[TrajectoryEmbedding] = []
    for i in range(len(args.task_dataset)):
        embedding_file_path = args.task_dataset.embedding_paths[i]
        file_path = args.task_dataset.dataset_paths[i]
        with h5py.File(file_path, "r", swmr=True) as data_file:
            with h5py.File(embedding_file_path, "r", swmr=True) as embedding_file:
                data_grp = get_demo_grp(
                    data_file, args.task_dataset.file_structure.demo_group
                )
                grp = get_demo_grp(
                    embedding_file, args.task_dataset.file_structure.demo_group
                )
                for traj_key in grp.keys():
                    embedding = np.stack(
                        [
                            grp[traj_key][args.model_key][image_key]
                            for image_key in args.image_keys
                        ],
                        axis=1,
                    )
                    embedding = np.mean(embedding, axis=1)
                    file_traj_key = traj_key  # if args.task_dataset.file_structure.demo_group is None else f"{args.task_dataset.file_structure.demo_group}/{traj_key}"
                    eef_poses = np.array(
                        data_grp[traj_key][
                            args.task_dataset.file_structure.obs_eef_pos_group
                        ]
                    )
                    results.append(
                        TrajectoryEmbedding(
                            embedding=embedding,
                            eef_poses=eef_poses,
                            file_path=file_path,
                            file_traj_key=file_traj_key,
                            file_model_key=args.model_key,
                            file_img_keys=args.image_keys,
                        )
                    )
    if args.num_demos > 0:
        # randomly pick that many trajectories
        results = random.sample(results, args.num_demos)
    return results


def load_single_embedding_into_memory(
    args: RetrievalArgs, demo_grp: h5py.File, traj_key, file_path
):
    embedding = np.stack(
        [
            demo_grp[traj_key][args.model_key][image_key]
            for image_key in args.image_keys
        ],
        axis=1,
    )
    embedding = np.mean(embedding, axis=1)

    result = TrajectoryEmbedding(
        embedding=embedding,
        eef_poses=None,
        file_path=file_path,
        file_traj_key=traj_key,
        file_model_key=args.model_key,
        file_img_keys=args.image_keys,
    )
    return result


def segment_trajectory_by_derivative(states, threshold=2.5e-3):
    # Calculate the absolute derivative of the first three states (X, Y, Z)
    diff = np.diff(states[:, :3], axis=0)
    abs_diff = np.sum(np.abs(diff), axis=1)

    # Find points where the derivative is below the threshold (indicating a stop)
    stops = np.where(abs_diff < threshold)[0]

    # Initialize the sub-trajectories list
    sub_trajectories = []
    start_idx = 0

    # Segment the trajectory at each stop point
    for stop in stops:
        sub_trajectories.append(states[start_idx : stop + 1])  # Add the segment
        start_idx = stop + 1  # Update start index

    # Append the last remaining segment
    if start_idx < len(states):
        sub_trajectories.append(states[start_idx:])

    return sub_trajectories


def merge_short_segments(segments, min_length=5):
    merged_segments = []
    current_segment = segments[0]

    for i in range(1, len(segments)):
        # If the current segment is too short, merge it with the next
        if len(current_segment) < min_length:
            current_segment = np.vstack((current_segment, segments[i]))
        else:
            merged_segments.append(
                current_segment
            )  # Save the segment if it's long enough
            current_segment = segments[i]  # Start a new segment

        prev_segment = current_segment

    # If the last segment is too short, merge it with the previous
    if len(current_segment) < min_length:
        merged_segments[-1] = np.vstack((merged_segments[-1], current_segment))
    else:
        merged_segments.append(current_segment)

    return merged_segments


"""
Code referenced and borrowed from https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html
"""


@dataclass
class TrajectoryMatchResult:
    start: int
    end: int
    cost: int
    file_path: str
    file_traj_key: str

    def __ge__(self, other):
        return self.cost >= other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __gt__(self, other):
        return self.cost > other.cost


def compare_distance_result(x, y):
    if x.index < y.index:
        return -1
    if x.index > y.index:
        return 1
    if x.start < y.start:
        return -1
    if x.start > y.start:
        return 1
    return 0


@nb.jit(nopython=True)
def get_distance_matrix(sub_trajectory, dataset_trajectory):
    """
    This is a fast calculation of the Euclidean distance matrix. It is compiled down to C using numba.
    Args:
        sub_trajectory: The sub_trajectory embedding vector that you want retrieve based off of. This
            can be also a full trajectory
        dataset_trajectory:  The dataset trajectory that you are comparing the sub_trajectory to

    Returns:
    This returns the Euclidean distance matrix.
    """
    sub_squared = np.sum(sub_trajectory**2, axis=1)[:, np.newaxis]
    dataset_squared = np.sum(dataset_trajectory**2, axis=1)[:, np.newaxis]

    cross_term = np.dot(sub_trajectory, dataset_trajectory.T)
    # since ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a * b
    distance_matrix = np.sqrt(sub_squared - 2 * cross_term + dataset_squared.T)

    return distance_matrix


# Most of this code was taken from this reference https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html
@nb.jit(nopython=True)
def compute_accumulated_cost_matrix_subsequence_dtw_21(C):
    """
    Args:
        C (np.ndarray): Cost matrix
    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N, M = C.shape
    D = np.zeros((N + 1, M + 2))
    D[0:1, :] = np.inf
    D[:, 0:2] = np.inf

    D[1, 2:] = C[0, :]

    for n in range(1, N):
        for m in range(0, M):
            if n == 0 and m == 0:
                continue
            D[n + 1, m + 2] = C[n, m] + min(
                D[n - 1 + 1, m - 1 + 2], D[n - 1 + 1, m - 2 + 2]
            )  # D[n-2+1, m-1+2],
    D = D[1:, 2:]
    return D


@nb.jit(nopython=True)
def compute_optimal_warping_path_subsequence_dtw_21(D, m=-1):
    """
    Args:
        D (np.ndarray): Accumulated cost matrix
        m (int): Index to start back tracking; if set to -1, optimal m is used (Default value = -1)

    Returns:
        P (np.ndarray): Optimal warping path (array of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n - 1, m - 1], D[n - 1, m - 2])  # D[n-2, m-1],
            if val == D[n - 1, m - 1]:
                cell = (n - 1, m - 1)
            # elif val == D[n-2, m-1]:
            #     cell = (n-2, m-1)
            else:
                cell = (n - 1, m - 2)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P


# ============================================================================
# Optimal Transport (OT) Functions - Alternative to DTW
# ============================================================================

@nb.jit(nopython=True)
def get_structure_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the structure matrix (pairwise distances within a trajectory).
    This captures the internal geometry of the trajectory.
    
    Args:
        embeddings: Trajectory embeddings of shape (T, D)
        
    Returns:
        Structure matrix of shape (T, T)
    """
    T = embeddings.shape[0]
    C = np.zeros((T, T))
    
    for i in range(T):
        for j in range(T):
            diff = embeddings[i] - embeddings[j]
            C[i, j] = np.sqrt(np.sum(diff ** 2))
    
    return C


def compute_fgw_distance(
    query_embedding: np.ndarray,
    target_embedding: np.ndarray,
    alpha: float = 0.5,
    numItermax: int = 100
) -> tp.Tuple[float, np.ndarray]:
    """
    Compute Fused Gromov-Wasserstein distance between query and target trajectories.
    
    Args:
        query_embedding: Query trajectory embeddings (T1, D)
        target_embedding: Target trajectory embeddings (T2, D)
        alpha: Weight between structure (0) and features (1). Default 0.5 balances both.
        numItermax: Maximum iterations for optimization
        
    Returns:
        (distance, transport_plan): FGW distance and optimal transport plan
    """
    T1, T2 = len(query_embedding), len(target_embedding)
    
    # Compute structure matrices
    C1 = get_structure_matrix(query_embedding)
    C2 = get_structure_matrix(target_embedding)
    
    # Compute feature distance matrix (reuse existing function)
    M = get_distance_matrix(query_embedding, target_embedding)
    
    # Uniform distributions
    p = ot.unif(T1)
    q = ot.unif(T2)
    
    # Compute FGW - Fixed parameter passing
    fgw_dist, log = ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p, q,
        loss_fun='square_loss',
        alpha=alpha,
        max_iter=numItermax,  # Changed from numItermax to max_iter
        log=True,
        verbose=False
    )
    
    transport_plan = log['T']
    
    return fgw_dist, transport_plan

def adaptive_window_fgw(
    query_embedding: np.ndarray,
    target_embedding: np.ndarray,
    min_length: int = 20,
    alpha: float = 0.5,
    top_k_windows: int = 10
) -> tp.Tuple[int, int, float]:
    """
    Adaptive sliding window FGW - faster than exhaustive search.
    First does coarse search, then refines around best candidates.
    
    Args:
        query_embedding: Query trajectory (T1, D)
        target_embedding: Target trajectory to search in (T2, D)
        min_length: Minimum subsequence length
        alpha: FGW weight parameter
        top_k_windows: Number of top candidates to refine
        
    Returns:
        (start_idx, end_idx, cost): Best matching subsequence indices and cost
    """
    T1 = len(query_embedding)
    T2 = len(target_embedding)
    
    if T1 > T2:
        return -1, -1, float('inf')
    
    # Phase 1: Coarse search
    coarse_stride = max(10, T1 // 4)
    candidates = []
    
    window_sizes = [T1, int(T1 * 1.2), int(T1 * 1.5), int(T1 * 2.0)]
    window_sizes = [w for w in window_sizes if w <= T2]
    
    for window_size in window_sizes:
        for start in range(0, T2 - window_size + 1, coarse_stride):
            end = start + window_size
            target_window = target_embedding[start:end]
            
            cost, _ = compute_fgw_distance(
                query_embedding,
                target_window,
                alpha=alpha,
                numItermax=30
            )
            
            normalized_cost = cost / window_size
            candidates.append((start, end, normalized_cost, cost))
    
    # Phase 2: Refine top candidates
    candidates.sort(key=lambda x: x[2])
    top_candidates = candidates[:top_k_windows]
    
    best_cost = float('inf')
    best_start = 0
    best_end = T1
    
    for cand_start, cand_end, _, _ in top_candidates:
        search_radius = coarse_stride
        fine_stride = max(1, coarse_stride // 5)
        
        start_range = range(
            max(0, cand_start - search_radius),
            min(T2, cand_start + search_radius),
            fine_stride
        )
        
        for start in start_range:
            for delta in range(-5, 6, 2):
                window_size = (cand_end - cand_start) + delta
                if window_size < min_length or start + window_size > T2:
                    continue
                    
                end = start + window_size
                target_window = target_embedding[start:end]
                
                cost, _ = compute_fgw_distance(
                    query_embedding,
                    target_window,
                    alpha=alpha,
                    numItermax=100
                )
                
                if cost < best_cost:
                    best_cost = cost
                    best_start = start
                    best_end = end
    
    return best_start, best_end, best_cost


def compute_ot_match(
    query_embedding,
    target_embedding,
    min_length=20,
    alpha=0.5,
    method="adaptive"
):
    """
    GPU-accelerated Hybrid FGW approximation using Sinkhorn OT + structural term.
    alpha ∈ [0,1] balances feature vs. structure alignment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to torch tensors on GPU
    query = torch.tensor(query_embedding, dtype=torch.float32, device=device)
    target = torch.tensor(target_embedding, dtype=torch.float32, device=device)

    # 1️⃣ Feature-level OT via Sinkhorn divergence (fast, GPU)
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    feature_cost = loss_fn(query, target)

    # 2️⃣ Structure-level alignment via intra-trajectory distance matrices
    C1 = torch.cdist(query, query)
    C2 = torch.cdist(target, target)
    structure_cost = torch.norm(C1.mean() - C2.mean())

    # 3️⃣ Combined hybrid FGW-like distance
    total_cost = alpha * feature_cost + (1 - alpha) * structure_cost
    cost_value = total_cost.item()

    # Return dummy indices (retrieval only ranks by cost)
    return 0, len(target_embedding), cost_value