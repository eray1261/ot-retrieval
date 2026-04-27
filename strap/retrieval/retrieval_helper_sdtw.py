import os
import typing as tp
from itertools import accumulate

import h5py
from tqdm.auto import tqdm
import numpy as np
from strap.utils.file_utils import get_demo_grp
from strap.utils.processing_utils import flatten_2d_array
from strap.utils.retrieval_utils import (
    RetrievalArgs,
    load_embeddings_into_memory,
    TrajectoryEmbedding,
    segment_trajectory_by_derivative,
    merge_short_segments,
    get_distance_matrix,
    compute_accumulated_cost_matrix_subsequence_dtw_21,
    compute_optimal_warping_path_subsequence_dtw_21,
    TrajectoryMatchResult,
)

from strap.utils.retrieval_utils import load_single_embedding_into_memory


def process_matches(
    args: RetrievalArgs, nested_match_list: tp.List[tp.List[TrajectoryMatchResult]]
):
    k = int(args.top_k / len(nested_match_list))
    if k == 0:
        print(
            "Requested to retrieve less segments than there are than query sub-trajectories. Defaulting"
            "to one match per query."
        )
        k = 1

    for matches in nested_match_list:
        matches.sort()
        matches[:] = matches[:k]
    return nested_match_list


def run_retrieval(
    args: RetrievalArgs,
) -> tp.Tuple[tp.List[TrajectoryMatchResult], tp.List[tp.List[TrajectoryMatchResult]]]:
    """
    Run retrieval to find matching trajectory segments from an offline dataset.

    Args:
        args (RetrievalArgs): Arguments specifying the retrieval configuration, including:
    Returns:
        tuple:
            - List[TrajectoryMatchResult]: Original full trajectories from task dataset
            - List[List[TrajectoryMatchResult]]: Nested list of retrieved trajectory segments,
              where outer list corresponds to subtask task trajectories and inner lists contain
              the top-k matching segments for that trajectory.
    """
    task_embeddings: tp.List[TrajectoryEmbedding] = load_embeddings_into_memory(args)
    original_demo_trajectories = [
        TrajectoryMatchResult(
            start=0,
            end=len(task_embedding),
            cost=0,
            file_path=task_embedding.file_path,
            file_traj_key=task_embedding.file_traj_key,
        )
        for task_embedding in task_embeddings
    ]
    task_embeddings = slice_embeddings(args, task_embeddings)
    nested_match_list = get_all_matches(args, task_embeddings)
    nested_match_list = process_matches(args, nested_match_list)
    return original_demo_trajectories, nested_match_list


def slice_embeddings(
    args: RetrievalArgs, task_embeddings: tp.List[TrajectoryEmbedding]
):
    new_task_embeddings = []
    for task_embedding in task_embeddings:
        # segment using state derivative heuristic
        segments = segment_trajectory_by_derivative(
            task_embedding.eef_poses, threshold=5e-3
        )
        merged_segments = merge_short_segments(
            segments, min_length=args.min_subtraj_len
        )

        # extract slice indexes
        seg_idcs = [0] + list(accumulate(len(seg) for seg in merged_segments))

        for i in range(len(seg_idcs) - 1):
            new_task_embeddings.append(
                TrajectoryEmbedding(
                    embedding=task_embedding.embedding[seg_idcs[i] : seg_idcs[i + 1]],
                    eef_poses=None,  # we do not need them anymore as we sliced already
                    file_path=task_embedding.file_path,
                    file_traj_key=task_embedding.file_traj_key,
                    file_model_key=task_embedding.file_model_key,
                    file_img_keys=task_embedding.file_img_keys,
                )
            )
    del task_embeddings
    return new_task_embeddings


def get_all_matches(
    args: RetrievalArgs, task_embeddings: tp.List[TrajectoryEmbedding]
) -> tp.List[tp.List[TrajectoryMatchResult]]:
    # Need to loop over offline data
    total_matches = 0
    if args.verbose:
        # calculate total number of matches to create
        for i in range(len(args.offline_dataset)):
            embedding_file_path = args.offline_dataset.embedding_paths[i]
            with h5py.File(embedding_file_path, "r", swmr=True) as embedding_file:
                grp = get_demo_grp(
                    embedding_file, args.offline_dataset.file_structure.demo_group
                )
                total_matches += len(grp.keys())
        total_matches *= len(task_embeddings)

    with tqdm(
        total=total_matches, disable=not args.verbose, desc="Finding Matches"
    ) as pbar:
        result_nested_list = [[] for _ in range(len(task_embeddings))]
        for i in range(len(args.offline_dataset)):
            file_path, embedding_path = (
                args.offline_dataset.dataset_paths[i],
                args.offline_dataset.embedding_paths[i],
            )
            with h5py.File(embedding_path, "r", swmr=True) as embedding_file:
                emb_grp = get_demo_grp(
                    embedding_file, args.offline_dataset.file_structure.demo_group
                )
                for traj_key in emb_grp.keys():
                    # load the trajectory into memory
                    off_traj_embd: TrajectoryEmbedding = (
                        load_single_embedding_into_memory(
                            args, emb_grp, traj_key, file_path=file_path
                        )
                    )
                    for j, sub_traj_embedding in enumerate(task_embeddings):
                        single_match = get_single_match(
                            sub_traj_embedding, off_traj_embd
                        )
                        pbar.update(1)
                        if single_match is None:
                            continue
                        result_nested_list[j].append(single_match)
    return result_nested_list


def get_single_match(
    sub_traj_embedding: TrajectoryEmbedding, off_traj_embd: TrajectoryEmbedding
) -> tp.Union[None, TrajectoryMatchResult]:
    if len(sub_traj_embedding) > len(off_traj_embd):
        # There cannot be a valid match
        return None

    distance_matrix = get_distance_matrix(
        sub_traj_embedding.embedding, off_traj_embd.embedding
    )
    accumulated_cost_matrix = compute_accumulated_cost_matrix_subsequence_dtw_21(
        distance_matrix
    )
    path = compute_optimal_warping_path_subsequence_dtw_21(accumulated_cost_matrix)
    start = path[0, 1]
    if start < 0:
        assert start == -1
        start = 0
    end = path[-1, 1]
    cost = accumulated_cost_matrix[-1, end]
    # Note that the actual end index is inclusive in this case so +1 to use python : based indexing
    end = end + 1
    return TrajectoryMatchResult(
        start=start,
        end=end,
        cost=cost,
        file_path=off_traj_embd.file_path,
        file_traj_key=off_traj_embd.file_traj_key,
    )


def save_results(
    args: RetrievalArgs,
    full_task_trajectory_results: tp.List[TrajectoryMatchResult],
    nested_match_list: tp.List[tp.List[TrajectoryMatchResult]],
) -> None:

    if os.path.isfile(args.output_path):
        print(f"Output file already exists, overwriting...")

    # make the output location if it doesn't exist
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    with h5py.File(args.output_path, "w") as f:
        # Global file-level metadata (task, model, keys, etc.)
        args.task_dataset.initalize_save_file_metadata(f, args.task_dataset)

        # Create (or reuse) the demo group
        demo_group_name = args.task_dataset.file_structure.demo_group
        if demo_group_name is not None:
            if demo_group_name not in f:
                demo_grp = f.create_group(demo_group_name)
            else:
                demo_grp = f[demo_group_name]
        else:
            demo_grp = f

        # Flatten nested [subtraj][matches] → [matches]
        nested_match_list = flatten_2d_array(nested_match_list)

        cur_idx = 0
        retrieved_keys: tp.List[str] = []
        gt_keys: tp.List[str] = []

        # ============= RETRIEVED SEGMENTS =============
        for match in tqdm(nested_match_list, desc="Saving Matches"):
            demo_key = f"demo_{cur_idx}"
            retrieved_keys.append(demo_key)

            # Let STRAP create demo_{i} inside demo_grp
            with h5py.File(match.file_path, "r", swmr=True) as data_file:
                args.offline_dataset.save_trajectory_match(
                    data_grp=data_file,
                    out_grp=demo_grp,              # parent group
                    result=match,
                    args=args,
                    dataset_config=args.task_dataset,
                    new_demo_key=demo_key,         # name of this demo
                )

            # Now access the newly created demo group and attach metadata
            save_grp = demo_grp[demo_key]
            meta = save_grp.create_group("metadata")
            meta.create_dataset("cost", data=match.cost)
            meta.create_dataset("start", data=match.start)
            meta.create_dataset("end", data=match.end)
            meta.create_dataset("source_file", data=np.string_(match.file_path))
            meta.create_dataset("source_traj_key", data=np.string_(match.file_traj_key))

            cur_idx += 1

        # ============= FULL GT TASK TRAJECTORIES =============
        for match in full_task_trajectory_results:
            demo_key = f"demo_{cur_idx}"
            gt_keys.append(demo_key)

            with h5py.File(match.file_path, "r", swmr=True) as data_file:
                args.offline_dataset.save_trajectory_match(
                    data_grp=data_file,
                    out_grp=demo_grp,
                    result=match,
                    args=args,
                    dataset_config=args.task_dataset,
                    new_demo_key=demo_key,
                )

            save_grp = demo_grp[demo_key]
            meta = save_grp.create_group("metadata")
            meta.create_dataset("cost", data=0.0)
            meta.create_dataset("start", data=match.start)
            meta.create_dataset("end", data=match.end)
            meta.create_dataset("source_file", data=np.string_(match.file_path))
            meta.create_dataset("source_traj_key", data=np.string_(match.file_traj_key))

            cur_idx += 1

        # ============= MASKS =============
        mask_grp = f.create_group("mask")
        mask_grp.create_dataset("demos", data=np.array(retrieved_keys, dtype="S"))
        mask_grp.create_dataset(
            "all", data=np.array(retrieved_keys + gt_keys, dtype="S")
        )
        mask_grp.create_dataset("retrieved", data=np.array(retrieved_keys, dtype="S"))

        print(f"Output file saved at {args.output_path}")
