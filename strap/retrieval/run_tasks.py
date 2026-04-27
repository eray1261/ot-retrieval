from strap.utils.retrieval_utils import RetrievalArgs
from strap.utils.constants import REPO_ROOT
from strap.retrieval.retrieval_helper_sdtw import run_retrieval, save_results
from copy import deepcopy
import numpy as np
import random

TASKS = {
    "stove-moka": ".*turn_on_the_stove_and_put_the_moka_pot_on_it.*",
    "soup-cheese": ".*alphabet_soup_and_the_cream_cheese_box.*",
    "mug-mug": ".*white_mug_on_the_left_plate.*",
    "moka-moka": ".*put_both_moka_pots_on_the_stove.*",
}

def get_args(task_name, pattern):
    from strap.configs.libero_hdf5_config import LIBERO_90_CONFIG, LIBERO_10_CONFIG

    return RetrievalArgs(
        task_dataset=deepcopy(LIBERO_10_CONFIG),
        offline_dataset=deepcopy(LIBERO_90_CONFIG),
        output_path=f"{REPO_ROOT}/data/retrieval_results/{task_name}.hdf5",
        model_key="DINOv2",
        image_keys="obs/agentview_rgb",
        num_demos=5,
        frame_stack=5,
        action_chunk=5,
        top_k=100,
        task_dataset_filter=pattern,
        offline_dataset_filter=None,
        min_subtraj_len=20,
    )


def main():
    for name, pattern in TASKS.items():
        print(f"\n🚀 Running task: {name}")

        args = get_args(name, pattern)

        np.random.seed(args.retrieval_seed)
        random.seed(args.retrieval_seed)

        full_task, retrieved = run_retrieval(args)
        save_results(args, full_task, retrieved)


if __name__ == "__main__":
    main()