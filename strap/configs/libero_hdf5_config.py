import h5py
import json

from strap.configs.libero_file_functions import initialize_libero_dataset
from strap.utils.file_utils import HDF5FileStructure, DatasetConfig
from strap.utils.constants import REPO_ROOT
from strap.configs.libero_file_functions import get_libero_lang_instruction, save_trajectory_result_libero


LIBERO_FILE_STRUCTURE = HDF5FileStructure(
        demo_group="data",
        obs_image_groups=["obs/agentview_rgb"], # ["obs/agentview_rgb", "obs/eye_in_hand_rgb"]
        obs_action_group="actions",
        obs_eef_pos_group="obs/ee_pos",
)


LIBERO_CONFIG = DatasetConfig(
    name="libero",
    absolute_dataset_folder=f"{REPO_ROOT}/data/LIBERO",
    file_structure=LIBERO_FILE_STRUCTURE,
    ds_match_regex="*.hdf5",
    embedding_extension="embeds.hdf5",
    get_language_instruction=get_libero_lang_instruction,
    save_trajectory_match=save_trajectory_result_libero,
    initalize_save_file_metadata=initialize_libero_dataset
)

LIBERO_90_CONFIG = DatasetConfig(
    name="libero",
    absolute_dataset_folder=f"{REPO_ROOT}/data/LIBERO",
    file_structure=LIBERO_FILE_STRUCTURE,
    ds_match_regex="*.hdf5",
    embedding_extension="embeds.hdf5",
    exclude_path=["libero_10"],
    get_language_instruction=get_libero_lang_instruction,
    save_trajectory_match=save_trajectory_result_libero,
    initalize_save_file_metadata=initialize_libero_dataset
)

LIBERO_10_CONFIG = DatasetConfig(
    name="libero",
    absolute_dataset_folder=f"{REPO_ROOT}/data/LIBERO",
    file_structure=LIBERO_FILE_STRUCTURE,
    ds_match_regex="*.hdf5",
    embedding_extension="embeds.hdf5",
    exclude_path=["libero_90"],
    get_language_instruction=get_libero_lang_instruction,
    save_trajectory_match=save_trajectory_result_libero,
    initalize_save_file_metadata=initialize_libero_dataset
)
