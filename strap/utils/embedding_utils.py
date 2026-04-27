from strap.utils.file_utils import (
    DatasetConfig,
    DatasetFileInfo,
    HDF5FileStructure,
)
import typing as tp
from strap.embedding.encoders import BaseEncoder
import os
import h5py




def get_all_datasets_info(
    dataset: DatasetConfig, encoders: tp.List[BaseEncoder], verbose=False
):
    """
    Gets all the dataset info for the given dataset and encoders
    Args:
        dataset (DatasetConfig): Dataset to embed
        encoders (List[BaseEncoder]): Encoders to use
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    Returns:
        tp.Dict[str, DatasetFileInfo]: Dictionary of dataset file path to DatasetFileInfo
    """

    dataset_info = {}

    model_keys = [encoder.embedding_file_key for encoder in encoders]
    img_keys = dataset.file_structure.obs_image_groups
    if verbose:
        print(f"Using model keys to save: {model_keys}")
        print(f"Using image keys to embed: {img_keys}")

    # Populate the dataset_info
    for i in range(len(dataset)):
        # prepare in and output paths
        df_path, df_save_path = dataset.dataset_paths[i], dataset.embedding_paths[i]
        is_file = os.path.isfile(df_save_path)

        with h5py.File(df_path, "r", swmr=True) as original_file:
            if not is_file:
                dataset_info[df_path] = get_dataset_file_info(
                    original_file,
                    None,
                    model_keys,
                    img_keys,
                    df_structure=dataset.file_structure,
                )
            else:
                with h5py.File(df_save_path, "a") as save_file:
                    dataset_info[df_path] = get_dataset_file_info(
                        original_file,
                        save_file,
                        model_keys,
                        img_keys,
                        df_structure=dataset.file_structure,
                    )

    return dataset_info


def get_dataset_file_info(
    file: h5py.File,
    output_file: tp.Union[None, h5py.File],
    model_keys,
    image_keys,
    df_structure: HDF5FileStructure,
):
    """
    This method takes in the output file, creates the "data" group if it doesn't exist
    and returns the information used about the dataset for future use.

    Args:
        file (h5py.File): The input dataset file
        output_file (h5py.File): The output dataset file
        model_keys (_type_): all model keys of interest
        image_keys (_type_): all image keys of interest
        key_0 (str, optional): This is the key of the first demo to check if already embedded. Defaults to "demo_1".

    Returns:
        DatasetInfo: Information about the dataset
    """

    if df_structure.demo_group is None:
        key_to_len = {
            k: len(file[k][df_structure.obs_action_group]) for k in file.keys()
        }
    else:
        key_to_len = {
            k: len(file[df_structure.demo_group][k][df_structure.obs_action_group])
            for k in file[df_structure.demo_group].keys()
        }

    model_to_image_to_done = {}

    # If there is no output file:
    if output_file is None:
        model_to_image_to_done = {
            model_key: {image_key: False for image_key in image_keys}
            for model_key in model_keys
        }
        return DatasetFileInfo(key_to_len, model_to_image_to_done)

    if (
        output_file.get(df_structure.demo_group) is None
        and df_structure.demo_group is not None
    ):
        grp = output_file.create_group(df_structure.demo_group)
    else:
        grp = (
            output_file[df_structure.demo_group]
            if df_structure.demo_group is not None
            else output_file
        )

    for model_key in model_keys:
        model_to_image_to_done[model_key] = {}
        for image_key in image_keys:
            model_to_image_to_done[model_key][image_key] = True
            for traj_key in (
                file[df_structure.demo_group].keys()
                if df_structure.demo_group is not None
                else file.keys()
            ):
                if (
                    (grp.get(traj_key) is None)
                    or (grp[traj_key].get(model_key) is None)
                    or (grp[traj_key][model_key].get(image_key) is None)
                ):
                    model_to_image_to_done[model_key][image_key] = False
                    break

    return DatasetFileInfo(key_to_len, model_to_image_to_done)
