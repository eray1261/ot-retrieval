from strap.utils.file_utils import (
    DatasetConfig,
    DatasetFileInfo,
    DatasetSaver,
    DatasetFilePointer,
)
from strap.utils.embedding_utils import get_all_datasets_info
from strap.utils.processing_utils import HDF5Dataset
import typing as tp
from strap.embedding.encoders import BaseEncoder
import os
from tqdm.auto import tqdm
import h5py
from torch.utils.data import DataLoader


def embed_dataset(
    dataset: DatasetConfig,
    encoders: tp.List[BaseEncoder],
    saver_threads: int,
    flip_images=False,
    batch_size=32,
    image_size=(224, 224),
    verbose: bool = False,
):
    """
    Embeds a dataset using the given encoders
    Args:
        dataset (DatasetConfig): Dataset to embed
        encoders (List[BaseEncoder]): Encoders to use
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    # create a map of file path to work needed to be

    dataset_info: tp.Dict[str, DatasetFileInfo] = get_all_datasets_info(
        dataset, encoders, verbose
    )

    # create the dataset saver
    saver = DatasetSaver(num_threads=saver_threads, verbose=verbose)
    for i in tqdm(
        range(len(dataset)), desc=f"Embedding datasets", disable=(not verbose)
    ):
        # prepare in and output paths
        df_path, df_save_path = dataset.dataset_paths[i], dataset.embedding_paths[i]

        # process image keys
        for img_key in dataset.file_structure.obs_image_groups:
            if verbose:
                print(f"Embedding dataset: {df_path}, img_key: {img_key}")

            hdf5_dataset = HDF5Dataset(
                dataset_path=df_path,
                file_structure=dataset.file_structure,
                get_language_instruction=dataset.get_language_instruction,
                img_key=img_key,
                img_size=image_size,
                flip_imgs=flip_images,
                verbose=verbose,
            )

            dataloader = DataLoader(
                hdf5_dataset,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=False,
            )

            # iter models
            for model in encoders:
                model_k = model.embedding_file_key

                # check if model embedding already exists i.e. info -> file -> model_key -> img_key is True means it is done
                if dataset_info[df_path].model_to_image_to_done[model_k][img_key]:
                    if verbose:
                        print(model_k, img_key, "already exists. skipping...")
                    continue

                features = model.encode_dataloader(dataloader, verbose=1)

                if saver_threads > 1:
                    saver.queue_save_job(
                        df_save_path,
                        dataset_info[df_path],
                        features,
                        model_k,
                        img_key,
                        dataset.file_structure,
                    )
                else:
                    # save one-by-one
                    if df_save_path not in saver.data_registry:
                        saver.data_registry[df_save_path] = DatasetFilePointer(
                            None,
                            df_save_path,
                            dataset_info[df_path],
                            dataset.file_structure.demo_group,
                        )
                    saver.save_job(
                        (
                            df_save_path,
                            features,
                            model_k,
                            img_key,
                            dataset.file_structure,
                        )
                    )

    saver.wait_until_saved()


