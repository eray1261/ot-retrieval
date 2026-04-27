import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import typing as tp

T = tp.TypeVar("T")


def flatten_2d_array(arr_2d: tp.List[tp.List[T]]) -> tp.List[T]:
    flattened = []
    for sublist in arr_2d:
        flattened.extend(sublist)
    return flattened


def resize_batch(imgs, img_size):
    resized_imgs = []
    for img in imgs:
        resized_imgs.append(cv2.resize(img, img_size))
    return np.stack(resized_imgs)


class HDF5Dataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        file_structure: "HDF5FileStructure",
        img_key: str,
        get_language_instruction: tp.Callable[[h5py.File, str], str],
        img_size=(224, 224),
        verbose=False,
        flip_imgs=False,
    ):

        self.data = []
        self.lang = []
        self.actions = []

        with h5py.File(dataset_path, "r", swmr=True) as f:

            demo_group = (
                f[file_structure.demo_group]
                if file_structure.demo_group is not None
                else f
            )

            demo_keys = list(demo_group.keys())

            images = []
            langs = []
            actions = []
            for k in tqdm(
                demo_keys,
                disable=True,
            ):
                observations = resize_batch(np.array(demo_group[k][img_key]), img_size)

                images.append(observations)
                language_instruction = get_language_instruction(f, k)

                languages = [language_instruction] * len(observations)

                langs.append(languages)
                acts = np.array(demo_group[k][file_structure.obs_action_group])
                actions.append(acts)
            langs = np.concatenate(langs, axis=0)
            images = np.concatenate(images, axis=0)
            actions = np.concatenate(actions, axis=0)

            # if imgs are upside down
            if flip_imgs:
                images = images[:, ::-1].copy()  # fix negative stride bug with copy
            # change from B x H x W x C to B x C x H x W
            if images.shape[3] == 3:
                images = images.transpose(0, 3, 1, 2)

            self.data = images
            self.lang = langs
            self.actions = actions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.lang[idx], self.actions[idx]
