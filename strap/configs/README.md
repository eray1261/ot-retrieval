### Dataset Configs and HDF5 File Structure
The first step to defining a dataset is to define the HDF5 file structure. This is done by defining a `HDF5FileStructure` object. This object is a dataclass that contains the following fields:
- `demo_group`: The path from the root of the HDF5 file to the group containing all demonstrations. For example, if your demonstrations are stored under "/data/demos" in the HDF5 file, this would be "data/demos". Set to None if the demonstrations are stored directly at the root level.
- `obs_image_groups`: A list of paths from a demo group to image data. For example, if your images are stored at "/data/demos/###/images/rgb1" and "/data/demos/###/images/rgb2", this would be ["images/rgb1", "images/rgb2"]. These images will be used for embedding and retrieval.
- `obs_action_group`: The path from a demo group to action data. For example, if your actions are stored at "/data/demos/###/actions", this would be "actions".
- `obs_eef_pos_group`: The path from a demo group to end-effector position data. For example, if your end-effector positions are stored at "/data/demos/###/obs/ee_pos", this would be "obs/ee_pos".


The second step is to define a `DatasetConfig` object. This represents a dataset of multiple HDF5 files. This object is a dataclass that contains the following fields:
- `name`: The name of the dataset.
- `absolute_dataset_folder`: The path to the root folder containing all of the HDF5 files.
- `file_structure`: The `HDF5FileStructure` object.
- `ds_match_regex`: A regex pattern to match the HDF5 file paths in the dataset folder. Defaults to "*.hdf5".
- `embedding_extension`: What to append on to each file to save the embeddings version as. Defaults to "embeds.hdf5".
- `exclude_path`: An optional list of regexes to exclude from the dataset.
- `get_language_instruction`: Helper function to get the language instructions from the dataset.
- `save_trajectory_match`:  Helper function to save the retrieved trajectories to the output dataset.
- `initalize_save_file_metadata`: Helper function to copy over the metadata from the task dataset to the output dataset.

We have defined this for the LIBERO dataset in `strap/configs/libero_hdf5_config.py`.

#### Config Helper Functions

The `DatasetConfig` object also contains two helper functions you can define to use new datasets.

- `get_language_instruction`: A function that takes a h5py file in your dataset, and a demo_key and returns the language instruction for that demo. For example, in LIBERO this extracts the task name from the demo metadata.
- `save_trajectory_result`: A function that takes a h5py file input, a new h5py file as output, a `TrajectoryMatchResult`, the args passed for retrieval, the `DatasetConfig` of the input dataset, and the demo key to save the result under in the new file. This function must then handle copying over the relevant data from the TrajectoryMatchResult into the new retrieved dataset. We chose to define this as a configurable function in order to allow for different file structures for different datasets.
- `initalize_save_file_metadata`: A function that takes a h5py file input and the `DatasetConfig` of the input dataset. This function must then handle copying over the relevant metadata from the input dataset to the output dataset, and initializing the file.
