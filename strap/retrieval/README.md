## Configuring Retrieval
The retrieval process is defined by the config created in the get_args() function. 
The parameters are:
- `task_dataset`: The `DatasetConfig` of the task dataset to use as the retrieval query.
- `offline_dataset`: The `DatasetConfig` of the dataset you want to retrieve from.
- `output_path`: The path to the output file.
- `model_key`: The key of the encoder to use for retrieval.
- `image_keys`: The keys of the images to use for retrieval. If it is multiple images, it will average the embeddings.
- `num_demos`: The number of demos to sample from the task dataset.
- `frame_stack`: How much to pad the sequence start by for models with frame stacks. In our libero experiments, we set this to 5, and then disable the robomimic padding.
- `action_chunk`: How much to pad the sequence end by for models with action chunking. In our libero experiments, we set this to 5.
- `top_k`: How many segments to retrieve.
- `task_dataset_filter`: A regex pattern or list of patterns to filter the demos in the task dataset.
- `offline_dataset_filter`: A regex pattern or list of patterns to filter the demos in the offline dataset.
- `min_subtraj_len`: The minimum length of a subtrajectory to create during slicing.
- `verbose`: Whether to print verbose logging statements.
- `retrieval_seed`: The seed to use for retrieval. Defaults to 42.