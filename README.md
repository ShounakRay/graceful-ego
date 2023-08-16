# TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios

[**Webpage**](https://metadriverse.github.io/trafficgen/) | 
[**Code**](https://github.com/metadriverse/trafficgen) |
[**Video**](https://youtu.be/jPS93-d6msM) |
[**Paper**](https://arxiv.org/pdf/2210.06609.pdf)



## Setup environment

### (Mostly) Original Instructions Below from Forked Repository
```bash
# Create virtual environment
conda create -n trafficgen python=3.8
conda activate trafficgen

# Read lines 5 and 6 in `requirements.txt` and enable/disable accordingly.

# You should install pytorch by yourself to make them compatible with your GPU. REFER TO: https://pytorch.org/get-started/locally/.
# # For cuda 11.0:
# python3 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# # For MacOS:
# python3 -m pip install torch torchvision torchaudio

# Install basic dependency
python3 -m pip install -e .
```
If you find error messages related to `geos` when installing `Shapely`, checkout [this post](https://stackoverflow.com/questions/19742406/could-not-find-library-geos-c-or-load-any-of-its-variants).

### Completely Additional Instructions after trouble-shooting

**Applicable for all users (OS-agnostic):**

For some reason, ```metadrive``` doesn't come installed, so complete the following instructions.
First navigate to ```trafficgen``` subdirectory in ```graceful-ego```.
From here, install the metadrive MODULE (as per <https://github.com/metadriverse/metadrive>).
```bash
python3 -m pip install metadrive-simulator
```
Be sure to include the ```python3 -m``` portion to ensure that you're installing the package to the proper conda-linked python3 and pip.

I wasn't able to get the above, simple pip install working for metadrive. So I had to create a submodule and clone the entire repository as follows:
```bash
# From root (graceful-ego), navigate to trafficgen/
cd trafficgen/
# Install metadrive
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
python3 -m pip install -e .

"""
# Now that we've installed all the pip dependencies, we're going to change up the structure a little bit so `generate.py` recognizes metadrive
# ONLY run the following after running `python3 -m pip install -e .`
# GENERAL IDEA: Notice that inside `graceful-ego/metadrive/` there is another folder called `metadrive`. This lowest-level subfolder has everything we need.
#               We want to move this folder one level up to `graceful-ego`.
#               After this operation, `graceful-ego/metadrive/metadrive` will no longer exist BUT `graceful-ego/metadrive/` will exist.
# COMMANDS (start from `graceful-ego` as you current directory):
"""
# Move the core metadrive folder one level up
mv metadrive/metadrive metadrive_main
# Rename the old metadrive wrapper to something else
mv metadrive metadrive_wrapper
# Rename the core metadrive folder to the proper package name (so we can work with it)
mv metadrive_main metadrive
# [OPTIONAL (NOT TESTED)] Deleted the old metadrive wrapper (no longer needed since we've already installed everything we need from its `requirements.txt` from the `python3 -m pip install -e .` command)
rm -r metadrive_main
```

**MacOS M1 Specific Advice:**
Refer to "trafficgen/MacOS_Additional_Commands.md" for necessary steps.

---
---

## Quick Start

You can run the following scripts to test whether the setup is correct. These scripts do not require
downloading data.

```bash
# Vehicle Placement Model
python3 train_init.py -c local
# Trajectory Generator Model
python3 train_act.py -c local 
```

---
---

## Download and Process Dataset and Pre-trained Model

### Download dataset for road and traffic

Download from Waymo Dataset
- Register your Google account in: https://waymo.com/open/
- [BE SURE TO USE V.110] Open the following link with your Google account logged in: https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0
- Download one or more proto files from `waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training_20s`
- Move download files to PATH_A, where you store the raw tf_record files.

Note: it is not necessary to download all the files from Waymo. You can download one of them for a simple test.

NOTE: [SR] The downloaded files you originally moved to PATH_A will be deleted once you run the following command. You will have a bunch of pickle files with format "^\d+$.pkl".
This is expected and necessary to run ```python3 generate.py```.

*[Changed from old path (it was wrong)]* Data Preprocess
```bash
python3 trafficgen/utils/trans20.py <PATH_A> <PATH_B> None
```
NOTE: PATH_B is where you store the processed data.


[//]: # (The processed data has the following attributes:)

[//]: # (- `id`: scenario id)

[//]: # (- `all_agent`: A `[190, n, 9]` array which contains 190 frames, n agents, 9 features `[coord, velocity, heading, length, width, type, validity]`)

[//]: # (- `traffic_light`: A list containing information about the traffic light)

[//]: # (- `lane`: A `[n,4]` array which contains n points and `[coord, type, id&#40;which lane this point belongs to&#41;]` features.)

[//]: # ()

### Download and retrieve pretrained TrafficGen model

Please download two models from this link: https://drive.google.com/drive/folders/1TbCV6y-vssvG3YsuA6bAtD9lUX39DH9C?usp=sharing

And then put them into `trafficgen/traffic_generator/ckpt` folder.

### Generate new traffic scenarios

Running following scripts will generate images and GIFs (if with `--gif`) visualizing the new traffic scenarios in
`traffic_generator/output/vis` folder.

```bash
# First, change the data usage and set the data dir in trafficgen/init/configs/local.yaml
# Then, you have to change working directory
cd trafficgen/
# Set `--gif` flag to generate GIF files.
python3 generate.py [--gif] [--save_metadrive]
```

#### Troubleshooting `generate.py` command
If you run into the error:
```
Traceback (most recent call last):
  File "generate.py", line 20, in <module>
    trafficgen.generate_scenarios(gif=args.gif, save_metadrive=args.save_metadrive)
  File "/Users/shounak/Documents/GitHub/graceful-ego/trafficgen/traffic_generator/traffic_generator.py", line 84, in generate_scenarios
    self.generate_traj(snapshot=True, gif=gif, save_metadrive=save_metadrive)
  File "/Users/shounak/Documents/GitHub/graceful-ego/trafficgen/traffic_generator/traffic_generator.py", line 276, in generate_traj
    save_as_metadrive_data(i, pred_i, data, data_dir)
  File "/Users/shounak/Documents/GitHub/graceful-ego/trafficgen/traffic_generator/utils/data_utils.py", line 221, in save_as_metadrive_data
    dtype=np.bool,
  File "/Users/shounak/miniconda3/envs/trafficgen/lib/python3.8/site-packages/numpy/__init__.py", line 305, in __getattr__
    raise AttributeError(__former_attrs__[attr])
AttributeError: module 'numpy' has no attribute 'bool'.
`np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```
and your numpy version is NOT 1.23.1 (you can check by running `python3 -m pip show numpy`), then as per <https://stackoverflow.com/a/74938548/9582712>, do:
```bash
python3 -m pip uninstall numpy
python3 -m pip install numpy==1.23.1
```
Refer to this <https://stackoverflow.com/a/71119218/9582712> if you get installation version conflicts.

---


## Connect TrafficGen with MetaDrive

### Create single-agent RL environment

After running `python3 generate.py --save_metadrive`,
a folder `trafficgen/traffic_generator/output/scene_pkl` will be created, and you will see many
pickle files. Each `.pkl` file is a scenario created by TrafficGen.

We provide a script to create single-agent RL environment with TrafficGen generated data.
Please refer to [trafficgen/run_metadrive.py](trafficgen/run_metadrive.py) for details.

We also provide pre-generated scenarios from TrafficGen, so you can kick off RL training
on TrafficGen-generated scenarios immediately. Please follow
[trafficgen/dataset/README.md](trafficgen/dataset/README.md)
to download the dataset.

```bash
cd trafficgen/

# Run generated scenarios:
python3 run_metadrive.py --dataset traffic_generator/output/scene_pkl

# Please read `trafficgen/dataset/README.md` to download pre-generated scenarios
# Then you can use them to create an RL environment:
python3 run_metadrive.py --dataset dataset/validation

# If you want to visualize the generated scenarios, with the ego car also replaying data, use:
python3 run_metadrive.py --dataset dataset/validation --replay

# If you want to create RL environment where traffic vehicles are not replaying 
# but are controlled by interactive IDM policy, use:
python3 run_metadrive.py --dataset dataset/validation --no_replay_traffic
```

NOTE: [SR] Stopped here, seems like a trivial fix, though.
Traceback when running `python3 run_metadrive.py --dataset traffic_generator/output/scene_pkl`.
```
Traceback (most recent call last):
  File "run_metadrive.py", line 61, in <module>
    env = WaymoEnv(config)
  File "/Users/shounak/Documents/GitHub/graceful-ego/trafficgen/metadrive/envs/real_data_envs/waymo_env.py", line 20, in __init__
    super(WaymoEnv, self).__init__(config)
  File "/Users/shounak/Documents/GitHub/graceful-ego/trafficgen/metadrive/envs/scenario_env.py", line 107, in __init__
    super(ScenarioEnv, self).__init__(config)
  File "/Users/shounak/Documents/GitHub/graceful-ego/trafficgen/metadrive/envs/base_env.py", line 218, in __init__
    merged_config = self._merge_extra_config(config)
  File "/Users/shounak/Documents/GitHub/graceful-ego/trafficgen/metadrive/envs/real_data_envs/waymo_env.py", line 23, in _merge_extra_config
    config = self.default_config().update(config, allow_add_new_key=False)
  File "/Users/shounak/Documents/GitHub/graceful-ego/trafficgen/metadrive/utils/config.py", line 122, in update
    raise KeyError(
KeyError: "'{'replay'}' does not exist in existing config. Please use config.update(...)
```

---
---
---
---

You can then kick off RL training by utilizing the created environment showcased in the script above.

### Train RL agents in TrafficGen-generated single-agent RL environment

```bash
# Dependencies:
python3 -m pip install ray==2.2.0
python3 -m pip install ray[rllib]==2.2.0

# Install pytorch by yourself and make it compatible with your CUDA
# ...

# Kickoff training
cd trafficgen

python3 run_rl_training.py --exp-name EXPERIMENT_NAME --num-gpus 1 
# You can also specify the path to dataset. Currently we set:

--dataset_train  dataset/1385_training
--dataset_test  dataset/validation

# by default. Check the file for more details about the arguments. 
```



## Training

### Local Debug
Use the sample data packed in the code repo directly
#### Vehicle Placement Model
````
python3 train_init.py -c local
````
#### Trajectory Generator Model
````
python3 train_act.py -c local
````


### Cluster Training
For training, we recommend to download all the files from: https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0

PATH_A is the raw data path

PATH_B is the processed data path

Execute the data_trans.sh:
```bash
sh utils/data_trans.sh PATH_A PATH_B
```
Note: This will take about 2 hours.

Then modify the 'data_path' in init/configs and act/configs to PATH_B, run:
```bash
python3 init/uitls/init_dataset.py
python3 act/uitls/act_dataset.py
```
to get a processed cache for the model.

Modify cluster.yaml. Change data_path, data_usage, run:
````
python3 train_act.py -c cluster -d 0 1 2 3 -e exp_name
````

-d denotes which GPU to use



## Reference

```latex
@article{feng2022trafficgen,
  title={TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios},
  author={Feng, Lan and Li, Quanyi and Peng, Zhenghao and Tan, Shuhan and Zhou, Bolei},
  journal={arXiv preprint arXiv:2210.06609},
  year={2022}
}
```



