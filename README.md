![argmax.ai](argmaxlogo.png)

*This repository is published by the Volkswagen Group Machine Learning Research Lab.*

*Learn more at https://argmax.ai.*


# [Constrained Probabilistic Movement Primitives for Robot Trajectory Adaptation](https://arxiv.org/abs/2101.12561)

This repository contains the source code of some of the experiments conducted for our paper "Constrained Probabilistic Movement Primitives for Robot Trajectory Adaptation", published in [TRO](https://ieeexplore.ieee.org/document/9655714) in 2021.
The relevant experiments can be found in Section III.B of our paper and their results are shown in the Tables II, III and IV.

[Video explaining the paper](https://youtu.be/7UI6QX-eZ3I)


## Folder Structure

```bash
  ├── docker
  │   ├── Dockerfile                  # Dockerfile for reducing the experiments
  │   └── tf_prob_patch_0.10.1        # patchfile for tf-probability
  ├── docker.build                    # convenience docker build script
  ├── docker.run                      # convenience docker run script
  ├── opt_pmp_utils                   # core implementation of the algorithm
  │   ├── __init__.py
  │   ├── basis_functions.py          # ProMP basis functions
  │   ├── constraints.py              # implementation of the presented constraints
  │   ├── cpmp.py                     # implementation of Constrained ProMPs and algorithm 1
  │   ├── plot_2d_normal.py           # custom plot function
  │   ├── promp.py                    # ProMP implementation
  │   ├── unscented_transform.py      # generic TF unscented transform
  │   └── utils.py                    # utilities
  ├── quant_experiment
  │   ├── 2d_env_vipmp_obsAv.py       # obstacle avoidance experiment script
  │   ├── 2d_env_vipmp_viaP.py        # via-point experiment script
  │   ├── 2d_env_vipmp_vWall.py       # virtual wall experiment script
  │   ├── analyse_obsAv.py            # obstacle avoidance analysis script
  │   ├── analyse_viaP.py             # via-point analysis script
  │   ├── analyse_vWall.py            # virtual wall analysis script
  │   └── trajectory_env.py           # 2D environment used in all experiments
  ├── README.md                       # this file
  └── setup.py                        # Python setup file
```


## Reproducing the Results of the Paper

This repository contains 3 experiment scripts in the `quant_experiment` folder.
These can be used to run an obstacle avoidance, a via-point and a virtual wall experiment.
They produce Figures similar to the Figures 6, 7 and 8 of our paper and they also store all data necessary to produce tables similar to the Tables II, III and IV.
We provide a docker environment and step-by-step instructions on how to run the respective experiments.

### Setup docker

Install `docker>=20.10.10` according to their [published manuals](https://docs.docker.com/get-docker/).
It helps if you have basic familiarity with docker, otherwise you can go through their [getting started manuals](https://docs.docker.com/get-started/).

### Build and Run the Docker Container

There are two convenience scripts, `docker.build` and `docker.run`, at the top level of the repository that you can execute to build and run the docker container, respectively.

```bash
cd <repository root folder>
./docker.build
./docker.run <absolute path to the output folder (optional)>
```

The `docker.run` command accepts an argument determining the output path for experiments which are run from the docker container.
This path has to be absolute.
If not specified, the output folder defaults to the `cpmp-exp-output` at the root of the repository.


### Run the Experiments

Inside the docker container, simply execute an experiment by running the corresponding script in the `quant_experiment` folder:

  - Obstacle avoidance experiment: `2d_env_vipmp_obsAv.py`
  - Via-point experiment: `2d_env_vipmp_viaP.py`
  - Virtual wall experiment: `2d_env_vipmp_vWall.py`

These scripts have a range of parameters that you can adjust, such as the number of experiments, the number of obstacles, scale of the smoothness penalty or the number of iterations you allow.
Note that you have to rebuild the docker container (by running `./docker.build`) after changing any of the parameters or the code in general for the changes to take effect.


### Evaluate the Experiments

If you want to look at the generated figures after running the experiments you can check out the experiment output folder you've specified in the `docker.run` command (or the `cpmp-exp-output` folder by default).
Every type of experiment (obstacle avoidance, via-points and virtual wall) will be in a separate folder and every individual experiment will be in a folder with the timestamp as its name.
The experiment output folder contains some images as well as checkpoints and a pickle file with the final state of the learned ProMP.
The final state images of every experiment are called `vipmp_final.png`

For producing tables similar to the ones presented in the paper, run the corresponding analysis script inside the docker container after running the experiment:

  - Obstacle avoidance experiment: `analyse_obsAv.py`
  - Via-point experiment: `analyse_viaP.py`
  - Virtual wall experiment: `analyse_vWall.py`


### Run & Evaluate All Experiments

```bash
cd <repository root folder>
./docker.build
./docker.run
# Now inside a shell that opens after executing docker.run
python 2d_env_vipmp_obsAv.py && python 2d_env_vipmp_viaP.py && python 2d_env_vipmp_vWall.py && python analyse_obsAv.py && python analyse_viaP.py && python analyse_vWall.py
```


## Related Publications

If you find the code useful for your research, please consider citing our work.

```BibTeX
@article{frank2021constrained,
  title={Constrained Probabilistic Movement Primitives for Robot Trajectory Adaptation},
  author={Frank, Felix and Paraschos, Alexandros and van der Smagt, Patrick and Cseke, Botond},
  journal={IEEE Transactions on Robotics},
  year={2021},
  publisher={IEEE}
}
```


## Disclaimer

The purpose of this source code is limited to bare demonstration of the experimental section of the related papers.
