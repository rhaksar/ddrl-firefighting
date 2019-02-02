# ddrl-firefighting

A repository to support the paper [Distributed Deep Reinforcement Learning for Fighting Forest Fires with a Network of Aerial Robots](https://ieeexplore.ieee.org/abstract/document/8593539).

Paper citation:
```
@InProceedings{8593539, 
    author={R. N. Haksar and M. Schwager}, 
    booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
    title={Distributed Deep Reinforcement Learning for Fighting Forest Fires with a Network of Aerial Robots}, 
    year={2018}, 
    pages={1067-1074}, 
    doi={10.1109/IROS.2018.8593539}, 
    ISSN={2153-0866}, 
    month={Oct},}
```

### Requirements
- Developed with Python 3.5
- Requires [`pytorch`](https://pytorch.org/) and `numpy`
- Requires the [simulators](https://github.com/rhaksar/simulators) repository: clone the repository into the root level of this repository

### Files
- `main.py`: Example usage of algorithm: train and test a network with simulations. 
- `madqn.py`: Implementation of training algorithm, policy architecture, and aerial vehicle model.
- `rlUtilities.py`: Helper utilities to simplify implementation of the reinforcement learning problem.
