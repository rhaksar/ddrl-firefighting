# ddrl-firefighting

A repository to support the paper [Distributed Deep Reinforcement Learning for Fighting Forest Fires with a Network of Aerial Robots](https://msl.stanford.edu/sites/g/files/sbiybj8446/f/haksar_iros2018_0.pdf).

Video:  
[![click to play](https://img.youtube.com/vi/bVWf2fJ2WRQ/0.jpg)](https://www.youtube.com/watch?v=bVWf2fJ2WRQ)

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
- Developed with Python 3.6
- Requires `numpy` and [`pytorch`](https://pytorch.org/) (tested with version 1.4.0)
- Requires the [simulators](https://github.com/rhaksar/simulators) repository

### Files
- `madqn.py`: Implementation of training algorithm, policy architecture, and aerial vehicle model.
- `main.py`: Example usage of algorithm: train and test a network with simulations. 
- `rlUtilities.py`: Helper utilities to simplify implementation of the reinforcement learning problem.
