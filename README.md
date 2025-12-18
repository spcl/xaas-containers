# XaaS Containers: Performance-Portable Representation With Source and IR Containers

![License](https://img.shields.io/github/license/spcl/xaas-containers)
![GitHub issues](https://img.shields.io/github/issues/spcl/xaas-containers)
![GitHub pull requests](https://img.shields.io/github/issues-pr/spcl/xaas-containers)

**Performance-portable containers for HPC systems through Source and IR Containers.**

XaaS (Acceleration as a Service) Containers provides a practical solution to achieve performance-portable containers in HPC environments.
By delaying performance-critical decisions until the target system specification is known, our containers combine the convenience of containerization with the performance of system-specialized builds.
We propose two new solutions:
- **Source Containers**: Deploy source of application and its environment to the final system, building only once hardware configuration and dependencies are known
- **IR (Intermediate Representation) Containers**: Distribute container images where application is compiled to LLVM IR, and then optimized and lowered once the target system is known.

If you use XaaS containers in your research, please cite our SC '25 paper:

```bibtex
@inproceedings{10.1145/3712285.3759868,
author = {Copik, Marcin and Alnuaimi, Eiman and Kamatar, Alok and Hayot-Sasson, Valerie and Madonna, Alberto and Gamblin, Todd and Chard, Kyle and Foster, Ian and Hoefler, Torsten},
title = {XaaS Containers: Performance-Portable Representation With Source and IR Containers},
year = {2025},
isbn = {9798400714665},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3712285.3759868},
doi = {10.1145/3712285.3759868},
booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
pages = {533–555},
numpages = {23},
keywords = {Containers, Intermediate Representation, Performance Portability},
series = {SC '25}
}
```

The complete artifact for reproducing experiments from our SC '25 paper is available at [Zenodo](https://doi.org/10.5281/zenodo.17115960).

## Installation

Requirements 
- Docker to build containers (support for Podman and Apptainer coming later)
- Python 3.11 or later

Install Python dependencies and XaaS:

```bash
pip install .
```

## Acknowledgments

XaaS Containers is developed by the [Scalable Parallel Computing Laboratory (SPCL)](https://spcl.inf.ethz.ch/) at ETH Zürich.

