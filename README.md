# ELEP

<!-- PROJECT LOGO & STATUS -->
<br />
<div align="center">
  <a href="https://github.com/congcy/ELEP">
    <img src="images/ELEP_logo.png" alt="Logo" width="100" height="80">
  </a>
  <h3 align="center">ELEP</h3>
  <p align="center">
    A ensemble-learning based toolkit for seismologists to make the best pick on earthquake phases by combining multiple predictions into the one. 
    <br />
    <br />
    <a href="https://github.com/congcy/ELEP/blob/main/LICENSE" alt="Liscence">
        <img src="https://badgen.net/badge/license/BSD-3-Clause/blue" /></a>
    <a href="https://github.com/congcy/ELEP/tree/main/docs" alt="Documentation Status">
        <img src="https://readthedocs.org/projects/ssec-python-project-template/badge/?version=latest" /></a>
    <a href="https://github.com/congcy/ELEP/tree/main/.github/workflows" alt="Test">
        <img src="https://github.com/uw-ssec/python-project-template/actions/workflows/test.yaml/badge.svg" /></a>
    <br />
    <a href="https://ssec-python-project-template.readthedocs.io/en/latest/?badge=latest">Tutorials</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</div>


## Prerequisites & installations

* create a new environment using [Anaconda](https://www.anaconda.com/) 
  ```python
  >>> conda create --name myenv
  ```
* install [Seisbench](https://github.com/seisbench/seisbench)
  ```python
  >>> pip install seisbench
  ```
* install other packages
  ```python
  >>> pip install numpy scipy pandas obspy h5py mpi4py matplotlib
  ```
* install our toolkit (under development)
  ```python
  >>> pip install ELEP
  ```

<p align="right">(<a href="https://github.com/congcy/ELEP">back to top</a>)</p>

## Features

![workflow](/images/ELEP_framework.png)

This toolkit contains the following features:

1. It provides broadband and multiband prediction workflows.
2. It provides three ensemble estimation or combination approaches.
3. It provides GPU-supported batch predictions on avilable datasets.
4. It supports parallel predictions for real-time earthquake monitoring.
5. It possess a good generalization capability

<p align="right">(<a href="https://github.com/congcy/ELEP">back to top</a>)</p>

## Demo

<p align="right">(<a href="https://github.com/congcy/ELEP">back to top</a>)</p>

## Citation 

<p align="right">(<a href="https://github.com/congcy/ELEP">back to top</a>)</p>

## Contact

<p align="right">(<a href="https://github.com/congcy/ELEP">back to top</a>)</p>
