<!-- PROJECT SHIELDS -->
[![arXiv][arxiv-shield]][arxiv-url]
[![DOI][doi-shield]][doi-url]
[![Documentation Status][docs-shield]][docs-url]
[![MIT License][license-shield]][license-url]

# [Port-Hamiltonian Dynamic Mode Decomposition][sisc-url]

We present a novel physics-informed system identification method to construct a passive linear time-invariant system. In more detail, for a given quadratic energy functional, measurements of the input, state, and output of a system in the time domain, we find a realization that approximates the data well while guaranteeing that the energy functional satisfies a dissipation inequality. To this end, we use the framework of port-Hamiltonian (pH) systems and modify the dynamic mode decomposition to be feasible for continuous-time pH systems. We propose an iterative numerical method to solve the corresponding least-squares minimization problem. We construct an effective initialization of the algorithm by studying the least-squares problem in a weighted norm, for which we present the analytical minimum-norm solution. The efficiency of the proposed method is demonstrated with several numerical examples.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#citing">Citing</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Citing
If you use this project for academic work, please consider citing our
[publication][sisc-url]:

    R. Morandin, J. Nicodemus, and B. Unger
    Port-Hamiltonian Dynamic Mode Decomposition
    SIAM Journal on Scientific Computing 2023 45:4, A1690-A1710

## Installation
A python environment is required with at least **Python 3.10**.

Install dependencies via `pip`:
   ```sh
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

There are two executable scripts `main.py`, `reduction.py` and a configuration file `config.py` in the `src` directory. 
- `main.py` execute the `pHDMD` algorithm for the current experiments, defined in `config.py`.
- `reduction.py` executes the `pHDMD` algorithm with model reduction step for the MIMO Mass-Spring-Damper experiment, defined in `config.py`. 
The algorithm is executed for different reduced orders, subsequently the H2 and Hinf errors are plotted of the reduced orders.

<!-- USAGE EXAMPLES -->
## Documentation

Documentation is available [online][docs-url]
or you can build it yourself from inside the `docs` directory
by executing:

    make html

This will generate HTML documentation in `docs/build/html`.
It is required to have the `sphinx` dependencies installed. This can be done by
   
    pip install -r requirements.txt
   
within the `docs` directory.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Jonas Nicodemus - jonas.nicodemus@simtech.uni-stuttgart.de

Benjamin Unger - benjamin.unger@simtech.uni-stuttgart.de\
Riccardo Morandin - morandin@math.tu-berlin.de 

Project Link: [https://github.com/Jonas-Nicodemus/phdmd][project-url]

[license-shield]: https://img.shields.io/github/license/Jonas-Nicodemus/phdmd.svg?style=for-the-badge
[license-url]: https://github.com/Jonas-Nicodemus/phdmd/blob/main/LICENSE
[doi-shield]: https://img.shields.io/badge/DOI-10.5281%20%2F%20zenodo.6497497-blue.svg?style=for-the-badge
[doi-url]: https://doi.org/10.5281/zenodo.6497497
[arxiv-shield]: https://img.shields.io/badge/arXiv-2204.13474-b31b1b.svg?style=for-the-badge
[arxiv-url]: https://arxiv.org/abs/2204.13474
[project-url]: https://github.com/Jonas-Nicodemus/phdmd
[docs-shield]: https://img.shields.io/badge/docs-online-blue.svg?style=for-the-badge
[docs-url]: https://jonas-nicodemus.github.io/phdmd/
[sisc-url]: https://doi.org/10.1137/22M149329X
