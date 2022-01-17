# Variational Quantum Eigensolver(VQE) example by MindQuantum
The variational quantum eigensolver is a hybridclassical-quantum algorithm that variationally determines the ground state energy of a Hamiltonian. In this tutorial, I am going to implement VQE for finding the ground state energy of ![eq1](https://render.githubusercontent.com/render/math?math=\color{green}H_2) by mindquantum.

MindQuantum is general quantum computing framework designed by [Mindspore](https://www.mindspore.cn/en) and [Hiq](https://hiq.huaweicloud.com/). MindQuantum can efficiently solve problem such as quantum machine learning, quantum chemistry simulation and so on.

<p align="center">
  <img width="400" src="mindquantum.png" alt="MindQuantum Architecture">
</p>

The problem we want to tackle is based on Ref. [[1]](#1). In this paper, O'Malley reported that their first experiment demonstrated the VQE proposed in 2014 [[2]](#2) on a real quantum computer. The schematic I am going to reproduce is the following, and I will implement the "software".

<p align="center">
  <img width="500" src="images/vqe_diagram.svg" alt="vqe">
</p>
<p align="center"> The figure is taken from Ref. [1] </p>

The molecular hydrogen Hamiltonian is 
<p align="center"> <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bgreen%7DH%3Dg_%7B0%7DI%20%2B%20g_%7B1%7DZ_%7B0%7D%2Bg_%7B2%7DZ_%7B0%7DZ_%7B1%7D%2Bg_%7B3%7DZ_%7B0%7DZ_%7B1%7D%2Bg_%7B4%7DY_%7B0%7DY_%7B1%7D%2Bg_%7B5%7DX_%7B0%7DX_%7B1%7D"> </p>
where {X_i, Z_i, Y_i} denote Pauli matrices acting on the *i*th qubit and the real scalars {g_i} are efficiently computable functions of hydrogen-hydrogen bond length R. Let’s build a Hamiltonian for the H2 molecule with different bond length. Firstly, we import mindquantum package.
```python
  import mindquantum as mq
```









## Reference
<a id="1">[1]</a> 
[O’Malley, Peter JJ, et al. "Scalable quantum simulation of molecular energies." Physical Review X 6.3 (2016): 031007.](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.031007)

<a id="2">[2]</a>
[Peruzzo, Alberto, et al. "A variational eigenvalue solver on a photonic quantum processor." Nature communications 5.1 (2014): 1-7.](https://www.nature.com/articles/ncomms5213)
