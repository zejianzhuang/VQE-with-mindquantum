# Variational Quantum Eigensolver(VQE) example by MindQuantum
The variational quantum eigensolver is a hybridclassical-quantum algorithm that variationally determines the ground state energy of a Hamiltonian. In this tutorial, I am going to implement VQE for finding the ground state energy of ![eq1](https://render.githubusercontent.com/render/math?math=\color{green}H_2) by mindquantum.

MindQuantum is general quantum computing framework designed by [Mindspore](https://www.mindspore.cn/en) and [Hiq](https://hiq.huaweicloud.com/). MindQuantum can efficiently solve problem such as quantum machine learning, quantum chemistry simulation and so on.

<img src="mindquantum.png" alt="MindQuantum Architecture" width="400"/>

The problem we want to tackle is based on Ref. [[1]](#1). In this paper, O'Malley reported that their first experiment demonstrated the VQE proposed in 2014 [[2]](#2) on a real quantum computer. The schematic I am going to reproduce is the following, and I will implement the "software".




## Reference
<a id="1">[1]</a> 
[O’Malley, Peter JJ, et al. "Scalable quantum simulation of molecular energies." Physical Review X 6.3 (2016): 031007.](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.031007)

<a id="2">[2]</a>
[Peruzzo, Alberto, et al. "A variational eigenvalue solver on a photonic quantum processor." Nature communications 5.1 (2014): 1-7.](https://www.nature.com/articles/ncomms5213)
