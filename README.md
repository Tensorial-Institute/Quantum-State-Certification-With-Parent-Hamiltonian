# Quantum State Certification via Effective Parent Hamiltonians from Local Measurement Data
This repository is provided to facilitate reproducibility of the numerical and experimental results from our paper, "Quantum State Certification via Effective Parent Hamiltonians from Local Measurement Data."

It contains three scripts.

- vqe_HPC_ansatz_generation.py is a script to find an ansatz to prepare the Dicke state as best as possible on a simulator. This was run on Calcul Quebec's HPC
- QPU_dicke_n7k3.ipynb is the script we used to quantify the fidelity of the Dicke state that we prepared, as seen in Appendix D
- QPU_w_state.ipynb is the script we used to quantify the fidelity of the W state as seen in Figure 2


## Citation

If you use this code in academic work, please cite the associated manuscript:

**Quantum State Certification via Effective Parent Hamiltonians from Local Measurement Data**  
Guy-Philippe Nadon, Guanyi Heng, Pacôme Gasnier, Antoine Lemelin, Camille Coti,  
Zeljko Zilic, Mikko Möttönen, Ville Kotovirta, Toni Annala, Ernesto Campos, Jacob Biamonte  

https://doi.org/10.48550/arXiv.2603.04499

### BibTeX

```bibtex
@article{nadon2026quantum,
  title   = {Quantum State Certification via Effective Parent Hamiltonians from Local Measurement Data},
  author  = {Nadon, Guy-Philippe and Heng, Guanyi and Gasnier, Pac{\^o}me and Lemelin, Antoine and Coti, Camille and Zilic, Zeljko and M{\"o}tt{\"o}nen, Mikko and Kotovirta, Ville and Annala, Toni and Campos, Ernesto and Biamonte, Jacob},
  year    = {2026},
  journal = {arXiv preprint arXiv:2603.04499},
  doi     = {10.48550/arXiv.2603.04499}
}
```
