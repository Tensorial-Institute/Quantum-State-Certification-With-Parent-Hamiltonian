# Quantum State Certification via Effective Parent Hamiltonians from Local Measurement Data
This repository is provided to facilitate reproducibility of the numerical and hardware results from our paper, "Quantum State Certification via Effective Parent Hamiltonians from Local Measurement Data."

All hardware runs were done on a Heron r2 with the ibm_quebec backend.

It contains three scripts.

- vqe_HPC_ansatz_generation.py is a script to find an ansatz to prepare the Dicke state as best as possible on a simulator. This was run on Calcul Quebec's HPC
- QPU_dicke_n7k3.ipynb is the script we used to quantify the fidelity of the Dicke state that we prepared, as seen in Appendix D
- QPU_w_state.ipynb is the script we used to quantify the fidelity of the W state as seen in Figure 2

## Reproducing the Results

| Result | File | Data |
|---|---|---|
| W-state hardware certification, Figure 2 | `QPU_w_state.ipynb` | `data/w_state/results_and_calibrations/` |
| Dicke-state hardware certification, Appendix D | `QPU_dicke_n7k3.ipynb` | `data/k2_k3_Dicke/results_and_calibrations/` |
| Dicke ansatz search | `vqe_HPC_ansatz_generation.py` | `data/k2_k3_Dicke/precomputed_circuits/` |

The circuits in `data/k2_k3_Dicke/precomputed_circuits/` are the optimized Dicke-state preparation circuits. After optimization, these are the Dicke-state circuits that were ultimately sent to the quantum computer.

## Repository Structure

```text
.
|-- QPU_w_state.ipynb
|-- QPU_dicke_n7k3.ipynb
|-- vqe_HPC_ansatz_generation.py
|-- data/
|   |-- w_state/
|   |   `-- results_and_calibrations/
|   `-- k2_k3_Dicke/
|       |-- precomputed_circuits/
|       `-- results_and_calibrations/
|-- requirements.txt
`-- .env.example
```

## Data Notes

The checked-in result folders contain IBM hardware job outputs, backend calibration snapshots, target instruction summaries, and compact JSON files with measured parent-Hamiltonian expectation values.

The checked-in data is sufficient to reproduce the plotted analyses in the notebooks. Submitting new IBM hardware jobs requires separate IBM Runtime credentials and may produce different numerical values because backend calibrations change over time.

For the W-state analysis, the fidelity lower-bound estimate is computed as

```text
F >= 1 - <H>
```

When `<H> > 1`, this lower-bound estimate is negative. These values are non-informative as fidelity lower bounds and are omitted from the fidelity-axis plot, while the corresponding expectation-value points remain part of the analysis.

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
