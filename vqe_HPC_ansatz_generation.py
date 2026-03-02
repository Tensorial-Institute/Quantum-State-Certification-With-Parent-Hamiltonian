# -*- coding: utf-8 -*-
"""
vqe_HPC_ansatz_generation.py

This is the code ran on Calcul Québec's HPC. Ran in order to find the optimal ansatz to run later on the QPU.

Progressive VQE for Dicke Hamiltonian.

Parallel multi-start L-BFGS-B:
- Parallelizes independent restarts per ansatz size using ProcessPoolExecutor.
- Each worker process creates its own EstimatorV2 once (initializer).
- RNG seed depends on --seed so SLURM array tasks differ.
- Robust output path joining + directory creation.

SELECTION POLICY:
- Wait for ALL parallel restarts to finish, then pick the best (lowest energy).

Ansatz used:
- checkerboard_ansatz_update (Rx + Rz + RZZ + Rx/Rz locals).
"""

from __future__ import annotations

import os
import csv
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0, help="random seed index")
    p.add_argument("--n", type=int, default=20, help="number of qubits")
    p.add_argument(
        "--outdir",
        type=str,
        default="/home/guanyih/links/projects/def-biamonte/guanyih/bfgs_best/bfgs_data_73",
        help="output directory",
    )
    p.add_argument("--numk", type=int, default=1, help="Dicke excitation number k")

    p.add_argument("--tries", type=int, default=3, help="number of full experiment repeats (best kept)")

    # Parallel multi-start knobs (defaults: use SLURM_CPUS_PER_TASK)
    p.add_argument("--restarts", type=int, default=0, help="multi-start restarts (0 => auto = workers)")
    p.add_argument("--workers", type=int, default=0, help="worker processes (0 => auto = SLURM_CPUS_PER_TASK)")
    return p.parse_args()


args = parse_args()
num_qubits = int(args.n)
numk = int(args.numk)
rng_seed = int(args.seed)

path = args.outdir
os.makedirs(path, exist_ok=True)

tries = int(args.tries)
t0 = time.time()


# ============================================================
# Ansatz: checkerboard_ansatz_update (Rx + Rz)
# ============================================================
def checkerboard_ansatz_update(n: int, num_two_qubit_gates: int):
    qc = QuantumCircuit(n)
    params = []
    used = 0
    layer = 0

    # Initial single-qubit layer: Rx + Rz
    for i in range(n):
        p_rx = Parameter(f"p_init_rx_{i}")
        p_rz = Parameter(f"p_init_rz_{i}")
        qc.rx(p_rx, i)
        qc.rz(p_rz, i)
        params += [p_rx, p_rz]

    # Checkerboard pattern using RZZ entangling gates
    while used < num_two_qubit_gates:
        if layer % 2 == 0:
            pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
        else:
            pairs = [(i, i + 1) for i in range(1, n - 1, 2)]

        for (a, b) in pairs:
            if used >= num_two_qubit_gates:
                break

            # RZZ entangling gate
            p_rzz = Parameter(f"p_rzz_{used}")
            qc.rzz(p_rzz, a, b)
            params.append(p_rzz)

            # Local single-qubit rotations: Rx + Rz on both qubits
            p_rx_a = Parameter(f"p_{used}_rx_a")
            p_rz_a = Parameter(f"p_{used}_rz_a")
            p_rx_b = Parameter(f"p_{used}_rx_b")
            p_rz_b = Parameter(f"p_{used}_rz_b")

            qc.rx(p_rx_a, a)
            qc.rz(p_rz_a, a)
            qc.rx(p_rx_b, b)
            qc.rz(p_rz_b, b)

            params += [p_rx_a, p_rz_a, p_rx_b, p_rz_b]
            used += 1

        layer += 1

    return qc, params

# ============================================================
# Hamiltonian (same as your original)
# ============================================================
def dicke_state_generator_hamiltonian(number_qubits: int, kth_dickie_state: int) -> SparsePauliOp:
    if number_qubits < 2:
        raise ValueError("Need at least two qubits.")

    paulis, coeffs = [], []

    first_term = (number_qubits / 2 - kth_dickie_state) ** 2 + (2 * number_qubits - 1) / 4
    second_term = -(number_qubits / 2 - kth_dickie_state)
    third_term = -(1.0 / (2.0 * number_qubits))
    fourth_term = 1 / 2 - (1 / (2 * number_qubits))

    for j in range(number_qubits):
        for k in range(j + 1, number_qubits):
            term = ["I"] * number_qubits
            term[j] = term[k] = "X"
            paulis.append("".join(term))
            coeffs.append(third_term)

            term = ["I"] * number_qubits
            term[j] = term[k] = "Y"
            paulis.append("".join(term))
            coeffs.append(third_term)

            term = ["I"] * number_qubits
            term[j] = term[k] = "Z"
            paulis.append("".join(term))
            coeffs.append(fourth_term)

    for j in range(number_qubits):
        term = ["I"] * number_qubits
        term[j] = "Z"
        paulis.append("".join(term))
        coeffs.append(second_term)

    paulis.append("I" * number_qubits)
    coeffs.append(first_term)

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


# ============================================================
# Output helpers
# ============================================================
def file_dump(line, name):
    outpath = os.path.join(path, name)
    with open(outpath, "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(line)


# ============================================================
# Multiprocessing-safe Estimator handling
# ============================================================
_EST = None


def _worker_init():
    global _EST
    _EST = EstimatorV2()


def expectation(theta_values, ansatz, params, H) -> float:
    global _EST
    if _EST is None:
        _EST = EstimatorV2()

    param_map = dict(zip(params, theta_values))
    bound = ansatz.assign_parameters(param_map, inplace=False)
    res = _EST.run([(bound, H)]).result()
    return float(res[0].data.evs)


def _run_one_lbfgsb(theta0, ansatz, params, H):
    def objective(theta):
        return expectation(theta, ansatz, params, H)
    bounds = [(0, np.pi / 2) if "rzz" in p.name else (-np.pi, np.pi) for p in params]
    return minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)



def run_vqe_parallel_best(ansatz, params, H, past_result, n_restarts: int, n_workers: int):
    """
    Run n_restarts independent L-BFGS-B optimizations in parallel,
    wait for ALL to finish, and return the one with the lowest energy.
    """
    inits = []
    n_past = len(past_result)
    new_params = params[n_past:]
    old_params = params[:n_past]
    for m in range(n_restarts):
        new_vals = np.array([
            np.random.uniform(0, np.pi / 2) if "rzz" in p.name
            else np.random.uniform(-np.pi, np.pi)
            for p in new_params
        ])
        if m < 8 or n_past == 0:
            warm = past_result
        else:
            # Small perturbation on warm-started parameters to explore other basins
            perturbation = np.array([
                np.random.uniform(-0.1, 0.1) for _ in old_params
            ])
            # Clip per-parameter: RZZ to [-pi/2, pi/2], others to [-pi, pi]
            warm = np.array([
                np.clip(v, 0, np.pi / 2) if "rzz" in p.name
                else np.clip(v, -np.pi, np.pi)
                for v, p in zip(past_result + perturbation, old_params)
            ])
        theta0 = np.concatenate([warm, new_vals])
        inits.append(theta0)

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        futs = [ex.submit(_run_one_lbfgsb, init, ansatz, params, H) for init in inits]

        best_result = None
        best_fun = float("inf")

        for fut in as_completed(futs):
            res = fut.result()

            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res

        return best_result


# ============================================================
# Experiment loop (progressively increasing entanglers)
# ============================================================
SAVE_THRESHOLDS = [
    (0.05,   "_firstbelow005"),
    (0.03,   "_firstbelow003"),
    (0.01,   "_firstbelow001"),
    (0.005,  "_firstbelow0005"),
    (0.001,  "_firstbelow0001"),
    (0.0005, "_firstbelow00005"),
    (0.0003, "_firstbelow00003"),
]
def _save_bound_qpy(ansatz, params, theta, suffix):
    bound = ansatz.assign_parameters(dict(zip(params, theta)))
    qpy_name = f"U_n_{num_qubits}_numk_{numk}_seed_{rng_seed}{suffix}.qpy"
    with open(os.path.join(path, qpy_name), "wb") as f:
        qpy.dump(bound, f)
    print(f"  Saved {qpy_name}")


def experiment(num_qubits: int, numk: int, rng_seed: int, try_idx: int = 0):
    H = dicke_state_generator_hamiltonian(num_qubits, numk)

    np.random.seed(rng_seed * 127 + try_idx * 136)

    exp_list = []
    parameters_history = []

    past_result = np.array([], dtype=float)

    # keep best snapshot across all progressive steps
    best_fun = float("inf")
    best_ansatz = None
    best_params = None
    best_theta = None
    saved_thresholds = set()

    num_2q_gates = numk * num_qubits * 10

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    n_workers = args.workers if args.workers > 0 else slurm_cpus
    n_restarts = args.restarts if args.restarts > 0 else max(1, n_workers)

    stop_point = 1e-4

    for num_2q_ansatz in range(0, num_2q_gates):
        ansatz, params = checkerboard_ansatz_update(num_qubits, num_2q_ansatz)

        result = run_vqe_parallel_best(ansatz=ansatz, params=params, H=H, past_result=past_result, n_restarts=n_restarts, n_workers=n_workers)

        exp_list.append(float(result.fun))
        past_result = result.x.copy()
        parameters_history.append(result.x.copy())

        # Track best snapshot encountered (for saving)
        if float(result.fun) < best_fun:
            best_fun = float(result.fun)
            best_ansatz = ansatz
            best_params = params
            best_theta = result.x.copy()

        # Save .qpy at each threshold the first time it is crossed (per try)
        for thresh_val, thresh_label in SAVE_THRESHOLDS:
            if thresh_val not in saved_thresholds and best_fun < thresh_val:
                saved_thresholds.add(thresh_val)
                _save_bound_qpy(best_ansatz, best_params, best_theta, f"{thresh_label}_try{try_idx}")

        if abs(result.fun) < stop_point:
            break

        print(result.fun)

    return exp_list, parameters_history, best_ansatz, best_params, best_theta, best_fun


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    best_exp = float("inf")
    best_circuit = None
    best_params_order = None
    best_theta = None
    best_exp_list = None
    best_parameters_history = None

    for try_idx in range(tries):
        exp_list, parameters_history, cand_circuit, cand_params, cand_theta, cand_fun = experiment(
            num_qubits=num_qubits, numk=numk, rng_seed=rng_seed, try_idx=try_idx
        )

        if cand_fun < best_exp:
            best_exp = cand_fun
            best_circuit = cand_circuit
            best_params_order = cand_params
            best_theta = cand_theta
            best_exp_list = exp_list
            best_parameters_history = parameters_history

        print("seed:", rng_seed, "ev:", cand_fun)
        # Write outputs
        file_dump(best_exp_list, f"In_exp_list_n_{num_qubits}_excitations_{numk}_seed_{rng_seed}_try{try_idx}.csv")

        # Bind optimal parameters so the .qpy is fully self-contained
        bound_circuit = best_circuit.assign_parameters(dict(zip(best_params_order, best_theta)))
        with open(os.path.join(path, f"In_U_n_{num_qubits}_numk_{numk}_seed_{rng_seed}_finalexp0e4_try{try_idx}.qpy"), "wb") as f:
            qpy.dump(bound_circuit, f)


    # Write outputs
    file_dump(best_exp_list, f"exp_list_n_{num_qubits}_excitations_{numk}_seed_{rng_seed}.csv")

    for pars in best_parameters_history:
        file_dump(list(pars), f"parameters_history_n_{num_qubits}_excitations_{numk}_seed_{rng_seed}.csv")

    # Bind optimal parameters so the .qpy is fully self-contained
    bound_circuit = best_circuit.assign_parameters(dict(zip(best_params_order, best_theta)))
    with open(os.path.join(path, f"U_n_{num_qubits}_numk_{numk}_seed_{rng_seed}_finalexp0e4.qpy"), "wb") as f:
        qpy.dump(bound_circuit, f)

    print(f"Done. Best energy={best_exp:.6f}. Total time={(time.time() - t0):.1f}s")
