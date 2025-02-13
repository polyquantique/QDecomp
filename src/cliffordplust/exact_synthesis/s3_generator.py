import numpy as np

from cliffordplust.rings import Domega
from cliffordplust.exact_synthesis import (
    exact_synthesis_alg,
    apply_sequence,
    random_sequence,
    H,
    T,
    T_inv,
    I,
    omega,
    W,
)

import json
import os


def generate_sequences(max_consecutive_t=7):
    valid_sequences = []
    for n_3 in range(0, max_consecutive_t + 1):
        if n_3 == 0:
            valid_sequences.append("H")
            valid_sequences.append("")
        else:
            for n_2 in range(0, max_consecutive_t + 1):
                if n_2 == 0:
                    valid_sequences.append("T" * n_3 + "H")
                    valid_sequences.append("H" + "T" * n_3 + "H")
                else:
                    for n_1 in range(0, max_consecutive_t + 1):
                        if n_1 == 0:
                            valid_sequences.append("T" * n_3 + "H" + "T" * n_2 + "H")
                            valid_sequences.append(
                                "H" + "T" * n_3 + "H" + "T" * n_2 + "H"
                            )
                        else:
                            valid_sequences.append(
                                "T" * n_3 + "H" + "T" * n_2 + "H" + "T" * n_1 + "H"
                            )
    return valid_sequences


def generate_s3(max_consecutive_t=7):
    s3_sequences = generate_sequences(max_consecutive_t)
    print(len(s3_sequences))
    s3_dict = {seq: convert_to_tuple(apply_sequence(seq)) for seq in s3_sequences}

    def serialize_dict(d):
        return {
            k: [[list(inner) for inner in outer] for outer in v] for k, v in d.items()
        }

    with open(os.path.join(os.path.dirname(__file__), "s3_table.json"), "w") as f:
        json.dump(serialize_dict(s3_dict), f, indent=4)


def convert_to_tuple(array):
    return tuple(
        tuple((Domega[i].num, Domega[i].denom) for i in range(4))
        for Domega in array[:, 0]
    )


def evaluate_omega_exponent(z_1, z_2):
    z_1_angle = np.angle(z_1.real() + 1j * z_1.imag())
    z_2_angle = np.angle(z_2.real() + 1j * z_2.imag())
    angle = z_1_angle - z_2_angle
    omega_exponent = int(np.round(angle / (np.pi / 4))) % 8
    return omega_exponent


def lookup_sequence(U):
    with open(os.path.join(os.path.dirname(__file__), "s3_table.json"), "r") as f:
        s3_dict = json.load(f)
        s3_dict = {
            k: tuple(tuple(tuple(inner) for inner in outer) for outer in v)
            for k, v in s3_dict.items()
        }
    for i in range(8):
        U_t = np.multiply(omega**i, U)
        for key, value in s3_dict.items():
            if convert_to_tuple(U_t) == value:
                print(f"Sequence : {key}")
                U_w = apply_sequence(key + "W" * (8 - i))
                k = evaluate_omega_exponent(U[1, 1], U[0, 0].complex_conjugate())
                k_pp = evaluate_omega_exponent(U_w[1, 1], U_w[0, 0].complex_conjugate())
                k_prime = (k - k_pp) % 8
                key += "T" * k_prime
                key += "W" * (8 - i)
                return key

    # print(f"Sequence not found : {convert_to_tuple(U)}")
    with open("S3missing.txt", "r") as f:
        lines = f.readlines()
        if f"{np.array(U, dtype=complex)}\n" not in lines:
            with open("S3missing.txt", "a") as f:
                f.write(f"{np.array(U, dtype=complex)}\n")
    print(np.array(U, dtype=complex))
    return None
