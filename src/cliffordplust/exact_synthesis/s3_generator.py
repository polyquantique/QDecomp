import numpy as np

from cliffordplust.Rings import Domega
from cliffordplust.exact_synthesis import (
    exact_synthesis_alg,
    apply_sequence,
    random_sequence,
    H,
    T,
    T_inv,
    I,
)

from itertools import product
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

    s3_sequences = valid_sequences
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


def lookup_sequence(U):
    with open(os.path.join(os.path.dirname(__file__), "s3_table.json"), "r") as f:
        s3_dict = json.load(f)
        s3_dict = {
            k: tuple(tuple(tuple(inner) for inner in outer) for outer in v)
            for k, v in s3_dict.items()
        }
    for key, value in s3_dict.items():
        if convert_to_tuple(U) == value:
            print(f"Sequence : {key}")
            return key
    # print(f"Sequence not found : {convert_to_tuple(U)}")
    with open("S3missing.txt", "r") as f:
        lines = f.readlines()
        # if f"{convert_to_tuple(U)}\n" not in lines:
    with open("S3missing.txt", "a") as f:
        f.write(f"{convert_to_tuple(U)}\n")
    return None


if __name__ == "__main__":
    # generate_sequences()
    sequence_not_found = 0

    for _ in range(10):
        for len in range(10, 100):
            # Initialise matrix to decompose
            init_seq = random_sequence(40)
            # print(f"Initial sequence : {init_seq}")
            U = apply_sequence(init_seq)
            # print(f"Initial gate : \n{U}")
            # Find the reduced U3 gate and its associated sequence
            sequence_u3, U_3 = exact_synthesis_alg(U)
            assert U.all() == apply_sequence(sequence_u3, U_3).all()
            norm_z = U_3[0, 0] * U_3[0, 0].complex_conjugate()
            print(norm_z.sde())
            # print(f"U_3 : {U_3}")
            # generate_sequences()
            if lookup_sequence(U_3) == None:
                print("Sequence not found")
                sequence_not_found = sequence_not_found + 1
    print(f"Sequence not found : {sequence_not_found}")
    with open("S3missing.txt", "r") as f:
        missing_sequences = f.readlines()
