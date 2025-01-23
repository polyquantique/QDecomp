import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Domega import Domega, H, I, T
from exact_synthesis import (
    apply_sequence,
    is_unitary,
    exact_synthesis_alg,
    random_sequence,
)

from itertools import product
import json


def generate_sequences(max_h=3, max_consecutive_t=4, max_length=19):
    valid_sequences = []

    def is_valid(sequence: tuple):
        h_count = sequence.count("H")
        sequence_str = "".join(sequence)
        if h_count > max_h:
            return False
        if "HH" in sequence_str:
            return False
        for t_group in "".join(sequence).split("H"):
            if len(t_group) > max_consecutive_t:
                return False
        return True

    for length in range(1, max_length + 1):
        for sequence in product("HT", repeat=length):
            if is_valid(sequence):
                valid_sequences.append("".join(sequence))

    return valid_sequences


def is_gate_equal(gate1, gate2):
    for elements in zip(gate1, gate2):
        if elements[0] != elements[1]:
            return False
    return True


if __name__ == "__main__":
    init_seq = random_sequence(10)
    print(f"Initial sequence : {init_seq}")
    U = apply_sequence(init_seq)
    print(f"Initial gate : \n{U}")
    sequence, U_f = exact_synthesis_alg(U)
    assert U.all() == apply_sequence(sequence, U_f).all()
    remaining_seq = init_seq.replace(sequence, "", 1)
    print(f"Remaining sequence : {remaining_seq}")
    sequences = generate_sequences()

    with open("exact_synthesis/sequences.txt", "w") as file:
        for seq in sequences:
            file.write(f"{seq}\n")

    s3_dict = {seq: apply_sequence(seq) for seq in sequences}

    for key, value in s3_dict.items():
        if np.array_equal(value, U_f):
            u_f_seq = key
            print(f"Sequence : {key}")
            break

    print(f"Sequence : {sequence + u_f_seq}")
    final_sequence = sequence + u_f_seq
    print(f"Matrix with s<3 : \n{U_f}")
    print(f"Final matrix : \n{apply_sequence(final_sequence)}")
