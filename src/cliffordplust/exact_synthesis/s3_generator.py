import json
import os

from cliffordplust.Rings import Domega
from cliffordplust.exact_synthesis import (
    apply_sequence,
    convert_to_tuple,
)


def generate_sequences(max_consecutive_t: int = 7) -> list:
    """Generate all valid sequences of T and H gates with a maximum number of consecutive T gates.

    Args:
        max_consecutive_t (int): Maximum number of consecutive T gates

    Returns:
        list: List of valid sequences of T and H gates in string format
    """
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


def generate_s3(max_consecutive_t: int = 7) -> None:
    """Generate the S3 table for the first column of Clifford+ gate set and write it to a JSON file.
    Args:
        max_consecutive_t (int): Maximum number of consecutive T gates
    """
    s3_sequences = generate_sequences(max_consecutive_t)
    s3_dict = {seq: convert_to_tuple(apply_sequence(seq)) for seq in s3_sequences}

    def serialize_dict(d):
        return {
            k: [[list(inner) for inner in outer] for outer in v] for k, v in d.items()
        }

    with open(os.path.join(os.path.dirname(__file__), "s3_table.json"), "w") as f:
        json.dump(serialize_dict(s3_dict), f, indent=4)
