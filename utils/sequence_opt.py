# Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
This module simplifies sequences to its most optimal
"""


def optimize_sequence(sequence: str) -> str:
    """
    Optimize a sequence of gates by removing redundant gates and combining consecutive gates.

    Args:
        sequence (str): The input sequence of gates as a string.

    Returns:
        str: The optimized sequence of gates.
    """
    # Replace HH by identity
    optimized_sequence = sequence.replace("HH", "")

    # Replace TTTT by Z
    optimized_sequence = optimized_sequence.replace("TTTT", "Z")

    # Replace TT by S
    optimized_sequence = optimized_sequence.replace("TT", "S")

    return optimized_sequence
