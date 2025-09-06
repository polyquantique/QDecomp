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

r"""
This module implements the Tonelli-Shanks algorithm to find square roots modulo a prime number.

The problem is stated as follows:
Given a prime number :math:`p` and an integer :math:`a`, find an integer :math:`r` such that
:math:`r^2 \equiv a\ (\text{mod p})`.
The complexity of this algorithm is :math:`O(\log^2 p)`. Its efficiency is due to the fact that it
exploits the decomposition of :math:`p-1` into the form :math:`p-1 = q \cdot 2^s`, where
:math:`q` is odd and :math:`s` is a non-negative integer.

More information on this algorithm and implementation can be found in :cite:`diophantine_tonelli_shanks`.
"""


def legendre_symbol(a: int, p: int) -> int:
    r"""
    Compute the Legendre symbol :math:`(a/p)` using Euler's criterion.

    The Legendre symbol is defined as follows:

    | - :math:`(a/p)` = 0 if :math:`a` is divisible by :math:`p`
    | - :math:`(a/p)` = 1 if :math:`a` is a quadratic residue modulo :math:`p` (i.e., there exists an integer :math:`x` such that :math:`x^2 \equiv a\ (\text{mod p})`)
    | - :math:`(a/p)` = -1 if :math:`a` is not a quadratic residue modulo :math:`p`

    Args:
        a (int): The integer for which the Legendre symbol is computed.
        p (int): A prime number.

    Returns:
        int: The Legendre symbol :math:`(a/p)`.
    """
    return pow(a, (p - 1) // 2, p)


def tonelli_shanks_algo(a, p):
    r"""
    Tonelli-Shanks algorithm to find the smallest square root of :math:`a` modulo :math:`p`.

    The problem solved by this function is stated as follows:
    Given a prime number :math:`p` and an integer :math:`a`, find an integer :math:`r` such that
    :math:`r^2 \equiv a\ (\text{mod p})`. If no solution exists, a `ValueError` is raised.

    Args:
        a (int): The integer for which the square root is computed.
        p (int): A prime number.

    Returns:
        int: The smallest non-negative integer :math:`r` such that :math:`r^2 \equiv a\ (\text{mod p})`.

    Raises:
        ValueError: If :math:`a` is not a quadratic residue modulo :math:`p`, i.e. no solution exists.
    """
    if legendre_symbol(a, p) != 1:
        raise ValueError(f"a = {a} is not a quadratic residue modulo p = {p}.")

    if p % 4 == 3:
        r = pow(a, (p + 1) // 4, p)
        return min(r, p - r)  # Return the smallest root

    # Decomposition of p-1 = q * 2^s
    s, q = 0, p - 1
    while q % 2 == 0:
        s += 1
        q //= 2

    # Find the quadratic non-residue z
    z = 2
    while legendre_symbol(z, p) != p - 1:
        z += 1

    c = pow(z, q, p)
    r = pow(a, (q + 1) // 2, p)
    t = pow(a, q, p)
    m = s

    while t != 1:
        i = 1
        temp = pow(t, 2, p)
        while temp != 1:
            temp = pow(temp, 2, p)
            i += 1

        b = pow(c, 2 ** (m - i - 1), p)
        r = (r * b) % p
        t = (t * pow(b, 2, p)) % p
        c = pow(b, 2, p)
        m = i

    return min(r, p - r)  # Return the smallest root
