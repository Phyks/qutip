# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
import qutip
from numpy.testing import assert_, run_module_suite
from qutip import floquet


class TestFloquet:
    """
    A test class for the QuTiP functions for Floquet formalism.
    """
    @staticmethod
    def _det(v1, v2):
        """
        Simple determinant of two vectors
        """
        return v1[0]*v2[1] - v1[1]*v2[0]

    def _get_simple_driven_qubit_parameters(self):
        """
        Parameters used in the simple driven qubit model.
        """
        omega_0 = 1.0 * 2 * np.pi
        Omega = 0.1 * 2 * np.pi
        omega_d = 1.5 * 2 * np.pi
        return omega_0, Omega, omega_d

    def _get_simple_driven_qubit_hamiltonian(self):
        """
        Most examples will use this very simple model of a driven qubit. It is
        interesting to note that quasi-energies can be computed analytically in
        an easy way for such a model, thus making testing easier.
        """
        # Parameters
        omega_0, Omega, omega_d = self._get_simple_driven_qubit_parameters()
        T = 2 * np.pi / omega_d

        # Hamiltonian
        H0 = omega_0 / 2.0 * qutip.sigmaz()  # Time-independent part
        H1 = Omega * qutip.sigmax()  # Time-dependent part
        H2 = Omega * qutip.sigmay()  # Time-dependent part

        # Time-dependent hamiltonian specification
        H = [H0, [H1, 'cos(w_d * t)'], [H2, 'sin(w_d * t)']]
        args = {'w_d': omega_d}

        return H, T, args

    def testFloquetModes(self):
        """
        Floquet: test computation of Floquet modes at time t=0
        """
        H, T, args = self._get_simple_driven_qubit_hamiltonian()
        # Compute Floquet modes at time t=0
        f_modes_0, f_energies = floquet.floquet_modes(H, T, args)

        # Compute reference quasienergies from analytical expressions
        omega_0, Omega, omega_d = self._get_simple_driven_qubit_parameters()
        f_energies_ref = np.array([
            np.sqrt(Omega**2 + (omega_d - omega_0)**2 / 4) - omega_d / 2,
            -np.sqrt(Omega**2 + (omega_d - omega_0)**2 / 4) + omega_d / 2
        ])
        f_modes_0_ref = [
            np.array([
                np.sqrt(
                    Omega**2 + (omega_d - omega_0)**2 / 4
                ) + (omega_0 - omega_d) / 2,
                Omega
            ]),
            np.array([
                -np.sqrt(
                    Omega**2 + (omega_d - omega_0)**2 / 4
                ) + (omega_0 - omega_d) / 2,
                Omega
            ]),
        ]
        for f_mode_0, f_mode_0_ref in zip(f_modes_0, f_modes_0_ref):
            assert_(np.abs(self._det(f_mode_0, f_mode_0_ref)) < 1e-4)
        assert_(np.allclose(np.sort(f_energies_ref), np.sort(f_energies)))

        # Check sorting
        indices = np.argsort(f_energies)
        f_modes_0_sorted, f_energies_sorted = floquet.floquet_modes(
            H, T, args, sort=True
        )
        for f_mode_0, f_mode_0_ref in zip(
            f_modes_0_sorted, np.array(f_modes_0)[indices]
        ):
            assert(np.isclose((f_mode_0 - f_mode_0_ref).norm(), 0))
        assert_(np.allclose(f_energies_sorted, np.array(f_energies)[indices]))

        # Check computation with a provided propagator
        U = qutip.propagator(H, T, c_op_list=[], args=args)
        f_modes_0_U, f_energies_U = floquet.floquet_modes(
            H, T, args, U=U, sort=True
        )
        for f_mode_0, f_mode_0_ref in zip(
            f_modes_0_U, f_modes_0_sorted
        ):
            assert(np.isclose((f_mode_0 - f_mode_0_ref).norm(), 0))
        assert_(np.allclose(f_energies_sorted, f_energies_U))

    def testFloquetModesT(self):
        """
        Floquet: test evolution of Floquet modes at a later time t.
        """
        omega_0, Omega, omega_d = self._get_simple_driven_qubit_parameters()
        H, T, args = self._get_simple_driven_qubit_hamiltonian()
        f_modes_0, f_energies = floquet.floquet_modes(H, T, args)

        # Compute Floquet modes at time t
        t = T / 100
        f_modes_t = floquet.floquet_modes_t(
            f_modes_0, f_energies, t, H, T, args
        )
        f_modes_t_ref = [
            np.array([
                np.exp(-1.0j * omega_d * t / 2) * (
                    np.sqrt(
                        Omega**2 + (omega_d - omega_0)**2 / 4
                    ) + (omega_0 - omega_d) / 2
                ),
                np.exp(1.0j * omega_d * t / 2) * Omega
            ]),
            np.array([
                np.exp(-1.0j * omega_d * t / 2) * (
                    -1.0 * np.sqrt(
                        Omega**2 + (omega_d - omega_0)**2 / 4
                    ) + (omega_0 - omega_d) / 2
                ),
                np.exp(1.0j * omega_d * t / 2) * Omega
            ]),
        ]
        for f_mode_t, f_mode_t_ref in zip(f_modes_t, f_modes_t_ref):
            assert_(np.abs(self._det(f_mode_t, f_mode_t_ref) < 1e-4))

    def testFloquetModesTable(self):
        """
        Floquet: test computation of table of values of Floquet modes at
        different times.
        """
        H, T, args = self._get_simple_driven_qubit_hamiltonian()
        f_modes_0, f_energies = floquet.floquet_modes(H, T, args)

        # Compute Floquet modes at time t
        tlist = [0, T / 100, 2 * T / 100, 5 * T / 100]
        f_modes_table = floquet.floquet_modes_table(
            f_modes_0, f_energies, tlist, H, T, args
        )
        for t, f_modes_t in zip(tlist, f_modes_table):
            f_modes_t_ref = floquet.floquet_modes_t(
                f_modes_0, f_energies, t, H, T, args
            )
            assert_(
                all(
                    (f_mode_t_ref - f_mode_t).norm() < 1e-4
                    for (f_mode_t, f_mode_t_ref) in zip(f_modes_t,
                                                        f_modes_t_ref)
                )
            )

    def testFloquetModesTLookup(self):
        """
        Floquet: test lookup of Floquet modes value at later time from a
        precomputed table.
        """
        H, T, args = self._get_simple_driven_qubit_hamiltonian()
        f_modes_0, f_energies = floquet.floquet_modes(H, T, args)

        # Compute Floquet modes at time t from lookup table
        tlist = np.linspace(0, T, 100)
        t = 0.5 * (tlist[20] + tlist[21])
        f_modes_table = floquet.floquet_modes_table(
            f_modes_0, f_energies, tlist, H, T, args
        )
        f_modes_t = floquet.floquet_modes_t_lookup(
            f_modes_table, t, T
        )

        # Reference
        f_modes_t_ref = floquet.floquet_modes_t(
            f_modes_0, f_energies, t, H, T, args
        )
        for f_mode_t, f_mode_t_ref in zip(f_modes_t, f_modes_t_ref):
            assert_(
                (f_mode_t_ref - f_mode_t).norm() < 1e-2
            )

    def testFloquetStatesT(self):
        """
        Floquet: test computation of the Floquet states at any time t.
        """
        H, T, args = self._get_simple_driven_qubit_hamiltonian()
        f_modes_0, f_energies = floquet.floquet_modes(H, T, args)
        t = T / 100

        f_modes_t = floquet.floquet_modes_t(
            f_modes_0, f_energies, t, H, T, args
        )
        f_states_t = floquet.floquet_states_t(f_modes_t, f_energies, t)
        f_states_t_ref = [
            np.exp(-1.0j * f_energy * t) * f_mode_t
            for (f_energy, f_mode_t) in zip(f_energies, f_modes_t)
        ]
        for f_state_t, f_state_t_ref in zip(f_states_t, f_states_t_ref):
            assert_(
                (f_state_t_ref - f_state_t).norm() < 1e-4
            )

    def testFloquetStateDecomposition(self):
        """
        Floquet: test the decomposition of any state at a given time on the
        Floquet states at the same time.
        """
        H, T, args = self._get_simple_driven_qubit_hamiltonian()
        f_modes_0, f_energies = floquet.floquet_modes(H, T, args)

        for i, f_mode in enumerate(f_modes_0):
            f_coeff = floquet.floquet_state_decomposition(
                f_modes_0, f_energies, f_mode
            )
            f_coeff_ref = np.zeros(len(f_modes_0))
            f_coeff_ref[i] = 1
            assert_(np.allclose(f_coeff_ref, f_coeff))

    def testFloquetWavefunctionT(self):
        """
        Floquet: test computation of the full wavefunction given an initial
        decomposition on Floquet modes and a time t.
        """
        H, T, args = self._get_simple_driven_qubit_hamiltonian()
        t = T / 10
        psi0 = qutip.rand_ket(2)

        f_modes_0, f_energies = floquet.floquet_modes(H, T, args)
        f_coeff = floquet.floquet_state_decomposition(
            f_modes_0, f_energies, psi0
        )
        f_modes_t = floquet.floquet_modes_t(
            f_modes_0, f_energies, t, H, T, args
        )
        sol = floquet.floquet_wavefunction_t(f_modes_t, f_energies, f_coeff, t)

        # compare with Schrodinger evolution
        sol_ref = qutip.mesolve(H, psi0, [0, t], [], [], args)

        assert_((sol - sol_ref.states[-1]).norm() < 1e-4)

    def testFloquetUnitary(self):
        """
        Floquet: test the full unitary evolution of time-dependent two-level
        system.
        """
        H, T, args = self._get_simple_driven_qubit_hamiltonian()
        tlist = np.linspace(0.0, 2 * T, 101)
        psi0 = qutip.rand_ket(2)

        e_ops = []
        # solve schrodinger equation with floquet solver
        sol = floquet.fsesolve(H, psi0, tlist, e_ops, T, args)
        # compare with results from standard schrodinger equation
        sol_ref = qutip.mesolve(H, psi0, tlist, c_ops=[], e_ops=e_ops,
                                args=args)
        assert_((sol.states[-1] - sol_ref.states[-1]).norm() < 1e-4)

        e_ops = [qutip.num(2)]
        # solve schrodinger equation with floquet solver
        sol = floquet.fsesolve(H, psi0, tlist, e_ops, T, args)
        # compare with results from standard schrodinger equation
        sol_ref = qutip.mesolve(H, psi0, tlist, c_ops=[], e_ops=e_ops,
                                args=args)
        assert_(np.linalg.norm(sol.expect[0] - sol_ref.expect[0]) < 1e-4)

    def testFloquetUnitaryMultiLevel(self):
        """
        Floquet: test unitary evolution of time-dependent multi-level system

        Cavity with a detuned drive.
        """
        # TODO: Not working
        wc = 1.0 * 2 * np.pi  # Cavity frequency
        wp = 1.5 * 2 * np.pi  # Drive frequency
        T = 2 * np.pi / wp
        N = 20  # Truncature
        nbar = 2
        epsilon_p = np.sqrt(4 * nbar) * (wp - wc)
        tlist = np.linspace(0.0, 2 * T, 101)

        a = qutip.destroy(N)
        H = [
            wc * a.dag() * a,
            [1.0j * epsilon_p * (a.dag() - a), 'cos(wp * t)']
        ]
        args = {
            'wp': wp
        }
        e_ops = [a.dag() * a]
        psi0 = qutip.basis(N, 0)

        # solve schrodinger equation with floquet solver
        sol = floquet.fsesolve(H, psi0, tlist, e_ops, T, args)

        # compare with results from standard schrodinger equation
        sol_ref = qutip.mesolve(H, psi0, tlist, [], e_ops, args)

        assert_(np.linalg.norm(sol.expect[0] - sol_ref.expect[0]) < 1e-4)

    def testFloquetSteadystate(self):
        """
        Floquet: test steadystate solution of time-dependent multi-level
        system
        """
        # TODO
        wc = 1.0 * 2 * np.pi  # Cavity frequency
        wp = 1.5 * 2 * np.pi  # Drive frequency
        T = 2 * np.pi / wp
        N = 20  # Truncature
        nbar = 2
        epsilon_p = np.sqrt(4 * nbar) * (wp - wc)  # Drive amplitude

        # noise power spectrum
        gamma1 = 0.05
        noise_spectrum = lambda omega: 0.5 * gamma1 * omega / (2 * np.pi)

        # Define time independent hamiltonian
        a = qutip.destroy(N)
        H = [
            wc * a.dag() * a,
            [1.0j * epsilon_p * (a.dag() - a), 'cos(wp * t)']
        ]
        args = {
            'wp': wp
        }

        # find the floquet modes at t=0 for the time-dependent hamiltonian
        f_modes_0, f_energies = floquet.floquet_modes(H, T, args, sort=True)

        # precalculate mode table at later times
        t_list_f_modes = np.linspace(0, T, 501)
        f_modes_table_t = floquet.floquet_modes_table(
            f_modes_0, f_energies,
            t_list_f_modes,
            H, T, args
        )

        # Compute transition matrix between Floquet modes
        _, _, _, Amat = floquet.floquet_master_equation_rates(
            f_modes_0, f_energies,
            # Caution: We must use an hermitian operator for the dissipation
            # description
            a + a.dag(),
            H, T, args, noise_spectrum,
            0,  # Temperature
            5, f_modes_table_t
        )

        # Compute steadystate
        rho_ss = floquet.floquet_master_equation_steadystate(Amat)

        # Compute reference steadystate with mesolve
        psi0 = qutip.basis(N, 0)
        rho_ss_ref = qutip.mesolve(
            H, psi0, np.linspace(0, 5 * T, 100), [], [], args
        )

        assert_((rho_ss - rho_ss_ref.states[-1]).norm() < 1e-4)


if __name__ == "__main__":
    run_module_suite()
