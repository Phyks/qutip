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

__all__ = ['floquet_modes', 'floquet_modes_t', 'floquet_modes_table',
           'floquet_modes_t_lookup', 'floquet_states_t',
           'floquet_wavefunction_t',
           'floquet_state_decomposition', 'fsesolve',
           'floquet_master_equation_rates', 'floquet_collapse_operators',
           'floquet_master_equation_tensor',
           'floquet_master_equation_steadystate', 'floquet_basis_transform',
           'floquet_markov_mesolve', 'fmmesolve']

from types import FunctionType

import numpy as np
import scipy
from scipy import angle, pi, exp, sqrt
from scipy.linalg import null_space
from qutip.qobj import Qobj, isket
from qutip.superoperator import vec2mat_index, mat2vec, vec2mat
from qutip.steadystate import steadystate
from qutip.states import ket2dm
from qutip.states import projection
from qutip.solver import Options
from qutip.propagator import propagator
from qutip.solver import Result, _solver_safety_check
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.expect import expect
from qutip.utilities import n_thermal


def floquet_modes(H, T, args=None, sort=False, U=None):
    """
    Calculate the initial Floquet modes :math:`\\Phi_\\alpha(0)`` at time
    :math:`t=0` for a driven system with period :math:`T`.

    Returns a list of :class:`qutip.qobj` instances representing the Floquet
    modes and a list of corresponding quasienergies, sorted by increasing
    quasienergy in the interval [-pi/T, pi/T]. The optional parameter `sort`
    decides if the output is to be sorted in increasing quasienergies or not.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian, time-dependent with period `T`.

    args : dictionary
        dictionary with variables required to evaluate :math:`H`.

    T : float
        The period of the time-dependence of the hamiltonian. The default value
        'None' indicates that the 'tlist' spans a single period of the driving.

    U : :class:`qutip.qobj`
        The propagator for the time-dependent Hamiltonian with period `T`.
        If U is `None` (default), it will be calculated from the Hamiltonian
        :math:`H` using :func:`qutip.propagator.propagator`.

    Returns
    -------

    output : list of kets, list of quasi energies

        Two lists: the Floquet modes as kets and the quasi energies.

    """
    # Default values
    if args is None:
        args = {}

    if U is None:
        # get the unitary propagator
        U = propagator(H, T, [], args)

    # find the eigenstates for the propagator
    evals, evecs = U.eigenstates()

    eargs = angle(evals)
    # note: angle is in the interval [-pi, pi], so that the quasi energy is in
    # the interval [-pi/T, pi/T] where T is the period of the driving.
    e_quasi = -eargs / T

    # sort by the quasi energy
    if sort:
        order = np.argsort(e_quasi)
    else:
        order = list(range(len(evals)))

    return [evecs[o] for o in order], e_quasi[order]


def floquet_modes_t(f_modes_0, f_energies, t, H, T, args=None):
    """
    Compute the Floquet modes at time `t`, :math:`\\Phi_\\alpha(t)`,
    by propagating the initial Floquet modes :math:`\\Phi_\\alpha(0)`.

    Parameters
    ----------

    f_modes_0 : list of :class:`qutip.qobj` (kets)
        Floquet modes at initial time.

    f_energies : list
        Floquet quasienergies.

    t : float
        The time at which we want to evaluate the Floquet modes.

    H : :class:`qutip.qobj`
        The system Hamiltonian, time-dependent with period `T`.

    args : dictionary
        dictionary with variables required to evaluate `H`.

    T : float
        The period of the time-dependence of the hamiltonian.

    Returns
    -------

    output : list

        The Floquet modes as kets at time :math:`t`

    """
    # Handle default values
    if args is None:
        args = {}

    # find t in [0,T] such that t_orig = t + n * T for integer n
    t = t - int(t / T) * T

    f_modes_t = []

    # get the unitary propagator from 0 to t
    if t > 0.0:
        U = propagator(H, t, [], args)

        for n in np.arange(len(f_modes_0)):
            f_modes_t.append(U * f_modes_0[n] * exp(1j * f_energies[n] * t))

    else:
        f_modes_t = f_modes_0

    return f_modes_t


def floquet_modes_table(f_modes_0, f_energies, tlist, H, T, args=None):
    """
    Pre-calculate the Floquet modes for a range of times spanning the driving
    period. Can later be used as a table to look up the floquet modes for
    any time.

    Parameters
    ----------

    f_modes_0 : list of :class:`qutip.qobj` (kets)
        Floquet modes at :math:`t`

    f_energies : list
        Floquet quasienergies.

    tlist : array
        The list of times at which to evaluate the floquet modes. tlist values
        should be in [O, T].

    H : :class:`qutip.qobj`
        system Hamiltonian, time-dependent with period `T`

    T : float
        The period of the time-dependence of the hamiltonian.

    args : dictionary
        dictionary with variables required to evaluate H

    Returns
    -------

    output : nested list

        A nested list of Floquet modes as kets for each time in `tlist`

    """
    # truncate tlist to the driving period
    tlist_period = np.array(tlist)[np.where(tlist <= T)]

    f_modes_table_t = [[] for t in tlist_period]

    Ulist = propagator(H, tlist, [], args)
    for t_idx, (t, Ut) in enumerate(zip(tlist_period, Ulist)):
        for (f_mode_0, f_energy) in zip(f_modes_0, f_energies):
            f_modes_table_t[t_idx].append(
                Ut * f_mode_0 * exp(1j * f_energy * t)
            )

    return f_modes_table_t


def floquet_modes_t_lookup(f_modes_table_t, t, T):
    """
    Lookup the Floquet mode at time t in the pre-calculated table of floquet
    modes in the first period of the time-dependence.

    Parameters
    ----------

    f_modes_table_t : nested list of :class:`qutip.qobj` (kets)
        A lookup-table of Floquet modes at times precalculated by
        :func:`qutip.floquet.floquet_modes_table`.

    t : float
        The time for which to evaluate the Floquet modes.

    T : float
        The period of the time-dependence of the hamiltonian.

    Returns
    -------

    output : nested list

        A list of Floquet modes as kets for the time that most closely matching
        the time `t` in the supplied table of Floquet modes.
    """

    # find t_wrap in [0,T] such that t = t_wrap + n * T for integer n
    t_wrap = t - int(t / T) * T

    # find the index in the table that corresponds to t_wrap (= tlist[t_idx])
    t_idx = int(t_wrap / T * len(f_modes_table_t))

    # XXX: might want to give a warning if the cast of t_idx to int discard
    # a significant fraction in t_idx, which would happen if the list of time
    # values isn't perfect matching the driving period
    # if debug: print "t = %f -> t_wrap = %f @ %d of %d" % (t, t_wrap, t_idx,
    # N)

    return f_modes_table_t[t_idx]


def floquet_states_t(f_modes_t, f_energies, t):
    """
    Evaluate the floquet states :math:`\\Psi_\\alpha(t)` at time `t` given the
    Floquet modes :math:`\\Phi_\\alpha(t)` at that time.

    Parameters
    ----------

    f_modes_t : list of :class:`qutip.qobj` (kets)
        A list of Floquet modes for time :math:`t`.

    f_energies : array
        The Floquet quasienergies.

    t : float
        The time for which to evaluate the Floquet states.

    Returns
    -------

    output : list

        A list of Floquet states for the time :math:`t`.

    """
    return [
        f_modes_t[i] * exp(-1j * f_energies[i] * t)
        for i in np.arange(len(f_energies))
    ]


def floquet_state_decomposition(f_states, f_energies, psi):
    """
    Decompose the wavefunction :math:`\\psi(t)` (typically an initial state) in
    terms of the Floquet states :math:`\\Psi_\\alpha(t)`,
    :math:`\\psi(t) = \\sum_\\alpha c_\\alpha \\psi_\\alpha(t)`.

    .. note::

        Floquet states and Floquet modes coincide at time :math:`t = 0`,
        :math:`\\Psi_\\alpha(0) = \\Phi_\\alpha(0)`.

    Parameters
    ----------

    f_states : list of :class:`qutip.qobj` (kets)
        A list of Floquet states. They should be given at the same time `t` as
        the wavefunction :math:`\\psi`.

    f_energies : array
        The Floquet quasienergies.

    psi : :class:`qutip.qobj`
        The wavefunction to decompose in the Floquet state basis.

    Returns
    -------

    output : array

        The coefficients :math:`c_\\alpha` in the Floquet state decomposition.

    """
    return [
        f_states[i].overlap(psi)
        for i in np.arange(len(f_energies))
    ]


def floquet_wavefunction_t(f_modes_t, f_energies, f_coeff, t):
    """
    Evaluate the wavefunction :math:`\\psi(t)` for a time `t` using the Floquet
    state decomposition, given the Floquet modes :math:`\\Phi_\\alpha(t)` at
    time `t`.

    Parameters
    ----------

    f_modes_t : list of :class:`qutip.qobj` (kets)
        A list of Floquet modes :math:`\\Phi_\\alpha(t)` (for time
        :math:`t`).

    f_energies : array
        The Floquet quasienergies.

    f_coeff : array
        The coefficients for Floquet decomposition :math:`c_\\alpha` of the
        initial wavefunction.

    t : float
        The time for which to evaluate the Floquet states.

    Returns
    -------

    output : :class:`qutip.qobj`

        The wavefunction for the time :math:`t`.

    """
    f_states_t = floquet_states_t(f_modes_t, f_energies, t)
    return sum([
        f_state * coeff
        for (f_state_t, coeff) in zip(f_states_t, f_coeff)
    ])


def fsesolve(H, psi0, tlist, e_ops=None, T=None, args=None, Tsteps=100):
    """
    Solve the Schrodinger equation using the Floquet formalism.

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        System Hamiltonian, time-dependent with period `T`.

    psi0 : :class:`qutip.qobj`
        Initial state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function
        list of operators for which to evaluate expectation values. If this
        list is empty, the state vectors for each time in `tlist` will be
        returned instead of expectation values.

    T : float
        The period of the time-dependence of the hamiltonian. If not provided,
        `tlist` should span exactly one period of the driving.

    args : dictionary
        Dictionary with variables required to evaluate H.

    Tsteps : integer
        The number of time steps in one driving period for which to
        precalculate the Floquet modes. `Tsteps` should be an even number.

    Returns
    -------

    output : :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which
        contains either an *array* of expectation values or an array of
        state vectors, for the times specified by `tlist`.
    """
    if Tsteps % 2 != 0:
        raise ValueError('Tsteps should be an even number.')

    # Handle default values
    if e_ops is None:
        e_ops = []
    if args is None:
        args = {}

    if not T:
        # assume that tlist span exactly one period of the driving
        T = tlist[-1]

    # find the Floquet modes for the time-dependent hamiltonian
    f_modes_0, f_energies = floquet_modes(H, T, args)

    # setup Result for storing the results
    output = Result()
    output.times = tlist
    output.solver = "fsesolve"

    if isinstance(e_ops, FunctionType):
        output.num_expect = 0
        expt_callback = True

    elif isinstance(e_ops, list):
        output.num_expect = len(e_ops)
        expt_callback = False

        if output.num_expect == 0:
            output.states = []
        else:
            output.expect = []
            for op in e_ops:
                if op.isherm:
                    output.expect.append(np.zeros(len(tlist)))
                else:
                    output.expect.append(np.zeros(len(tlist), dtype=complex))

    else:
        raise TypeError("e_ops must be a list Qobj or a callback function")

    f_coeff = floquet_state_decomposition(f_modes_0, f_energies, psi0)
    for t_idx, t in enumerate(tlist):
        f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, args)
        psi_t = floquet_wavefunction_t(f_modes_t, f_energies, f_coeff, t)

        if expt_callback:
            # use callback method
            e_ops(t, psi_t)
        else:
            # calculate all the expectation values, or output psi if
            # no expectation value operators where defined
            if output.num_expect == 0:
                output.states.append(Qobj(psi_t))
            else:
                for e_idx, e in enumerate(e_ops):
                    output.expect[e_idx][t_idx] = expect(e, psi_t)

    return output


def floquet_master_equation_rates(f_modes_0, f_energies, c_op, H, T,
                                  args, J_cb, w_th, kmax=5,
                                  f_modes_table_t=None, nT=100):
    """
    Calculate the rates and matrix elements for the Floquet-Markov master
    equation.

    Parameters
    ----------

    f_modes_0 : list of :class:`qutip.qobj` (kets)
        A list of initial Floquet modes.

    f_energies : array
        The Floquet quasienergies.

    c_op : :class:`qutip.qobj`
        The collapse operators describing the dissipation.

    H : :class:`qutip.qobj`
        System Hamiltonian, time-dependent with period `T`.

    T : float
        The period of the time-dependence of the hamiltonian.

    args : dictionary
        Dictionary with variables required to evaluate H.

    J_cb : callback functions
        A callback function that computes the noise power spectrum, as
        a function of frequency, associated with the collapse operator `c_op`.

    w_th : float
        The temperature in units of frequency.

    k_max : int
        The truncation of the number of sidebands (default 5).

    f_modes_table_t : nested list of :class:`qutip.qobj` (kets)
        A lookup-table of Floquet modes at times precalculated by
        :func:`qutip.floquet.floquet_modes_table` (optional).

    nT : int
        Number of steps to take in the numerical integration.

    Returns
    -------

    output : list

        A list (Delta, X, Gamma, A) containing the matrices Delta, X, Gamma
        and A used in the construction of the Floquet-Markov master equation.

    """

    N = len(f_energies)
    M = 2 * kmax + 1

    omega = (2 * pi) / T

    Delta = np.zeros((N, N, M))
    X = np.zeros((N, N, M), dtype=complex)
    Gamma = np.zeros((N, N, M))
    A = np.zeros((N, N))

    dT = T / nT
    tlist = np.arange(dT, T + dT / 2, dT)

    if f_modes_table_t is None:
        f_modes_table_t = floquet_modes_table(f_modes_0, f_energies,
                                              np.linspace(0, T, nT + 1), H, T,
                                              args)

    c_op = c_op.full()
    for t in tlist:
        # Use numpy representation to compute overlaps, which is about 50 times
        # more efficient.
        f_modes_t = [
            f.full() for f in floquet_modes_t_lookup(f_modes_table_t, t, T)
        ]
        for a in range(N):
            bra_a = np.dot(np.conj(f_modes_t[a].T), c_op)
            for b in range(N):
                scalar_product = np.asscalar(np.dot(
                    bra_a,
                    f_modes_t[b]
                ))
                k_idx = 0
                for k in range(-kmax, kmax + 1, 1):
                    X[a, b, k_idx] += (dT / T) * exp(-1j * k * omega * t) * \
                        scalar_product
                    k_idx += 1

    for a in range(N):
        for b in range(N):
            k_idx = 0
            for k in range(-kmax, kmax + 1, 1):
                Delta[a, b, k_idx] = f_energies[a] - f_energies[b] + k * omega
                Gamma[a, b, k_idx] = (
                    2 * pi * np.heaviside(Delta[a, b, k_idx], 0.5) *
                    J_cb(Delta[a, b, k_idx]) * abs(X[a, b, k_idx])**2
                )
                k_idx += 1

    for a in range(N):
        for b in range(N):
            for k in range(-kmax, kmax + 1, 1):
                k1_idx = k + kmax
                k2_idx = -k + kmax
                A[a, b] += (
                    Gamma[a, b, k1_idx] +
                    n_thermal(abs(Delta[a, b, k1_idx]), w_th) * (
                        Gamma[a, b, k1_idx] + Gamma[b, a, k2_idx]
                    )
                )

    return Delta, X, Gamma, A


def floquet_collapse_operators(A):
    """
    Construct collapse operators (in Floquet basis!) corresponding to the
    Floquet-Markov master-equation rate matrix :math:`A`.

    .. note::

        The Lindlad equation obtained using only these collapse operators
        (``liouvillian(None, c_ops=floquet_collapse_operator(A))``) is the
        master equation from eq.(251) in
        [Grifoni et al, Phys.Rep. 304 229 (1998)].

    Parameters
    ----------

    A : array
        The rate matrix :math:`A` used in the construction of the
        Floquet-Markov master equation.

    Returns
    -------

    c_ops : list

        A list of collapse operators (in Floquet basis!) corresponding to the
        Floquet-Markov master-equation rate matrix :math:`A`.

    """
    c_ops = []

    N, M = np.shape(A)

    #
    # Here we really need a master equation on Bloch-Redfield form, or perhaps
    # we can use the Lindblad form master equation with some rotating frame
    # approximations? ...
    #
    for a, b in zip(range(M), range(N)):
        if a != b and abs(A[a, b]) > 0.0:
            # only relaxation terms included...
            c_ops.append(sqrt(A[a, b]) * projection(N, a, b))

    return c_ops


def floquet_master_equation_tensor(Alist, f_energies):
    # TODO: Review
    """
    Construct a tensor that represents the master equation in the floquet
    basis (with constant Hamiltonian and collapse operators).

    Simplest RWA approximation [Grifoni et al, Phys.Rep. 304 229 (1998)]

    Parameters
    ----------

    Alist : list
        A list of Floquet-Markov master equation rate matrices.

    f_energies : array
        The Floquet quasienergies.

    Returns
    -------

    output : array

        The Floquet-Markov master equation tensor `R`.

    """

    if isinstance(Alist, list):
        # Alist can be a list of rate matrices corresponding
        # to different operators that couple to the environment
        N, M = np.shape(Alist[0])
    else:
        # or a simple rate matrix, in which case we put it in a list
        Alist = [Alist]
        N, M = np.shape(Alist[0])

    Rdata_lil = scipy.sparse.lil_matrix((N * N, N * N), dtype=complex)
    for I in range(N * N):
        a, b = vec2mat_index(N, I)
        for J in range(N * N):
            c, d = vec2mat_index(N, J)

            R = -1.0j * (f_energies[a] - f_energies[b])*(a == c)*(b == d)
            Rdata_lil[I, J] = R

            for A in Alist:
                s1 = s2 = 0
                for n in range(N):
                    s1 += A[a, n] * (n == c) * (n == d) - A[n, a] * \
                        (a == c) * (a == d)
                    s2 += (A[n, a] + A[n, b]) * (a == c) * (b == d)

                dR = (a == b) * s1 - 0.5 * (1 - (a == b)) * s2

                if dR != 0.0:
                    Rdata_lil[I, J] += dR

    return Qobj(Rdata_lil, [[N, N], [N, N]], [N*N, N*N])


def floquet_master_equation_steadystate(A, H=None):
    """
    Returns the steadystate density matrix (in the Floquet basis!) for the
    Floquet-Markov master equation.

    .. note::

        This is based on the simplest RWA approximation from
        [Grifoni et al, Phys.Rep. 304 229 (1998)] (eq.251).

    Parameters
    ----------
    A : array
        The rate matrix :math:`A` used in the construction of the
        Floquet-Markov master equation.

    H : :class:`qutip.qobj`
        An extra Hamiltonian to consider. When `H` is ``None``, this is exactly
        the steadystate of eq.(251) in
        [Grifoni et al, Phys.Rep. 304 229 (1998)].

    Returns
    -------
    rho_ss : :class:`qutip.qobj`
        Steady state density matrix (in the Floquet basis!).
    """
    if H is not None:
        # When H is provided, rely on the steadystate solver
        c_ops = floquet_collapse_operators(A)
        rho_ss = steadystate(H, c_ops)
        return rho_ss
    else:
        # When H is not provided, we can rely on the fact that eq.(251) from
        # Grifoni, 1998 can be written as an eigenvalue problem for an
        # auxiliary B matrix, which is much more efficient. This is possible
        # since the steady state density matrix will be diagonal in this case.
        B = np.zeros_like(A)
        M, N = B.shape
        for (a, b) in zip(range(M), range(N)):
            if a != b:
                B[a, b] = A[a, b]  # TODO: A[b, a]?
            else:
                B[a, b] = -1.0 * np.sum([
                    A[nu, a] for nu in range(A.shape[0]) if nu != a
                ])  # TODO: A[a, nu]?

        # Look for steadystate in the kernel of B
        Z = null_space(B)
        if Z.shape[1] > 1:
            raise ValueError('Steady state is not unique. This is currently '
                             'unsupported.')
        else:
            # Null space of dimension 1
            rho_ss = Z.flatten()
            if not np.allclose(rho_ss, 0) or np.any(np.real(rho_ss) < 0):
                raise ValueError('Found steady state is not a valid density '
                                 'matrix.')

        # Normalize the steady state in L1 norm
        rho_ss = rho_ss / np.sum(np.abs(rho_ss))
        return Qobj(np.diag(rho_ss))


def floquet_basis_transform(f_modes, rho0):
    """
    Make a basis transform that takes rho0 from the floquet basis to the
    computational basis.
    """
    return rho0.transform(f_modes, True)


# -----------------------------------------------------------------------------
# Floquet-Markov master equation
#
#
def floquet_markov_mesolve(R, ekets, rho0, tlist, e_ops, f_modes_table=None,
                           options=None, floquet_basis=True):
    # TODO: Review
    """
    Solve the dynamics for the system using the Floquet-Markov master equation.
    """

    if options is None:
        opt = Options()
    else:
        opt = options

    if opt.tidy:
        R.tidyup()

    #
    # check initial state
    #
    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = ket2dm(rho0)

    #
    # prepare output array
    #
    n_tsteps = len(tlist)
    dt = tlist[1] - tlist[0]

    output = Result()
    output.solver = "fmmesolve"
    output.times = tlist

    if isinstance(e_ops, FunctionType):
        n_expt_op = 0
        expt_callback = True

    elif isinstance(e_ops, list):

        n_expt_op = len(e_ops)
        expt_callback = False

        if n_expt_op == 0:
            output.states = []
        else:
            if not f_modes_table:
                raise TypeError("The Floquet mode table has to be provided " +
                                "when requesting expectation values.")

            output.expect = []
            output.num_expect = n_expt_op
            for op in e_ops:
                if op.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))

    else:
        raise TypeError("Expectation parameter must be a list or a function")

    #
    # transform the initial density matrix to the eigenbasis: from
    # computational basis to the floquet basis
    #
    if ekets is not None:
        rho0 = rho0.transform(ekets)

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(cy_ode_rhs)
    r.set_f_params(R.data.data, R.data.indices, R.data.indptr)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # start evolution
    #
    rho = Qobj(rho0)

    t_idx = 0
    for t in tlist:
        if not r.successful():
            break

        rho = Qobj(vec2mat(r.y), rho0.dims, rho0.shape)

        if expt_callback:
            # use callback method
            if floquet_basis:
                e_ops(t, Qobj(rho))
            else:
                f_modes_table_t, T = f_modes_table
                f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
                e_ops(t, Qobj(rho).transform(f_modes_t, True))
        else:
            # calculate all the expectation values, or output rho if
            # no operators
            if n_expt_op == 0:
                if floquet_basis:
                    output.states.append(Qobj(rho))
                else:
                    f_modes_table_t, T = f_modes_table
                    f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
                    output.states.append(Qobj(rho).transform(f_modes_t, True))
            else:
                f_modes_table_t, T = f_modes_table
                f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
                for m in range(0, n_expt_op):
                    output.expect[m][t_idx] = \
                        expect(e_ops[m], rho.transform(f_modes_t, False))

        r.integrate(r.t + dt)
        t_idx += 1

    return output


# -----------------------------------------------------------------------------
# Solve the Floquet-Markov master equation
#
#
def fmmesolve(H, rho0, tlist, c_ops=[], e_ops=[], spectra_cb=[], T=None,
              args={}, options=Options(), floquet_basis=True, kmax=5,
              _safe_mode=True):
    # TODO: Review
    """
    Solve the dynamics for the system using the Floquet-Markov master equation.

    .. note::

        This solver currently does not support multiple collapse operators.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian.

    rho0 / psi0 : :class:`qutip.qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_ops : list of :class:`qutip.qobj`
        list of collapse operators.

    e_ops : list of :class:`qutip.qobj` / callback function
        list of operators for which to evaluate expectation values.

    spectra_cb : list callback functions
        List of callback functions that compute the noise power spectrum as
        a function of frequency for the collapse operators in `c_ops`.

    T : float
        The period of the time-dependence of the hamiltonian. The default value
        'None' indicates that the 'tlist' spans a single period of the driving.

    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

        This dictionary should also contain an entry 'w_th', which is
        the temperature of the environment (if finite) in the
        energy/frequency units of the Hamiltonian.  For example, if
        the Hamiltonian written in units of 2pi GHz, and the
        temperature is given in K, use the following conversion

        >>> temperature = 25e-3 # unit K
        >>> h = 6.626e-34
        >>> kB = 1.38e-23
        >>> args['w_th'] = temperature * (kB / h) * 2 * pi * 1e-9

    options : :class:`qutip.solver`
        options for the ODE solver.

    k_max : int
        The truncation of the number of sidebands (default 5).

    Returns
    -------

    output : :class:`qutip.solver`

        An instance of the class :class:`qutip.solver`, which contains either
        an *array* of expectation values for the times specified by `tlist`.
    """

    if _safe_mode:
        _solver_safety_check(H, rho0, c_ops, e_ops, args)

    if T is None:
        T = max(tlist)

    if len(spectra_cb) == 0:
        # add white noise callbacks if absent
        spectra_cb = [lambda w: 1.0] * len(c_ops)

    f_modes_0, f_energies = floquet_modes(H, T, args)

    f_modes_table_t = floquet_modes_table(f_modes_0, f_energies,
                                          np.linspace(0, T, 500 + 1),
                                          H, T, args)

    # get w_th from args if it exists
    if 'w_th' in args:
        w_th = args['w_th']
    else:
        w_th = 0

    # TODO: loop over input c_ops and spectra_cb, calculate one R for each set

    # calculate the rate-matrices for the floquet-markov master equation
    Delta, X, Gamma, Amat = floquet_master_equation_rates(
        f_modes_0, f_energies, c_ops[0], H, T, args, spectra_cb[0],
        w_th, kmax, f_modes_table_t)

    # the floquet-markov master equation tensor
    R = floquet_master_equation_tensor(Amat, f_energies)

    return floquet_markov_mesolve(R, f_modes_0, rho0, tlist, e_ops,
                                  f_modes_table=(f_modes_table_t, T),
                                  options=options,
                                  floquet_basis=floquet_basis)
