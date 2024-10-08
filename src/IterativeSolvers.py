import numpy as np
from numpy.linalg import qr
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, gmres, bicgstab, spilu, LinearOperator
from scipy import linalg
from copy import copy


class LUSolver:

    def __init__(self, A):
        A_sp = csc_matrix(A)
        self.lu = splu(A_sp)

    def solve(self, b):
        return self.lu.solve(b)


class QRSolver:

    def __init__(self, A):
        self.Q, self.R = qr(A, mode='complete')
        self.Q_H = np.conj(self.Q).T

    def solve(self, b):
        return linalg.solve_triangular(self.R, self.Q_H.dot(b))


class IterativeRefinement:

    def __init__(self, solver_type='LU', tol=1.e-5, maxiter=100, verbose=True):
        """
        :param solver_type: [string] the direct solver in the iterative refinement loop, allowed values = {'LU', 'QR'}
        :param tol:  the allowed tolerance
        :param maxiter: [int] max number of iteration for convergence.

        **Reference:**
            Burden, R.L. and Faires, J.D., 2011. Numerical analysis.

        """
        self.type = solver_type
        self.tol = tol
        self.maxiter = maxiter
        self.verbose = verbose

    def _solver(self, A):
        if self.type == 'LU':
            return LUSolver(A)
        elif self.type == 'QR':
            return QRSolver(A)
        else:
            return None

    def solver_name(self):
        return 'iterRef_{}'.format(self.type)

    def solve(self, A, b):
        """
        Solve the equation a x = b for x using Iterative Refinement method.

        :param A: [(M,M) array_like] A square matrix.
        :param b: [(M,) array like]Right-hand side matrix in a x = b.

        :return x: (M,) or (M, N) ndarray
            Solution to the system a x = b. Shape of the return matches the shape of b.
        """


        A_sp = csc_matrix(A)

        solver = self._solver(A)

        #_b = _b
        # declarations
        _n = len(b)
        xx = np.zeros_like(b)
        r = np.zeros_like(b)

        x = solver.solve(b)
        res = np.sum(A_sp.dot(x)-b)
        #    print("res :: A * x - b = {:e}".format(res))

        # check if converged
        if np.abs(res) < self.tol:
            print("IR ::: A * x - b = {:e}".format(res))
            return x

        k = 1                                       # step 1
        while (k <= self.maxiter):                       # step 2
            r = b - A_sp.dot(x)                        # step 3
            y = (solver.solve(r))                         # step 4
            xx = copy(x + y)                        # step 5

            if k == 1:                              # step 6
                # t = 16
                # COND = np.linalg.norm(y)/np.linalg.norm(xx) * 10**t
                # print("cond is ::::", COND)
                COND = np.linalg.cond(A_sp.toarray())

            norm_ = np.linalg.norm(x-xx) # * np.linalg.norm(x) * 1e10
            print("iteration {:3d}, norm = {:e}".format(k, norm_))

            if norm_ < self.tol:                         # step 7

                if self.verbose:
                    #                 print("Conditional number of matrix A is: {:e}".format(COND))
                    print("The procedure was successful.")
                    print("IR: A * x - b = {:e}".format(np.sum(A_sp.dot(xx)-b)))
                    print(f"number of iteration is : {k:d}")
                    print(" ")
                return xx

            k += 1                                 # step 8
            x = copy(xx)                           # step 9

        print("Max iteration exceeded.")
        print("The procedure was not successful.")
        print("Conditional number of matrix A is: {:e}".format(COND))
        print(" ")

        return None


class PreconditionedIterativeRefinement:

    def __init__(self, preconditioner, solver):
        self.preconditioner = preconditioner
        self.solver = solver

    def solve(self, A, b):
        M = self.preconditioner.precondition_matrix(A)
        A_new = np.matmul(A, M.dot(np.eye(M.shape[0])))
        y = self.solver.solve(A_new, b)
        return M.dot(y)

    def solver_name(self):
        return '{}_precond_{}'.format(self.solver.solver_name(), self.preconditioner.name())


class ILUPreconditioner:

    def precondition_matrix(self, A):
        A_sp = csc_matrix(A)
        P = spilu(A_sp)
        return LinearOperator(A_sp.shape, P.solve)

    def name(self):
        return 'ILU'


class BiCGSTABSolver:

    def __init__(self, preconditioner_type='ILU', tol=1.e-5):
        self.preconditioner = ILUPreconditioner()
        if preconditioner_type != 'ILU':
            raise ValueError('Pre-conditioner type {} is not implemented'.format(preconditioner_type))
        self.tol = tol

    def solve(self, A, b):
        A_sp = csc_matrix(A)
        M = self.preconditioner.precondition_matrix(A)
        x_0 = M.dot(b)
        x, exit_code = bicgstab(A_sp, b, x0=x_0, M=M, tol=self.tol)
        if exit_code != 0:
            print('bicgstab unsuccessful, exit code = {}, resid = {}'.format(exit_code, np.sum(np.abs(A_sp.dot(x)-b))))
            print('allclose = {}'.format(np.allclose(A_sp.dot(x), b)))
        return x

    def solver_name(self):
        return '{}_precond_{}'.format('BiCGSTAB', self.preconditioner.name())


class GMResSolver:

    def __init__(self, preconditioner_type='ILU', tol=1.e-5):
        self.preconditioner = ILUPreconditioner()
        if preconditioner_type != 'ILU':
            raise ValueError('Pre-conditioner type {} is not implemented'.format(preconditioner_type))
        self.tol = tol

    def solve(self, A, b):
        A_sp = csc_matrix(A)
        M = self.preconditioner.precondition_matrix(A)
        x_0 = M.dot(b)
        x, exit_code = gmres(A_sp, b, x0=x_0, M=M, tol=self.tol)
        if exit_code != 0:
            print('gmres unsuccessful, exit code = {}, resid = {}'.format(exit_code, np.sum(np.abs(A_sp.dot(x)-b))))
            print('allclose = {}'.format(np.allclose(A_sp.dot(x), b)))
        return x

    def solver_name(self):
        return '{}_precond_{}'.format('GMRes', self.preconditioner.name())
