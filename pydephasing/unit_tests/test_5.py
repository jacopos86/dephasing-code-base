import numpy as np
from petsc4py import PETSc
from utilities.log import log

#
#   TEST MATRIX ODE
#

def test_petsc_installation():
    # Simple 2x2 matrix
    A_np = np.array([[1, 2], [3, 4]], dtype='d')
    # Convert numpy matrix to PETSc matrix
    A = PETSc.Mat().createDense(size=(2, 2), array=A_np)
    # Create a solution vector (size 2)
    b = PETSc.Vec().createSeq(2)  # PETSc vector with 2 elements
    b.setArray([1.0, 1.0])  # Set the vector values
    # Multiply the matrix by the vector
    x = A * b
    # Define the expected result manually: A * b
    x_exp = np.array([3.0, 7.0])  # Expected result of the multiplication
    # Use pytest's assert to compare the result with the expected values
    np.testing.assert_allclose(x, x_exp, atol=1e-8)

def rhs(ts, t, y, f):
    A = ts.getRHSJacobian()[0]
    A.mult(y, f)

def test_matrix_ode():
    # Matrix A
    A_np = np.array([[1, 2], [3, 4]], dtype='d')  # 2x2 matrix
    A = PETSc.Mat().createDense(size=(2, 2), array=A_np)
    A.assemble()
    # Initial condition
    y0 = PETSc.Vec().createSeq(2)
    y0.setArray([1.0, 1.0])
    # Time stepper
    ts = PETSc.TS().create()
    ts.setType(ts.Type.EULER)
    ts.setRHSFunction(rhs)
    ts.setRHSJacobian(A, A)
    ts.setTimeStep(0.1)
    ts.setMaxSteps(1)
    # set y and solve
    y = y0.copy()
    ts.setSolution(y)
    ts.solve(y)
    result = y.getArray()
    expected = np.array([1.3, 1.7])  # Forward Euler: y + dt*A*y with dt=0.1
    np.testing.assert_allclose(result, expected)

def test_matrix_ode_2():
    # Matrix A
    A_np = np.array([[0.1, -0.1], [0.3, 0.4]], dtype='d')
    A = PETSc.Mat().createDense(size=(2, 2), array=A_np)
    A.assemble()
    # Initial condition
    y0 = PETSc.Vec().createSeq(2)
    y0.setArray([0.1, 0.1])
    # Time stepper
    ts = PETSc.TS().create()
    ts.setType(ts.Type.EULER)
    ts.setRHSFunction(rhs)
    ts.setRHSJacobian(A, A)
    ts.setTimeStep(0.1)
    ts.setMaxSteps(1)
    # set solver
    y = y0.copy()
    ts.setSolution(y)
    ts.solve(y)
    result = y.getArray()
    expected = np.array([0.1, 0.107])  # Forward Euler: y + dt*A*y with dt=0.1
    np.testing.assert_allclose(result, expected)