# Parallel Cubic Spline Interpolation Solver (C, OpenMPI)

Solves the cubic spline interpolation problem by reducing it to a
tridiagonal system and solving that system in parallel with MPI.

## Approach
- Partition the tridiagonal system across processes.
- Reduce it to a smaller system of size p (one block per process).
- Solve the reduced system with cyclic reduction.
- Back-substitute to recover the full solution.

Assumes a fully connected network of processes.

## Build
mpicc -o spline main.c -lm

## Run
mpiexec -np 8 ./spline

## Expected output
For f(x) = sin(x), convergence is 4th order on nodes and 3rd order
on general points as n doubles. For f(x) = x^2, the solver returns
the exact solution (machine precision).

Default output is for sin.

## Files
- main.c

## Limitations
- No pivoting; assumes a diagonally dominant system.
- No fault tolerance; all processes must remain alive.
- Tested on up to 8 processes.
