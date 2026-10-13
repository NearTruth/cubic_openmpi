Cubic spline interpolation solver written using C and OpenMPI. 

The problem reduces to solving tridiagonal systems using cubic basis functions. 

The large matrix is first partitioned and reduced into a smaller tridiagonal system of size n where n is the number of participating computers in the network.
The reduced system is solved using cyclic reduction.
The network topology is assumed to be a fully connected network

Compiling and linking requires use of mpicc. 

