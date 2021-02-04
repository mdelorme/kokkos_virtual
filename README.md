# kokkos_virtual
Playing with virtual functions and alternatives in Kokkos using a simple nbody direct summation code.

The objective of this miniapp is two-fold

 1. Become a playground to experiment on Nbody direct-summation
 2. Try out various solutions to implement user-friendliness in kokkos kernels.
 
# Experiments on Nbody direct-summation
The general idea for point no1 is that the acceleration calculations are taking a lot of time while being deceptively simple. The algorithm is quadratic hence perfect for GPU calculations (simple but intensive).
However the final acceleration for a particle has to be reduced over the various threads contributing. While reduction on scalars is pretty trivial with Kokkos, a reduction on a vector is more difficult to guarantee coalescence and effectiveness.

# Virtualization and User-Friendliness

The second point looks at how to add functionalities to a Kokkos program without having to hard-code the links between the various parts. This is part of an ongoing research on how to make simulation codes more abstract and on how to separate physics interfaces from algorithmic interfaces. 
Physicists are not interested by the actual parallelisation and the data traversal algorithms, but rather by the operations to be done on computational elements (particles, cells, etc.). While Kokkos simplifies a lot of things on that side, concentrating the essential part of calculations in kernels, the actual definition of the kernels, and how they are launched, on which device, using specific memory layouts, etc. can be very tricky and impressive to people editing a code.
Another point is that while doing experimentations on computations the users might want to modify or add treatments without having to duplicate code.

One solution to that point is to provide the user interfaces to implement their physics. This is something very well done in the [Pluto](http://plutocode.ph.unito.it/) code for instance where a user only has to modify the `init.c` file to add most of the info required for a simulation.
While this specific solution can be efficient "user-wise" it is mostly implemented via the build system which is not adequate in terms of software design and analysis. Another solution, more adapted to this would be to provide base classes from which the user implements their own versions ("User policies"). The user classes are then automatically added to a Factory which can select the appropriate policy at runtime without having to recompile the program.

This approach, while standard in computer-science is made very difficult in Kokkos because of [virtualization](https://github.com/kokkos/kokkos/wiki/Kokkos-and-Virtual-Functions) and the way functors are allocated on devices. Such an approach would work perfectly on a CPU-only simulation but using GPUs, the allocation must be done carefully on the GPUs to make sure the virtual-tables of inheriting classes are not pointing to memory space outside the device.

This miniapp is made to test different solutions using various degrees of complexity to tackle this problem in a generic manner.
