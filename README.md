# JP_Brain_mod
A simulation of development of fetus brain during gestation period.

Note: This code is written based on the step-44 tutorial of deal.ii library: [Step-44 tutorial by Jean-Paul Pelteret and Andrew McBride](https://dealii.org/current/doxygen/deal.II/step_44.html)

**Software requirements:**
1. deal.ii -version 9.4.0 (atleast)
2. cmake -version 3.26.3
3. gcc -version 11.4.0

**Implementation steps:**
1. In the CMakeLists.txt check for required source file name "hyper_ellipsoid.cc" or "hyper_shell.cc" based on requirement.
2. In the current repository run the following command to build the dependencies and run the binary file:

       cmake -DDEAL_II_DIR=/path/to/installed/deal.II .
       
       make
       
       make run

**Output:**
1. A set of .vtk files, one for each timestep. Suitable visulisation tool can be used to open these files. eg.: Paraview
2. convergence.txt file, used to observe the convergence behaviour of the solver.

**Keypoints:**
1. A partial section of the full brain of fetus at the 24th week of gestation period is considered. As seen in the below image the surface of the brain is smooth without any wrinkles or folds. We had considered two representative geometries, quarter hyper ellipsoid and quarter hyper shell.
   
2. We had considered homogeneous boundary conditions.
   
   Neumann boundary condition:

       no traction forces acting on the boundary 1

   Dirichlet boundary condition:
   
       zero displacement at boundary 0,

       zero displacement in x direction at boundary 2

       zero displacement in y direction at boundary 3

<img width="682" height="537" alt="mesh_ellipse" src="https://github.com/user-attachments/assets/024acd51-a218-4d29-97bd-a75e9740f5f8" />

3. An important consideration in this project, is with respect to the material properties:
   Biological tissues possess high water content which makes them quasi-incompressible. Therefore, here we shall impose quasi-incompressibility by considering a constraint on the dilatation
   $\widetilde{J}$ = 1.

   Trying to solve this problem with general single-field weak formulation will lead to stiff behaviour, where the computed deformations are much smaller than the actual values

   Therefore, we use a lagrangian-multiplier method, which introduces pressure as lagrangian multiplier into the objective function to be minimized. The final formulation is called Jacobian-Pressure formulation.

4. Though the brain tissue is considered to have visco-hyperelastic material behaviour, we here consider only hyperelastic behaviour. This because the viscous behaviour is more prevalent during short time periods.

5. The deformation gradient **F** is computed by a multiplicative decomposition into growth and elastic parts.

   $\mathbf{F} = \mathbf{F}^e . \mathbf{F}^g$

   $\mathbf{F}^e = \mathbf{F} . {\mathbf{F}^g}^{-1}$

   Where, the $\mathbf{F}$ is computed every newton iteration and the $\mathbf{F}^g$ is computed once every timestep.

   $\mathbf{F}^g = \nu_t  (\mathbf{I} - \mathbf{N} \otimes \mathbf{N}) + \nu_r  (\mathbf{N} \otimes \mathbf{N})$

   where,

   $\nu_t = \sqrt{1+(\nu_{max}-\nu_{min})\left(1-e^{-\frac{t}{\tau}}\right)}$

   $\nu_r = \sqrt{1+\frac{1}{\beta_k}(\nu_{max}-\nu_{min})\left(1-e^{-\frac{t}{\tau}}\right)}$

   The high growth rate in the tangential direction in the cortex layer leads to high compressive stresses and hence mechanical instability or buckling phenomenon. Therefore, causing morphological deformation in the form of folds ors wrinkles.

<img width="1280" height="720" alt="growth_continuum" src="https://github.com/user-attachments/assets/6213d189-3edc-4466-be4f-ac38fd238c23" />

<img width="1612" height="784" alt="gm1 5gr1 5sr10ct0 1_p_folds" src="https://github.com/user-attachments/assets/dc12253c-b264-4a0c-bb5f-350213f1a675" />

6. In general mechanical instabilities lead to phenomena like buckling due to presence of imperfections. But in case of numerical simulation we tend to consider perfect bodies. Therefore, in case of perfect geometries like hyper shell, we need to explicitly induce imperfections. For example increasing the shear modulus (because it is a material property) by a small amount at random quadrature points.

7. To study the pressure variation across the cortex morphology two points are considered on the outer surface at timestep 0 as shown in the below image. It was assumed that P1 indicated gyri and P2 indicates sulci.

<img width="649" height="501" alt="pressure_points_t0" src="https://github.com/user-attachments/assets/4ddd8095-3a9c-40d3-8968-c5b02dda1f02" />

8. Upon plotting the pressure values through the simulation at points P1 and P2, it was observed that at the point where the system reaches instability the pressure values bifurcated in the form of a pitch-fork. Where the pressure at gyri increased in positive direction and the pressure at sulci increased in negative direction.

   With varying the parameter, stiffness ratio, the behaviour of the bifurcation differed as seen in the following plot

<img width="1280" height="720" alt="stiffness_pressure" src="https://github.com/user-attachments/assets/b925ea50-a97c-4631-b631-8ecb0e87461a" />
