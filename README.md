# Testing autodiff for solving the Poisson equation
The code in this repository is used to test the autodiff package for solving the Poisson equation ($\Delta u = f$) on arbitrary surfaces. We use an MLP to approximate the function $u$ and the loss function is $\|\Delta u_{MLP} - f\|$.
The experiments are set up to check if one observes a convergence pattern as the number of degrees of freedom (tunable weights of the MLP) are increased.

## Experiments
We primarily built different classes of experiments, with each one being a little more involved than the previous one, to check where autodiff might fail. 
Our experiments can be summarised in the following table
| NN Domain<br> \ <br> Parameterization with map to target domain | Plane | Sphere |
| --------  | ------- | ----- |
| No | Convergence | Convergence |
| Yes (exact parameterization) | Convergence | Convergence |
| Yes (MLP learns the exact parameterization) | Convergence | Not Converging|
| Yes (computed parameterization) | - | -|

1. The first experiment is to solve the Poisson equation directly on a plane with Dirichlet boundary and on the surface of the sphere
2. The second experiment is to use an exact parameterization from the plane/sphere to surface
3. Same as the second experiment but instead an MLP is used to learn the exact parameterization and the MLP is used to solve the Poisson equation. The MLP is learned using the same method as in _"Neural Geometry Processing via Spherical Neural Surfaces" by Williamson et al._ and _"Neural Surface Maps" by Morreale et al._

## Setup
This code is tested on _x64 linux_ platform using _Python 3.12.4_.
The code should run out of the box if all the required packages are installed.

The repository has two directories
1. __test_exact__ : Consists of two files -- `test_autodiff_2d.py` and `test_autodiff_sphere.py` that solves the Poisson equation on the plane and sphere respectively (experiment 1).
2. __test_exact_parameterization__: Consists of the file `test_parameterization.py` that is responsible for experiment 2 and experiment 3.

## Run
For the file __test_exact/test_autodiff_2d.py__, run
```sh
python3 test_autodiff_2d.py [train | plot] [1 | 2 ]
```
For the file __test_exact/test_autodiff_sphere.py__, run
```sh
python3 test_autodiff_sphere.py [train | plot] [1 | 2 ]
```
The first argument is responsible to train the model or plot the results.
The second argument is used to choose what example to run.

For the file __test_exact_parameterization/test_parameterization.py__, run
```sh
python3 test_parameterization_2d.py [train|plot] [1|2|3] [hemisphere|heightfield|ellipsoid] [noNN|withNN]
```
The third argument is to select the surface to solve the Poisson equation on. For hemisphere and heightfield, the parameterization from 2d to hemisphere/heightfield is used. In the code, we choose 0.5*(x^2 + y^2) from [-1, 1] as the heightfield. If ellipsoid is chosen, then the parameterization from the unit sphere to ellipsoid is chosen to solve the Poisson equation.
The fourth argument is used to choose whether to choose the exact parameterization (experiment 2) or to choose a learned parameterization (experiment 3)


## Pretrained models
We test using multiple analytical functions for all experiments. The pretrained models are stored in the corresponding directories. The models are trained using the `train` command.
The directory also contains the convergence plots for each example using the `plot` command on the trained models.
