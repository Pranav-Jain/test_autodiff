# Testing autodiff for solving the Poisson equation
The code in this repository is used to test the autodiff package for solving the Poisson equation in 2D and on a surface of a sphere. 
In case of 2D , the domain is a R^2 and we use an analytical function to test the accuracy of the solution. 
In case of the sphere, the domain is S^2 and we use an analytical function to test the accuracy of the solution. 

## Setup
This code is tested on _x64 linux_ platform using _Python 3.12.4_.
The code should run out of the box if all the required packages are installed.

## Run
Use the following command to run the code
```sh
python3 test_autodiff_2d.py [train | plot] [1 | 2]
python3 test_autodiff_sphere.py [train | plot] [1]
```
The first argument is responsible to train the model or plot the results. The second argument is used to choose the example to run.

## Pretrained models
We test using two different unctions each in 2D and the sphere. The pretrained models are stored in the `poisson_results_2d` and `poisson_results_sphere` directory. The models are trained using the `train` command.
The directory also contains the convergence plots using the `plot` command on the trained models.
