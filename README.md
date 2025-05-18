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
python3 test_autodiff_sphere.py [train | plot] [1 | 2] [ | exactvalue | zerointegral]
```
The first argument is responsible to train the model or plot the results.
The second argument is used to choose what example to run.
The third argument is to select what loss to use for training. If the third argument is empty, the loss is simply L2 loss. If it's __exactvalue__, an extra L2 loss term is added which ensures that the solution is exactly satisfied at random discrete points. Finally, if it's __zerointegral__, then an extra L2 loss term is added which ensures that the integral of the solution over the domain is zero.
*Note that the third argument is applicable only to __test_autodiff_sphere.py__*

To generate a combined plot for all different losses, use the following command
```sh
python3 plot_all.py [1 | 2]
```

## Pretrained models
We test using two different functions each in 2D and on the sphere. The pretrained models are stored in the corresponding directories. The models are trained using the `train` command.
The directory also contains the convergence plots using the `plot` command on the trained models.

