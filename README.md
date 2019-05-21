# OOGAN based on vanilla GAN

1. code: oogan_models.py, oogan_modules.py
2. to run the code, 
    first edit the "config.py" for all hyperparameter settings, templetes are provided inside
    then run "python train.py"
   the training will print log on terminal and save the generated models and images during training.

# OOGAN based on styleGAN

1. code: ./oogan_stylegan/oo_stylegan_train.py, 
         ./oogan_stylegan/oo_stylegan_modules.py
2. to run the code, run "python oo_stylegan_train.py /path/to/images"
   the training will print log on terminal and save the generated models and images during training.

## dependent libraries
cuda 10.1
python 3.7
-python libraries:
 --pytorch 1.1.0
 --torchvision
 --math
 --scipy
 --numpy
 --numbers
 --tqdm


