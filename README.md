# CSS-490a-HW2
Implementing the VGG-16 neural network and training on a Rock-Paper-Scissors dataset.

## Dataset
Rock Paper Scissors Dataset
https://laurencemoroney.com/datasets.html#rock-paper-scissors-dataset
* 300x300 pixels
* 24-bit color
* Plain white background
* Image count:
  * 2525 train
  * 377 test
  * 33 validate

## Setup
```bash
# Set up Python environment
virtualenv --python $(which python3) venv
source venv/bin/activate
pip install -r requirements.txt

# Download dataset
sh download_dataset.sh
```

## Todo
Use the below file as reference to make my code object oriented.

https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py

## References
* https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
