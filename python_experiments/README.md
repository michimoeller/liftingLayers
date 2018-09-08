Lifting Layers: Python Experiments
================

Installation
-------------------

1. Install the following packages (preferably in a Python virtual environment) for Python 3.6:
    1. Execute `pip3 install -r requirements.txt`.
    3. Execute ` pip3 install https://github.com/timmeinhardt/pytorch_tools/archive/lifting_layers.zip`.
    2. Install PyTorch 0.3.1 for your CUDA version from [the offical webpage](https://pytorch.org/previous-versions/).
2. Load denoising datasets from [here](http://vision.in.tum.de/_media/downloads/lifting_layers_denoising.zip) and unzip the `*.h5` files to the `data/denoising` directory.
3. (**Optional**, for storing experiments in a database) Install MongoDB and PyMongo and start a MongoDB daemon (`mongod`).

Usage - Maxout (MNIST classification) Experiments
-------------------

1. Start Tensorboard with `tensorboard --logdir=logs` (default: [localhost:6006](localhost:6006)).
2. Run experiment for one of three activation functions (`RELU`, `LIFT`, `MAXOUT`) by executing for example `python src/train_mnist.py LIFT`.

Usage - Denoising Experiments
-------------------

1. Start visdom server with `python -m visdom.server -env_path logs/visdom` (default: [localhost:8097](localhost:8097)).
2. Test an existing model with: `python src/train_denoising.py with seed=1 test_model_path=models/25_dncnn_s_17_best_val.model`
3. Train a new model with: `CUDA_VISIBLE_DEVICES=0 python src/train_denoising.py with seed=1 model_name=LiftNet nn_train.data_cfg.noise.stddev=25.0`.
4. See `config/train_denoising.yaml` or execute `python src/train_denoising.py print_config` for further configuration options.

Layer Implementation
--------------------

We included the Python implementation of the scaled lifting for three labels (`L=3`) as described in the paper in Section 3.4 and Equation 12. The general form of a lifting layer is so far only available in our MATLAB implementation.

