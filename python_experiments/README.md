Lifting Layers: Python Experiments
================

Installation
-------------------

1. Install the following packages (preferably in a Python virtual environment) for Python 3.6:
    1. Execute `pip3 install -r requirements.txt`.
    2. Install PyTorch from [the offical webpage](http://pytorch.org/).
2. Load datasets from [here](https://drive.google.com/drive/folders/14OUgEqUcCkZC26Zu_FPXurlpIZZmOXHV?usp=sharing) and extract them to the `data` directory.
3. (**Optional**, for storing experiments in a database) Install MongoDB and PyMongo and start a MongoDB daemon (`mongod`).

Usage - Maxout (MNIST classfication) Experiments
-------------------

1. Run experiment for one of three activation functions (`RELU`, `LIFT`, `MAXOUT`) by executing for example `python src/train_mnist.py LIFT`.

Usage - Denoising Experiments
-------------------

1. Start visdom server with `python -m visdom.server -env_path logs/visdom` at [localhost:8097](localhost:8097).
2. Test an existing model with: `python src/train_denoising.py with seed=1 test_model_path=models/25_dncnn_s_17_best_val.model`
3. Train a new model with: `CUDA_VISIBLE_DEVICES=0 python src/train_denoising.py with seed=1 model_name=LiftNet nn_train.data_cfg.noise.stddev=25.0`.
4. See `config/train_denoising.yaml` or execute `python src/train_denoising.py print_config` for further configuration options.
