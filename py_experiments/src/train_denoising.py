""""""
from math import log10
import numpy as np
import torch

from sacred import Experiment
from sacred.observers import MongoObserver

from pytorch_tools import Solver
from pytorch_tools.vis import ImgVis, LineVis
from pytorch_tools.ingredients import (MONGODB_PORT, torch_ingredient,
                                       save_model_to_db, save_model_to_path)

from data import init_data_loaders
from models import LiftNet, DNCNN


ex = Experiment('train_denoising', ingredients=[torch_ingredient])
ex.add_config('config/train_denoising.yaml')

if MONGODB_PORT is not None:
    ex.observers.append(MongoObserver.create(db_name='network_lifting',
                                             port=MONGODB_PORT))

def psnr(output, target, data):
    """
    : returns: number of correct predictions
    : rtype: torch.IntTensor of size 1
    """
    # img = noisy_img - noise
    img = data - target
    # pred_img = noisy_img - pred_noise
    pred_img = torch.clamp(data - output, 0.0, 1.0)
    mse = torch.pow(img - pred_img, 2).mean()
    return torch.Tensor([10 * log10(1.0 / mse)])


def mse_batch_loss(output, target):
    return torch.nn.MSELoss(size_average=False)(output, target).div(output.size()[0] * 2)


@ex.automain
def main(model_name, save_model_path, vis_env_name, kaiming_init, vis, test_model_path,
         nn_train, torch_cfg, seed, _run, _log):
    # random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # data
    train_loader, val_loader, test_loader, (channels, _, _) = \
        init_data_loaders(nn_train['data_cfg'], torch_cfg, logger=_log.info)
    init_test_psnr = torch.cat([psnr(torch.zeros(noise.size()), noise, noisy_img)
                                for (noisy_img, noise) in test_loader]).mean()

    solver_cfg = nn_train['solver_cfg']
    Solver.compute_acc = staticmethod(psnr)
    solver = Solver(solver_cfg['optim'],
                    solver_cfg['optim_kwargs'][solver_cfg['optim']],
                    solver_cfg['ada_lr'],
                    solver_cfg['early_stopping'],
                    solver_cfg['epochs'],
                    mse_batch_loss, )

    # only test
    if test_model_path is not None:
        solver.best_val_model = torch.load(test_model_path, map_location=lambda s, l: s)
        if torch_cfg['cuda']:
            solver.best_val_model.cuda()
        _, test_psnr = solver.test(test_loader)
        _log.info(f"INIT/FINAL TEST PSNR: {init_test_psnr:.3f}/{test_psnr[0]:.3f}")
        return

    #
    # vis
    #
    if vis:
        # if MongoDB observer is available the vis env name is set according to
        # the database/run id
        if _run._id is not None:
            vis_env_name = f"{_run._id}_{_run.experiment_info['name']}"

        train_opts = dict(title=f"TRAIN METRICS", xlabel="EPOCHS",
            width=700, legend=["BATCH LOSS", "BATCH PSNR", "LR"],)
        test_opts = dict(title=f"VAL METRICS", xlabel="EPOCHS", ylabel='PSNR',
            width=700, legend=["PSNR", "EPOCH TIME"],)
        train_vis = LineVis(train_opts, env=vis_env_name)
        test_vis = LineVis(test_opts, env=vis_env_name)
        test_img_vis = ImgVis(dict(title="TEST IMAGES", width=700), env=vis_env_name)
        train_img_vis = ImgVis(dict(title="TRAIN IMAGES", width=700, height=100), env=vis_env_name)

        train_vis.plot([0.0, 0.0, 0.0], 0)
        test_vis.plot([init_test_psnr, 0.0], 0)

        # vis first image example
        for (noisy_img, noise) in test_loader:
            test_img_vis.plot([noisy_img - noise, noise, torch.clamp(noisy_img, 0.0, 1.0)])
            break
        for (noisy_img, noise) in train_loader:
            train_img_vis.plot([noisy_img[0] - noise[0], noise[0], torch.clamp(noisy_img[0], 0.0, 1.0)])
            break

    # model and solver
    if model_name == "DnCNN-S-17":
        model = DNCNN(channels, num_blocks=15, features=64, kaiming_init=kaiming_init)
    elif model_name == "DnCNN-S-20":
        model = DNCNN(channels, num_blocks=18, kaiming_init=kaiming_init)
    elif model_name == "LiftNet":
        model = LiftNet(channels, features=46, kaiming_init=kaiming_init)
    else:
        raise NotImplementedError
    if torch_cfg['cuda']:
        model.cuda()
    _log.info(f"NUM MODEL PARAMS: {model.num_params}")

    # callbacks
    def train_vis_callback(solver, batch_loss, batch_acc, data, target, output, batch_id):
        if vis:
            noisy_img, noise, pred_noise = data, target, output

            # vis every iteration slows down training
            if not batch_id % 20:
                run_epochs = batch_id * len(data) / train_loader.num_samples + solver.trained_epochs
                train_vis.plot([batch_loss, batch_acc, solver.optim.param_groups[0]['lr']], run_epochs)
                train_img_vis.plot([noisy_img[0] - noise[0],
                                    noise[0],
                                    torch.clamp(noisy_img[0], 0.0, 1.0),
                                    pred_noise[0],
                                    torch.clamp((noisy_img - pred_noise)[0], 0.0, 1.0), ])

    def vis_callback(solver, epoch, epoch_time):
        save_model_to_path(model, f"epoch_{epoch}.model", save_model_path)
        save_model_to_path(solver.best_val_model, f"best_val.model", save_model_path)
        save_model_to_db(model, f"epoch_{epoch}.model", ex)
        save_model_to_db(solver.best_val_model, f"best_val.model", ex)
        if vis:
            _, _, _, test_psnr = solver.last_metrics
            test_vis.plot([test_psnr, epoch_time], epoch)

            test_img_vis.close()
            model.eval()
            for (noisy_img, noise) in test_loader:
                noisy_img, noise = solver._data_loader_to_variables(model, noisy_img, noise, volatile=True)
                pred_noise = model(noisy_img)
                test_img_vis.plot([noisy_img - noise,
                                   noise,
                                   torch.clamp(noisy_img, 0.0, 1.0),
                                   pred_noise,
                                   torch.clamp(noisy_img - pred_noise, 0.0, 1.0), ])
                break
            model.train()

    # train
    solver.init_optim(model)
    solver.train_val(model, train_loader, val_loader,
                     vis_callback=vis_callback,
                     train_vis_callback=train_vis_callback,
                     reinfer_train_loader=False)

    # test
    _, test_psnr = solver.test(test_loader)
    _log.info(f"FINAL TEST PSNR: {test_psnr[0]:.3f}")

