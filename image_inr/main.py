import copy
import os
import random
import warnings
from pydoc import locate

import dataloader
import einops
import hydra
import lightning as L
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from utils import data_process, helper

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


for k, v in os.environ.items():
    if "slurm" in k.lower():
        os.environ.pop(k, None)


def get_loader(cfg, val=False):

    img_dataset = dataloader.single_image_dataset(cfg)
    # create torch dataset for one image.
    loader = DataLoader(img_dataset, batch_size=1, shuffle=False, num_workers=0)
    return loader


def get_trainer(cfg, logger, callbacks):

    trainer = L.Trainer(
        accelerator="gpu",
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.num_iters,
        default_root_dir=cfg.logging.checkpoint.logdir,
        fast_dev_run=cfg.trainer.debug,
        logger=logger,
        enable_checkpointing=(not cfg.logging.checkpoint.skip_save_model),
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.trainer.eval_every,
        devices=cfg.trainer.devices,
    )
    return trainer


def get_ckpt_callback(cfg, path=None):

    checkpoint_callback = ModelCheckpoint(
        monitor="val_psnr",
        dirpath=cfg.logging.checkpoint.logdir + "/",
        save_top_k=1,
        mode="max",
        filename="best_model",
    )

    return checkpoint_callback


class lightning_model(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        model_name = self.cfg.network.model_name
        self.model_class = locate("models." + model_name + "." + model_name)
        self.model = self.model_class(cfg=self.cfg)
        self.coords = None

        self.proc = data_process.DataProcessor(
            self.cfg.data, device="cpu"
        )  # dataloading on cpu
        self.size = self.cfg.network.sidelength
        self.load_loss()

        self.log_dir = self.cfg.logging.checkpoint.logdir

        self.img_save_folder = (
            self.cfg.logging.checkpoint.logdir + "/" + "reconstructions/"
        )
        helper.make_dir(self.img_save_folder)

    def load_loss(self):
        loss_cfg = self.cfg.trainer.losses
        loss_list = loss_cfg.loss_list
        self.loss_functions = {}
        self.loss_weights = {}

        for k in loss_list:
            if "vae" in k:
                loss_list.remove(k)
                loss_list += ["mse", "kl_loss"]
                break

        for idx, loss in enumerate(loss_list):
            loss_name = loss.lower()
            loss_class = locate("losses." + loss_name)

            loss_func = loss_class(self.cfg)
            self.loss_weights[loss_name] = loss_cfg.loss_weights[idx]
            self.loss_functions[loss_name] = loss_func

            print(
                "Using loss function : ",
                loss_name,
                " with weight: ",
                self.loss_weights[loss_name],
            )

    def apply_loss(self, x, pred, **kwargs):

        loss = 0
        for k, v in self.loss_functions.items():
            loss_weight = self.loss_weights[k]
            loss += v(x, pred) * loss_weight
        return loss

    def forward(self, batch, kwargs, reshape=False):
        data = batch["data"]
        features = batch["features"].squeeze()
        features_shape = batch["features_shape"].squeeze().tolist()

        # Image not used in most architectures. check model file.
        out = self.model(self.coords, img=data, **kwargs)
        pred = out["predicted"]

        loss = self.apply_loss(features, pred.squeeze(), **kwargs)
        psnr = helper.get_clamped_psnr(features, pred)

        if reshape:
            if self.cfg.data.patch_shape is not None:
                pred = pred.view(-1, 3, *self.cfg.data.patch_shape)
                N, C, H, W = data.shape
                num_patches_h = H // self.cfg.data.patch_shape[0]
                num_patches_w = W // self.cfg.data.patch_shape[1]
                pred = einops.rearrange(
                    pred,
                    "(h w) c ph pw -> c (h ph) (w pw)",
                    h=num_patches_h,
                    w=num_patches_w,
                )
                pred = self.proc.unpad(pred, (H, W))
            else:
                pred = self.proc.process_outputs(
                    pred,
                    input_img_shape=batch["data_shape"].squeeze().tolist(),
                    features_shape=features_shape,
                    patch_shape=self.cfg.data.patch_shape,
                )

        return loss, psnr, pred

    def training_step(self, batch, batch_idx):
        x = batch["data"]
        features_shape = batch["features_shape"].squeeze().tolist()

        kwargs = {}

        if batch_idx == 0 or self.coords is None:
            self.coords = self.proc.get_coordinates(
                data_shape=features_shape,
                patch_shape=self.cfg.data.patch_shape,
                split=self.cfg.data.coord_split,
                normalize_range=self.cfg.data.coord_normalize_range,
            )
            self.coords = self.coords.to(x)

        loss, psnr, pred = self.forward(batch, kwargs)

        self.log_dict(
            {"train_loss": loss, "train_psnr": psnr},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        x = batch["data"]
        features_shape = batch["features_shape"].squeeze().tolist()

        kwargs = {}

        if batch_idx == 0 or self.coords is None:

            self.coords = self.proc.get_coordinates(
                data_shape=features_shape,
                patch_shape=self.cfg.data.patch_shape,
                split=self.cfg.data.coord_split,
                normalize_range=self.cfg.data.coord_normalize_range,
            )
            self.coords = self.coords.to(x)

        if (
            (self.current_epoch + 1) % self.cfg.logging.checkpoint.save_every == 0
        ) and (batch_idx == 0):
            loss, psnr, pred = self.forward(batch, kwargs, reshape=True)

            pred = pred.squeeze().clamp(0, 1)
            pred_img = helper.tensor_to_cv(pred)
            helper.save_tensor_img(
                pred,
                self.img_save_folder
                + "/reconstruction_epoch_"
                + str(self.current_epoch)
                + ".png",
            )

            if self.cfg.logging.wandb.enabled:
                self.logger.log_image(
                    key="predictions_val", images=[pred_img], step=self.current_epoch
                )
        else:
            loss, psnr, pred = self.forward(batch, kwargs, reshape=False)

        self.log_dict(
            {"val_loss": loss, "val_psnr": psnr},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_fit_end(self):
        if self.current_epoch < self.cfg.trainer.eval_every:
            print("return")
            return

        best_score = self.trainer.checkpoint_callback.best_model_score.item()
        params = sum(p.numel() for p in self.parameters())
        info = {"best_psnr": best_score, "params": params}
        helper.dump_to_json(info, self.log_dir + "/info.json")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.trainer.lr,
            betas=self.cfg.trainer.optimizer.betas,
        )
        return optimizer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    torch.set_float32_matmul_precision("medium")

    # get overrides and merge.
    hydra_cfg = HydraConfig.get()
    hydra_info = {}
    if "id" in hydra_cfg.job:
        print("multi-run: ", hydra_cfg.job.id, hydra_cfg.mode)
        print("ouptut_dir: ", hydra_cfg.runtime.output_dir)
        hydra_info["hydra_id"] = hydra_cfg.job.id
    else:
        print("single-run: ", hydra_cfg.mode)
        print("ouptut_dir: ", hydra_cfg.runtime.output_dir)

    hydra_info["hydra_mode"] = str(hydra_cfg.mode)
    hydra_info["output_dir"] = hydra_cfg.runtime.output_dir

    if cfg.trainer.resume:
        print("Resuming from checkpoint.")
        save_dir = cfg.logging.checkpoint.logdir
        ckpt_path = helper.find_ckpt(save_dir)
        if os.path.exists(save_dir + "/exp_config.yaml"):
            exp_cfg = OmegaConf.load(save_dir + "/exp_config.yaml")
            print("Loaded existing config from: ", save_dir + "/exp_config.yaml")

            OmegaConf.resolve(exp_cfg)
            cfg = copy.deepcopy(exp_cfg)

            overrides = HydraConfig.get().overrides
            overrides = [str(item) for item in overrides.task]
            temp_cfg = OmegaConf.from_dotlist(overrides)

            # get new overrides and merge, for training args.
            cfg.trainer = OmegaConf.merge(exp_cfg.trainer, temp_cfg.trainer)
    else:
        ckpt_path = None

    helper.make_dir(cfg.logging.checkpoint.logdir)

    if cfg.common.seed == -1:
        cfg.common.seed = random.randint(0, 10000)

    if cfg.trainer.num_iters_first is None:
        cfg.trainer.num_iters_first = cfg.trainer.num_iters

    print("Seed: ", cfg.common.seed)
    torch.manual_seed(cfg.common.seed)
    torch.cuda.manual_seed_all(cfg.common.seed)

    # load data.
    save_folder = cfg.logging.checkpoint.logdir + "/" + "reconstructions/"
    helper.make_dir(save_folder)
    train_loader = get_loader(cfg)
    val_loader = get_loader(cfg, val=True)

    if val_loader is None:
        val_loader = train_loader

    # Update image sizes.
    batch = next(iter(train_loader))
    data = batch["data"]
    N, C, H, W = data.shape
    cfg.data.data_shape = (H, W)
    cfg.network.sidelength = max(data.shape)

    model = lightning_model(cfg)

    if cfg.logging.wandb.enabled:
        wandb_opt = cfg.logging.wandb
        wandb_config = helper.flatten_dictconfig(cfg)

        # just passing resume=true should not crash.
        if (cfg.trainer.resume) and (ckpt_path is not None):
            wandb_id = cfg.logging.wandb.id
            resume = "must"
        else:
            wandb_id = wandb.util.generate_id()
            resume = None  #'allow'

        logger = WandbLogger(
            project=wandb_opt.project,
            group=wandb_opt.group,
            save_dir=cfg.logging.wandb.dir,
            entity=wandb_opt.entity,
            config=wandb_config,
            resume=resume,
            id=wandb_id,
        )
        cfg.logging.wandb.id = wandb_id

    else:
        logger = TensorBoardLogger(
            save_dir=cfg.logging.checkpoint.logdir, name="tensorboard_logs"
        )

    # log model config. Used to infer trained models.
    OmegaConf.save(cfg, cfg.logging.checkpoint.logdir + "/exp_config.yaml")

    if cfg.logging.wandb.enabled:
        # log hydra info
        for key, value in hydra_info.items():
            logger.log_metrics({key: value})

    trainer = get_trainer(cfg, logger, callbacks=[get_ckpt_callback(cfg)])

    if cfg.trainer.eval_only:
        trainer.validate(model, val_loader)
        return

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    if cfg.logging.wandb.enabled:
        wandb.finish()

    print("FFN Model: \n", model.model)


if __name__ == "__main__":
    main()
