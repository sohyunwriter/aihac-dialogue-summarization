import os
import pytorch_lightning as pl
import hydra
import torch
import pathlib
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import _METRIC
from transformers import (
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    get_cosine_schedule_with_warmup,
    AutoModel,
    PegasusForConditionalGeneration, 
    PegasusTokenizer,
    PegasusConfig,
    MBartConfig,
    MBartForConditionalGeneration,
)

from transformers.tokenization_utils_base import BatchEncoding

import logging
from pathlib import Path
import nsml
from typing import Dict

from dialogue_summarization.datasets import SummarizationDataset, text_infil


import warnings
warnings.filterwarnings(action='ignore')

loss_info = {}
loss_info["val"]={"loss":0}
loss_info["train"]={"loss":0}


val_batch_sz = 0
tr_batch_sz = 0

def nsml_save(filepath, **kwargs):
    print(os.path.join(filepath))
    trainer.save_checkpoint(os.path.join(filepath, "model.ckpt"), False)


class NsmlModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        filepath = self.CHECKPOINT_NAME_LAST
        filepath = Path(str(filepath)).stem
        print(filepath)
        epoch = trainer.current_epoch
        global loss_info
        global val_batch_sz
        global tr_batch_sz


        if tr_batch_sz !=0:
            loss_info["train"]["loss"] = loss_info["train"]["loss"]/tr_batch_sz
            if val_batch_sz != 0:
                loss_info["val"]["loss"] = loss_info["val"]["loss"]/val_batch_sz
                print("logging....")
                print("mean_loss : ", loss_info)
            else:
                print("logging....")
                print("mean_loss : ", loss_info["train"]["loss"])
                
            nsml.report(epoch=epoch,summary=True,**loss_info)

        # nsml.report(epoch=epoch,summary=True,**loss_info)
        # epoch_name = str(epoch)+"_"+ str(round(loss_info["train"]["loss"],3))
        nsml.save(
            epoch, save_fn=nsml_save)

        loss_info["train"]["loss"] = 0
        loss_info["val"]["loss"] = 0
        tr_batch_sz = 0
        val_batch_sz = 0


nsml.bind(save = nsml_save)


class SummarizationModel(pl.LightningModule):
    def __init__(self, config_path, config_name, train_data=None, valid_data=None,pre_model=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.config = self.load_config(config_path, config_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.train_data = train_data
        self.valid_data = valid_data
        global loss_info
        global val_batch_sz
        global tr_batch_sz

        print ("using kkangtong")
        self.model = pre_model 

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        global loss_info
        global val_batch_sz
        global tr_batch_sz
        loss_info["train"]["loss"]+=loss.item()
        tr_batch_sz +=1
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        global loss_info
        global val_batch_sz
        global tr_batch_sz
        loss_info["val"]["loss"]+=loss.item()
        val_batch_sz += 1
        return loss

    def fit(self) -> None:
        trainer_options = {
            "callbacks": [
                LearningRateMonitor(
                    logging_interval="step",
                    log_momentum=False,
                ),
                ModelCheckpoint(
                    monitor=self.config.save_monitor,
                    dirpath=self.config.save_path,
                    filename="model.epoch={epoch:02d}.loss={valid_loss:.3f}",
                    save_top_k=self.config.num_save_ckpt,
                    mode=self.config.save_mode,

                    save_weights_only=True
                ),
                NsmlModelCheckpoint()
            ],
        }
        if self.config.deepspeed is not None:
            deepspeed_config = self.config.deepspeed
            deepspeed_config["train_batch_size"] = self.config.batch_size
            trainer_options["plugins"] = DeepSpeedPlugin(deepspeed_config)

        global trainer
        trainer = Trainer(
            **self.config.trainer_args,
            **trainer_options,
        )


        train_dataset = SummarizationDataset(
            # filename=self.config.train_data_path,
            datafile=self.train_data,
            separator=self.config.separator,
            meta_sep=self.config.meta_sep,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,  ## 모델 뺌
            strategy=self.config.strategy,
        )

        valid_dataset = SummarizationDataset(
            # filename=self.config.valid_data_path,
            datafile=self.valid_data,
            separator=self.config.separator,
            meta_sep=self.config.meta_sep,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,  ## 모델 뺌
            strategy=self.config.strategy,
            is_val=1,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.config.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
        )

        trainer.fit(
            model=self,
            train_dataloader=train_dataloader,
            val_dataloaders=valid_dataloader,
            #train_dataset=train_dataset,
            #valid_dataset=valid_dataset,
            #data_collator=data_collator,
        )

    def collate_fn(self, batch):
        # batch["input_ids"], batch["labels"] = text_infil(
        #     batch["input_ids"], special_tokens_mask=None
        # )
        #print(batch)
        input_ids = [batch[i]["input_ids"] for i in range(len(batch))]
        labels = [batch[i]["labels"] for i in range(len(batch))]

        inputs, _ = text_infil(
            self.tokenizer,
            input_ids, 
            special_tokens_mask=None
        )

        # encoder = self.tokenizer.pad(
        #     {"input_ids": input_ids},
        #     padding="max_length",
        #     max_length=self.config.max_length,  # 모델 뺌
        # )
        
        encoder = self.tokenizer.pad(
            {"input_ids": BatchEncoding({"input_ids" : inputs})["input_ids"]},
            padding="max_length",
            max_length=self.config.max_length,  # 모델 뺌
        )

        #encoder = self.tokenizer({"input_ids": BatchEncoding({"input_ids" : input_ids})['input_ids']}, padding=True, truncation=True, return_tensors="pt")
        decoder = self.tokenizer.pad(
            {"input_ids": labels},
            padding="max_length",
            max_length=self.config.max_length, # 모델 뺌
        )

        return {
            "input_ids": encoder["input_ids"],
            "attention_mask": encoder["attention_mask"],
            "labels": decoder["input_ids"],
        }

    def configure_optimizers(self):
        self.optimizer = AdamW(
            self.optimized_params(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.sechduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.num_training_steps
            * self.config.num_warmup_steps,
            num_training_steps=self.config.num_training_steps,
        )

        return [self.optimizer], [self.sechduler]

    def optimized_params(self):
        no_decay = ["bias", "LayerNorm.weight"]
        model = self.model
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_grouped_parameters

    @staticmethod
    def load_config(path: str, name: str) -> DictConfig:
        GlobalHydra().clear()
        hydra.initialize(path, name)
        return hydra.compose(name)
