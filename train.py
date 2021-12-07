import os
import argparse
import datetime
import re
import time

import torch
import numpy as np
import random
import wandb
import yaml
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from path import Path

from dataset.utils import fetch_datasets
from models.teasel import TeaselPretrain

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config.yaml file')

args = parser.parse_args()

def read_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(config_file)
    
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=loader)
        config['MAX_AUDIO_LENGTH'] = config['MAX_TIME'] * config['SAMPLING_RATE']
    
    return config

def set_random_seeds(seed):
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fetch_model_optim_sched(config, device):
    model = TeaselPretrain(device).to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_proportion'] * config['num_train_optimization_steps'],
        num_training_steps=config['num_train_optimization_steps'],
    )

    return model, optimizer, scheduler

def train_loop(train_dataloader, model, optimizer, scheduler, device, config):
    training_loss = 0
    model.train()
    start_time = time.monotonic()
    for step, batch in enumerate(tqdm(train_dataloader)):
        audio_input, masked_text_input, attention_mask, gt_text_input = batch
        audio_input = audio_input.to(device)
        masked_text_input = masked_text_input.to(device)
        attention_mask = attention_mask.to(device)
        gt_text_input = gt_text_input.to(device)

        maskedlm_output = model(audio_input, masked_text_input, attention_mask, gt_text_input)

        loss = maskedlm_output.loss

        if config['gradient_accumulation_step'] > 1:
            loss = loss / config['gradient_accumulation_step']

        loss.backward()

        training_loss += loss.item()

        if (step + 1) % config['gradient_accumulation_step'] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        wandb.log({
            'train_batch_loss': loss,
        })
    
    print("Batch time:", time.monotonic()-start_time)
    
    return training_loss / len(train_dataloader)

def validation_loop(val_dataloader, model, optimizer, scheduler, device, config):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            audio_input, masked_text_input, attention_mask, gt_text_input = batch
            audio_input = audio_input.to(device)
            masked_text_input = masked_text_input.to(device)
            attention_mask = attention_mask.to(device)
            gt_text_input = gt_text_input.to(device)

            maskedlm_output = model(audio_input, masked_text_input, attention_mask, gt_text_input)

            loss = maskedlm_output.loss

            if config['gradient_accumulation_step'] > 1:
                loss = loss / config['gradient_accumulation_step']

            val_loss += loss.item()

            wandb.log({
                'val_batch_loss': loss,
            })
    
    return val_loss / len(val_dataloader)

def main():
    ## Create Wandb object
    wandb.init(project="teasel-pretraining", entity='multi-modal-mosi')
    config = read_config(args.config)
    ## Set All Random Seeds
    set_random_seeds(config['seed'])

    ##Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ",device)

    ##Create Output Directory if it doesn't exists
    config['model_internal_dir'] = datetime.datetime.now().strftime('%m%d%Y_%H%M%S')
    if not os.path.exists(config['model_output_dir']):
        os.mkdir(config['model_output_dir'])
    
    os.mkdir(Path(config['model_output_dir']) / config['model_internal_dir'])

    print("Model Dir: ", Path(config['model_output_dir']) / config['model_internal_dir'])

    ## Create All Data Loaders
    print("Preparing Datasets and DataLoaders")
    train_dataset, val_dataset, _ = fetch_datasets(config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
    
    print("Datasets and DataLoaders Done! :D")
    
    config['num_train_optimization_steps'] = (int(len(train_dataset) / config['batch_size'] / config['gradient_accumulation_step'])* config['num_epochs'])
    
    wandb.config.update(config)
    ## Fetch Model, Optimizer and Scheduler for Training
    print("Creating Model")
    model, optimizer, scheduler = fetch_model_optim_sched(config, device)
    print("Model Created! :D")

    wandb.watch(model, log_freq=100)

    ## Training Loop Over Epochs
    print("Starting Training")
    for epoch in range(config['num_epochs']):
        print(f"Epoch: {epoch}")
        ## Train Loop on train set
        train_loss = train_loop(train_loader, model, optimizer, scheduler, device, config)
        
        ## Eval Loop on validation set
        val_loss = validation_loop(val_loader, model, optimizer, scheduler, device, config)
        
        ## Save Model
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'last_train_loss': train_loss,
                'last_val_loss': val_loss,
            }, Path(config['model_output_dir']) / config['model_internal_dir'] / f'model_ep_{epoch}.pt')

        ## Log Metrics on Wandb
        wandb.log({
            'train_loss': train_loss,
            'validation_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

        print("Finished Training! :D")

if __name__=="__main__":
    main()