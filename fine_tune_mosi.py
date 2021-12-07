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
from transformers import AdamW, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from path import Path
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score

from dataset.utils import fetch_mosi_datasets
from models.teasel import TeaselFineTuneMOSI

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning Rate')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Set Warmup Proportion')
parser.add_argument('--num_epochs', default=5, type=int, help='Set Number of Epochs')
parser.add_argument('--batch_size', default=16, type=int, help='Set Batch Size')
parser.add_argument('--max_text_length', default=75, type=int, help='Set Max Text Length')
parser.add_argument('--max_time', default=30, type=int, help='Set Max Time')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'], help='Train Or Test Mode')
#parser.add_argument('--config', type=str, help='Path to the config.yaml file')

args = parser.parse_args()

def read_config(config_file, args):
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
    
    ## Override Args Values
    config['num_epochs'] = args.num_epochs
    config['MAX_TEXT_LENGTH'] = args.max_text_length
    config['MAX_TIME'] = args.max_time
    config['learning_rate'] = args.learning_rate
    config['warmup_proportion'] = args.warmup_proportion
    config['batch_size'] = args.batch_size
    
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
    model = TeaselFineTuneMOSI(config['pretrained_model_file'], device).to(device)

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
    gt_sentiment_scores = None
    pred_scores = None
    model.train()
    start_time = time.monotonic()
    for step, batch in enumerate(tqdm(train_dataloader)):
        audio_input, input_ids, attention_mask, gt_scores = batch
        audio_input = audio_input.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        gt_scores = gt_scores.reshape((-1,1)).to(device)

        output = model(audio_input, input_ids, attention_mask, gt_scores)

        loss = output.loss

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
            "train_batch": step
        })

        if pred_scores is None:
            pred_scores = output.logits.detach().to(torch.device('cpu'))
            gt_sentiment_scores = gt_scores.detach().to(torch.device('cpu'))
        else:
            pred_scores = torch.vstack((pred_scores, output.logits.detach().to(torch.device('cpu')))).to(torch.device('cpu'))
            gt_sentiment_scores = torch.vstack((gt_sentiment_scores, gt_scores.detach().to(torch.device('cpu')))).to(torch.device('cpu'))
    
    print("Training Batch time:", time.monotonic()-start_time)
    
    return training_loss / len(train_dataloader), gt_sentiment_scores, pred_scores

def validation_loop(val_dataloader, model, device, config):
    val_loss = 0
    gt_sentiment_scores = None
    pred_scores = None
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dataloader)):
            audio_input, input_ids, attention_mask, gt_scores = batch
            audio_input = audio_input.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            gt_scores = gt_scores.reshape((-1,1)).to(device)

            output = model(audio_input, input_ids, attention_mask, gt_scores)

            loss = output.loss

            if config['gradient_accumulation_step'] > 1:
                loss = loss / config['gradient_accumulation_step']

            val_loss += loss.item()

            wandb.log({
                'val_batch_loss': loss,
                'val_batch': step
            })

            if pred_scores is None:
                pred_scores = output.logits.detach().to(torch.device('cpu'))
                gt_sentiment_scores = gt_scores.detach().to(torch.device('cpu'))
            else:
                pred_scores = torch.vstack((pred_scores, output.logits.detach().to(torch.device('cpu')))).to(torch.device('cpu'))
                gt_sentiment_scores = torch.vstack((gt_sentiment_scores, gt_scores.detach().to(torch.device('cpu')))).to(torch.device('cpu'))
    
    return val_loss / len(val_dataloader), gt_sentiment_scores, pred_scores

def compute_metrics(pred_scores, gt_sentiment_scores):
    pred_scores = pred_scores.flatten().detach().numpy()
    gt_sentiment_scores = gt_sentiment_scores.flatten().detach().numpy()
    
    mae = mean_absolute_error(y_true=gt_sentiment_scores, y_pred=pred_scores)
    binary_acc = accuracy_score(y_true=gt_sentiment_scores>0, y_pred=pred_scores>0)
    f1 = f1_score(y_true=gt_sentiment_scores>0, y_pred=pred_scores>0)
    corr = np.corrcoef(pred_scores, gt_sentiment_scores)[0][1]
    
    return mae, binary_acc, f1, corr

def main():
    ## Create Wandb object
    config = read_config('config_mosi.yaml', args)#args.config)
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
    train_dataset, val_dataset, test_dataset = fetch_mosi_datasets(config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
    print("Datasets and DataLoaders Done! :D")
    
    config['num_train_optimization_steps'] = (int(len(train_dataset) / config['batch_size'] / config['gradient_accumulation_step']) * config['num_epochs'])
    
    #Wandb
    print("Initializing Wandb")
    wandb.init(project="teasel-mosi-fine-tune", entity='multi-modal-mosi', config=config)
    print(f"Wandb Initialized Run Name: {wandb.run.name}")
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
        tloss, gt_train_scores, train_preds = train_loop(train_loader, model, optimizer, scheduler, device, config)
        vloss, gt_val_scores, val_preds = validation_loop(val_loader, model, device, config)
        test_loss, gt_test_scores, test_preds = validation_loop(test_loader, model, device, config)
        
        tr_mae, tr_acc, tr_f1, tr_corr = compute_metrics(train_preds, gt_train_scores)
        
        val_mae, val_acc, val_f1, val_corr = compute_metrics(val_preds, gt_val_scores)
        
        test_mae, test_acc, test_f1, test_corr = compute_metrics(test_preds, gt_test_scores)
        
        print(f"Epoch: {epoch} Train Loss: {tloss} Valid_Loss: {vloss} Test Loss: {test_loss}")
        print(f"Train\tMAE: {tr_mae}\tAcc: {tr_acc}\tF1: {tr_f1}\tCorr: {tr_corr}")
        print(f"Valid\tMAE: {val_mae}\tAcc: {val_acc}\tF1: {val_f1}\tCorr: {val_corr}")
        print(f"Test\tMAE: {test_mae}\tAcc: {test_acc}\tF1: {test_f1}\tCorr: {test_corr}")
        
        ## Save Model
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': tloss,
                'train_mae': tr_mae,
                'train_acc': tr_acc,
                'train_corr': tr_f1,
                'train_f1': tr_corr,
                'validation_loss': vloss,
                'val_mae': val_mae,
                'val_acc': val_acc,
                'val_corr': val_corr,
                'val_f1': val_f1,
                'test_loss': test_loss,
                'test_mae': test_mae,
                'test_acc': test_acc,
                'test_corr': test_corr,
                'test_f1': test_f1,
            }, Path(config['model_output_dir']) / config['model_internal_dir'] / f'model_ep_{epoch}.pt')

        ## Log Metrics on Wandb
        wandb.log({
            'train_loss': tloss,
            'train_mae': tr_mae,
            'train_acc': tr_acc,
            'train_corr': tr_f1,
            'train_f1': tr_corr,
            'validation_loss': vloss,
            'val_mae': val_mae,
            'val_acc': val_acc,
            'val_corr': val_corr,
            'val_f1': val_f1,
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_acc': test_acc,
            'test_corr': test_corr,
            'test_f1': test_f1,
            'learning_rate': optimizer.param_groups[0]['lr'],
            "global_step": epoch,
        })

        print("Finished Training! :D")

if __name__=="__main__":
    main()