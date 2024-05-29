import logging
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from utils import load_config
from transformers import default_data_collator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score

from accelerate import notebook_launcher
import gc
import wandb
from datasetevi import load_data
from model import ClaimVerification
from loss import focal_loss
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
def load_models(config):
        model = ClaimVerification(config)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        count_parameters(model)
        
        return model, tokenizer
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids, 'attention_mask': attention_mask}

def setting_optimizer(config):
    # Initialize the optimizer
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        # https://huggingface.co/docs/bitsandbytes/main/en/optimizers

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    return optimizer_cls
def main():
    
    logger = get_logger(__name__, log_level="INFO")
    
    ## config global
    path_config  = "./config.yaml"
    
    config = load_config(path_config)

    wandb.login(key=config.wandb['key_wandb'])

    logging_dir = os.path.join(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        
        log_with=config.report_to,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        # gradient_accumulation_plugin=gradient_accumulation_plugin,
        project_config=accelerator_project_config,
    )
    
    accelerator.init_trackers(
        project_name = config.wandb['project'],
        init_kwargs={"wandb": {"entity": "davidnguyen", 'tags': config.wandb['tags'], 'name': config.wandb['name']}}
        
    )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)


    model, tokenizer = load_models(config)
    

    
    optimizer_cls = setting_optimizer(config=config)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    train_dataset, test_dataset = load_data(config, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=config.train_batch_size,
    )
    
    eval_dataloader = DataLoader(
        test_dataset,
        collate_fn=default_data_collator,
        batch_size=config.train_batch_size
    )
    lambda1 = lambda epoch: 1/math.sqrt(epoch + 1) 
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    
    if config.path_fineturn_model:
        print(f"Update weight {config.path_fineturn_model}")
        accelerator.load_state(config.path_fineturn_model)
        # load_model(model, f"{config.path_fineturn_model}/model.safetensors")
        # Clean memory
        torch.cuda.empty_cache()
        gc.collect()
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )


    print("Running trainings")
    global_step = 0

    initial_global_step = 0
    
    progress_bar = tqdm(
        # range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    loss_fn = focal_loss(config.alpha, config.gamma, config.reduction )
    min_f1 = 0
    
    for epoch in range(config.num_train_epochs):

        model.train()
        
        train_loss = 0.0
        eval_loss = 0.0
        true_labels_train = []
        predicted_labels_train = []
        true_labels_eval = []
        predicted_labels_eval = []
        for step, batch in enumerate(train_dataloader):
     
            with accelerator.accumulate(model):
                
               
                y_true = batch['targets'].to(accelerator.device)
                targets = F.one_hot(batch['targets'], 2)
                input_ids = batch['input_ids'].to(accelerator.device)
                attention_mask = batch['attention_masks'].to(accelerator.device)
                targets = targets.to(accelerator.device, dtype=torch.float)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss = loss_fn(outputs, targets)
                
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                
                _, pred = torch.max(outputs, dim=1)
                true_labels_train.extend(y_true.cpu().numpy())
                predicted_labels_train.extend(pred.cpu().numpy())
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                
                optimizer.zero_grad()
                
            
            logs = {"step": f",{step}/{len(train_dataloader)}", "step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

           
            global_step+=1
            # if step == 1: break
        lr_scheduler.step()
        
        train_loss = round(train_loss/len(train_dataloader), 4)

        model.eval()
        predictions = []
        for step, batch in enumerate(eval_dataloader):
     
            with torch.no_grad():
                
                y_true = batch['targets'].to(accelerator.device)
                targets = F.one_hot(batch['targets'], 2)

                input_ids = batch['input_ids'].to(accelerator.device)
                attention_mask = batch['attention_masks'].to(accelerator.device)
                targets = targets.to(accelerator.device, dtype=torch.float)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                _, pred = torch.max(outputs, dim=1)

                loss = loss_fn(outputs, targets)
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                eval_loss += avg_loss.item() / config.gradient_accumulation_steps

                true_labels_eval.extend(y_true.cpu().numpy())
                predicted_labels_eval.extend(pred.cpu().numpy())
                
        f1_train = f1_score(true_labels_train, predicted_labels_train)
        acc_train = accuracy_score(true_labels_train, predicted_labels_train)
        
        f1_eval = f1_score(true_labels_eval, predicted_labels_eval)
        acc_eval = accuracy_score(true_labels_eval, predicted_labels_eval)
        
        accelerator.log(
            {
                
                'Train loss': train_loss, 
                "lr_current": optimizer.param_groups[0]['lr'],
                "Eval loss": eval_loss,
                "Eval F1": f1_eval,
                "Eval Acc": acc_eval,
                "Train F1": f1_train,
                "Train Acc": acc_train,
                # 'Test loss': test_loss
            },
            step=epoch
        )
        
        if f1_eval >= min_f1:
            save_path = os.path.join(config.output_dir, f"best")
            accelerator.save_state(save_path)
            min_f1 = f1_eval
            print("Save model")
            
            
        print({
                'epoch':epoch, 
                'Train loss': train_loss, 
            })

        train_loss = 0.0
    accelerator.end_training()


if __name__ == "__main__":
    # notebook_launcher()
    # main()
    notebook_launcher(main, args=(), num_processes=2)

