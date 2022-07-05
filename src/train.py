import time, datetime, os, json
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_entry import GraphTextDataset
from models.my_model import MyModel
from global_variables import *
from utils import *


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == '__main__':
    seed_all()

    logger.info(f"Device: {config.device}")
    logger.info(f'Count of using GPUs: {torch.cuda.device_count()}')

    model = MyModel(config)
    model = model.to(config.device)
    
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.lr
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_dataset = GraphTextDataset(config, split='train')
    val_dataset = GraphTextDataset(config, split='val')
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config.batch_size
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=config.batch_size
    )

    if not config.debug:
        os.makedirs(join(RUNS_PATH, config.name), exist_ok=True)
        writer_path = join(RUNS_PATH, config.name)
        writer = SummaryWriter(writer_path)
        logger.info(f"Create a writer: {writer_path}")

        os.makedirs(join(CKPT_PATH, config.name), exist_ok=True)
        ckpt_path = join(CKPT_PATH, config.name)

        os.makedirs(join(TRAIN_STATS_PATH, config.name), exist_ok=True)

    total_t0 = time.time()
    save_t0 = time.time()
    training_stats = {}
    for epoch in range(1, config.max_epoch + 1):
        logger.info("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch, config.max_epoch))
        logger.info(f"Model name: {config.name}")
        logger.info(f'Train using {config.cuda_devices}')

        t0 = time.time()
        total_train_loss = 0
        train_correct = 0
        model.train()

        for step, data in enumerate(train_dataloader):
            graph, text, labels = data
            graph = graph.to(config.device)
            input_ids = text['input_ids'].to(config.device)
            attention_mask = text['attention_mask'].to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                graph=graph,
                return_dict=True
            )

            loss = outputs.loss
            logits = outputs.logits
            total_train_loss += loss.item()

            batch_correct = (logits.argmax(1) == labels).type(torch.int).sum().item()
            train_correct += batch_correct


            if step % 1000 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                logger.info(f"  Batch {step} of {len(train_dataloader)}. Elapsed {elapsed}")
                logger.info(f" Batch correct: {batch_correct} / {config.batch_size}")
    
            loss.backward()
            optimizer.step()

            if config.debug:
                break
        
        scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)      
        train_accuracy = train_correct / len(train_dataset)      
        
        training_time = format_time(time.time() - t0)

        logger.info("")
        logger.info(f"  Accuracy: {train_correct} / {len(train_dataset)} = {train_accuracy:.4f}")
        logger.info("  Average training loss: {0:.4f}".format(avg_train_loss))
        logger.info("  Total training loss: {0:.4f}".format(total_train_loss))
        logger.info("  Training epcoh took: {:}".format(training_time))

        if not config.debug:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Acc/train", train_accuracy, epoch)


        logger.info("")
        logger.info("Run Validation")

        t0 = time.time()
        model.eval()

        eval_correct = 0
        total_eval_loss = 0

        for data in val_dataloader:
            
            graph, text, labels = data
            graph = graph.to(config.device)
            input_ids = text['input_ids'].to(config.device)
            attention_mask = text['attention_mask'].to(config.device)
            labels = labels.to(config.device)
            
            with torch.no_grad():
                result = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    graph=graph,
                    return_dict=True
                )

            loss = result.loss
            logits = result.logits
                
            total_eval_loss += loss.item()

            eval_correct += (logits.argmax(1) == labels).type(torch.int).sum().item()
            

            if config.debug:
                break

        eval_accuracy = eval_correct / len(val_dataset)
        logger.info(f"  Accuracy: {eval_correct} / {len(val_dataset)} = {eval_accuracy:.2f}")

        avg_eval_loss = total_eval_loss / len(val_dataloader)
        
        validation_time = format_time(time.time() - t0)
        
        logger.info("  Validation Loss: {0:.4f}".format(avg_eval_loss))
        logger.info("  Total validation loss: {0:.4f}".format(total_eval_loss))
        logger.info("  Validation took: {:}".format(validation_time))

        training_stats[epoch] = {
            'avg train loss': avg_train_loss,
            'total train loss': total_train_loss,
            'train acc': train_accuracy,
            'avg eval loss': avg_eval_loss,
            'total eval loss': total_eval_loss,
            'eval acc': eval_accuracy,
        }

        if not config.debug:
            writer.add_scalar("Loss/val", avg_eval_loss, epoch)
            writer.add_scalar("Acc/val", eval_accuracy, epoch)

            with open(join(TRAIN_STATS_PATH, f"{config.name}.json"), 'w') as f:
                json.dump(training_stats, f)

            if True: # time.time() - save_t0 > 3600:
                save_t0 = time.time()

                model_name = join(ckpt_path, f"epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, model_name)
                logger.info("")
                logger.info(f"Save model as {model_name}")


if not config.debug:
    writer.close()
logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
