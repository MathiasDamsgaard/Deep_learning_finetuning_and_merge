from typing import Optional

import torch
import torch.nn as nn

import wandb
from loguru import logger
from tqdm import tqdm, trange

from matplotlib import pyplot as plt
from peft import LoraConfig # type: ignore
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

class ModelTrainer:
    """
    Class for managing training of the models in the project.
    
    The purpose of this class is to ensure modularization of the training
    """
    def __init__(self, 
                 model: Optional[nn.Module] = None,
                 dataloader: Optional[DataLoader] = None,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 epochs: Optional[int] = 1,
                 batch_size: Optional[int] = 32) -> None:
        
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        
    def train_model(self, do_profiling: Optional[bool] = False) -> tuple[int, float]:
        """Train the model on the dataset."""
        if do_profiling and self.model.device == "cuda":
            logger.info("Profiling the model.")
            profiler = torch.autograd.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ], with_stack=True, profile_memory=True)
            profiler.__enter__()
            
        elif do_profiling:
            profiler = torch.profiler.profile(with_stack=True, profile_memory=True)
            
        else:
            profiler = None
            
        epoch, loss = self._train_model(profiler)
        
        if profiler:
            profiler.__exit__(None, None, None)
            print("CUDA PROFILING")
            print(profiler.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=15))
            print("CPU PROFILING")
            print(profiler.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=15))
            profiler.export_stacks("profiler_stacks.txt", "self_cpu_time_total")
            
        logger.success(f"Training completed! Observed loss: {loss} at epoch {epoch}.")
        
        return epoch, loss
    
    def _train_model(self, profiler: Optional[torch.profiler.profile] = None) -> tuple[int, float]:
        """Train the model on the dataset."""
        # for epoch in trange(self.epochs, desc="Training model", unit="epoch"):
        for epoch in range(self.epochs):
            running_loss = 0.0
            no_batches = 0
            for batch in tqdm(self.dataloader, total=len(self.dataloader), desc=f"Epoch {epoch+1}/{self.epochs}"):
                inputs, labels = batch
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                # TODO: Maybe add gradient clipping here
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                running_loss += loss.item()
                no_batches += 1
                
                # For debugging, only process 5 batches
                # if no_batches >= 5:
                #     break
                
            if profiler:
                profiler.step()
            
        return epoch, running_loss / len(self.dataloader)
    
    def save_model(self, path: str = "models/demo_run.pth") -> None:
        """Save the trained model to the specified path."""
        torch.save(self.model.state_dict(), path)
        logger.success(f"Model saved to: {path}")