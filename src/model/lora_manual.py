import os
import random
from typing import Tuple, Dict, Optional

import bitsandbytes as bnb
import numpy as np
import torch
import wandb
import tabulate
from loguru import logger
from ptflops import get_model_complexity_info
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.optimizers import create_loraplus_optimizer
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from transformers import (
    ViTForImageClassification,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

from src.config.config import *
from src.model.baseline_model import load_dataset, load_model

class LoRATrainer:
    """
    A custom trainer class for training and evaluating models, including saving, loading, and inference.

    Attributes:
        model (ViTForImageClassification): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training and evaluation.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
    """
    def __init__(self, model: ViTForImageClassification, 
                 optimizer: torch.optim.Optimizer, 
                 train_dataset: torch.utils.data.Dataset, 
                 val_dataset: torch.utils.data.Dataset, 
                 epochs: int, 
                 batch_size: int = 8, 
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 type_: str = "baseline",
                 do_profiling: bool = False) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(self.device)
        self.epoch_accuracies = []
        self.epoch_losses = []
        self.scheduler = scheduler
        self.type_ = type_
        self.do_profiling = do_profiling
        self.save_model_available = False

    def train(self) -> None:
        """
        Trains the model on the training dataset for a specified number of epochs.
        """
        if self.do_profiling:
            self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=1, wait=1, warmup=1, active=2, repeat=2
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("hf-training-trainer"),
            profile_memory=True,
            with_stack=True,
            record_shapes=True)
            
            self.profiler.__enter__()
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc=f"Training {self.epochs} epochs"):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(pixel_values)
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * pixel_values.size(0)
                preds = outputs.logits.argmax(dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)
                
            if self.scheduler:
                self.scheduler.step()
                
            if self.do_profiling:
                self.profiler.step()

            epoch_loss = running_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples
            self.epoch_losses.append(epoch_loss)
            self.epoch_accuracies.append(epoch_accuracy)

            wandb.log({"epoch_loss": epoch_loss, "epoch_accuracy": epoch_accuracy, "epoch": epoch})

            eval_results = self.evaluate(in_train=True, epoch_num=epoch)
        
        if self.do_profiling:
            self.profiler.__exit__(None, None, None)

    def evaluate(self, in_train: bool = False, epoch_num: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluates the model on the validation dataset.

        Args:
            in_train (bool): Whether the model is in training mode.

        Returns:
            Dict[str, float]: A dictionary containing evaluation loss and accuracy.
        """
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        running_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(pixel_values)
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
                running_loss += loss.item() * pixel_values.size(0)

                preds = outputs.logits.argmax(dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

        eval_loss = running_loss / total_samples
        eval_accuracy = correct_predictions / total_samples
        if in_train:
            self.model.train()
        
        if epoch_num is not None:
            wandb.log({"eval_loss": eval_loss, "eval_accuracy": eval_accuracy, "epoch": epoch_num})
            return {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy, "epoch": epoch_num}
        else:
            wandb.log({"eval_loss": eval_loss, "eval_accuracy": eval_accuracy})
            return {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
    
    @staticmethod
    def load_model(path: str, BM: bool = False) -> None:
        """
        Load a trained model from a specified path.

        Args:
            path (str): The path to load the model from.
        """
        model = ViTForImageClassification.from_pretrained(path).to(DEVICE)
    
        # Change id2label to match the new number of classes
        train_df = pd.read_csv(TRAIN_CSV)

        # Extract unique labels
        unique_labels = train_df['Labels'].unique()

        # Create mappings
        id2label = {int(idx): str(label) for idx, label in enumerate(unique_labels)}
        label2id = {str(label): int(idx) for idx, label in enumerate(unique_labels)}

        # Update model classifier layer and mapping functions
        model.config.num_labels = len(id2label)
        model.num_labels = len(id2label)
        model.classifier = nn.Linear(model.config.hidden_size, len(id2label)).to(DEVICE)
        model.config.id2label = id2label
        model.config.label2id = label2id
        
        # logger.debug(model.config.id2label)

        # Freeze base layers
        for param in model.base_model.parameters():
            param.requires_grad = BM
            
        return model

    def save_model(self, path: str) -> None:
        """
        Save the model to a specified path.

        Args:
            path (str): The path to save the model.
        """
        self.model.save_pretrained(path)
        torch.save(self.model, f"{path}_model.pth")
        self.model.id = path.split("/")[-1].split("_")[0]
        logger.info(f"Model saved to {path}")
        
    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(
            f"trainable params: {trainable_params} || all params: {all_params} || "
            f"trainable (%): {100 * trainable_params / all_params:.2f}"
    )
        
    def get_trainable_parameters(self):
        """
        Returns the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return trainable_params, all_params
        
    def load_trained_model(self, path: str) -> None:
        """
        Load a trained model from a specified path.

        Args:
            path (str): The path to load the model from.
        """
        logger.info(f"Loading model from {path}")
        self.model = ViTForImageClassification.from_pretrained(path).to(self.device)
        self.model.id = path.split("/")[-1].split("_")[0]
        logger.info(f"Model loaded from {path}")

    def model_infer(self, test_dataset: torch.utils.data.Dataset) -> list:
        """
        Perform inference on a given test dataset.

        Args:
            test_dataset (torch.utils.data.Dataset): The dataset to run inference on.

        Returns:
            list: List of predictions for each sample in the test dataset.
        """
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        test_df = pd.read_csv(TEST_CSV)
        self.model.eval()
        preds = []
        correct_predictions = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Running Inference"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(pixel_values)
                predicted_class_idx = outputs.logits.argmax(dim=1).cpu().tolist()
                predicted_class = [self.model.config.id2label[idx] for idx in predicted_class_idx]
                # logger.debug(f"Predicted class: {predicted_class} from indices: {predicted_class_idx}")
                correct_predictions += sum([1 for idx in range(len(predicted_class)) if predicted_class[idx] == labels[idx]])
                preds.extend(predicted_class)
        try:
            logger.info(f"Inferring from model id: {self.model.id}")
        except AttributeError:
            self.model.id = random.randint(0, 1000)
            logger.info(f"Inferring from model with id {self.model.id}")
            
        col_name = f'{self.type_}_{self.model.id}'
        idx = 1
        while col_name in test_df.columns:
            col_name = f'{self.type_}_{self.model.id}_{idx}'
            idx += 1
        
        test_df[f'{col_name}'] = preds
        test_df.to_csv(TEST_CSV, index=False)
        
        return preds, col_name

    def eval_predictions(self, col_name: str = None) -> float:
        """
        Evaluate the accuracy of predictions against ground truth labels.

        Args:
            test_csv (str): Path to the CSV file containing ground truth labels.
            col_name (str): Name of the column containing the predictions.

        Returns:
            float: Accuracy of the predictions.
        """
        col_name = col_name if col_name else f'{self.type_}'
        test_df = pd.read_csv(TEST_CSV)
        correct = 0
        for _, row in test_df.iterrows():
            if row['Labels'] == row[f'{col_name}']:
                correct += 1
        accuracy = correct / len(test_df)
        logger.info(f"Prediction accuracy for infernce column {col_name}: {accuracy:.4f}")
        return accuracy
    
    def analyze_misclassifications(self, col_name: str, verbose: bool = True) -> dict:
        """
        Analyze misclassifications from the CSV file and count misclassifications by true labels.

        Args:
            col_name (str): Column name containing the predictions.

        Returns:
            dict: Dictionary summarizing misclassification counts by label.
        """
        test_df = pd.read_csv(TEST_CSV)

        # Filter rows where predictions are incorrect
        misclassified_df = test_df[test_df['Labels'] != test_df[col_name]]

        # Count misclassifications for each true label
        misclassification_summary = defaultdict(int)
        for _, row in misclassified_df.iterrows():
            misclassification_summary[row['Labels']] += 1

        # Sort misclassifications by count for better visualization
        sorted_summary = dict(sorted(misclassification_summary.items(), key=lambda x: x[1], reverse=True))
        
        # Print top k misclassifications as a tabulate table
        if verbose:
            logger.info("Misclassification counts by true label:")
            logger.info(tabulate.tabulate(sorted_summary.items(), headers=["True Label", "Misclassification Count"], tablefmt="pretty"))

        return sorted_summary, misclassification_summary

    def visualize_misclassifications(self, col_name: str, image_root_dir: str = TEST_DIR, num_examples: int = 5, saliency: bool = False, images_or_hist: str = "images") -> None:
        """
        Visualize misclassified images for each label.

        Args:
            col_name (str): Column name containing the predictions.
            image_root_dir (str): Path to the root directory containing the images.
            num_examples (int): Number of misclassified examples to visualize per category.
            saliency (bool): Whether to compute and display saliency maps.
            images_or_hist (str): Whether to display images or histograms.
        """
        assert images_or_hist in ["images", "histograms"], "Invalid value for images_or_hist. Must be 'images' or 'histograms'."
        
        if images_or_hist == "histograms":
            sorted_summary, misclassification_summary = self.analyze_misclassifications(col_name, verbose = False)
            plt.bar(misclassification_summary.keys(), misclassification_summary.values())
            plt.xlabel("True Label")
            plt.ylabel("Misclassification Count")
            if self.type_ == "baseline":
                plt.title("Baseline Misclassification Counts by True Label")
            elif self.type_ == "lora":
                plt.title("LoRA Misclassification Counts by True Label")
            elif self.type_ == "lora_plus":
                plt.title("LoRA+ Misclassification Counts by True Label")
            elif self.type_ == "Q_lora":
                plt.title("Quantized LoRA Misclassification Counts by True Label")
            elif self.type_ == "Q_lora_plus":
                plt.title("Quantized LoRA+ Misclassification Counts by True Label")
            else:
                plt.title("Misclassification Counts by True Label")
            plt.savefig(f"models/{self.type_}/misclassification_histogram.png")
        
        elif images_or_hist == "images":
            # 1. Get misclassification summaries
            sorted_summary, misclassification_summary = self.analyze_misclassifications(col_name, verbose=False)

            # Load the test dataframe
            test_df = pd.read_csv(TEST_CSV)

            # 2. Compute per-class correct counts
            class_correct_counts = defaultdict(int)
            # We'll also get all classes that appear in the dataset
            all_classes = set(test_df['Labels'].unique())

            for _, row in test_df.iterrows():
                label = row['Labels']
                pred = row[col_name]
                if label == pred:
                    class_correct_counts[label] += 1

            # 3. Find the most misclassified class with at least num_examples correct classifications.
            # sorted_summary is already sorted by misclassification count descending.
            # It's a dict like: {class_label: mis_count, ...} ordered by mis_count descending.
            most_misclassified_class = None
            for cls, mis_count in sorted_summary.items():
                if class_correct_counts[cls] >= num_examples:
                    most_misclassified_class = cls
                    break

            # 4. Find the least misclassified class with at least num_examples incorrect classifications.
            # We'll sort by misclassification ascending for this.
            classes_by_least_mis = sorted(misclassification_summary.items(), key=lambda x: x[1])
            least_misclassified_class = None
            for cls, mis_count in classes_by_least_mis:
                if mis_count >= num_examples:
                    least_misclassified_class = cls
                    break

            if most_misclassified_class is None or least_misclassified_class is None:
                logger.warning("Could not find suitable classes based on the given criteria.")
                return

            logger.info(f"Most misclassified class with >=5 correct: {most_misclassified_class}")
            logger.info(f"Least misclassified class with >=5 incorrect: {least_misclassified_class}")

            # 5. Get 5 correct and 5 misclassified examples for each identified class
            def get_examples_for_class(cls: int, correct: bool, limit: int) -> pd.DataFrame:
                """Get examples for a given class.
                
                Args:
                    cls (int): Class label.
                    correct (bool): Whether to get correct or incorrect examples.
                    limit (int): Number of examples to get.
                    
                Returns:
                    pd.DataFrame: DataFrame containing examples for the given class.
                """
                if correct:
                    subset_df = test_df[(test_df['Labels'] == cls) & (test_df[col_name] == cls)]
                else:
                    subset_df = test_df[(test_df['Labels'] == cls) & (test_df[col_name] != cls)]
                return subset_df.head(limit)

            mm_correct_examples = get_examples_for_class(most_misclassified_class, correct=True, limit=num_examples)
            mm_incorrect_examples = get_examples_for_class(most_misclassified_class, correct=False, limit=num_examples)

            lm_correct_examples = get_examples_for_class(least_misclassified_class, correct=True, limit=num_examples)
            lm_incorrect_examples = get_examples_for_class(least_misclassified_class, correct=False, limit=num_examples)

            # 6. Plot the images and saliency maps
            def plot_class_examples(cls_name: str, correct_df: pd.DataFrame, incorrect_df: pd.DataFrame, title_prefix: str):
                """Plot examples for a given class.

                Args:
                    cls_name (str): Class name.
                    correct_df (pd.DataFrame): DataFrame containing correct examples.
                    incorrect_df (pd.DataFrame): DataFrame containing incorrect examples.
                    title_prefix (str): Prefix for the plot title.
                """

                fig, axes = plt.subplots(4, num_examples, figsize=(num_examples * 3, 12))
                fig.suptitle(f"{title_prefix} Class: {cls_name}", fontsize=16)
                
                # Ensure axes is a 2D array even if num_examples = 1
                if num_examples == 1:
                    # axes would be a 1D array of shape (4,)
                    # Convert it to a 2D array (4, 1) for consistency
                    axes = axes[:, np.newaxis]

                # Plot correct examples
                for i, (_, row) in enumerate(correct_df.iterrows()):
                    img_path = os.path.join(image_root_dir, row['filename'])
                    img = Image.open(img_path).convert("RGB")
                    pred = row[col_name]
                    if saliency:
                        sal_map = self.compute_saliency_map(img, predicted_label=pred)
                    else:
                        sal_map = None
                    
                    # Image on the top row (row 0)
                    axes[0, i].imshow(img)
                    axes[0, i].axis('off')
                    axes[0, i].set_title(f"Correct (Label: {row['Labels']}, Pred: {pred})", fontsize=10)

                    # Saliency map below (row 1)
                    if sal_map is not None:
                        axes[1, i].imshow(sal_map, cmap='hot')
                    axes[1, i].axis('off')

                # Plot incorrect examples
                for i, (_, row) in enumerate(incorrect_df.iterrows()):
                    img_path = os.path.join(image_root_dir, row['filename'])
                    img = Image.open(img_path).convert("RGB")
                    pred = row[col_name]
                    if saliency:
                        sal_map = self.compute_saliency_map(img, predicted_label=pred)
                    else:
                        sal_map = None
                    
                    # Images of incorrect examples go on row 2
                    axes[2, i].imshow(img)
                    axes[2, i].axis('off')
                    axes[2, i].set_title(f"Incorrect (Label: {row['Labels']}, Pred: {pred})", fontsize=10)

                    # Saliency maps for incorrect on row 3
                    if sal_map is not None:
                        axes[3, i].imshow(sal_map, cmap='hot')
                    axes[3, i].axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                fig.savefig(f"models/{self.type_}/{title_prefix}_{cls_name}_misclassifications.png")
                plt.close(fig)


            # Plot for the most misclassified class (with >=num_examples correct)
            plot_class_examples(
                most_misclassified_class,
                mm_correct_examples,
                mm_incorrect_examples,
                title_prefix="Most_Misclassified"
            )

            # Plot for the least misclassified class (with >=num_examples incorrect)
            plot_class_examples(
                least_misclassified_class,
                lm_correct_examples,
                lm_incorrect_examples,
                title_prefix="Least_Misclassified"
            )            

        else:
            raise ValueError("Invalid value for images_or_hist. Must be 'images' or 'histograms'.")
                    
    def compute_saliency_map(self, image: Image.Image, predicted_label: int) -> np.ndarray:
        """
        Compute the saliency map for a given image and predicted label.

        Args:
            image (Image.Image): The input image.
            predicted_label (int): The predicted label for the image.

        Returns:
            np.ndarray: The computed saliency map.
        """
        # Convert image to tensor and send to device
        pixel_values = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        pixel_values.requires_grad = True

        # Forward pass
        self.model.eval()
        outputs = self.model(pixel_values)
        loss = torch.nn.functional.cross_entropy(outputs.logits, torch.tensor([predicted_label]).to(self.device))

        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()

        # Extract the saliency map from the image gradients
        saliency, _ = torch.max(pixel_values.grad.data.abs(), dim=1)
        saliency_map = saliency.squeeze().cpu().numpy()

        return saliency_map

def get_lora_config(type_: str = "baseline",
                    trainer: LoRATrainer = LoRATrainer(ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"), 
                                                       torch.optim.Adam(ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").parameters(), lr=1e-5), 
                                                       load_dataset(TRAIN_CSV, TRAIN_DIR), 
                                                       load_dataset(VAL_CSV, VAL_DIR), 
                                                       epochs=30, 
                                                       batch_size=8),
                    r: int = 8, 
                    lora_alpha: float = 1.0, 
                    learning_rate: float = 1e-5, 
                    loraplus_lr_ratio: int = 16, 
                    dropout: float = 0.01, 
                    step_size: int = 10, 
                    gamma: float = 0.1, 
                    **kwargs) -> Tuple[ViTForImageClassification, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Configures and returns a LoRA model, an optimizer, and a learning rate scheduler based on the type specified.

    Args:
        type_ (str): Type of LoRA model to configure.
        r (int): Rank of the LoRA adaptation.
        lora_alpha (float): Scaling factor for LoRA.
        learning_rate (float): Learning rate for the optimizer.
        loraplus_lr_ratio (int): Ratio for LoRA+ optimizer learning rate.
        dropout (float): Dropout rate for LoRA.
        step_size (int): Step size for the learning rate scheduler.
        gamma (float): Multiplicative factor for the learning rate scheduler.

    Returns:
        Tuple[ViTForImageClassification, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]: Configured model, optimizer, and scheduler.
    """
    base_model = trainer.load_model(MODEL)
    logger.info(f"Loading {type_} model with parameters: \nr={r}, lora_alpha={lora_alpha}, learning_rate={learning_rate}, loraplus_lr_ratio={loraplus_lr_ratio}, step_size={step_size}, gamma={gamma}, dropout={dropout}")

    if type_ == "lora":
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["query", "value"],
            lora_dropout=dropout
        )
        model = get_peft_model(base_model, lora_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return model, optimizer, scheduler

    elif type_ == "Q_lora":
        Q_lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["query", "value"],
        )
        # Quantization config from QLoRA paper
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
        model = ViTForImageClassification.from_pretrained(
            MODEL, quantization_config=bnb_config, device_map={"": 0}
        )
        # Get the dataset to extract unique labels
        train_df = pd.read_csv(TRAIN_CSV)
        unique_labels = train_df['Labels'].unique()

        # Create mappings
        id2label, label2id = {}, {}
        for idx, label in enumerate(unique_labels):
            id2label[int(idx)] = str(label)
            label2id[str(label)] = int(idx)

        # Update model classifier layer and mapping functions
        model.config.num_labels = len(id2label)
        model.num_labels = len(id2label)
        model.classifier = nn.Linear(model.config.hidden_size, len(id2label)).to(DEVICE)
        model.config.id2label = id2label
        model.config.label2id = label2id

        # Freeze base layers
        for param in model.base_model.parameters():
            param.requires_grad = False
        model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, Q_lora_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return model, optimizer, scheduler

    elif type_ == "lora_plus":
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["query", "value"],
            lora_dropout=dropout
        )
        model = get_peft_model(base_model, lora_config)
        optimizer = create_loraplus_optimizer(
            model=model,
            optimizer_cls=bnb.optim.Adam8bit,
            lr=learning_rate,
            loraplus_lr_ratio=loraplus_lr_ratio
        )
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return model, optimizer, scheduler

    elif type_ == "Q_lora_plus":
        Q_lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["query", "value"],
            lora_dropout=dropout
        )
        # Quantization config from QLoRA paper
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
        model = ViTForImageClassification.from_pretrained(
            MODEL, quantization_config=bnb_config, device_map={"": 0}
        )
        # Get the dataset to extract unique labels
        train_df = pd.read_csv(TRAIN_CSV)
        unique_labels = train_df['Labels'].unique()

        # Create mappings
        id2label, label2id = {}, {}
        for idx, label in enumerate(unique_labels):
            id2label[int(idx)] = str(label)
            label2id[str(label)] = int(idx)

        # Update model classifier layer and mapping functions
        model.config.num_labels = len(id2label)
        model.num_labels = len(id2label)
        model.classifier = nn.Linear(model.config.hidden_size, len(id2label)).to(DEVICE)
        model.config.id2label = id2label
        model.config.label2id = label2id

        # Freeze base layers
        for param in model.base_model.parameters():
            param.requires_grad = False
        model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, Q_lora_config)
        optimizer = create_loraplus_optimizer(
            model=model,
            optimizer_cls=bnb.optim.Adam8bit,
            lr=learning_rate,
            loraplus_lr_ratio=loraplus_lr_ratio
        )
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return model, optimizer, scheduler

    elif type_ == "baseline":
        base_model = trainer.load_model(MODEL, BM=True)
        optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return base_model, optimizer, scheduler

    else:
        raise ValueError(f"Invalid type: {type_}")


def train_model_lora(epochs: int, type_: str, train_indices: Optional[np.ndarray] = None, val_indices: Optional[np.ndarray] = None, lora_train_config: Optional[dict] = None) -> LoRATrainer:
    """
    Trains a LoRA model with the specified configuration.

    Args:
        epochs (int): Number of epochs to train.
        type_ (str): Type of LoRA model to train.
        train_indices (Optional[np.ndarray]): Training indices for subset training.
        val_indices (Optional[np.ndarray]): Validation indices for subset training.
        lora_train_config (Optional[dict]): Configuration for the LoRA model.

    Returns:
        LoRATrainer: A trainer object for the trained model.
    """
    if lora_train_config is None:
        model, optimizer, scheduler = get_lora_config(type_)
    else:
        model, optimizer, scheduler = get_lora_config(type_, **lora_train_config)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lora_train_config["learning_rate"])

    if train_indices is None or val_indices is None:
        train_subset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
        val_subset = load_dataset(csv_file=VAL_CSV, root_dir=VAL_DIR)
    else:
        train_dataset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
        val_dataset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

    trainer = LoRATrainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_subset,
        val_dataset=val_subset,
        epochs=epochs,
        batch_size=lora_train_config["batch_size"] if lora_train_config else 8,
        scheduler=scheduler,
        type_=type_
    )

    logger.info("Training model...")
    trainer.train()
    # logger.debug(trainer.model.config.id2label)
    logger.success("Training complete")
    return trainer

def run_sweep(type_: str) -> None:
    def train_model_lora_wandb():
        wandb.init(reinit=True)
        config = wandb.config
        model, optimizer, scheduler = get_lora_config(type_ = config.type, learning_rate=config.learning_rate, r=config.r, lora_alpha=config.lora_alpha, loraplus_lr_ratio=config.loraplus_lr_ratio, dropout=config.dropout)
        logger.info(f"Training model with configuration: {config}")
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        train_subset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
        val_subset = load_dataset(csv_file=VAL_CSV, root_dir=VAL_DIR)

        trainer = LoRATrainer(
            model=model,
            optimizer=optimizer,
            train_dataset=train_subset,
            val_dataset=val_subset,
            epochs=config.epochs,
            batch_size=config.batch_size,
            scheduler=scheduler,
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        wandb.log(eval_results)
        
    def train_model_lora_wandb_baseline():
        wandb.init(reinit=True)
        config = wandb.config
        model, optimizer, scheduler = get_lora_config("baseline", learning_rate=config.learning_rate)
        logger.info(f"Training model with configuration: {config}")

        train_subset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
        val_subset = load_dataset(csv_file=VAL_CSV, root_dir=VAL_DIR)

        trainer = LoRATrainer(
            model=model,
            optimizer=optimizer,
            train_dataset=train_subset,
            val_dataset=val_subset,
            epochs=config.epochs,
            batch_size=config.batch_size,
            scheduler=scheduler
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        wandb.log(eval_results)
        
    if type_ == "baseline":
        sweep_config = {
        "method": "bayes",
        "metric": {"name": "eval_accuracy", "goal": "maximize"},
        "parameters": { # hyperparams: lr should be continous
            "learning_rate": {"min": 1e-5, "max": 1e-3},
            "batch_size": {"values": [16]},
            "epochs": {"values": [15]},
            },
        }
        sweep_id = wandb.sweep(sweep_config, project="lora-sweep")
        wandb.agent(sweep_id, function=train_model_lora_wandb_baseline, count=50)
    else:
        if type_ == "lora_plus" or type_ == "Q_lora_plus":
            sweep_config = {
            "method": "bayes",
            "metric": {"name": "eval_accuracy", "goal": "maximize"},
            "parameters": { # hyperparams: lr should be continous
                "learning_rate": {"min": 1e-6, "max": 1e-4},
                "batch_size": {"values": [16]},
                "epochs": {"values": [15]},
                "type": {"values": [type_]},
                "r": {"values": [8, 16, 32, 64, 128]}, # Rank of the LORA matrix
                "lora_alpha": {"min": 2.0, "max": 16.0}, # Alpha parameter for LORA
                "loraplus_lr_ratio": {"min": 20, "max": 40}, # Ratio of the learning rate for LORA+ compared to LORA
                "dropout": {"values": [0.2]}
                },
            }
        else:
            sweep_config = {
            "method": "bayes",
            "metric": {"name": "eval_accuracy", "goal": "maximize"},
            "parameters": { # hyperparams: lr should be continous
                "learning_rate": {"min": 1e-4, "max": 1e-2},
                "batch_size": {"values": [16]},
                "epochs": {"values": [15]},
                "type": {"values": [type_]},
                "r": {"values": [8, 16, 32, 64]}, # Rank of the LORA matrix
                "lora_alpha": {"min": 2.0, "max": 16.0}, # Alpha parameter for LORA
                "loraplus_lr_ratio": {"values": [0]},
                "dropout": {"values": [0.2]}
                },
            }
        sweep_id = wandb.sweep(sweep_config, project="lora-sweep")
        wandb.agent(sweep_id, function=train_model_lora_wandb, count=50)
    
    # Get the best configuration from wandb
    api = wandb.Api()
    sweep = api.sweep(f"magnusgp/lora-sweep/sweeps/{sweep_id}")
    best_config = sweep.best_run().config
    logger.info(f"Best configuration: {best_config}")
    torch.save(best_config, f"data/best_config_{type_}.pth")

def manage_cv(epochs: int, type_: str, lora_train_config: dict = None, num_folds: int = 5) -> float:
    """
    Manages the cross-validation process for the LORA models.
    For each fold, trains the model and evaluates it on the validation set.
    If the number of folds is 1, the model is trained on the entire training set.
    If the number of folds is greater than 1, the model is trained on each fold 
    and evaluated on the validation set.
    
    Args:
        - epochs (int): number of epochs to train the model
        - type_ (str): type of LORA model to train
        - lora_train_config (dict): configuration for the LORA model
        - num_folds (int): number of folds for cross-validation
    
    Returns:
        - avg_accuracy (float): average accuracy across all folds
    """
    run_id = random.randint(0, 100000)
    
    if num_folds == 1:
        logger.info("Training model on entire training set...")
        from_checkpoint = False
        if not from_checkpoint:
            wandb.init(project="lora-cv", 
                   reinit=True, 
                   group=f"{run_id}_cv_{type_}_{lora_train_config['r']}_{int(lora_train_config['lora_alpha'])}_{lora_train_config['loraplus_lr_ratio']}", 
                   name=f"run_{run_id}", 
                   config=lora_train_config,
                #    mode="disabled")
                   mode="online")
            trainer = train_model_lora(epochs, type_, lora_train_config=lora_train_config)
            if trainer.do_profiling:
                print(trainer.profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
                print(trainer.profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
                print(trainer.profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
                print(trainer.profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
                trainer.profiler.export_chrome_trace("hf-training-trainer-all-data.json")
            # save the model
            eval_results = trainer.evaluate()
            try:
                trainer.save_model(f"models/{type_}/{run_id}_1_{eval_results['eval_accuracy']*100:.3f}")
                trainer.save_model_available = True
            except AttributeError: # Some deep pickle error happens with the Quantized LoRA models
                trainer.save_model_available = False
                pass
        else:
            wandb.init(mode="disabled")
            trainer = LoRATrainer(ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"), 
                                  torch.optim.Adam(ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").parameters(), lr=1e-5), 
                                  load_dataset(TRAIN_CSV, TRAIN_DIR), 
                                  load_dataset(VAL_CSV, VAL_DIR), 
                                  epochs=30, 
                                  batch_size=8,
                                  type_=type_)
            if trainer.save_model_available:
                trainer.load_trained_model("/dtu/blackhole/15/155381/Deep_learning_finetuning_and_merge/models/lora/44971_58.386")
            # trainer.analyze_misclassifications(f"{type_}_44971")
            # trainer.visualize_misclassifications(f"{type_}_44971", image_root_dir=TEST_DIR, saliency=True)
        res = trainer.evaluate()
        logger.info(res["eval_accuracy"])
        # _, col_name = trainer.model_infer(load_dataset(TEST_CSV, TEST_DIR))
        # acc = trainer.eval_predictions(col_name)
        # logger.success(f"Model trained on entire training set. Inference results: {acc}")
        wandb.finish()
        # return eval_results["eval_accuracy"]
        return acc, trainer

    else:
        fold_results = []
        dataset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        max_accuracy = 0
        for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
            logger.info(f"Training fold {fold + 1}...")
            wandb.init(project="lora-cv", 
                       reinit=True, 
                       group=f"{run_id}_cv_{type_}_{lora_train_config['r']}_{int(lora_train_config['lora_alpha'])}_{lora_train_config['loraplus_lr_ratio']}", 
                       name=f"{run_id}_fold_{fold + 1}", 
                       config=lora_train_config,
                    #    mode="disabled")
                       mode="online")
            trainer = train_model_lora(epochs, type_, train_indices, val_indices, lora_train_config)
            # logger.debug(trainer.model.config.id2label)
            if trainer.do_profiling:
                print(trainer.profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
                print(trainer.profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
                print(trainer.profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
                print(trainer.profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
                trainer.profiler.export_chrome_trace(f"hf-training-trainer-fold{fold+1}.json")
            eval_results = trainer.evaluate()
            logger.success(f"Fold {fold + 1} results: {eval_results}")
            fold_results.append(eval_results)
            if eval_results["eval_accuracy"] > max_accuracy:
                max_accuracy = eval_results["eval_accuracy"]
                # save the model
                try:
                    trainer.save_model(f"models/{type_}/{run_id}_{fold}_{eval_results['eval_accuracy']*100:.3f}")
                    trainer.save_model_available = True
                except AttributeError: # Some deep pickle error happens with the Quantized LoRA models
                    trainer.save_model_available = False
                    pass
                # save model metadata
                model_metadata = {
                    "type": type_,
                    "run_id": run_id,
                    "fold": fold,
                    "accuracy": eval_results["eval_accuracy"]
                }
            # wandb.log(eval_results)
            trainable_params, all_params = trainer.get_trainable_parameters()
            macs, params = get_model_complexity_info(trainer.model, (3, 224, 224), as_strings=True, backend='pytorch',
                                            print_per_layer_stat=False, verbose=False)
            _, col_name = trainer.model_infer(load_dataset(TEST_CSV, TEST_DIR))
            acc = trainer.eval_predictions(col_name)
            wandb.log({"trainable_params": trainable_params, "all_params": all_params, "macs": macs, "params": params, "test_accuracy": acc})
            wandb.finish()
        
        # Load the best model and evaluate on the test set
        # if trainer.save_model_available:
        #     trainer.load_trained_model(f"/dtu/blackhole/15/155381/Deep_learning_finetuning_and_merge/models/{model_metadata['type']}/{model_metadata['run_id']}_{model_metadata['fold']}_{model_metadata['accuracy']*100:.3f}")
        # _, col_name = trainer.model_infer(load_dataset(TEST_CSV, TEST_DIR))
        # acc = trainer.eval_predictions(col_name)
        avg_accuracy = np.mean([result['eval_accuracy'] for result in fold_results])
        # logger.info(f"Average cross-validated accuracy: {avg_accuracy}")
        # logger.success(f"Final Test Accuracy: {acc}")
        return avg_accuracy, trainer

def lora_loop(type_: str, epochs: int = None, do_sweep: bool = False, num_folds: int = 1, config: dict = None) -> float:
    if do_sweep:
        run_sweep(type_)
        config_path = f"data/best_config_{type_}.pth"
        if os.path.exists(config_path):
            config = torch.load(config_path)
            logger.info(f"Loaded best configuration: {config}")
            avg_accuracy, trainer = manage_cv(epochs, type_, lora_train_config=config)
        else: # Just use the default config if the sweep fails
            avg_accuracy, trainer = manage_cv(epochs=epochs if epochs is not None else 30, type_=type_)
        logger.info(f"Final model trained with optimal configuration: {config_path}. \n Achieved accuracy: {avg_accuracy}")
    else:
        avg_accuracy, trainer = manage_cv(epochs if epochs is not None else 30, type_, lora_train_config=config, num_folds=num_folds)
        return avg_accuracy, trainer
