from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.optimizers import create_loraplus_optimizer
from transformers import ViTImageProcessor, ViTModel, ViTForImageClassification, BitsAndBytesConfig, AutoModelForCausalLM
import bitsandbytes as bnb
import evaluate
import random
from copy import deepcopy
import numpy as np
import wandb
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.config.config import *
from src.model.baseline_model import load_dataset, load_model
from transformers import Trainer, TrainerCallback, TrainingArguments, PrinterCallback
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

class LoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and state.global_step > 0:
            # Access the Trainer instance
            trainer = kwargs["trainer"]

            # Log training metrics (e.g., accuracy, loss)
            logs = {}
            # Access the latest logged training metrics from state.log_history
            logs["training_accuracy"] = state.log_history[-1].get("training_accuracy", "N/A")
            logs["step"] = state.global_step

            # Run evaluation
            eval_metrics = trainer.evaluate()
            logs.update({f"eval_{k}": v for k, v in eval_metrics.items()})

            # Log to WandB
            wandb.log(logs)

class AccuracyResetCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        trainer = kwargs["model"]
        trainer.total_correct = 0
        trainer.total_samples = 0

class EvaluateCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_predictions = []
        self.epoch_labels = []
        self.epoch_loss = []
        self.epoch_accuracies = []

    def compute_loss(self, model, inputs, num_items_in_batch):
        """
        MAX: Subclassed to compute training accuracy.

        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if "labels" in inputs:
            preds = outputs.logits.detach()

            # Log accuracy
            acc = (
                (preds.argmax(axis=1) == inputs["labels"])
                .type(torch.float)
                .mean()
                .item()
            )
            # Uncomment it if you want to plot the batch accuracy
            wandb.log({"batch_accuracy": acc})  # Log accuracy

            # Store predictions and labels for epoch-level metrics
            self.epoch_predictions.append(preds.cpu().numpy())
            self.epoch_labels.append(inputs["labels"].cpu().numpy())

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # Uncomment it if you want to plot the batch loss
            wandb.log({"batch_loss": loss})
            self.epoch_loss.append(loss.item())  # Store loss for epoch-level metrics

        # return (loss, outputs) if return_outputs else loss
        return loss

class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        # Aggregate predictions and labels for the entire epoch
        epoch_predictions = np.concatenate(self._trainer.epoch_predictions)
        epoch_labels = np.concatenate(self._trainer.epoch_labels)

        # Compute accuracy
        accuracy = np.mean(epoch_predictions.argmax(axis=1) == epoch_labels)
        print(f"Shape of epoch_predictions: {epoch_predictions.shape}, shape of epoch_labels: {epoch_labels.shape}")
        logger.info(f"First 5 preds: {epoch_predictions.argmax(axis=1)[:5]}, first 5 labels: {epoch_labels[:5]}")
        logger.info(accuracy)
        self._trainer.epoch_accuracies.append(accuracy)

        # Compute mean loss
        mean_loss = np.mean(self._trainer.epoch_loss)

        # # Compute precision, recall, and F1-score
        # precision, recall, f1, _ = precision_recall_fscore_support(
        #     epoch_labels, epoch_predictions.argmax(axis=1), average="weighted"
        # )

        # Log epoch-level metrics
        wandb.log({"epoch_accuracy": accuracy, "epoch_loss": mean_loss})
        # wandb.log({"precision": precision, "recall": recall, "f1": f1})

        # Clear stored predictions, labels, and loss for the next epoch
        self._trainer.epoch_predictions = []
        self._trainer.epoch_labels = []
        self._trainer.epoch_loss = []
        return None


def get_lora_config(type_: str, r: int = 16) -> LoraConfig:
    processor = ViTImageProcessor.from_pretrained(MODEL)
    # base_model = ViTForImageClassification.from_pretrained(MODEL).to(DEVICE)
    
    base_model = load_model()
    
    if type_ == "lora":
        lora_config = LoraConfig(
            r = r,
            lora_alpha = 2 * r,
            init_lora_weights="gaussian",
            target_modules=["query", "value"]
            )
        return get_peft_model(base_model, lora_config), None
    
    elif type_ == "Q_lora":
        Q_lora_config = LoraConfig(
            r = r,
            lora_alpha = 2 * r,
            init_lora_weights="gaussian",
            target_modules=["query", "value"],
            task_type="CASUAL_LM"
            )
        # Quantization config from QLoRA paper
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="torch.bfloat16"
            )
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map={"": 0})
        model = prepare_model_for_kbit_training(model)
        return get_peft_model(model, Q_lora_config), None
    
    elif type_ == "lora_plus":
        optimizer = create_loraplus_optimizer(
            model=get_peft_model(base_model, lora_config),
            optimizer_cls=bnb.optim.Adam8bit,
            lr=5e-5,
            loraplus_lr_ratio=16,
        )
        lora_config = LoraConfig(
            r = r,
            lora_alpha = 2 * r,
            init_lora_weights="gaussian",
            target_modules=["query", "value"]
            )
        return get_peft_model(base_model, lora_config), optimizer

    elif type_ == "Q_lora_plus":
        optimizer = create_loraplus_optimizer(
            model=get_peft_model(base_model, Q_lora_config),
            optimizer_cls=bnb.optim.Adam8bit,
            lr=5e-5,
            loraplus_lr_ratio=16,
        )
        Q_lora_config = LoraConfig(
            r = r,
            lora_alpha = 2 * r,
            init_lora_weights="gaussian",
            target_modules=["query", "value"],
            )
        # Quantization config from QLoRA paper
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="torch.bfloat16"
            )
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map={"": 0})
        model = prepare_model_for_kbit_training(model)
        return get_peft_model(model, Q_lora_config), optimizer

    else:
        raise ValueError(f"Invalid type: {type_}")

def compute_metrics(pred):
    metric = evaluate.get_metric("accuracy")
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Argmax for classification
    wandb.log(accuracy_score(labels, preds))
    return metric.compute(predictions=preds, references=labels)

# train loop
def train_model_lora(
    epochs: int,
    type_: str,
    r: int,
    model=None,
) -> ViTForImageClassification:
    """
    Train the model using the optimizer and return the trained model.
    """
    model, optimizer = get_lora_config(type_, r)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def collator_fn(features):
        batch = {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.tensor([f["label"] for f in features], dtype=torch.long)
        }
        return batch

    run_id = random.randint(0, 1000000)

    args=TrainingArguments(num_train_epochs=epochs, 
                           output_dir="hf-training-trainer",
                           report_to="wandb",
                           run_name=f"lora-run-{run_id}",
                           logging_steps=50,
                           do_eval=True,
                           eval_strategy="steps",
                           eval_steps=50,)

    trainer = Trainer(
        model=model,
        optimizers=(optimizer, None),
        train_dataset=load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR),
        eval_dataset=load_dataset(csv_file=VAL_CSV, root_dir=VAL_DIR),
        data_collator=collator_fn,
        compute_metrics=compute_metrics,
        args=args,
    )

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            skip_first=3, wait=1, warmup=1, active=2, repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("hf-training-trainer"),
        profile_memory=True,
        with_stack=True,
        record_shapes=True)

    # trainer.add_callback(LoggerCallback())
    trainer.add_callback(ProfCallback(prof = profiler))
    # trainer.add_callback(AccuracyResetCallback())
    trainer.remove_callback(PrinterCallback)
    trainer.train()
    if profiler:
        profiler_data = profiler.key_averages().table(sort_by="self_cuda_time_total")

        return model, profiler_data
    else:
        return model, None


def test_model_lora(model) -> float:
    """
    Test the model and return the accuracy.
    """
    test_dataloader = load_dataset(csv_file=TEST_CSV, root_dir=TEST_DIR)
    correct = 0
    total = 0
    model = model.to(DEVICE)
    i = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = torch.tensor(batch['label']).to(DEVICE)

            outputs = model(pixel_values[None, ...], labels=labels)
            predictions = torch.argmax(torch.nn.functional.softmax(outputs.logits, dim=-1), dim=-1)  # Example for classification

            correct += (predictions == labels).sum().item()
            try:
                total += len(labels.item())
            except TypeError: # If the batch size is 1
                total += 1

    accuracy = correct / total
    wandb.log({"test_accuracy": accuracy})
    return accuracy

def train_model_lora_wandb():
    """
    Train the model using W&B sweep parameters and return the trained model.
    """
    wandb.init(reinit=True)  # Ensures reinitialization for sweeps

    config = wandb.config
    model, optimizer = get_lora_config(config.type)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def collator_fn(features):
        batch = {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.tensor([f["label"] for f in features], dtype=torch.long)
        }
        return batch

    run_id = random.randint(0, 1000000)
    
    args = TrainingArguments(
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        output_dir="hf-training-trainer",
        report_to="wandb",
        run_name=f"lora-run-{run_id}",
        logging_steps=10,
        do_eval=True,
        eval_steps=50,
    )

    trainer = CustomTrainer(
        model=model,
        optimizers=(optimizer, None),
        train_dataset=load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR),
        eval_dataset=load_dataset(csv_file=VAL_CSV, root_dir=VAL_DIR),
        data_collator=collator_fn,
        compute_metrics=compute_metrics,
        args=args,
    )
    
    trainer.add_callback(CustomCallback(trainer))

    logger.info("Training model...")
    trainer.train()
    logger.success("Training complete, evaluating...")
    eval_results = trainer.evaluate()
    logger.success(f"Eval results: {eval_results}")

    wandb.log(eval_results)  # Log the results for W&B Sweep tracking
    
    logger.success(f"Training accuracy: {trainer.epoch_accuracies}")
    
    return trainer.epoch_accuracies[-1] # target metric

def lora_loop(type_: str, epochs: int, do_sweep: bool = False) -> float:
    """
    Train and test the model and return the accuracy.
    """
    model, profiler_data = train_model_lora(epochs, type_, r=16)
    return test_model_lora(model), profiler_data
