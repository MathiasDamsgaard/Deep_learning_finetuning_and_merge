from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import ViTImageProcessor, ViTModel, ViTForImageClassification
import bitsandbytes as bnb
import random
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.config.config import *
from src.model.baseline_model import load_dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
import torch.nn as nn

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

def get_lora_config(type_: str, r: int) -> LoraConfig:
    # Load the model and processor
    processor = ViTImageProcessor.from_pretrained(MODEL)
    base_model = ViTForImageClassification.from_pretrained(MODEL).to(DEVICE)

    # Get the dataset to extract unique labels
    train_df = pd.read_csv(TRAIN_CSV)
    unique_labels = train_df['Labels'].unique()

    # Create mappings
    id2label = {}, label2id = {}
    for idx, label in enumerate(unique_labels):
        id2label[int(idx)] = str(label)
        label2id[str(label)] = int(idx)

    # Update model classifier layer and mapping functions
    base_model.classifier = nn.Linear(base_model.config.hidden_size, len(id2label)).to(DEVICE)
    base_model.config.id2label = id2label
    base_model.config.label2id = label2id

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
            target_modules="all-linear"
            )
        return get_peft_model(base_model, Q_lora_config), None
    
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
            target_modules="all-linear"
            )
        return get_peft_model(base_model, Q_lora_config), optimizer

    else:
        raise ValueError(f"Invalid type: {type_}")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Argmax for classification
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


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
            "labels": torch.tensor([f["label"] for f in features], dtype=torch.long),
        }
        return batch

    run_id = random.randint(0, 1000000)

    args=TrainingArguments(num_train_epochs=epochs, 
                           output_dir="hf-training-trainer",
                           report_to="wandb",
                           run_name=f"lora-run-{run_id}",
                           logging_steps=100)

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

            # i += 1  # For debugging purposes      
            # if i == 100:
            #     break

    accuracy = correct / total
    wandb.log({"test_accuracy": accuracy})
    return accuracy

def lora_loop(type_: str, epochs: int, r: int) -> float:
    """
    Train and test the model and return the accuracy.
    """
    model, profiler_data = train_model_lora(epochs, type_, r)
    return test_model_lora(model), profiler_data
