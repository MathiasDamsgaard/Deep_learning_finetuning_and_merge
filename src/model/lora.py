from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import ViTImageProcessor, ViTModel, ViTForImageClassification
import bitsandbytes as bnb
from src.config.config import *
from src.model.baseline_model import load_dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from torch.profiler import profile, record_function, ProfilerActivity

# # Show running device
# print("Device:", DEVICE)


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


def get_lora_config(type_: str) -> LoraConfig:
    processor = ViTImageProcessor.from_pretrained(MODEL)
    base_model = ViTForImageClassification.from_pretrained(MODEL).to(DEVICE)
    
    if type_ == "lora":
        lora_config = LoraConfig(init_lora_weights="gaussian", target_modules=["query", "value"])
        return get_peft_model(base_model, lora_config), None
    
    elif type_ == "Q_lora":
        Q_lora_config = LoraConfig(init_lora_weights="gaussian", target_modules="all-linear")
        return get_peft_model(base_model, Q_lora_config), None
    
    elif type_ == "lora_plus":
        optimizer = create_loraplus_optimizer(
            model=get_peft_model(base_model, lora_config),
            optimizer_cls=bnb.optim.Adam8bit,
            lr=5e-5,
            loraplus_lr_ratio=16,
        )
        lora_config = LoraConfig(init_lora_weights="gaussian", target_modules=["query", "value"])
        return get_peft_model(base_model, lora_config), optimizer

    elif type_ == "Q_lora_plus":
        optimizer = create_loraplus_optimizer(
            model=get_peft_model(base_model, Q_lora_config),
            optimizer_cls=bnb.optim.Adam8bit,
            lr=5e-5,
            loraplus_lr_ratio=16,
        )
        Q_lora_config = LoraConfig(init_lora_weights="gaussian", target_modules="all-linear")
        return get_peft_model(base_model, Q_lora_config), optimizer

    else:
        raise ValueError(f"Invalid type: {type_}")


# train loop
def train_model_lora(epochs: int, type_:str, model = None) -> ViTForImageClassification:
    """
    Train the model using the optimizer and return the trained model.
    """
    # train_dataloader = load_dataset(batch_size=32, csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
    # val_dataloader = load_dataset(batch_size=32, csv_file=VAL_CSV, root_dir=VAL_DIR)

    # for epoch in range(epochs):
    #     model.train()
    #     for inputs, labels in train_dataloader:
    #         inputs = inputs.to(DEVICE)
    #         labels = labels.to(DEVICE)
    #         optimizer.zero_grad()
    #         outputs = model(**inputs, labels=labels)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()

    #     model.eval()
    #     for inputs, labels in val_dataloader:
    #         inputs = inputs.to(DEVICE)
    #         labels = labels.to(DEVICE)
    #         outputs = model(**inputs, labels=labels)
    #         loss = outputs.loss
    #         print(f"Epoch: {epoch}, Loss: {loss.item()}")

    model, optimizer = get_lora_config(type_)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    trainer = Trainer(
        model=model,
        optimizers=(optimizer, None),
        train_dataset=load_dataset(batch_size=32, csv_file=TRAIN_CSV, root_dir=TRAIN_DIR),
        eval_dataset=load_dataset(batch_size=32, csv_file=VAL_CSV, root_dir=VAL_DIR),
        args=TrainingArguments(num_train_epochs=epochs, output_dir="hf-training-trainer"),
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


    trainer.add_callback(ProfCallback(prof = profiler))
    trainer.train()
    profiler_data = profiler.key_averages().table(sort_by="self_cuda_time_total")

    return model, profiler_data

def test_model_lora(model) -> float:
    """
    Test the model and return the accuracy.
    """
    test_dataloader = load_dataset(batch_size=32, csv_file=TEST_CSV, root_dir=TEST_DIR)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def lora_loop(type_: str, epochs: int) -> float:
    """
    Train and test the model and return the accuracy.
    """

    model, profiler_data = train_model_lora(epochs, type_)
    return test_model_lora(model, type_), profiler_data
