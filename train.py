from transformers import Trainer, TrainingArguments

def train(model, train_dataset, eval_dataset):
    args = TrainingArguments(
        output_dir="./model",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()