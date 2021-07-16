import config
import torch

import pandas as pd
import numpy as np

from dataset import EntityDataset
from model import EntityModel
from engine import *

from sklearn import preprocessing
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def get_optimizer(model):

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    return AdamW(optimizer_parameters, lr=3e-5)

def get_scheduler(num_samples, optimizer):

    num_train_steps = int(num_samples / config.EPOCHS * config.BATCH_SIZE)

    return get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_train_steps
    ) 

def train(df):

    ##################### STAGE 1 #####################
    label_enc = preprocessing.LabelEncoder()
    df["label"] = label_enc.fit_transform(df["NER_tag"].values)

    sentences = list(df.groupby("id")["word"].apply(list).values)
    labels = list(df.groupby("id")["label"].apply(list).values)

    train_dataset = EntityDataset(sentences, labels)

    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = config.BATCH_SIZE
    )

    ##################### STAGE 2 #####################
    num_tags = df["label"].nunique()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    entity_model = EntityModel(num_tags).to(device)
    optimizer = get_optimizer(entity_model)
    scheduler = get_scheduler(df.shape[0], optimizer)

    ##################### STAGE 2 #####################
    best_loss = np.inf
    training_loss = []
    for epoch in range(config.EPOCHS):
        print(f"Epoch# {epoch+1}...")
        total_loss = train_one_epoch(entity_model, train_data_loader, optimizer, device, scheduler)

        print(f"\n loss: {round(total_loss, 5)}")
        training_loss.append(total_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(entity_model.state_dict(), config.MODEL_FILE)
            

if __name__ == "__main__":
    main_df = pd.read_csv(config.TRAIN_FILE, na_filter=False)
    train(main_df)