import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import perf_counter
import os

from load_config import load_config
from data import get_dataset
from dataloader import get_dataloader
from model import Model
from logger import Logger

CONFIG        = load_config()
CUDA          = CONFIG["cuda"]
LOG_CONFIG    = CONFIG["log"]
LOG_ROOT      = LOG_CONFIG["root"]
SAVE_NUM      = LOG_CONFIG["save_num"]
logger        = Logger(LOG_ROOT, SAVE_NUM)

TRAIN_CONFIG  = CONFIG["train"]
LEARNING_RATE = TRAIN_CONFIG["learning_rate"]
EPOCHS        = TRAIN_CONFIG["epochs"]
SHOW_CONFIG   = CONFIG["show"]
ACC_IMG       = os.path.join(logger.root, SHOW_CONFIG["acc_img"])
LOSS_IIMG     = os.path.join(logger.root, SHOW_CONFIG["loss_img"])

def train(model, optimizer, ceriterion, train_dataloader, logger):
    start = perf_counter()
    tot_loss = 0
    epoch_list = []
    loss_list  = []
    acc_list   = []

    for epoch in range(1, EPOCHS + 1):
        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.cuda() if CUDA else inputs
            labels = labels.cuda() if CUDA else labels

            outputs = model(inputs)
            optimizer.zero_grad()
            loss = ceriterion(outputs, labels)
            loss.backward()
            tot_loss += loss.cpu().item()
            optimizer.step()

        loss = tot_loss / len(train_dataloader)
        tot_loss = 0
        time  = perf_counter() - start
        start = perf_counter()
        logger.info("\n------")
        logger.info(f"Epoch:{epoch:3d} Loss:{loss:10.6f} Time:{time:6.2f}s.")
        logger.save_model(model, f"Epoch_{epoch}.pth")
        logger.info(f"Save model as Epoch_{epoch}.pth")
        epoch_list.append(epoch)
        loss_list.append(loss)

    logger.info("Finished training!")
    return  epoch_list, loss_list

def draw(epoch_list, loss_list):
    plt.plot(epoch_list, loss_list, label="Loss")
    plt.title("Loss Img")
    plt.legend()
    plt.savefig(LOSS_IIMG)
    plt.close()
    return




if __name__ == "__main__":
    logger.info("Logger init.")

    train_dataset = get_dataset(CONFIG, mod="train")

    train_dataloader = get_dataloader(CONFIG, train_dataset, mod="train")
    logger.info("Load data.")

    model = Model()
    model = model.cuda() if CUDA else model
    logger.info("Build model.")

    optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ceriterion = nn.BCEWithLogitsLoss()

    epoch_list, loss_list = train(model, optimizer, ceriterion, train_dataloader, logger)
    draw(epoch_list, loss_list)