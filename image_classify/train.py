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

CONFIG        = load_config()
CUDA          = CONFIG["cuda"]
TRAIN_CONFIG  = CONFIG["train"]
LEARNING_RATE = TRAIN_CONFIG["learning_rate"]
EPOCHS        = TRAIN_CONFIG["epochs"]
SHOW_CONFIG   = CONFIG["show"]
ACC_IMG       = SHOW_CONFIG["acc_img"]
LOSS_IIMG     = SHOW_CONFIG["loss_img"]

def test(model, test_dataloader):
    cnt = 0
    tot = 0

    for inputs, labels in tqdm(test_dataloader):
        inputs = inputs.cuda() if CUDA else inputs
        labels = labels.cuda() if CUDA else labels

        outputs = model(inputs)
        predict = torch.max(outputs, 1)[1] == labels
        predict = predict.sum().cpu().item()

        cnt += predict
        tot += labels.shape[0]

    acc =  cnt / tot
    return acc

def train(model, optimizer, ceriterion, train_dataloader, test_dataloader):
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
        acc = test(model, test_dataloader) * 100
        time  = perf_counter() - start
        start = perf_counter()
        print(f"[INFO] Epoch:{epoch} Loss:{loss:.6f} Acc:{acc:.2f}% Time:{time:.2f}s")
        epoch_list.append(epoch)
        loss_list.append(loss)
        acc_list.append(acc)

    print("[INFO] Finished training!")
    return  epoch_list, loss_list, acc_list

def draw(epoch_list, loss_list, acc_list):
    if not os.path.isdir("log"):
        os.mkdir("log")

    plt.plot(epoch_list, loss_list, label="Loss")
    plt.title("Loss Img")
    plt.legend()
    plt.savefig(LOSS_IIMG)
    plt.close()

    plt.plot(epoch_list, acc_list, label="Acc")
    plt.title("Acc Img")
    plt.legend()
    plt.savefig(ACC_IMG)
    plt.close()
    return




if __name__ == "__main__":

    train_dataset = get_dataset(CONFIG, mod="train")
    test_dataset  = get_dataset(CONFIG, mod="test")

    train_dataloader = get_dataloader(CONFIG, train_dataset, mod="train")
    test_dataloader  = get_dataloader(CONFIG, test_dataset,  mod="test")

    model = Model()
    model = model.cuda() if CUDA else model
    optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ceriterion = nn.CrossEntropyLoss()

    epoch_list, loss_list, acc_list = train(model, optimizer, ceriterion, train_dataloader, test_dataloader)
    draw(epoch_list, loss_list, acc_list)