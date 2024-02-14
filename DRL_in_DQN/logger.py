import os
import time
import torch

class Logger():
    def __init__(self, root, save_num):
        name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.root = os.path.join(root, name)
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.log_path  = os.path.join(self.root, "log.txt")
        self.save_path = []
        self.save_num  = save_num
        self.cur_num   = 0
        return
    
    def info(self, info):
        info = "[INFO] " + info
        with open(self.log_path, "a+") as log_file:
            log_file.write(info + "\n")
        print(info)
        return
    
    def save_model(self, model, save_name):
        path = os.path.join(self.root, save_name)
        torch.save(model, path)
        self.save_path.append(path)
        if len(self.save_path) > self.save_num:
            os.remove(self.save_path[0])
            del self.save_path[0]
        return