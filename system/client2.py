import socket
import os
import sys
import time
import argparse
# !/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import torch.nn as nn
import numpy as np
import json
from flcore.servers.serverbase import Server
from threading import Thread

import torch

from flcore.clients.clientbase import Client
from utils.privacy import *

from flcore.servers.serveravg import FedAvg

from flcore.trainmodel.models import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
emb_dim = 32

##### 1.建立连接
# SRV = os.getenv('SERVER_ADDRESS')
# PORT = int(os.getenv('SERVER_PORT'))
SRV = "10.181.26.104"
PORT = 8900

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (SRV, PORT)

sock.connect(server_address)
print('连接到: Connecting to {} port {}'.format(*server_address))
sock.close()
#####


##### 2.收到训练数并转回源格式
# 转成string
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(server_address)
str_dic_encode = sock.recv(102400)
sock.close()
str_dic = str_dic_encode.decode()

# 转成args这个class
dic = eval(str_dic)
args = argparse.Namespace(**dic)


def run(args):
    settings = copy.copy(args)
    for i in range(args.prev, args.times):

        print("server端生成模型中...")
        time_list = []
        reporter = MemReporter()
        model_str = args.model
        print("args", args)

        ##### 第一轮需要生成初始模型，其它轮不再需要这个步骤

        start = time.time()

        # Generate args.model
        if True:
            if model_str == "mlr":  # convex
                if "mnist" in args.dataset:
                    args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
                elif "Cifar10" in args.dataset:
                    args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
                else:
                    args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

            elif model_str == "cnn":  # non-convex
                if "mnist" in args.dataset:
                    args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
                elif "Cifar10" in args.dataset:
                    args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
                elif "omniglot" in args.dataset:
                    args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                    # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
                elif "Digit5" in args.dataset:
                    args.model = Digit5CNN().to(args.device)
                else:
                    args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)


            elif model_str == "dnn":  # non-convex
                if "mnist" in args.dataset:
                    args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
                elif "Cifar10" in args.dataset:
                    args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
                else:
                    args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

            elif model_str == "resnet":
                args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

                # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
                # feature_dim = list(args.model.fc.parameters())[0].shape[1]
                # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

                # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

            elif model_str == "alexnet":
                args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

                # args.model = alexnet(pretrained=True).to(args.device)
                # feature_dim = list(args.model.fc.parameters())[0].shape[1]
                # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            elif model_str == "googlenet":
                args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                          num_classes=args.num_classes).to(args.device)

                # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
                # feature_dim = list(args.model.fc.parameters())[0].shape[1]
                # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            elif model_str == "mobilenet_v2":
                args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

                # args.model = mobilenet_v2(pretrained=True).to(args.device)
                # feature_dim = list(args.model.fc.parameters())[0].shape[1]
                # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            elif model_str == "lstm":
                args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                    args.device)

            elif model_str == "bilstm":
                args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim,
                                                       output_size=args.num_classes,
                                                       num_layers=1, embedding_dropout=0, lstm_dropout=0,
                                                       attention_dropout=0,
                                                       embedding_length=emb_dim).to(args.device)

            elif model_str == "fastText":
                args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                    args.device)

            elif model_str == "TextCNN":
                args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                                     num_classes=args.num_classes).to(args.device)

            elif model_str == "Transformer":
                args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim, nlayers=2,
                                              num_classes=args.num_classes).to(args.device)

            elif model_str == "AmazonMLP":
                args.model = AmazonMLP().to(args.device)

            elif model_str == "harcnn":
                if args.dataset == 'har':
                    args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                        pool_kernel_size=(1, 2)).to(args.device)
                elif args.dataset == 'pamap':
                    args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                        pool_kernel_size=(1, 2)).to(args.device)

            else:
                raise NotImplementedError

            # select algorithm
            if args.algorithm == "FedAvg":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedAvg(args, i)

            elif args.algorithm == "Local":
                server = Local(args, i)

            elif args.algorithm == "FedMTL":
                server = FedMTL(args, i)

            elif args.algorithm == "PerAvg":
                server = PerAvg(args, i)

            elif args.algorithm == "pFedMe":
                server = pFedMe(args, i)

            elif args.algorithm == "FedProx":
                server = FedProx(args, i)

            elif args.algorithm == "FedFomo":
                server = FedFomo(args, i)

            elif args.algorithm == "FedAMP":
                server = FedAMP(args, i)

            elif args.algorithm == "APFL":
                server = APFL(args, i)

            elif args.algorithm == "FedPer":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedPer(args, i)

            elif args.algorithm == "Ditto":
                server = Ditto(args, i)

            elif args.algorithm == "FedRep":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedRep(args, i)

            elif args.algorithm == "FedPHP":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedPHP(args, i)

            elif args.algorithm == "FedBN":
                server = FedBN(args, i)

            elif args.algorithm == "FedROD":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedROD(args, i)

            elif args.algorithm == "FedProto":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedProto(args, i)

            elif args.algorithm == "FedDyn":
                server = FedDyn(args, i)

            elif args.algorithm == "MOON":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = MOON(args, i)

            elif args.algorithm == "FedBABU":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedBABU(args, i)

            elif args.algorithm == "APPLE":
                server = APPLE(args, i)

            elif args.algorithm == "FedGen":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedGen(args, i)

            elif args.algorithm == "SCAFFOLD":
                server = SCAFFOLD(args, i)

            elif args.algorithm == "FedDistill":
                server = FedDistill(args, i)

            elif args.algorithm == "FedALA":
                server = FedALA(args, i)

            elif args.algorithm == "FedPAC":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedPAC(args, i)

            elif args.algorithm == "LG-FedAvg":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = LG_FedAvg(args, i)

            elif args.algorithm == "FedGC":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedGC(args, i)

            elif args.algorithm == "FML":
                server = FML(args, i)

            elif args.algorithm == "FedKD":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedKD(args, i)

            elif args.algorithm == "FedPCL":
                args.model.fc = nn.Identity()
                server = FedPCL(args, i)

            elif args.algorithm == "FedCP":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = FedCP(args, i)

            elif args.algorithm == "GPFL":
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = GPFL(args, i)

            else:
                raise NotImplementedError

        server.party = "client"
        server.settings = settings
        global server_address
        server.server_address = server_address

        server.train()
        '''##### 3.接收模型参数并生成模型
        # 接收tensor
        embedding_back = server.socket_receive_model()
        model_init = copy.deepcopy(server.global_model)
        model_unflatten_1 = server.unflatten(model_init, embedding_back)
        print("从server端收到的模型: \n", model_unflatten_1)

        server.global_model = model_unflatten_1
        print("开始训练\n")
        server.train()
        #####



        ##### 5.发送生成的参数（1.编号uploaded_ids 2.权重uploaded_weights 3.每个client的模型 4.test_metrics 5.test_metrics）

        server.socket_send_parameters(server.test_metrics())
        server.socket_send_parameters(server.train_metrics())
        server.socket_send_parameters(server.uploaded_ids)
        server.socket_send_parameters(server.uploaded_weights)

        ##### 发送第一个模型的参数长度
        server.socket_send_model(server.uploaded_models[0])
        server.socket_send_model(server.uploaded_models[1])'''


run(args)





