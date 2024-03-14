# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import copy
import torch

def deep_leakage_from_gradients(model, origin_grad):
    dummy_data = torch.randn(size=(28,28))
    dummy_label = torch.randn(size=(28,28))
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    for iters in range(300):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = criterion(dummy_pred, F.softmax(dummy_label, dim=-1))
            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = sum(((dummy_grad - origin_grad) ** 2).sum() \
                            for dummy_g, origin_g in zip(dummy_grad, origin_grad))

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

    return dummy_data, dummy_label


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            ##### 3.接收模型参数并生成模型
            embedding_back = self.socket_receive_model()
            model_init = copy.deepcopy(self.global_model)
            model_unflatten_1 = self.unflatten(model_init, embedding_back)
            print("从server端收到的模型: \n", model_unflatten_1)

            self.global_model = model_unflatten_1
            print("开始训练\n")
            #####
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            
            
            
            
            
            ##### 5.发送生成的参数（1.编号uploaded_ids 2.权重uploaded_weights 3.每个client的模型 4.test_metrics 5.test_metrics）

            self.socket_send_parameters(self.test_metrics())
            self.socket_send_parameters(self.train_metrics())
            self.socket_send_parameters(self.uploaded_ids)
            self.socket_send_parameters(self.uploaded_weights)

            ##### 发送第一个模型的参数长度
            for i in range(self.num_clients):
                self.socket_send_model(self.uploaded_models[i])
            
            
            
            self.selected_clients = self.select_clients()
            self.send_models()


        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
            
