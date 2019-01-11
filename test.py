# import dynet as dy
# m = dy.Model()
# W = m.add_parameters((4,3))
# b = m.add_parameters(4)
# x = dy.inputVector([1,2,3])
# y = b + W * x
# print(x.dim())
# z = dy.affine_transform([b, W , x])
# print(y.dim(), z.dim())
#
# lookup = m.add_lookup_parameters((10, 20))
# lookup.init_row()
#
# x = 4
# assert x == 3


# import numpy as np
# # Note: please import dynet_config before import dynet
# import dynet_config
# # set random seed to have the same result each time
# dynet_config.set(random_seed=0)
# import dynet as dy
#
# x = dy.inputTensor(np.random.rand(10,10,1))
# model = dy.Model()
# #
# W = model.add_parameters((5, 5, 1, 10))
# b = model.add_parameters((10,1))
# q = model.add_parameters((10, 1))
# q2 = model.add_parameters((10))
# print(b.dim())
# print(q.dim())
# print(q2.dim()[0][0])
# convds = dy.conv2d_bias(x, W, b, stride=(1,1))
#
# print(convds.value())
# print(convds.dim())
# dy.renew_cg()
# x = [0, 2, 1, 2, 1]
# model = dy.Model()
#
# vocab_size = 3
# emb_size = 6
# num_filer = 4
# win_size = 3
#
# emb = model.add_lookup_parameters((vocab_size, emb_size))
# # print(emb.value())
# # print(emb.dim())
#
# print(emb[x[0]].dim()[0][0])
#
# print(emb[x[0]].value())
#
# # q = dy.reshape(emb[x[0]], (1, 1, emb[x[]]))
# # print(q.dim())
#
# W_cnn = model.add_parameters((1, win_size, emb_size, num_filer))
# b_cnn = model.add_parameters((num_filer))
# #
# cnn_in = dy.concatenate([dy.reshape(emb[i], (1, 1, emb[i].dim()[0][0]))  for i in x], d=1)
# print(cnn_in.dim())
# #
# cnn_out = dy.conv2d_bias(cnn_in, W_cnn, b_cnn,stride=(1,1))
#
# print(cnn_out.dim())
#
# # x = dy.inputVector([3,2,1])
# print(x.dim()[0])


import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda:0')
# word_embedding = nn.Embedding(10, 25).to(device)


num_indexs = 100
x= torch.randn(num_indexs)
print(x.size())
num_row = 15
num_edges = 6


iter = 1000000
import time
start = time.time()
for i in range(iter):
    index = torch.randint(0, 100, (num_row, num_edges), dtype=torch.long)
    x_view = x.view(1, -1).expand(num_row, -1)
    x_expr = torch.gather(x_view, 1, index)

end = time.time()
print("time is {:.5f}s".format(end-start))

start = time.time()
for i in range(iter):
    index = torch.randint(0, 100, (num_row, num_edges), dtype=torch.long)
    # x_view = x.view(1, -1).expand(num_row, -1)
    # x_expr = torch.gather(x_view, 1, index)
    x_expr = torch.take(x, index)
end = time.time()

print("time is {:.5f}s".format(end-start))

