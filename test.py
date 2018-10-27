import dynet as dy
m = dy.Model()
W = m.add_parameters((4,3))
b = m.add_parameters(4)
x = dy.inputVector([1,2,3])
y = b + W * x
print(x.dim())
z = dy.affine_transform([b, W , x])
print(y.dim(), z.dim())

