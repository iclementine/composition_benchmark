import pdx as fx
import paddle


x = paddle.rand([3, 8])
x.stop_gradient = False
y = paddle.rand([3, 1])
y.stop_gradient = False

# broadcast_tensors has a bug
x2, y2 = paddle.broadcast_tensors((x, y))
z = x2 + y2

print(z)
z.sum().backward()
print(y.grad)