import unittest
import torch
from layers import *


class LayerTests(unittest.TestCase):
    def test_Linear(self):
        torch.manual_seed(42)

        batch_size, n_in, n_out = 3, 4, 5
        for _ in range(100):
            torch_layer = torch.nn.Linear(n_in, n_out)
            custom_layer = Linear(n_in, n_out)
            custom_layer.weights = torch_layer.weight.data
            custom_layer.bias = torch_layer.bias.data

            layer_input = torch.rand((batch_size, n_in), dtype=torch.float32, requires_grad=True)
            next_layer_grad = torch.rand((batch_size, n_out), dtype=torch.float32)

            # 1. Check layer output
            custom_layer_output = custom_layer.forward(layer_input)
            torch_layer_output = torch_layer(layer_input)
            self.assertTrue(torch.allclose(torch_layer_output.data, custom_layer_output, atol=1e-6))

            # 2. Check layer input gradient
            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output.backward(next_layer_grad)
            self.assertTrue(torch.allclose(layer_input.grad.data, custom_layer_grad, atol=1e-6))

            # 3. Check layer parameters gradient
            self.assertTrue(torch.allclose(torch_layer.weight.grad.data, custom_layer.grad_weights, atol=1e-6))
            self.assertTrue(torch.allclose(torch_layer.bias.grad.data, custom_layer.grad_bias, atol=1e-6))

    def test_Softmax(self):
        torch.manual_seed(42)

        batch_size, n_in = 3, 6
        for _ in range(100):
            torch_layer = torch.nn.Softmax(dim=1)
            custom_layer = Softmax()

            layer_input = torch.rand((batch_size, n_in), dtype=torch.float32, requires_grad=True)
            next_layer_grad = torch.rand((batch_size, n_in), dtype=torch.float32)
            next_layer_grad.div_(next_layer_grad.sum(-1, keepdims=True))
            next_layer_grad.clip_(1e-5, 1.)
            next_layer_grad = 1. / next_layer_grad

            # 1. Check layer output
            custom_layer_output = custom_layer.forward(layer_input)
            torch_layer_output = torch_layer(layer_input)
            self.assertTrue(torch.allclose(torch_layer_output.data, custom_layer_output, atol=1e-5))

            # 2. Check layer input gradient
            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output.backward(next_layer_grad)
            self.assertTrue(torch.allclose(layer_input.grad.data, custom_layer_grad, atol=1e-5))

    def test_LogSoftmax(self):
        torch.manual_seed(42)

        batch_size, n_in = 3, 6
        for _ in range(100):
            torch_layer = torch.nn.LogSoftmax(dim=1)
            custom_layer = LogSoftmax()

            layer_input = torch.rand((batch_size, n_in), dtype=torch.float32, requires_grad=True)
            next_layer_grad = torch.rand((batch_size, n_in), dtype=torch.float32)
            next_layer_grad.div_(next_layer_grad.sum(-1, keepdims=True))

            # 1. Check layer output
            custom_layer_output = custom_layer.forward(layer_input)
            torch_layer_output = torch_layer(layer_input)
            self.assertTrue(torch.allclose(torch_layer_output.data, custom_layer_output, atol=1e-6))

            # 2. Check layer input gradient
            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output.backward(next_layer_grad)
            self.assertTrue(torch.allclose(layer_input.grad.data, custom_layer_grad, atol=1e-6))

    def test_BatchNorm2d(self):
        torch.manual_seed(42)

        batch_size, in_channels, height, width = 32, 3, 16, 16
        for _ in range(100):
            momentum = 0.9
            custom_layer = BatchNorm2d(in_channels, momentum=momentum)
            custom_layer.train()
            torch_layer = torch.nn.BatchNorm2d(in_channels, eps=custom_layer.eps, momentum=0.1)
            custom_layer.running_mean = torch_layer.running_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            custom_layer.running_var = torch_layer.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            custom_layer.gammas = torch_layer.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            custom_layer.betas = torch_layer.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            layer_input = torch.rand((batch_size, in_channels, height, width),
                                     requires_grad=True,
                                     dtype=torch.float32
                                     )

            next_layer_grad = torch.rand((batch_size, in_channels, height, width),
                                         dtype=torch.float32)

            # # 1. Check layer output
            # custom_layer_output = custom_layer.forward(layer_input)
            torch_layer_output = torch_layer(layer_input)
            # self.assertTrue(torch.allclose(torch_layer_output.data, custom_layer_output, atol=1e-4))

            # 2. Check layer input grad
            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output.backward(next_layer_grad)
            print(layer_input.grad.data[0][0][0])
            print(custom_layer_grad[0][0][0])
            self.assertTrue(torch.allclose(layer_input.grad.data, custom_layer_grad, atol=1e-5))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(LayerTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
