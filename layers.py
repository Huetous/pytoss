import torch
import numpy as np
from utils import listify

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Module:
    """
    The main class for all layers
    """

    def __init__(self):
        self.output = None  # output of the layer
        self.grad_input = None  # gradient w.r.t. to the layer input
        self.training = True  # two modes of the layer: training or evaluate(inference)

    def forward(self, input_):
        """
        Performs the transformation
        :param input_: input tensor
        :return: tensor
        """
        pass

    def backward(self, input_, grad_output):
        """
        Computes gradients w.r.t. to the layer parameters and input
        :param input_: tensor
        :param grad_output: tensor
        :return: tensor (returns only the grad_input)
        """
        return self.grad_input

    def get_params(self):
        """
        Returns a list of the layer`s parameters
        :return: tensor of tensors
        """
        return torch.tensor([])

    def get_params_grad(self):
        """
        Returns a list of gradients computed w.r.t to the layer's parameters
        :return: tensor of tensors
        """
        return torch.tensor([])

    def zero_grad(self):
        """
        Zeros out the gradients of the layer's parameters
        """
        pass

    def train(self):
        """
        Switches the layer to its training mode
        """
        self.training = True

    def evaluate(self):
        """
        Switches the layer to its evaluating mode
        """
        self.training = False

    def __repr__(self):
        """
        Human-oriented representation of the layer
        """
        return "Module"


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = listify(args)

    def forward(self, input_):
        for module in self.modules:
            input_ = module.forward(input_)

        self.output = input_
        return self.output

    def backward(self, input_, grad_output):
        for i in range(len(self.modules) - 1, 0, -1):
            grad_output = self.modules[i].backward(self.modules[i - 1].output, grad_output)

        self.grad_input = self.modules[0].backward(input_, grad_output)
        return self.grad_input

    def get_params(self):
        return torch.stack([m.get_params() for m in self.modules])

    def get_params_grad(self):
        return torch.stack([m.get_params_grad() for m in self.modules])

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        self.training = False
        for module in self.modules:
            module.evaluate()

    def __repr__(self):
        repr = "Sequential(\n"
        repr += "".join(['\t' + str(m) + '\n' for m in self.modules])
        repr += ")\n"
        return repr

    def __getitem__(self, i):
        return self.modules.__getitem__(i)


class Linear(Module):
    """
    For detailed explanation, please visit https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        std_ = 1. / in_features
        self.weights = torch.empty((out_features, in_features), device=device,
                                   dtype=torch.float32).uniform_(-std_, std_)
        self.bias = torch.empty(out_features, device=device,
                                dtype=torch.float32).uniform_(-std_, std_) if bias else None

        self.grad_weights = None
        self.grad_bias = None

    def forward(self, input_):
        self.output = input_ @ self.weights.t() + self.bias
        return self.output

    def backward(self, input_, grad_output):
        self.grad_weights = grad_output.t() @ input_
        self.grad_bias = grad_output.sum(0)
        self.grad_input = grad_output @ self.weights
        return self.grad_input

    def zero_grad(self):
        self.grad_weights.zero_()
        self.grad_bias.zero_()

    def get_params(self):
        return torch.stack([self.weights, self.bias])

    def get_params_grad(self):
        return torch.stack([self.grad_weights, self.grad_bias])

    def __repr__(self):
        in_features, out_features = self.weights.shape
        return f"Linear(in_features={in_features}, out_features={out_features}, bias={self.bias is not None})"


class Softmax(Module):
    """
    For detailed explanation, please visit https://pytorch.org/docs/master/generated/torch.nn.Softmax.html
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_):
        # subtract max feature value for each object to provide numerical stability
        self.output = torch.sub(input_, input_.max(1, keepdim=True)[0])
        self.output.exp_()

        # Sum of exponents in each row (for each object in the batch)
        sum_exp = torch.sum(self.output, 1, keepdims=True)
        self.output.div_(sum_exp)

        return self.output

    def backward(self, input_, grad_output):
        batch_size = self.output.shape[0]

        # For each row of softmax matrix create a matrix with derivatives
        self.grad_input = torch.stack([
            torch.diag(self.output[i]) - torch.outer(self.output[i], self.output[i])
            for i in range(batch_size)
        ])

        # Multiply exactly one row of grad_output with exactly one derivative matrix.
        # To perform the operation we add required dimension and then remove it
        # shapes: (batch_size, 1, n_features) @ (batch_size, n_features, n_features)
        # result: (batch_size, 1, n_features)
        self.grad_input = (grad_output.unsqueeze(1) @ self.grad_input).squeeze()
        return self.grad_input

    def __repr__(self):
        return "Softmax()"


class LogSoftmax(Module):
    """
    For detailed explanation, please visit https://pytorch.org/docs/master/generated/torch.nn.LogSoftmax.html
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_):
        # subtract max feature value for each object to provide numerical stability
        self.output = torch.sub(input_, input_.max(1, keepdim=True)[0])

        # we store softmax values for future use in backward pass
        self.softmax = torch.exp(self.output)
        sum_exp = torch.sum(self.softmax, 1, keepdim=True)
        self.softmax.div_(sum_exp)

        self.output.sub_(torch.log(sum_exp))
        return self.output

    def backward(self, input_, grad_output):
        batch_size, n_features = input_.shape

        # For each row of logsoftmax matrix create a matrix with derivatives
        self.grad_input = torch.stack([
            torch.eye(n_features) - self.softmax[i] for i in range(batch_size)
        ])

        # Multiply exactly one row of grad_output with exactly one derivative matrix.
        # To perform the operation we add required dimension and then remove it
        # shapes: (batch_size, 1, n_features) @ (batch_size, n_features, n_features)
        # result: (batch_size, 1, n_features)
        self.grad_input = (grad_output.unsqueeze(1) @ self.grad_input).squeeze()
        return self.grad_input

    def __repr__(self):
        return "LogSoftmax()"


class BatchNorm2d(Module):
    """
    For detailed explanation, please visit https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """

    def __init__(self, n_filters, momentum=0., eps=1e-3):
        super().__init__()

        self.momentum = momentum
        self.eps = eps

        self.running_mean = torch.zeros((1, n_filters, 1, 1), device=device, dtype=torch.float32)
        self.running_var = torch.ones((1, n_filters, 1, 1), device=device, dtype=torch.float32)

        # scaling factors
        std_ = 1. / np.sqrt(n_filters)
        self.gammas = torch.empty((n_filters, 1, 1), device=device, dtype=torch.float32).uniform_(-std_, std_)
        self.betas = torch.empty((n_filters, 1, 1), device=device, dtype=torch.float32).uniform_(-std_, std_)

        self.grad_gammas = torch.zeros_like(self.gammas, device=device, dtype=torch.float32)
        self.grad_betas = torch.zeros_like(self.betas, device=device, dtype=torch.float32)

    def forward(self, input_):
        self.output = input_

        if self.training:
            output = self.output.detach()
            mean = output.mean((0, 2, 3), keepdim=True)
            var = output.var((0, 2, 3), keepdim=True)

            self.running_mean.lerp_(mean, self.momentum)
            self.running_var.lerp_(var, self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        # normalize input
        # self.output.sub_(mean)
        self.output = self.output - mean
        # self.output.div_((var + self.eps).sqrt())
        self.output = self.output / (var + self.eps).sqrt()

        # rescale input
        # self.output.mul_(self.gammas)
        # self.output.add_(self.betas)
        self.output = self.output * self.gammas - self.betas
        return self.output

    def backward(self, input_, grad_output):
        # 1. Compute gradient w.r.t. to gammas and betas
        t = grad_output * input_
        self.grad_betas = grad_output.sum((0, 2, 3))
        self.grad_gammas = torch.sum(t, (0, 2, 3))


        grad_output.mul_(self.gammas)

        # 2. Compute gradient w.r.t to input
        mean = input_.mean((0, 2, 3), keepdim=True)
        var = input_.var((0, 2, 3), keepdim=True)
        batch_size = input_.shape[0]

        grad_var = torch.sum((-grad_output / 2.) * (input_ - mean) * (var + self.eps) ** (-1.5), (0, 2, 3),
                             keepdim=True)

        grad_mean = torch.sum(-grad_output / torch.sqrt(var + self.eps), (0, 2, 3),
                              keepdim=True) - 2. * grad_var * torch.mean(input_ - mean, (0, 2, 3), keepdim=True)
        # grad_mean.sub_(2 * grad_var * torch.mean(input_ - mean, (0, 2, 3), keepdim=True))

        self.grad_input = grad_output / torch.sqrt(var + self.eps) + 2. * (grad_var / batch_size) * (
                input_ - mean) + grad_mean / batch_size
        # self.grad_input.sum_(2 * (grad_var / batch_size) * (input_ - mean))
        # self.grad_input.sum_(grad_mean / batch_size)

        return self.grad_input

    def zero_grad(self):
        self.grad_gammas.fill_(0)
        self.grad_betas.fill_(0)

    def get_params(self):
        return torch.stack([self.gammas, self.betas])

    def get_params_grad(self):
        return torch.stack([self.grad_gammas, self.grad_betas])

    def __repr__(self):
        n_filters = self.gammas.shape[0]
        return f"BatchNorm2d(n_filters={n_filters}, momentum={self.momentum}, eps={self.eps})"
