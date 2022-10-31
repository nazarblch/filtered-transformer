from torch import nn


class Accumulator(nn.Module):

    def __init__(self, module: nn.Module, module_acc: nn.Module, decay=0.9):
        super().__init__()

        self.module = module
        self.module_acc = module_acc
        self.module_acc.load_state_dict(module.state_dict())
        self.decay = decay

    def accumulate(self):
        params = dict(self.module.named_parameters())
        acc_params = dict(self.module_acc.named_parameters())

        for k in params.keys():
            acc_params[k].data.mul_(self.decay)
            acc_params[k].data += (1 - self.decay) * params[k].data

    def forward(self, *args, **kw):
        return self.module_acc.forward(*args, **kw)

    def dump(self):
        params = dict(self.module.named_parameters())
        acc_params = dict(self.module_acc.named_parameters())

        for k in params.keys():
            params[k].data = params[k].data * 0 + acc_params[k].data
