import random
import torch
from torchvision import transforms, datasets
import numpy as np
import torchvision.transforms.functional as F


torch.manual_seed(10)
random.seed(10)
np.random.seed(10)


transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
])
perm1 = np.random.permutation(28 * 28)
perm2 = np.random.permutation(3136)


class PermMNISTTaskGenerator:

    def __init__(self, is_train, padding=False, do_perm=True):
        dataset = datasets.MNIST('/tmp/data', train=is_train, download=True, transform=transform)

        self._curr_episode_idx = 0

        self.data = []

        self.n = len(dataset)
        print("n=", self.n)
        for i in range(self.n):
            image = dataset[i][0].reshape(-1, 1).numpy()
            label = dataset[i][1]
            pad_size = 14 if padding else 0
            if pad_size > 0:
                image = F.to_tensor(F.pad(F.to_pil_image(image), pad_size)).reshape(-1, 1).numpy()
            if do_perm:
                perm = perm2 if padding else perm1
                image = image[perm]
            self.data.append({"x": image,
                              "s": np.array((label,), dtype=np.int64)})

        self.data = np.asarray(self.data, dtype=object)
        np.random.shuffle(self.data)

        self.pos = 0


    def gen_trajectory(self):

        element = self.data[self.pos]
        p = self.pos
        self.pos += 1
        self.pos = self.pos % self.n

        return {"x": element["x"], "s": element["s"], "pos": p}


