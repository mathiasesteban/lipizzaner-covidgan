from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils.data import Dataset
from data.data_loader import DataLoader
from torchvision.utils import save_image
from PIL import Image
import torch
from helpers.pytorch_helpers import denorm


BATCH_SIZE = 10
WIDTH = 28
HEIGHT = 28


class CovidDataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=1, n_batches=0, shuffle=False):
        super().__init__(COVIDDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return WIDTH*HEIGHT

    @staticmethod
    def save_images(images, shape, filename):

        img_view = images.view(images.size(0), 1, WIDTH, HEIGHT)
        save_image(denorm(img_view.data), filename)


class COVIDDataSet(Dataset):

    def __init__(self, **kwargs):
        transforms = [Grayscale(num_output_channels=1), Resize(size=[WIDTH, HEIGHT], interpolation=Image.NEAREST), ToTensor()]
        dataset = ImageFolder(root="datasets/covid", transform=Compose(transforms))

        tensor_list = []

        for img in dataset:
            tensor_list.append(img[0])

        # Remuevo los ultimos elementos que no completan un batch
        stacked_tensor = torch.stack(tensor_list)
        reminder = len(dataset) % BATCH_SIZE
        stacked_tensor = stacked_tensor[:-reminder]

        print("*************************")
        print(len(dataset))
        print(len(stacked_tensor))
        print("*************************")

        self.data = stacked_tensor


    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return len(self.data)
