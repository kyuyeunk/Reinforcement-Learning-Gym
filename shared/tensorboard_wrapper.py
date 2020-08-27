from torch.utils.tensorboard import SummaryWriter
import torchvision


class TensorboardWrapper:
    def __init__(self, game, algorithm, now_str):
        self.writer = SummaryWriter("runs/{}_{}_{}".format(game, algorithm, now_str))

    def save_images(self, torch_array, name, epoch):
        for i in range(torch_array.shape[1]):
            img = torchvision.utils.make_grid(torch_array[:, [i], :, :])
            self.writer.add_image("{} {}".format(name, i), img, epoch)
        img = torchvision.utils.make_grid(torch_array[:, :, :, :])
        self.writer.add_image(name, img, epoch)

    def save_scalars(self, data, epoch):
        for key in data:
            self.save_scalar(key, data[key], epoch)

    def save_scalar(self, name, data, epoch):
        if type(data) == list:
            if len(data) > 0:
                avg = sum(data) / len(data)
                self.writer.add_scalar(name, avg, epoch)
        else:
            self.writer.add_scalar(name, data, epoch)

    def save_hparam(self, param, metric):
        self.writer.add_hparams(param, metric)

    def save_model(self, policy, dataset):
        self.writer.add_graph(policy, dataset)

    def close(self):
        self.writer.close()
