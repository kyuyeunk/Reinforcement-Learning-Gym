import cv2
import torch
from shared.utils import data_to_torch


class ImageProcessDimension:
    def __init__(self, height, width, crop_bottom, crop_top, crop_left, crop_right):
        self.height = height
        self.width = width
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_bottom = crop_bottom
        self.crop_top = crop_top


# input data is in [height][width][channel]
# output data is in [frame_number][height][width] (color channel is converted to grey scale)
# last frame is the latest frame
class ImageProcessor:
    def __init__(self, n_stack, dimension, device):
        self.n_stack = n_stack

        self.height = dimension.height
        self.width = dimension.width

        self.crop_left = dimension.crop_left
        self.crop_right = dimension.crop_right
        self.crop_bottom = dimension.crop_bottom
        self.crop_top = dimension.crop_top

        self.device = device

        self.stack = self.get_start_screen()

    def get_start_screen(self):
        return torch.zeros([self.n_stack, self.height, self.width], device=self.device)

    def process_screen(self, data):
        cropped_data = data[self.crop_bottom:self.crop_top, self.crop_left:self.crop_right]
        resized_data = cv2.resize(cropped_data, (self.width, self.height))
        grey_data = cv2.cvtColor(resized_data, cv2.COLOR_RGB2GRAY)

        torch_data = data_to_torch([grey_data/255], torch.float32, self.device)

        self.stack = torch.cat((self.stack[1:], torch_data))
        return self.stack.unsqueeze(0)

    def write_processed_data(self, data):
        cropped_data = data[self.crop_bottom:self.crop_top, self.crop_left:self.crop_right]
        resized_data = cv2.resize(cropped_data, (self.width, self.height))
        grey_data = cv2.cvtColor(resized_data, cv2.COLOR_RGB2GRAY)

        cv2.imwrite("data.png", data)
        cv2.imwrite("cropped.png", cropped_data)
        cv2.imwrite("resized.png", resized_data)
        cv2.imwrite("grey.png", grey_data)

    def write_stack_data(self):
        for i, image in enumerate(self.stack):
            cv2.imwrite("stack{}.png".format(i), (image * 255).cpu().numpy())

    def reset_screen(self):
        self.stack = self.get_start_screen()


