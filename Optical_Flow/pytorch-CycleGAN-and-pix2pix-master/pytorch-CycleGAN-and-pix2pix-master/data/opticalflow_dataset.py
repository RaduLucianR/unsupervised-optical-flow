import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np


class OpticalflowDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A' + '/frame1')  # create a path '/path/to/data/trainA'
        self.dir_flow = os.path.join(opt.dataroot, opt.phase + 'A' + '/flow')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA/frame1'
        self.flow_paths = sorted(make_dataset(self.dir_flow, opt.max_dataset_size))   # load images from '/path/to/data/trainA/flow'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.flow_size = len(self.flow_paths) 
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_flow = get_transform(self.opt, grayscale=(input_nc == 1), tensor = True)
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        flow_path = self.flow_paths[index % self.flow_size]

        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)

        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        flow_img = self.load_flow_from_png(flow_path)
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        #print(A_img.size)
        A = self.transform_A(A_img)
        flow = torch.from_numpy(flow_img)
        #print(flow.size())
        flow = flow.permute(2, 1, 0) #300 1200 3
        #print(flow.size())
        flow = self.transform_flow(flow)
        B = self.transform_B(B_img)

        #print(A.size())
        #print(flow.size())
        #print(B.size())

        #concatenation
        #breakpoint()
        A = torch.cat((A,flow), 0)
        # print(A.size())
        #breakpoint()

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def load_flow_from_png(self, png_path):
        # The -1 is here to specify not to change the image depth (16bit), and is compatible
        # with both OpenCV2 and OpenCV3
        flo_file = cv2.imread(png_path, -1)
        flo_img = flo_file[:,:,2:0:-1].astype(np.float32)
        invalid = (flo_file[:,:,0] == 0)
        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img[invalid, :] = 0
        return(flo_img) #Image.fromarray(flo_img, "I;16")
