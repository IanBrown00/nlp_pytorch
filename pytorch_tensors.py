from loguru import logger
import numpy as np
import torch


def describe(x) -> None:
    logger.info("Type: {}".format(x.type()))
    logger.info("Shape: {}".format(x.shape))
    logger.info("Values: \n{}".format(x))


def make_tensor() -> None:
    describe(torch.FloatTensor(2, 3))


def make_rand_tensors() -> None:
    describe(torch.rand(2, 3))
    describe(torch.randn(4, 2))


def make_const_tensors() -> None:
    describe(torch.ones(1, 6))
    describe(torch.ones(1, 6).fill_(3))


def make_tensor_from_np() -> None:
    describe(torch.from_numpy(np.random.rand(3, 2)))


def make_tensor_from_list() -> None:
    describe(torch.FloatTensor([[1, 2, 3], [3, 2, 1]]))


if __name__ == "__main__":
    make_tensor()
    make_rand_tensors()
    make_const_tensors()
    make_tensor_from_np()
    make_tensor_from_list()

