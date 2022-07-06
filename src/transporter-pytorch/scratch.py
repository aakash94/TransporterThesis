import matplotlib.pyplot as plt
import numpy as np

from train_pong import get_data_loader_wo_config




def main():
    loader = get_data_loader_wo_config()
    # print(len(loader))

    for itr, (xt, xtp1) in enumerate(loader):
        print(itr)
        print("xt", type(xt), xt.size())
        print("xtp1", type(xtp1), xtp1.size())
        show_tensor(xt)
        print("done")
        break


if __name__ == "__main__":
    main()
