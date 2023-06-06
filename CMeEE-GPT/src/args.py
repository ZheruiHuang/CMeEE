import argparse
from easydict import EasyDict


def GPTParser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--Debug", "--debug", default=False, action="store_true", help="whether to debug")
    parser.add_argument("--VPN", "--vpn", default=False, action="store_true", help="whether to use VPN")
    parser.add_argument("--port", type=int, default=33210, help="port")

    parser.add_argument("-c", "--c", type=str, default="random", help="the method we pick cluster")
    parser.add_argument("-n", "--num_shots", type=int, default=2, help="num of shots we pick")
    parser.add_argument("-p", "--prompt", type=str, default="base", help="the way we add prompt")
    parser.add_argument("--data", type=str, default="select_50", help="the data we use")

    opt = parser.parse_args()
    opt = EasyDict(opt.__dict__)
    return opt
