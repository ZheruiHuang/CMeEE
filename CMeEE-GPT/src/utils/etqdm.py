from tqdm import tqdm

def etqdm(iterable, **kwargs):
    return tqdm(iterable, bar_format="{l_bar}{bar:20}{r_bar}", colour="yellow", **kwargs)