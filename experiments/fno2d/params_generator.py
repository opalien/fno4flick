import itertools
import os


params_path = "experiments/fno2d/params"

n_layers = [2, 4, 6]
hidden_channels = [32, 64, 128]
n_modes = [32, 64, 128, 256]
epochs = 1000

if os.path.exists(params_path):
    os.remove(params_path)

with open(params_path, "a") as f:
    for _ in range(1):
        for nl, hc, nm, bs in [[2, 64, 64, 32], [4, 64, 64, 16], [4, 16, 16, 64], [4, 32, 32, 32], [4, 128, 32, 16], [2, 128, 128, 16]]: #itertools.product(n_layers, hidden_channels, n_modes):
            f.write(f"-l {nl} -m {nm} -c {hc} -e {epochs} -d data/ -n fno2d_nl_{nl}_hc_{hc}_nm_{nm} -b {bs} \n")
