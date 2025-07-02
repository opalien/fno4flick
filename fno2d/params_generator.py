import itertools
import os


params_path = "fno2d/params"

n_layers = [4, 8, 16]
hidden_channels = [32, 64, 128]
n_modes = [64, 128, 256]
epochs = 500

if os.path.exists(params_path):
    os.remove(params_path)

with open(params_path, "a") as f:
    for _ in range(10):
        for nl, hc, nm in itertools.product(n_layers, hidden_channels, n_modes):
            f.write(f"-l {nl} -c {hc} -m {nm} -e {epochs}\n")
