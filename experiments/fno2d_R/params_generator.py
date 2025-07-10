import itertools
import os


params_path = "experiments/fno2d_R/params"

n_layers = [4, 8, 16]
hidden_channels = [16, 32, 64]
n_modes = [32, 64, 128]
epochs = 1000

if os.path.exists(params_path):
    os.remove(params_path)


with open(params_path, "a") as f:
    for _ in range(10):
        for nl, hc, nm in itertools.product(n_layers, hidden_channels, n_modes):

            if nm == 128 and hc == 64:
                continue

            f.write(f"-l {nl} -c {hc} -m {nm} -e {epochs}\n")
