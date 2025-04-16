import os
import click

all_config = [
    ("wave_oval", 0.0, 3, 12, 1.0, 128, 128, 20),
    ("wave_oval", 0.1, 3, 12, 2.0, 512, 256, 20),
    ("wave_oval", 0.2, 3, 12, 1.0, 512, 128, 20),
    ("wave_z", 0.0, 4, 8, 2.0, 512, 256, 20),
    ("wave_z", 0.1, 2, 8, 2.0, 64, 256, 20),
    ("wave_z", 0.2, 4, 8, 1.0, 12, 128, 20),
    ("darcy_curve", 0.0, 4, 32, 1.0, 32, 256, 20),
    ("darcy_curve", 0.1, 4, 16, 2.0, 128, 256, 20),
    ("darcy_curve", 0.2, 4, 12, 1.0, 64, 256, 20),
    ("darcy_line", 0.0, 4, 32, 1.0, 64, 256, 20),
    ("darcy_line", 0.1, 4, 12, 1.0, 12, 128, 20),
    ("darcy_line", 0.2, 4, 12, 1.0, 12, 256, 20),
    ("ns", 0.0, 4, 8, 1.0, 12, 64, 20),
    ("ns", 0.1, 3, 8, 1.0, 64, 256, 20),
    ("ns", 0.2, 4, 8, 1.0, 64, 128, 20),
]
def get_config(index):
    config_l = []
    count = 0

    for seed in range(5):
        for config in all_config:
            dst, NL, nLayers, mode, beta, rank, hidden, padding = config
            config_l.append((NL, nLayers, mode, beta, rank, hidden, padding, dst, count, seed))
            count += 1
    return config_l[index]


def run(index):
    NL, nLayers, mode, beta, rank, hidden, padding, dst, count, seed = get_config(index)

    if dst == "darcy_line":
        os.system(
            f"python darcy_line.py --modes={mode} --hidden={hidden} --beta={beta} --nl={NL} --rank={rank} --kl=0.000001 --padding={padding} --epochs-VAE=1000 \
                                 --epochs-IFNO=600 --epochs=500 --n-train=300 --n-test=500 --batchsize=10 --batchsize2=100 --batchsize3=20 --lr-VAE=0.001 \
                                 --lr-IFNO=0.00025 --lr-forward=0.000005 --lr-backward=0.000005 --seed={seed} --count={count} --n-layers={nLayers} --n-valid=100")
    elif dst == "darcy_curve":
        os.system(
            f"python darcy_curve.py --modes={mode} --hidden={hidden} --beta={beta} --nl={NL} --rank={rank} --kl=0.000001 --padding={padding} --epochs-VAE=1000 \
                                 --epochs-IFNO=400 --epochs=100 --n-train=800 --n-test=200 --batchsize=10 --batchsize2=100 --batchsize3=20 --lr-VAE=0.001 \
                                 --lr-IFNO=0.00025 --lr-forward=0.000005 --lr-backward=0.000005 --seed={seed} --count={count} --n-layers={nLayers} --n-valid=100")
    elif dst == "ns":
        os.system(
            f"python ns.py --modes={mode} --hidden={hidden} --beta={beta} --nl={NL} --rank={rank} --kl=0.01 --padding={padding} --epochs-VAE=100 --epochs-IFNO=500 \
                                 --epochs=100 --n-train=1000 --n-test=200 --batchsize=10 --batchsize2=200 --batchsize3=50 --batchsize4=50 --lr-VAE=0.0001 \
                                 --lr-IFNO=0.0005 --lr-forward=0.00005 --lr-backward=0.00005 --seed={seed} --count={count} --n-layers={nLayers} --n-valid=100")
    elif dst == "wave_oval":
        os.system(
            f"python wave_oval.py --modes={mode} --hidden={hidden} --beta={beta} --nl={NL} --rank={rank} --kl=0.01 --padding={padding} --epochs-VAE=200 \
                                 --epochs-IFNO=120 --epochs=300 --n-train=400 --n-test=100 --batchsize=10 --batchsize2=100 --batchsize3=10 --lr-VAE=0.0001 \
                                 --lr-IFNO=0.0005 --lr-forward=0.0001 --lr-backward=0.0001 --seed={seed} --count={count} --n-layers={nLayers} --n-valid=100")
    elif dst == "wave_z":
        os.system(
            f"python wave_Z.py --modes={mode} --hidden={hidden} --beta={beta} --nl={NL} --rank={rank} --kl=0.01 --padding={padding} --epochs-VAE=1000 \
                                 --epochs-IFNO=600 --epochs=1000 --n-train=400 --n-test=100 --batchsize=10 --batchsize2=100 --batchsize3=10 --lr-VAE=0.0001 \
                                 --lr-IFNO=0.005 --lr-forward=0.0001 --lr-backward=0.0001 --seed={seed} --count={count} --n-layers={nLayers} --n-valid=100")
    else:
        raise NotImplementedError

@click.command()
@click.option("--index", type=int, required=True, help="Which grid index to run.")
def main(index):
    start = 15 * index
    end = start + 15
    for i in range(start, end):
        run(i)


if __name__ == "__main__":
    main()
