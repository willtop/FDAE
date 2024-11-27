import numpy as np
import matplotlib.pyplot as plt

model_dir = "logs/2024-11-26-12-29-24-celeba_FDAE_seed0_rank0/"
loss_file = "progress_celeba_FDAE_seed0_rank0.csv"

loss_names_to_examine = ['content_decorrelation_loss',
                         'lg_loss_scale',
                         'loss',
                         'mask_entropy_loss',
                         'mse']


if __name__ == "__main__":
    with open(model_dir+loss_file) as f:
        loss_names = f.readline()
    loss_names = loss_names.split(',')

    # don't use names=True from genfromtxt since it would result in tuple returned
    losses = np.genfromtxt(model_dir+loss_file, delimiter=',', skip_header=1, filling_values=0)

    fig, axes = plt.subplots(nrows=1, ncols=len(loss_names_to_examine), 
                             figsize=(len(loss_names_to_examine)*5, 5))
    ax_counter = 0
    for i, loss_name in enumerate(loss_names):
        if loss_name in loss_names_to_examine:
            loss_vals = losses[:,i]
            axes[ax_counter].set_title(f"Loss curve for {loss_name}")
            axes[ax_counter].plot(loss_vals)
            ax_counter += 1
    plt.savefig(model_dir+"loss_curves.pdf", format="pdf", bbox_inches='tight')
    print("script finished!")