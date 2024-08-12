import torch
import matplotlib.pyplot as plt


def display_generated_samples(model, class_labels, latent_size, num_samples=5, device='cuda'):
    model.eval()

    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, num_samples * 3))

        for i in range(num_samples):
            z = torch.randn(1, latent_size).to(device)

            c_one_hot = torch.zeros(1, model.class_size).to(device)
            c_one_hot[0, class_labels[i]] = 1.0

            generated_signal = model.decode(z, c_one_hot).cpu().numpy().flatten()

            ax = axes[i]
            ax.plot(generated_signal[200:400])
            ax.set_title(f"Generated Signal - Class {class_labels[i]}")
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')

        plt.tight_layout()
        plt.show()
