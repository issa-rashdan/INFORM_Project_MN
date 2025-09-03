import numpy as np
import matplotlib.pyplot as plt


def visualize_cumulative_histogram(data, min_val=-100, max_val=100, percentile=99):
    """
    Visualizes a cumulative histogram for data intensities.

    Parameters:
        data (numpy.ndarray): Data array; can be 2D or 3D (if multi-channel).
        min_val (int, optional): Minimum intensity value to consider. Default is -100.
        max_val (int, optional): Maximum intensity value to consider. Default is 100.
        percentile (float, optional): Percentile value to mark (default: 99).
    """
    # Determine the number of channels: if data is 3D, the third dimension represents channels.
    num_channels = data.shape[2] if data.ndim == 3 else 1

    plt.close('all')
    fig, axes = plt.subplots(1, num_channels, figsize=(6 * num_channels, 5))
    if num_channels == 1:
        axes = [axes]

    for channel_index in range(num_channels):
        # Get data for the current channel; if data is 2D, use the whole data.
        data_channel = data[:, :, channel_index].flatten() if data.ndim == 3 else data.flatten()

        # Filter intensity values within the specified range.
        valid_mask = (data_channel >= min_val) & (data_channel <= max_val)
        valid_data = data_channel[valid_mask]

        # Compute the percentile threshold value.
        threshold_value = np.percentile(valid_data, percentile)
    
        # Plot normalized cumulative histogram for the current channel.
        axes[channel_index].hist(
            valid_data,
            bins=50,
            range=(min_val, max_val),
            color='blue',
            alpha=0.7,
            cumulative=True,
            density=True
        )
        # Draw a dashed red line at the computed percentile.
        axes[channel_index].axvline(threshold_value, color='red', linestyle='dashed', linewidth=2)
        axes[channel_index].set_title(f'Cumulative Histogram - Channel {channel_index}\n'
                                      f'{percentile}th percentile: {threshold_value:.2f}')
        axes[channel_index].set_xlabel('Intensity')
        axes[channel_index].set_ylabel('Cumulative Frequency')
        axes[channel_index].set_xlim(min_val, max_val)

    plt.tight_layout()
    plt.savefig("cumulative_histogram.jpg", format="jpg", dpi=300)
    plt.show()
