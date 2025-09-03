from data.load_data import Echogram
import numpy as np
from data.plotting import setup_matplotlib
import matplotlib.colors as mcolors
from method.pca import PCAonGPU, rescale_each_channel_for_visualization


class EchogramRGB():
    def __init__(self,
                 e: Echogram) -> None:
        self.e = e
        self.pca_features = None

    def extract_pca_features(self):
        self.pca_features = []  # Reset the PCA features list
        for label_chunk, data_db_img_chunk, seabed_chunk, time_vector_chunk, mask_chunk in self.e.generate_echogram():
            data_rgb_tensor, pca = PCAonGPU(data_db_img_chunk, mask_chunk, device=self.e.device)
            self.pca_features.append({
                'label': label_chunk,
                'data_rgb_tensor': data_rgb_tensor,
                'seabed': seabed_chunk,
                'time_vector': time_vector_chunk,
                'mask': mask_chunk,
                'raw_data': data_db_img_chunk
            })

    def visualize_rgb(self,
                      predictions=None,
                      frequencies=None,
                      draw_seabed=True,
                      show_grid=True,
                      show_name=True,
                      show_labels_str=True,
                      return_fig=True,
                      figure=None):
        """ Visualize echogram, and optionally predictions """

        ### Parameters
        # predictions (2D numpy array): each value is a proxy for probability of fish at given pixel coordinate
        # pred_contrast (positive float): exponent for prediction values to adjust contrast (predictions -> predictions^pred_contrast)
        ###

        if self.pca_features is None:
            self.extract_pca_features()
                
        # Get data
        if frequencies is None:
            frequencies = self.e.frequencies_of_interest


        # Tick labels Y
        tick_labels_y = self.e.range_vector
        tick_labels_y = tick_labels_y - np.min(tick_labels_y)
        tick_idx_y = np.arange(start=0, stop=len(tick_labels_y), step=int(len(tick_labels_y) / 4))

        # Format settings
        color_seabed = {'seabed': 'white'}
        lw = {'seabed': 1}
        cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'red', 'green'])
        boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
        norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

        global_title = ''
        if show_name:
            global_title += self.e.name + ' '

        for j, features in enumerate(self.pca_features):
            label = features['label']
            data_rgb_tensor = features['data_rgb_tensor']
            seabed = features['seabed']
            time_vector = features['time_vector']
            mask = features['mask']

            # Initialize plot
            plt = setup_matplotlib()
            if figure is not None:
                plt.clf()
            plt.tight_layout()

            # Tick labels X
            tick_labels_x = time_vector * 24 * 60
            tick_labels_x = tick_labels_x - np.min(tick_labels_x)
            tick_idx_x = np.arange(start=0, stop=len(tick_labels_x), step=int(len(tick_labels_x) / 6))

            # Number of subplots
            n_plts = 2 + 2 # data_rgb + scaled_rgb + selected mask + label
            if predictions is not None:
                if type(predictions) is np.ndarray:
                    n_plts += 1
                elif type(predictions) is list:
                    n_plts += len(predictions)

            # Channels
            plt.figure(figsize=(label.shape[1]//50, 
                                label.shape[0]//50*(len(frequencies)+2+(sum(1 for var in [predictions] if var is not None)))))
            major_font_size = label.shape[1]//250
            minor_font_size = major_font_size//2

            # 3ch shift without scaling to range [0, ]
            i = 0
            wo_scaling = data_rgb_tensor.cpu().numpy()
            wo_scaling[mask] -= wo_scaling.min()

            main_ax = plt.subplot(n_plts, 1, i + 1)
            plt.suptitle(global_title, fontsize=major_font_size)
            plt.text(0.5, 0.9, 'w/o channel scaling', fontsize=major_font_size, ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
            plt.imshow(wo_scaling, cmap='jet', vmin=0, aspect='auto')
            if not show_grid:            # Hide grid
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=minor_font_size)
                plt.xticks(tick_idx_x, [int(tick_labels_x[j]) for j in tick_idx_x], fontsize=minor_font_size)
                # plt.ylabel("Depth\n[meters]", fontsize=minor_font_size)
            if draw_seabed:
                plt.plot(np.arange(label.shape[1]), seabed, c=color_seabed['seabed'], lw=lw['seabed'])

            # 3ch with rescaling to range [0, 1]
            i += 1 
            scaled_rgb_tensor= rescale_each_channel_for_visualization(data_rgb_tensor)
            w_scaling = np.zeros_like(scaled_rgb_tensor.cpu().numpy())
            w_scaling[mask] = scaled_rgb_tensor.cpu().numpy()[mask]

            plt.subplot(n_plts, 1, i + 1, sharex = main_ax, sharey = main_ax)
            plt.text(0.5, 0.9, 'w/ channel scaling', fontsize=major_font_size, ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
            plt.imshow(w_scaling, cmap='jet', aspect='auto')
            if not show_grid:            # Hide grid
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=minor_font_size)
                plt.xticks(tick_idx_x, [int(tick_labels_x[j]) for j in tick_idx_x], fontsize=minor_font_size)
                # plt.ylabel("Depth\n[meters]", fontsize=minor_font_size)
            if draw_seabed:
                plt.plot(np.arange(label.shape[1]), seabed, c=color_seabed['seabed'], lw=lw['seabed'])

            # Labels
            i += 1
            plt.subplot(n_plts, 1, i + 1, sharex = main_ax, sharey = main_ax)
            plt.imshow(label, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            if show_labels_str:
                plt.title("Annotations", fontsize=major_font_size)
            if draw_seabed:
                plt.plot(np.arange(label.shape[1]), seabed, c=color_seabed['seabed'], lw=lw['seabed'])
            if not show_grid:             # Hide grid
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=minor_font_size)
                plt.xticks(tick_idx_x, [int(tick_labels_x[j]) for j in tick_idx_x], fontsize=minor_font_size)
                # plt.ylabel("Depth\n[meters]", fontsize=minor_font_size)

            # selected mask
            i += 1
            plt.subplot(n_plts, 1, i + 1, sharex = main_ax, sharey = main_ax)
            plt.imshow(mask, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            if show_labels_str:
                plt.title("Annotations", fontsize=major_font_size)
            if draw_seabed:
                plt.plot(np.arange(label.shape[1]), seabed, c=color_seabed['seabed'], lw=lw['seabed'])
            if not show_grid:            # Hide grid
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=minor_font_size)
                plt.xticks(tick_idx_x, [int(tick_labels_x[j]) for j in tick_idx_x], fontsize=minor_font_size)
                plt.ylabel("Depth\n[meters]", fontsize=minor_font_size)
            plt.xlabel("Time [minutes]", fontsize=minor_font_size)
            plt.tight_layout()

            if return_fig:
                plt.savefig('%s_%d_rgb.png' % (self.e.name, j))
                pass
            else:
                plt.show()
    
