import cv2
import numpy as np
from typing import List, Any, Generator, Tuple
from data.echosounder_data.dataloader import GroupedGenerator
from functools import partial

def generate_patch_batches(echograms: List,
                           split_patch_size: int,
                           output_patch_size: int,
                           data_transforms=None,
                           batch_size: str = "full",
                           verbose: bool = False) -> Tuple[List, List[int]]:
    """
    Generates patch generators and their corresponding patch counts for each echogram.

    Args:
        echograms (List): List of input echogram objects.
        split_patch_size (int): Size to split each echogram into patches.
        output_patch_size (int): Final size of output patches.
        data_transforms (callable, optional): Transformations applied to each patch.

    Returns:
        Tuple[List, List[int]]:
            - generators: List of batch generators for each echogram.
            - num_patches_per_echogram: Number of patches generated from each echogram.
    """
    generators = []
    num_patches_per_echogram = []

    for echo in echograms:
        split_and_resize = SplitResizeEchogram(
            echogram=echo,
            split_patch_size=split_patch_size,
        )
        gen_fn = partial(
            split_and_resize,
            output_patch_size=output_patch_size,
            data_transforms=data_transforms,
            batch_size=batch_size,
            verbose=verbose
        )
        generators.append(gen_fn)

        num_patches = (
            split_and_resize.ROI['num_patch_height'] *
            split_and_resize.ROI['num_patch_width']
        )
        num_patches_per_echogram.append(num_patches)


    return generators, num_patches_per_echogram

class SplitResizeEchogram:
    def __init__(self,
                 echogram: Any,
                 split_patch_size: int
                 ):
        """
        This class handles the resizing and splitting of large echogram images into patches. Each patch is used as an image
        input for a ViT, resulting in, 14x14 embeddings per image.
        The default patch flattening uses 'C' (row-major) order, indicating that the patches align from the top to the bottom rows.
        """
        self.order = 'C'  # 'F' for Fortran order, 'C' for C order, 'A' to keep the same order as input
        self.top_margin = 10  # Number of rows to remove from the top of the echogram
        self.bottom_margin = 5  # Number of rows to remove from the bottom of the echogram
        self.echogram = echogram
        self.split_patch_size = split_patch_size  # Height of each patch in the original scale
        self.seabed_bottom = echogram.get_seabed().max()  # Maximum row index of the seabed
        self.raw_height, self.raw_width = echogram.shape[:2]
        self.trim_height()
        self.adjust_split_patch_size(split_patch_size)
        self.set_height_of_interest()
        self.set_width_of_interest()

        self.ROI: dict[str, int] = {
            'height_start': (self.seabed_bottom + self.bottom_margin) - self.height_of_interest, 
            'height_end': (self.seabed_bottom + self.bottom_margin), 
            'width_start': 0, 
            'width_end': self.width_of_interest, 
            'split_patch_size': self.split_patch_size,
            'num_patch_width': self.num_patch_width,
            'num_patch_height': self.num_patch_height, 
            'raw_width': self.raw_width, 
            'raw_height': self.raw_height}

    def __call__(self,
                 output_patch_size: int,
                 data_transforms: List,
                 batch_size: str,
                 verbose: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Process the echogram by:
          1. Setting the region of interest parameters.
          2. Adjusting the patch size to fit the image height.
          3. Computing the number of patches in each dimension.
          4. Resizing and slicing each component (label, mask, data) into uniform patches.
        
        Args:
            echogram: The echogram object providing image data and associated methods.
            split_patch_size: The initial patch height used to compute patch configurations.
            output_patch_size: The desired patch dimensions in the final output.
        
        Yields:
            Tuples of patches (label_patch, data_patch, mask_patch) for each patch unit.
        """
        # Get raw image dimensions and determine seabed bottom.

        label_patches = self.resize_and_split_2d(self.echogram.label_numpy(), output_patch_size=output_patch_size)
        mask_patches = self.resize_and_split_2d(self.echogram.select_mask(), output_patch_size=output_patch_size)
        data_patches = self.resize_and_split_2d(self.echogram.data_numpy_transformed(data_transforms=data_transforms), output_patch_size=output_patch_size)

        if batch_size == "full":
            batch_size = self.num_patch_width * self.num_patch_height
        elif batch_size == "row":    
            batch_size = self.num_patch_width 
        else:   
            raise ValueError("Invalid batch size. Use 'full' or 'row'.")

        if verbose:
            print(self.ROI)

        batch = []
        for patch_group in zip(label_patches, data_patches, mask_patches):
            batch.append(patch_group)
            if len(batch) == batch_size:
                labels, data, masks = zip(*batch)
                yield np.array(labels), np.array(data), np.array(masks)
                batch = []
        if batch:
            labels, data, masks = zip(*batch)
            yield np.array(labels), np.array(data), np.array(masks)

    def trim_height(self):
        """
        Adjusts the region of interest used for patch generation by cropping the echogram based on the seabed position.
        The method performs the following steps:
          - Retrieves the maximum seabed row index.
          - Adjusts the bottom margin if the seabed position plus the margin would extend beyond the image height.
          - Computes the effective height by subtracting the specified top and bottom margins.
          - Sets the full image width for subsequent patch processing.
        """
        # Ensure the bottom margin does not extend beyond the image height.
        if (self.seabed_bottom + self.bottom_margin) > self.raw_height: 
            self.bottom_margin = self.raw_height - self.seabed_bottom
        
        # Calculate the effective region for patch extraction.
        self.height_of_interest = (self.seabed_bottom + self.bottom_margin) - self.top_margin

    def adjust_split_patch_size(self, split_patch_size: int):
        """
        Adjust the split patch size based on the effective image height and a 0.5 rounding criterion.
        The number of patch rows is determined by rounding (height_of_interest / split_patch_size) 
        with a threshold of 0.5. For example:
          - A ratio of 1.4 results in 1 patch row.
          - A ratio between 1.6 and 2.4 results in 2 patch rows.
        
        Updates:
            self.split_patch_size: Adjusted height of each patch in the original scale.
       
        Args:
            split_patch_size: The initial patch height used for computing the patch configuration.
        """
        self.num_patch_height = max(1, int((self.height_of_interest / split_patch_size) + 0.5))
        self.split_patch_size = self.height_of_interest // self.num_patch_height

    def set_height_of_interest(self):
        """
        Sets the effective height for patch extraction and checks that the
        region of interest is large enough to generate patches.
        This method:
          1. Determines the number of full patches that fit vertically based on the raw image height.
          2. Calculates the effective height as an exact multiple of the split patch size.
        """ 
        self.height_of_interest = self.num_patch_height * self.split_patch_size

    def set_width_of_interest(self):
        """
        Sets the effective width for patch extraction and checks that the 
        region of interest is large enough to generate patches.
        
        This method:
          1. Determines the number of full patches that fit horizontally based on the raw image width.
          2. Calculates the effective width as an exact multiple of the split patch size.
        """
        # Calculate the number of horizontal patches that can be formed.
        self.num_patch_width = self.raw_width // self.split_patch_size
        # Define the width of interest as an exact multiple of the patch size.
        self.width_of_interest = self.num_patch_width * self.split_patch_size


    def resize_and_split_2d(self, component: np.ndarray, output_patch_size: int) -> np.ndarray:
        """
        Crop, resize, and split a 2D image component into uniformly shaped patches.
        The method:
          1. Crops the component based on the seabed and computed patchified dimensions.
          2. Resizes the cropped portion using the scaling factor, applying appropriate interpolation.
          3. Reshapes and transposes the resized image to create a collection of patches.
        
        Args:
            component: A 2D (or 3D) numpy array representing one channel or the full echogram component.
        
        Returns:
            A numpy array of patches. For multi-channel data, each patch has shape 
            (output_patch_size, output_patch_size, channels); for 2D data, shape is 
            (output_patch_size, output_patch_size).
        """
        # Update scaling factor to convert from split patch size to output patch size.
        self.output_patch_size = output_patch_size

        # Crop the image component to the area of interest.
        if component.ndim == 3:
            cropped_component = component[
                (self.seabed_bottom + self.bottom_margin) - self.height_of_interest : (self.seabed_bottom + self.bottom_margin),
                :self.width_of_interest,
                :
            ]
        else: # 2D case
            # Ensure the component is cropped to the specified dimensions.
            cropped_component = component[
                (self.seabed_bottom + self.bottom_margin) - self.height_of_interest : (self.seabed_bottom + self.bottom_margin),
                :self.width_of_interest
            ]

        # Compute new dimensions directly from the number of patches and output_patch_size.
        new_width = self.num_patch_width * self.output_patch_size
        new_height = self.num_patch_height * self.output_patch_size
        
        # Resize using different interpolation for boolean and numeric types.
        if cropped_component.dtype == np.bool_:
            comp_uint8 = cropped_component.astype(np.uint8) * 255
            resized_uint8 = cv2.resize(comp_uint8, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            resized = resized_uint8 > 127
        else:
            resized = cv2.resize(cropped_component, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Reshape and transpose the resized image into patch arrays.
        # return (B, C, H, W)
        if resized.ndim == 3:
            patches_array = resized.reshape(
            self.num_patch_height, self.output_patch_size,
            self.num_patch_width, self.output_patch_size, -1
            )
            channels = patches_array.shape[-1]
            patches_array = patches_array.swapaxes(1, 2).reshape(self.num_patch_height * self.num_patch_width, self.output_patch_size, self.output_patch_size, channels)
            return patches_array.transpose(0, 3, 1, 2)
        else:# 2D case
            patches_array = resized.reshape(
                self.num_patch_height, self.output_patch_size,
                self.num_patch_width, self.output_patch_size, order=self.order
            )
            patches_array = patches_array.transpose(0, 2, 1, 3)
            return patches_array.reshape(-1, self.output_patch_size, self.output_patch_size, order=self.order)


