# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
# Save feature map of conv layers with names ending in '_*0' for a given image to files
import re
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from matplotlib import pyplot as plt
from numpy import expand_dims
from keras.layers import Conv2D

# Load the InceptionV3 model
model = InceptionV3()

# Load the image with the required shape
img = load_img('sheep.png', target_size=(299, 299))
# Convert the image to an array
img = img_to_array(img)
# Expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# Prepare the image (e.g., scale pixel values for inception)
img = preprocess_input(img)

# Define a regex pattern to match layer names ending with 'conv' followed by any characters and ending with a '0'
pattern = re.compile(".*conv2d_.*0$")

# Loop through the layers of the model and find matches with the regex pattern
for layer in model.layers:
    if isinstance(layer, Conv2D) and re.match(pattern, layer.name):
        # Redefine model to output right after the current conv layer
        intermediate_model = Model(inputs=model.inputs, outputs=layer.output)
        
        # Get feature map for the current conv layer
        feature_maps = intermediate_model.predict(img)

        print(f'Feature maps of layer {layer.name} have shape {feature_maps.shape}')

        # Determine the number of feature maps (filters)
        num_feature_maps = feature_maps.shape[-1]

        # Set up the subplot grid
        columns = 8  # For example, we take 8 columns for visualization
        rows = num_feature_maps // columns + (num_feature_maps % columns > 0)

        # Plotting the feature maps
        fig, axs = plt.subplots(rows, columns, figsize=(columns*2, rows*2))
        fig.suptitle(f'Feature maps of layer {layer.name}')
        
        for ix in range(num_feature_maps):
            row = ix // columns
            col = ix % columns
            ax = axs[row, col] if num_feature_maps > 1 else axs[col]
            ax.set_xticks([])
            ax.set_yticks([])
            # Plot filter channel in grayscale
            if ix < num_feature_maps:
                ax.imshow(feature_maps[0, :, :, ix], cmap='gray')
            ax.axis('off')
        
        # Adjust layout and save the figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f'sheep_layer_{layer.name}.png')
        plt.close(fig)

# Ensure that the file 'bird.jpg' exists in your current directory or provide the correct path to the image.
