# Cat and Dog Classifier

This project is a Convolutional Neural Network (CNN) based classifier to distinguish between images of cats and dogs. The classifier is built using TensorFlow and Keras, leveraging image data augmentation techniques for improved model performance.

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Installation

To run this project, ensure you have Python installed. Then, install the required libraries using pip:

```bash
pip install tensorflow keras numpy pandas matplotlib
```

## Dataset

The dataset used for this project is the Dogs vs. Cats dataset. It contains separate folders for training and testing images.

- Training data: `path/to/train`
- Testing data: `path/to/test`

Ensure the directory structure is as follows:

```
/path/to/dataset
    /train
        /cats
        /dogs
    /test
        /cats
        /dogs
```

## Model Architecture

The CNN model consists of the following layers:

1. Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation.
2. MaxPooling layer with pool size of 2x2.
3. Flattening layer to convert the 2D matrices into a vector.
4. Fully connected (Dense) layer with 128 units and ReLU activation.
5. Output layer with 1 unit and sigmoid activation for binary classification.

## Training

The model is trained using the `ImageDataGenerator` for data augmentation. The augmentation techniques applied include rescaling, shear transformation, zoom, and horizontal flipping.



## Evaluation

The model's performance can be evaluated using the test set to measure accuracy and other relevant metrics. Ensure the test images are preprocessed in the same way as the training images.

## Usage

To use the trained model to make predictions on new images, you can load the model and use the `predict` method. Here's an example:

```python
from keras.preprocessing import image
import numpy as np

test_image = image.load_img('path/to/image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements or have found a bug.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

