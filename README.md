COMPANY : CODTECH IT SOLUTIONS

NAME : PATEL SUNNY KANAKKUMAR

INTERN ID : CT04DG881

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEK

MENTOR : NEELA SANTOSH

DESCRIPTION OF TASK :This project focuses on the implementation of a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. It demonstrates a complete deep learning pipeline using TensorFlow and Keras, covering data preprocessing, model design, training, evaluation, and performance visualization.

The MNIST dataset is a standard benchmark in the field of computer vision and machine learning. It contains 70,000 grayscale images of handwritten digits from 0 to 9, where each image is of size 28x28 pixels. The dataset is already split into a training set of 60,000 images and a test set of 10,000 images.
 
Objective
The main objective of this project is to build a deep learning model that can accurately classify handwritten digits. This type of digit recognition has many practical applications such as in postal code recognition, cheque processing, digitized form reading, and more.
 
 Project Workflow
1. Data Loading and Preprocessing
The MNIST dataset is loaded using TensorFlow’s built-in Keras API. The images are reshaped to add a single color channel ((28, 28, 1)) and normalized by dividing each pixel value by 255 to bring all values in the range [0, 1], which helps in faster and stable training. The labels are one-hot encoded to convert the categorical class values into binary vectors, which is a requirement for multi-class classification with neural networks.

2. Model Architecture
The CNN model is built using the Sequential API from Keras and includes the following layers:

Conv2D Layer: Applies 32 filters of size (3x3) to extract low-level features such as edges and curves.

MaxPooling2D Layer: Reduces the spatial dimensions by taking the maximum value from a 2x2 window, helping in downsampling and reducing computation.

Flatten Layer: Converts the 2D feature maps into a 1D vector.

Dense Layers: A fully connected layer with 128 neurons followed by an output layer with 10 neurons (one for each digit class) using softmax activation for multi-class classification.

3. Model Compilation and Training
The model is compiled with the Adam optimizer, categorical crossentropy loss function (since this is a multi-class classification problem), and accuracy as the performance metric. It is trained for 5 epochs using the training data and validated on the test data during training.

4. Model Evaluation
After training, the model’s performance is evaluated on the test dataset using the .evaluate() method, which returns the final test accuracy and loss. A high accuracy indicates the model’s effectiveness in generalizing on unseen data.

5. Visualization
The training and validation accuracy and loss values over epochs are plotted using Matplotlib. These plots help in understanding the learning behavior of the model — whether it is underfitting, overfitting, or training well.


output:<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/636c5e9d-44a2-4f0f-9216-716690161fd6" />
![Image](https://github.com/user-attachments/assets/82782b1d-4c0e-4a89-9eed-55dc41e2a597)
