# Vehicle_Plate_Classifier

The folder is supposed to have 3 Files
*vehicle_plate_classifier.h5*,
  *VPClassifier.py*,
  *DataSet*
  .Due to Size issue, I could't upload vehicle_plate_classifier.h5, complete project folder is present in Google Drive Link given below
- https://drive.google.com/drive/folders/1IAAcW1dGmdvLWLPG5UcgMNIsvFijwd1y?usp=sharing
Regarding the Dataset, due to sensitive information such as multiple number plates and vehicle images through junction points, uploading of images was not allowed for confidentiality reasons.
## Model Architecture:
The CNN architecture is designed to effectively learn the distinguishing features between white and yellow license plates. The model comprises multiple layers, including convolutional layers, max-pooling layers, and dense layers. The convolutional layers are responsible for extracting essential features from the input images, while the max-pooling layers reduce the dimensionality of the data. The dense layers provide the classification output.

### Model Architecture Elaboration
The core of this project lies in the construction of a Convolutional Neural Network (CNN), a deep learning technique specifically designed for image processing tasks. The architecture of the CNN plays a crucial role in extracting meaningful features from the input images, facilitating accurate classification between private and commercial vehicle license plates.

### Convolutional Layers
The CNN begins with a series of convolutional layers, each configured to detect specific visual features within the input images. In this particular model, three convolutional layers are employed. The first layer uses 32 filters of size (3, 3) to convolve over the input images, each filter generating a distinct feature map that highlights different aspects of the image. The subsequent layers, with 64 and 128 filters respectively, continue to extract higher-level features as the network becomes deeper. The activation function 'ReLU' (Rectified Linear Unit) is applied to introduce non-linearity, allowing the network to learn complex patterns within the data.

### Max-Pooling Layers
Following each set of convolutional layers, max-pooling layers are integrated to downsample the feature maps, reducing the spatial dimensions and controlling overfitting. The pooling layers, with a size of (2, 2), effectively select the maximum value within each region, preserving the most prominent features while discarding irrelevant details. The consecutive application of these layers aids in capturing the most salient information from the images.

### Flatten Layer
After the convolutional and max-pooling layers, the data is flattened into a one-dimensional vector, preparing it for input into the densely connected layers. This process ensures that the information extracted from the images is organized in a manner suitable for further analysis and classification.

### Dense Layers
The flattened data is then fed into a series of densely connected layers. The first dense layer consists of 512 units and utilizes the 'ReLU' activation function, enabling the network to learn intricate relationships between the extracted features. Furthermore, to mitigate overfitting, a dropout layer with a rate of 0.5 is introduced, randomly deactivating a portion of the neurons during each training iteration. This technique enhances the model's generalizability and robustness.

### Output Layer
The final layer of the model is a dense layer with a single unit, activated by the 'sigmoid' function. This configuration is suitable for binary classification tasks, where the network outputs a probability score between 0 and 1, indicating the likelihood of an image belonging to a particular class. A threshold of 0.5 is used to classify the images into private or commercial vehicle categories. By combining these various layers, the CNN effectively learns intricate patterns and features from the input images, enabling accurate classification of vehicle license plates. The architecture is optimized to strike a balance between capturing relevant details and preventing overfitting, ultimately ensuring the model's robust performance.

### Model Training and Evaluation
The compiled model is trained using the training dataset, with the `adam` optimizer and `binary_crossentropy` as the loss function. During the training process, the model's performance is continuously evaluated using the testing dataset to monitor its accuracy and potential overfitting. The model is trained for 10 epochs to ensure an adequate convergence of the training process.

### Model Saving and Loading
After the training process, the model is saved in the `.h5` format, allowing for easy retrieval and reuse without the need for retraining. The saved model is then loaded for subsequent use in classifying unsorted images.

## Data Set Modelling

### Image Classification of Unsorted Images:
The saved model is employed to classify unsorted images stored in a designated folder. Each image is loaded, preprocessed, and fed into the model for classification. Based on the prediction output, images are sorted into two categories: those belonging to private vehicles with white plates and those belonging to commercial vehicles with yellow plates. The `shutil` library is utilized to move the images to their respective sorted folders.


### Optimizing Image Input for Enhanced Results

In this code, the process of inputting images is crucial for ensuring optimal performance and accurate classification results. There are several strategies that can be employed to improve the input image handling, thereby enhancing the overall performance of the model.

#### Preprocessing and Rescaling
   Before feeding the images into the model, it is essential to preprocess them to ensure uniformity and consistency in the data. One critical step is rescaling the images, as observed in the code with the line `img_tensor /= 255.0`. This operation normalizes the pixel values within the range of 0 to 1, facilitating smoother convergence during the training process. Additional preprocessing steps such as noise reduction, contrast adjustment, and resizing to a standardized dimension can further improve the model's robustness to variations in input data.

#### Data Augmentation
   Implementing data augmentation techniques can significantly enrich the training dataset and enhance the model's ability to generalize to different scenarios. Techniques such as rotation, flipping, zooming, and shifting can be applied to create additional variations of the existing images, effectively expanding the dataset without collecting new samples. This process helps prevent overfitting and improves the model's ability to recognize patterns under diverse conditions.

#### Quality Assurance and Image Integrity
   It is vital to ensure the quality and integrity of the input images to avoid any potential biases or inaccuracies during the classification process. Conducting a thorough quality check of the images, including checks for blurriness, distortion, or poor lighting, can help maintain the consistency and reliability of the dataset. Furthermore, ensuring that the images are representative of the real-world scenarios the model is intended to encounter can significantly enhance its practical applicability.

#### Handling Data Imbalance
   In scenarios where the dataset is imbalanced, meaning one class has significantly more samples than the other, employing techniques such as data resampling, synthetic data generation, or adjusting class weights during training can help the model achieve a more balanced and accurate classification. Handling data imbalance is crucial for preventing the model from being biased toward the majority class and ensuring it accurately captures patterns from both classes.

By implementing these strategies for image input, the model can achieve enhanced performance, improved generalization, and increased robustness, leading to more accurate and reliable classifications of vehicle license plates.

### Separating or annotating exceptional images
Due to certain ambiguous images, the model's accuracy tends to decrease, necessitating a proactive approach to address these cases and incorporate them into the training data to prevent potential errors.

#### Multiple vehicles
![image](https://github.com/user-attachments/assets/252391d8-d9c8-4273-b810-3a126b75c347)

(There are some images that contain multiple vehicles and it leads to which number plate.)



*For privacy protection, the number plates have been intentionally blurred in the images. However, the abundance of private vehicles in the dataset poses a potential challenge, as the model might erroneously identify them. Thus, the model should be designed with the capability to prioritize the recognition of commercial vehicles when they are present.*

#### Unclear images
![image](https://github.com/user-attachments/assets/5802972b-47cd-4e5a-a773-1fa7216cc762)









(Sometime due to multiple reason the images gets blurred due to being in the end of frame or other reasons. Due to that, the model ignores the number plate leading to loss in accuracy.)


*The number plates are blurred but the blur on yellow number plate is just a little space with 20% intensity blur to disable readability. In this image the blurry number plate disables the model to recognize this image into category of commercial vehicles. In order to improve this again annotation can help a lot which for the tester model was done for images in the 
way given below.*

 This way of annotation enables the model to read the annotated and focus on it.

(Note: For this type of annotation usually a classifier is attached in order to focus on the annotated only but after conducting multiple tests, the model works better if classifier is not applied at all.)

#### Hidden number plate or multiple number plates
![image](https://github.com/user-attachments/assets/c9ca71fa-30f5-40b5-bbd3-beaed990e511)


(There contains some images which consists of images containing vehicles with number plates either hidden or there are multiple number plates on one vehicle causes some confusion and leads to avoiding that image.)

Given image is an example to vehicles containing multiple number plates which can serve as a good data for plate classification but it causes some confusion leading to increasing confusion in automation.


*This image is blurred for privacy purposes but the number plate is partially visible causing it to be unreadable. These images are not read by the model 95% of the time leading to disruption in confusion matrix. Therefore it can help if annotation is done properly but still it is hard to recognize these images.*

### Creating Dataset and Usage 
#### The initial usage of the code was started with only the training data which used to give biased data towards private vehicles due to the presence of large amount of data in private files figuratively. This data lead to cause an imbalance which decreases accuracy. In order to tackle the problem the data was divided equally(500 images) for the first test. The accuracy was increased.
#### The second approach to increase the accuracy was to create testing dataset. The testing dataset is used by model to verify the the training effect and if there are some accuracy issues, the changes are done within so but any minute changes in the dataset creates an impact on the accuracy(good or bad). The total allotted images were 2000 for the first test.
#### Within this dataset an accuracy of around 93% is reached, or it can be said that within every 1000 images there are less than 100 images that are errors or confusion data.
#### The main data to focus is yellow images as it can highly affect by discrepancies due to dataset. Although unconfirmed the data can show major errors if there are unequal images in testing or training data.


## Algorithm: 
### Step 1: Prepare the data
#### •	Initialize the ImageDataGenerator for both the training and testing data.
#### •	Specify the directory paths for the training and testing data.
#### •	Set the target image size and batch size.
#### •	Define the class mode for binary classification.

### Step 2: Build the CNN model
#### •	Initialize a Sequential model.
#### •	Add the Conv2D layers with specified filters, kernel size, and activation function.
#### •	Add MaxPooling2D layers to downsample the feature maps.
#### •	Flatten the data to prepare for the fully connected layers.
#### •	Add Dense layers with ReLU activation and a dropout layer for regularization.
#### •	Add the final output layer with a sigmoid activation for binary classification.

### Step 3: Compile the model
#### •	Specify the optimizer, loss function, and evaluation metrics for the model.

### Step 4: Train the model
#### •	Fit the model using the training data and specify the number of epochs.
#### •	Evaluate the model using the testing data to monitor its performance.

### Step 5: Save the model
#### •	Save the trained model in the '.h5' format for future use.

### Step 6: Load the saved model for classification
#### •	Load the saved model using the load_model function from TensorFlow.

### Step 7: Classify unsorted images
#### •	Define the paths for the unsorted and sorted image folders.
#### •	Iterate through each image in the unsorted folder.
#### •	Load the image and preprocess it to match the input format of the model.
#### •	Use the loaded model to predict the class of the image.
#### •	Based on the prediction, move the image to the corresponding sorted folder.

### Step 8: End
