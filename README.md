
# Object Oriented Image Classifier

Image classification is a common problem in deep learning. In this notebook, we aim to develop a generic convolutional network which can work on multiple datasets. In addition to the model, supporting routines such as classifying user upload and finding similar images will also be developed.

## Package Import


```
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras import layers, models
from PIL import Image
from google.colab import files
```


```
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

    Num GPUs Available:  1


## Image Classifier


```
def preprocess(ds, img_height, img_width, metadata, batch_size = 32, shuffle=True):
    """ a dataset wrapper that resizes images """
    # resize images as the specified height and width
    def resize_img(image, label):
        image = tf.image.resize(image, (img_height, img_width))
        image = tf.cast(image, tf.float32)
        label = tf.one_hot(label, metadata.features["label"].num_classes)
        return image, label 
    ds = ds.map(resize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # provide images as batches and enable prefetching (and shuffling)
    if shuffle:
        ds = ds.shuffle(10000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
```

Under the class of ImageClassifier, we define the convolutional neural network and and various support routines such as:
- Plotting learning curve
- Plotting confusion matrix
- Predict label
- Finding similar images




```
class ImageClassifier:
    def __init__(self, train_ds, test_ds, metadata, 
                 img_height, img_width, img_depth=3, batch_size=32, 
                 shuffle=True, augment=False, dropout=False):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.metadata = metadata
        self.batch_size = batch_size
        # Preprocess data sources
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_pds = preprocess(self.train_ds, img_height, img_width, 
                                    metadata, batch_size, shuffle)
        self.test_pds = preprocess(self.test_ds, img_height, img_width, 
                                   metadata, batch_size, shuffle)
        # Create and compile the model
        model = models.Sequential()
        # Standardize the data
        model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, img_depth)))
        # Data augmentation
        if augment:
            model.add(layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)))
            model.add(layers.experimental.preprocessing.RandomRotation(0.1))
            model.add(layers.experimental.preprocessing.RandomZoom(0.1))
            model.add(layers.experimental.preprocessing.RandomContrast(0.3))
            model.add(layers.experimental.preprocessing.RandomFlip())
        # Convolution and pooling to extract features
        model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())
        # Dropout to reduce overfitting
        if dropout:
            model.add(layers.Dropout(0.2))
        # Dense layer for learning
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.metadata.features["label"].num_classes, activation='softmax'))
        # Compile the model
        model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])
        self.model = model
        self.history = None
        self.model.summary()
        
    def train(self, epochs=15):
        # fit the model       
        history = self.model.fit(self.train_pds, validation_data=self.test_pds, epochs=epochs)
        self.history = history
        self.epochs = epochs

    def plot_learning_curve(self):
        if self.history == None:
            print("Train the model first!")
            return
        # Create a figure with two subsplots
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs_range = range(self.epochs)
        # Accuracy
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
    
    def evaluate(self, test_ds=None):
        print("Evaluation:")
        if test_ds == None:
            test_ds = self.test_pds
        result = self.model.evaluate(test_ds)
        print(dict(zip(self.model.metrics_names, result)))

    def print_confusion_matrix(self, test_ds=None, alt_label=False):
        print("Confusion matrix:")
        if test_ds == None:
            test_ds = self.test_pds
        if alt_label == True:
            class_names = self.metadata.features["label"].alt_names
        else:
            class_names = self.metadata.features["label"].names
        # Create confusion matrix from the test dataset
        y_true = []
        for image, label in test_ds: # for each batch (batch_size, 10)
            label = tf.argmax(label, axis=1) # decode the label in a batch, return (1, batch_size)
            y_true.append(label.numpy()) # gives (num_batch, batch_size) list
        y_true = np.concatenate(y_true, axis=None) #concat and flatten
        y_pred = tf.argmax(self.model.predict(test_ds), axis=1)
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=class_names, yticklabels=class_names, annot=True, fmt='g')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()
    
    def query(self, path, alt_label=False):
        # obtain image
        #img = tf.keras.preprocessing.image.load_img(path, target_size=(self.img_height, self.img_width))
        img = Image.open(path)
        img = img.resize((self.img_width, self.img_height))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # obtain label
        if alt_label == True:
            class_names = self.metadata.features["label"].alt_names
        else:
            class_names = self.metadata.features["label"].names
        # make prediction on the image
        prediction = self.model.predict(tf.expand_dims(img_array, 0))
        label = tf.argmax(prediction, axis=1)
        label_as_name = class_names[int(label.numpy())]
        confidence = np.array(prediction).flatten()[label]*100
        print(f"This image most likely belongs to {label_as_name} with a {confidence:.2f} percent confidence.")
        return (img, label_as_name) # return pillow instance
    
    def find_similar(self, path, no_of_images=5, alt_label=False, 
                     search_bound=1000, tolerance=50):
        # obtain label
        if alt_label == True:
            class_names = self.metadata.features["label"].alt_names
        else:
            class_names = self.metadata.features["label"].names
        # call query() to get the image array and predicted label of the input image file
        image, label_target = self.query(path, alt_label)
        # Calculate the average color [r, g, b] of the input image
        width, height = image.size
        r = 0
        g = 0
        b = 0
        count = 0
        for i in range(width):
            for j in range(height):
                pixel = image.getpixel((i,j))
                r += pixel[0]
                g += pixel[1]
                b += pixel[2]
                count += 1
        colour_mean_target = np.array((r/count, g/count, b/count))
        # Show the image with title showing average RGB values w/o axes
        plt.imshow(image)
        plt.title(str(np.trunc(colour_mean_target)))
        plt.axis('off')
        plt.show()
        print('Similar images:')
        # Scan the original dataset for images
        n = 0
        for image, label in self.train_ds:
            im = Image.fromarray(image.numpy())
            width, height = im.size
            r = 0
            g = 0
            b = 0
            count = 0
            for i in range(width):
                for j in range(height):
                    pixel = im.getpixel((i,j))
                    r += pixel[0]
                    g += pixel[1]
                    b += pixel[2]
                    count += 1
            colour_mean = np.array((r/count, g/count, b/count))
            label_as_name = class_names[int(label.numpy())]
            # same class label and same RGB
            if label_as_name == label_target and np.allclose(colour_mean, colour_mean_target, atol=tolerance):
                plt.imshow(im)
                plt.title(str(np.trunc(colour_mean)))
                plt.axis('off')
                plt.show()
                n += 1
            if n == no_of_images:
               break
        # if not found
        if n == 0:
            print('No similar images found!')
```

## Client Code

The model and supporting functions can be applied to various dataset.

### Imagenette

Imagenette is a subset of 10 easily classified classes from the Imagenet dataset. We use this dataset to quickly test the capabilities of the generic network.


```
# create training and testing datasets
(train_ds, test_ds), metadata = tfds.load( 
    'imagenette/320px-v2',
    split=['train', 'validation'],
    with_info=True,
    as_supervised=True,
)
print("train:", len(train_ds))
print("test: ", len(test_ds))
fig = tfds.show_examples(train_ds, metadata)
print(metadata.features["label"].names)
# ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 
#  'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
# add alternative names for labels
metadata.features["label"].alt_names = [
    'tench', 'English springer', 'cassette player', 'chain saw', 'church', 
    'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
    ]
print(train_ds)
```

    train: 9469
    test:  3925



![svg](classifier_files/classifier_12_1.svg)


    ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
    <PrefetchDataset shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>



```
# client code for testing
IMG_SIZE = 180
imagenette = ImageClassifier(train_ds, test_ds, metadata, IMG_SIZE, IMG_SIZE, 
                             augment=False, dropout=False)
imagenette.train(epochs=15)
imagenette.plot_learning_curve()
imagenette.evaluate()
imagenette.print_confusion_matrix(alt_label=True)
```


```
# test effectiveness of data augmentation and dropout
IMG_SIZE = 180
imagenette = ImageClassifier(train_ds, test_ds, metadata, IMG_SIZE, IMG_SIZE, 
                             augment=True, dropout=True)
imagenette.train(epochs=15)
imagenette.plot_learning_curve()
imagenette.evaluate()
imagenette.print_confusion_matrix(alt_label=True)
```

    Model: "sequential_27"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    sequential_28 (Sequential)   (None, 180, 180, 3)       0         
    _________________________________________________________________
    rescaling_7 (Rescaling)      (None, 180, 180, 3)       0         
    _________________________________________________________________
    conv2d_28 (Conv2D)           (None, 180, 180, 16)      448       
    _________________________________________________________________
    max_pooling2d_28 (MaxPooling (None, 90, 90, 16)        0         
    _________________________________________________________________
    conv2d_29 (Conv2D)           (None, 90, 90, 32)        4640      
    _________________________________________________________________
    max_pooling2d_29 (MaxPooling (None, 45, 45, 32)        0         
    _________________________________________________________________
    conv2d_30 (Conv2D)           (None, 45, 45, 64)        18496     
    _________________________________________________________________
    max_pooling2d_30 (MaxPooling (None, 22, 22, 64)        0         
    _________________________________________________________________
    conv2d_31 (Conv2D)           (None, 22, 22, 128)       73856     
    _________________________________________________________________
    max_pooling2d_31 (MaxPooling (None, 11, 11, 128)       0         
    _________________________________________________________________
    flatten_7 (Flatten)          (None, 15488)             0         
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 15488)             0         
    _________________________________________________________________
    dense_14 (Dense)             (None, 512)               7930368   
    _________________________________________________________________
    dense_15 (Dense)             (None, 10)                5130      
    =================================================================
    Total params: 8,032,938
    Trainable params: 8,032,938
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/15
    296/296 [==============================] - 25s 85ms/step - loss: 1.8652 - accuracy: 0.3462 - val_loss: 1.5248 - val_accuracy: 0.4968
    Epoch 2/15
    296/296 [==============================] - 25s 84ms/step - loss: 1.5018 - accuracy: 0.4928 - val_loss: 1.3922 - val_accuracy: 0.5345
    Epoch 3/15
    296/296 [==============================] - 25s 84ms/step - loss: 1.3743 - accuracy: 0.5391 - val_loss: 1.3447 - val_accuracy: 0.5434
    Epoch 4/15
    296/296 [==============================] - 25s 85ms/step - loss: 1.2654 - accuracy: 0.5730 - val_loss: 1.2657 - val_accuracy: 0.5822
    Epoch 5/15
    296/296 [==============================] - 25s 85ms/step - loss: 1.1844 - accuracy: 0.6047 - val_loss: 1.1394 - val_accuracy: 0.6303
    Epoch 6/15
    296/296 [==============================] - 25s 84ms/step - loss: 1.1456 - accuracy: 0.6177 - val_loss: 1.1841 - val_accuracy: 0.6074
    Epoch 7/15
    296/296 [==============================] - 25s 84ms/step - loss: 1.0812 - accuracy: 0.6396 - val_loss: 1.1489 - val_accuracy: 0.6257
    Epoch 8/15
    296/296 [==============================] - 25s 84ms/step - loss: 1.0327 - accuracy: 0.6574 - val_loss: 1.0767 - val_accuracy: 0.6522
    Epoch 9/15
    296/296 [==============================] - 25s 84ms/step - loss: 1.0041 - accuracy: 0.6586 - val_loss: 0.9905 - val_accuracy: 0.6856
    Epoch 10/15
    296/296 [==============================] - 25s 84ms/step - loss: 0.9535 - accuracy: 0.6854 - val_loss: 1.1159 - val_accuracy: 0.6446
    Epoch 11/15
    296/296 [==============================] - 25s 84ms/step - loss: 0.9320 - accuracy: 0.6900 - val_loss: 0.9708 - val_accuracy: 0.6943
    Epoch 12/15
    296/296 [==============================] - 25s 84ms/step - loss: 0.8981 - accuracy: 0.6984 - val_loss: 0.9652 - val_accuracy: 0.6838
    Epoch 13/15
    296/296 [==============================] - 25s 84ms/step - loss: 0.8744 - accuracy: 0.7103 - val_loss: 0.9406 - val_accuracy: 0.7057
    Epoch 14/15
    296/296 [==============================] - 25s 84ms/step - loss: 0.8288 - accuracy: 0.7252 - val_loss: 1.0631 - val_accuracy: 0.6578
    Epoch 15/15
    296/296 [==============================] - 25s 84ms/step - loss: 0.8182 - accuracy: 0.7252 - val_loss: 0.8853 - val_accuracy: 0.7210



![png](classifier_files/classifier_14_1.png)


    Evaluation:
    123/123 [==============================] - 5s 38ms/step - loss: 0.8853 - accuracy: 0.7210
    {'loss': 0.8852530121803284, 'accuracy': 0.7210190892219543}
    Confusion matrix:



![png](classifier_files/classifier_14_3.png)



```
# make some predictions
URL = 'https://upload.wikimedia.org/wikipedia/commons/e/ef/Golf_ball_near_green.jpg'
golf_path1 = tf.keras.utils.get_file('golfball1.jpg', origin=URL)
imagenette.query(golf_path1, alt_label=True)

URL = 'https://contents.mediadecathlon.com/p1345747/k$8bfca050583b3bfb22ace6db8769f696/bong-golf-inesis-100-x12-trang.jpg?format=auto&f=700x700'
golf_path2 = tf.keras.utils.get_file('golfball2.jpg', origin=URL)
imagenette.query(golf_path2, alt_label=True)

URL = 'https://cdn.techexplorist.com/wp-content/uploads/2018/12/parachute.jpg'
parachute_path1 = tf.keras.utils.get_file('parachute1.jpg', origin=URL)
imagenette.query(parachute_path1, alt_label=True)

URL = 'https://media.npr.org/assets/img/2018/12/21/parachute_custom-14f30a9f6c9cd40ce0c2079732f3cf6122206945-s800-c85.jpg'
parachute_path2 = tf.keras.utils.get_file('parachute2.jpg', origin=URL)
imagenette.query(parachute_path2, alt_label=True)

URL = 'https://spacenews.com/wp-content/uploads/2019/12/spacexchutes-dec19.jpg'
parachute_path3 = tf.keras.utils.get_file('parachute3.jpg', origin=URL)
imagenette.query(parachute_path3, alt_label=True)

print("Finding similar figures:")
imagenette.find_similar(golf_path1, 5, alt_label=True, tolerance=20, search_bound=2000)
imagenette.find_similar(parachute_path1, 5, alt_label=True, tolerance=20, search_bound=3000)

```

    This image most likely belongs to golf ball with a 87.16 percent confidence.
    This image most likely belongs to golf ball with a 90.36 percent confidence.
    This image most likely belongs to parachute with a 100.00 percent confidence.
    This image most likely belongs to parachute with a 100.00 percent confidence.
    This image most likely belongs to parachute with a 91.65 percent confidence.
    Finding similar figures:
    This image most likely belongs to golf ball with a 87.16 percent confidence.



![png](classifier_files/classifier_15_1.png)


    Similar images:



![png](classifier_files/classifier_15_3.png)



![png](classifier_files/classifier_15_4.png)



![png](classifier_files/classifier_15_5.png)



![png](classifier_files/classifier_15_6.png)



![png](classifier_files/classifier_15_7.png)


    This image most likely belongs to parachute with a 100.00 percent confidence.



![png](classifier_files/classifier_15_9.png)


    Similar images:



![png](classifier_files/classifier_15_11.png)



![png](classifier_files/classifier_15_12.png)



![png](classifier_files/classifier_15_13.png)



![png](classifier_files/classifier_15_14.png)



![png](classifier_files/classifier_15_15.png)


### Cats vs Dogs

Cats vs Dogs is a binary image classification problem which is fun to play with.


```
# create training and testing datasets
(train_ds, test_ds), metadata = tfds.load( 
    'cats_vs_dogs',
    split=['train[:50%]', 'train[:10%]'],
    with_info=True,
    as_supervised=True,
)

print("train:", len(train_ds))
print("test: ", len(test_ds))
fig = tfds.show_examples(train_ds, metadata)
print(metadata.features["label"].names)
```

    train: 11631
    test:  2326



![png](classifier_files/classifier_17_1.png)


    ['cat', 'dog']



```
# client code for testing
IMG_SIZE = 150
cats_vs_dogs = ImageClassifier(train_ds, test_ds, metadata, IMG_SIZE, IMG_SIZE, 
                               augment=True, dropout=True)
cats_vs_dogs.train(epochs=15)
cats_vs_dogs.plot_learning_curve()
cats_vs_dogs.evaluate()
cats_vs_dogs.print_confusion_matrix()
```

    Model: "sequential_31"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    sequential_32 (Sequential)   (None, 150, 150, 3)       0         
    _________________________________________________________________
    rescaling_8 (Rescaling)      (None, 150, 150, 3)       0         
    _________________________________________________________________
    conv2d_32 (Conv2D)           (None, 150, 150, 16)      448       
    _________________________________________________________________
    max_pooling2d_32 (MaxPooling (None, 75, 75, 16)        0         
    _________________________________________________________________
    conv2d_33 (Conv2D)           (None, 75, 75, 32)        4640      
    _________________________________________________________________
    max_pooling2d_33 (MaxPooling (None, 37, 37, 32)        0         
    _________________________________________________________________
    conv2d_34 (Conv2D)           (None, 37, 37, 64)        18496     
    _________________________________________________________________
    max_pooling2d_34 (MaxPooling (None, 18, 18, 64)        0         
    _________________________________________________________________
    conv2d_35 (Conv2D)           (None, 18, 18, 128)       73856     
    _________________________________________________________________
    max_pooling2d_35 (MaxPooling (None, 9, 9, 128)         0         
    _________________________________________________________________
    flatten_8 (Flatten)          (None, 10368)             0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 10368)             0         
    _________________________________________________________________
    dense_16 (Dense)             (None, 512)               5308928   
    _________________________________________________________________
    dense_17 (Dense)             (None, 2)                 1026      
    =================================================================
    Total params: 5,407,394
    Trainable params: 5,407,394
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/15
    364/364 [==============================] - 25s 68ms/step - loss: 0.6909 - accuracy: 0.5515 - val_loss: 0.6774 - val_accuracy: 0.5830
    Epoch 2/15
    364/364 [==============================] - 25s 67ms/step - loss: 0.6461 - accuracy: 0.6222 - val_loss: 0.6412 - val_accuracy: 0.6414
    Epoch 3/15
    364/364 [==============================] - 25s 67ms/step - loss: 0.5938 - accuracy: 0.6833 - val_loss: 0.6058 - val_accuracy: 0.6844
    Epoch 4/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.5563 - accuracy: 0.7142 - val_loss: 0.5435 - val_accuracy: 0.7253
    Epoch 5/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.5446 - accuracy: 0.7240 - val_loss: 0.5179 - val_accuracy: 0.7489
    Epoch 6/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.5241 - accuracy: 0.7466 - val_loss: 0.4838 - val_accuracy: 0.7653
    Epoch 7/15
    364/364 [==============================] - 25s 67ms/step - loss: 0.5076 - accuracy: 0.7503 - val_loss: 0.4986 - val_accuracy: 0.7592
    Epoch 8/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.4906 - accuracy: 0.7612 - val_loss: 0.4749 - val_accuracy: 0.7764
    Epoch 9/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.4786 - accuracy: 0.7740 - val_loss: 0.4417 - val_accuracy: 0.7863
    Epoch 10/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.4615 - accuracy: 0.7814 - val_loss: 0.4233 - val_accuracy: 0.8044
    Epoch 11/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.4534 - accuracy: 0.7833 - val_loss: 0.4357 - val_accuracy: 0.8031
    Epoch 12/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.4444 - accuracy: 0.7895 - val_loss: 0.3973 - val_accuracy: 0.8173
    Epoch 13/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.4270 - accuracy: 0.8006 - val_loss: 0.3942 - val_accuracy: 0.8263
    Epoch 14/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.4247 - accuracy: 0.8059 - val_loss: 0.3925 - val_accuracy: 0.8259
    Epoch 15/15
    364/364 [==============================] - 24s 67ms/step - loss: 0.4129 - accuracy: 0.8110 - val_loss: 0.4096 - val_accuracy: 0.8194



![png](classifier_files/classifier_18_1.png)


    Evaluation:
    73/73 [==============================] - 2s 30ms/step - loss: 0.4096 - accuracy: 0.8194
    {'loss': 0.4095933139324188, 'accuracy': 0.8194324970245361}
    Confusion matrix:



![png](classifier_files/classifier_18_3.png)


With the trained network, we can make predictions on new images, and find similar images according to their class and colour palette.


```
URL = 'https://i.insider.com/5484d9d1eab8ea3017b17e29?width=600&format=jpeg&auto=webp'
dog_path1 = tf.keras.utils.get_file('dog1', origin=URL)
cats_vs_dogs.query(dog_path1)

URL = 'https://www.sciencenewsforstudents.org/wp-content/uploads/2020/07/070720_bo_dogage_feat-1028x579.jpg'
dog_path2 = tf.keras.utils.get_file('dog2', origin=URL)
cats_vs_dogs.query(dog_path2)

URL = 'https://images.theconversation.com/files/319652/original/file-20200310-61148-vllmgm.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=754&fit=clip'
dog_path3 = tf.keras.utils.get_file('dog3', origin=URL)
cats_vs_dogs.query(dog_path3)

cats_vs_dogs.find_similar(dog_path1, 5, tolerance=15)
cats_vs_dogs.find_similar(dog_path2, 5, tolerance=15)
cats_vs_dogs.find_similar(dog_path3, 5, tolerance=15)
```

    This image most likely belongs to dog with a 100.00 percent confidence.
    This image most likely belongs to dog with a 87.17 percent confidence.
    This image most likely belongs to dog with a 86.99 percent confidence.
    This image most likely belongs to dog with a 100.00 percent confidence.



![png](classifier_files/classifier_20_1.png)


    Similar images:



![png](classifier_files/classifier_20_3.png)



![png](classifier_files/classifier_20_4.png)



![png](classifier_files/classifier_20_5.png)



![png](classifier_files/classifier_20_6.png)



![png](classifier_files/classifier_20_7.png)


    This image most likely belongs to dog with a 87.17 percent confidence.



![png](classifier_files/classifier_20_9.png)


    Similar images:



![png](classifier_files/classifier_20_11.png)



![png](classifier_files/classifier_20_12.png)



![png](classifier_files/classifier_20_13.png)



![png](classifier_files/classifier_20_14.png)



![png](classifier_files/classifier_20_15.png)


    This image most likely belongs to dog with a 86.99 percent confidence.



![png](classifier_files/classifier_20_17.png)


    Similar images:
    No similar images found!


### Malaria

Malaria is a disease caused by a parasite which is spread to humans through the bites of infected mosquitoes. This dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells from the thin blood smear slide images of segmented cells, labelled parasitized and uninfected.


```
# create training and testing datasets
(train_ds, test_ds), metadata = tfds.load( 
    'malaria',
    split=['train[:50%]', 'train[:10%]'],
    with_info=True,
    as_supervised=True,
)

print("train:", len(train_ds))
print("test: ", len(test_ds))
fig = tfds.show_examples(train_ds, metadata)
print(metadata.features["label"].names)
```

    [1mDownloading and preparing dataset malaria/1.0.0 (download: 337.08 MiB, generated: Unknown size, total: 337.08 MiB) to /root/tensorflow_datasets/malaria/1.0.0...[0m



    Dl Completed...: 0 url [00:00, ? url/s]



    Dl Size...: 0 MiB [00:00, ? MiB/s]



    Extraction completed...: 0 file [00:00, ? file/s]


    
    
    



    0 examples [00:00, ? examples/s]


    Shuffling and writing examples to /root/tensorflow_datasets/malaria/1.0.0.incompleteHM3ELV/malaria-train.tfrecord



      0%|          | 0/27558 [00:00<?, ? examples/s]


    [1mDataset malaria downloaded and prepared to /root/tensorflow_datasets/malaria/1.0.0. Subsequent calls will reuse this data.[0m
    train: 13779
    test:  2756



![png](classifier_files/classifier_22_9.png)


    ['parasitized', 'uninfected']



```
IMG_SIZE = 150
malaria = ImageClassifier(train_ds, test_ds, metadata, IMG_SIZE, IMG_SIZE, 
                               augment=True, dropout=True)
malaria.train(epochs=15)
malaria.plot_learning_curve()
malaria.evaluate()
malaria.print_confusion_matrix()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     rescaling_1 (Rescaling)     (None, 150, 150, 3)       0         
                                                                     
     random_flip_2 (RandomFlip)  (None, 150, 150, 3)       0         
                                                                     
     random_rotation_1 (RandomRo  (None, 150, 150, 3)      0         
     tation)                                                         
                                                                     
     random_zoom_1 (RandomZoom)  (None, 150, 150, 3)       0         
                                                                     
     random_contrast_1 (RandomCo  (None, 150, 150, 3)      0         
     ntrast)                                                         
                                                                     
     random_flip_3 (RandomFlip)  (None, 150, 150, 3)       0         
                                                                     
     conv2d_3 (Conv2D)           (None, 150, 150, 16)      448       
                                                                     
     max_pooling2d_3 (MaxPooling  (None, 75, 75, 16)       0         
     2D)                                                             
                                                                     
     conv2d_4 (Conv2D)           (None, 75, 75, 32)        4640      
                                                                     
     max_pooling2d_4 (MaxPooling  (None, 37, 37, 32)       0         
     2D)                                                             
                                                                     
     conv2d_5 (Conv2D)           (None, 37, 37, 64)        18496     
                                                                     
     max_pooling2d_5 (MaxPooling  (None, 18, 18, 64)       0         
     2D)                                                             
                                                                     
     dropout_1 (Dropout)         (None, 18, 18, 64)        0         
                                                                     
     flatten_1 (Flatten)         (None, 20736)             0         
                                                                     
     dense_2 (Dense)             (None, 128)               2654336   
                                                                     
     dense_3 (Dense)             (None, 2)                 258       
                                                                     
    =================================================================
    Total params: 2,678,178
    Trainable params: 2,678,178
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/15
    431/431 [==============================] - 27s 29ms/step - loss: 0.5307 - accuracy: 0.7176 - val_loss: 0.2747 - val_accuracy: 0.8999
    Epoch 2/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.2454 - accuracy: 0.9133 - val_loss: 0.1891 - val_accuracy: 0.9398
    Epoch 3/15
    431/431 [==============================] - 7s 17ms/step - loss: 0.1926 - accuracy: 0.9381 - val_loss: 0.1607 - val_accuracy: 0.9546
    Epoch 4/15
    431/431 [==============================] - 9s 20ms/step - loss: 0.1775 - accuracy: 0.9442 - val_loss: 0.1516 - val_accuracy: 0.9539
    Epoch 5/15
    431/431 [==============================] - 8s 18ms/step - loss: 0.1647 - accuracy: 0.9488 - val_loss: 0.1422 - val_accuracy: 0.9561
    Epoch 6/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.1568 - accuracy: 0.9495 - val_loss: 0.1328 - val_accuracy: 0.9561
    Epoch 7/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.1492 - accuracy: 0.9531 - val_loss: 0.1300 - val_accuracy: 0.9565
    Epoch 8/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.1442 - accuracy: 0.9540 - val_loss: 0.1284 - val_accuracy: 0.9601
    Epoch 9/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.1436 - accuracy: 0.9544 - val_loss: 0.1263 - val_accuracy: 0.9597
    Epoch 10/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.1355 - accuracy: 0.9568 - val_loss: 0.1244 - val_accuracy: 0.9597
    Epoch 11/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.1357 - accuracy: 0.9562 - val_loss: 0.1246 - val_accuracy: 0.9572
    Epoch 12/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.1338 - accuracy: 0.9558 - val_loss: 0.1278 - val_accuracy: 0.9594
    Epoch 13/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.1303 - accuracy: 0.9567 - val_loss: 0.1243 - val_accuracy: 0.9579
    Epoch 14/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.1324 - accuracy: 0.9569 - val_loss: 0.1262 - val_accuracy: 0.9583
    Epoch 15/15
    431/431 [==============================] - 7s 16ms/step - loss: 0.1339 - accuracy: 0.9554 - val_loss: 0.1214 - val_accuracy: 0.9601



![png](classifier_files/classifier_23_1.png)


    Evaluation:
    87/87 [==============================] - 1s 6ms/step - loss: 0.1214 - accuracy: 0.9601
    {'loss': 0.12139755487442017, 'accuracy': 0.9600870609283447}
    Confusion matrix:



![png](classifier_files/classifier_23_3.png)

