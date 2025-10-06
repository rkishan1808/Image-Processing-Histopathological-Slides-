import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Attention, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import numpy as np
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Permute
import cv2

# Define data generators for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('/Users/ravikishan/Desktop/HE/Train', target_size=(224, 224), batch_size=32, class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory('/Users/ravikishan/Desktop/HE/Test', target_size=(224, 224), batch_size=32, class_mode='binary')

# Define a custom transformer-based pooling layer
class TransformerPooling(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(TransformerPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(self.output_dim, activation='softmax')
        super(TransformerPooling, self).build(input_shape)

    def call(self, inputs):
        attention_probs = self.dense(inputs)
        attention_mul = Multiply()([inputs, attention_probs])
        pooled_output = GlobalAveragePooling2D()(attention_mul)
        return pooled_output


# Define the ResNet50 model for feature extraction
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Replace the GlobalAveragePooling2D layer with the custom TransformerPooling layer

x = resnet.output
x = TransformerPooling(2048)(x)

# Add a final dense layer for binary classification
outputs = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=resnet.input, outputs=outputs)

# Freeze layers in ResNet50 for transfer learning
for layer in resnet.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Implement callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# Train the model with advanced configurations
model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f'Validation loss: {loss}, Validation accuracy: {accuracy}')



# Calculate predictions and probabilities for ROC curve
y_true = val_generator.classes
y_pred = model.predict(val_generator).flatten()

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate F1 score
y_pred_binary = (y_pred > 0.5).astype(int)
f1 = f1_score(y_true, y_pred_binary)
print(f'F1 score: {f1}')

def get_grad_cam(model, img_array):
    # Identify the last convolutional layer in ResNet50
    last_conv_layer_name = 'conv5_block3_out'
    
    # Create a model that maps the input image to the activations of the last conv layer
    # and the final predictions
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Assuming binary classification and interest in class 0

    # Compute the gradient of the loss with respect to the convolutional layer
    grads = tape.gradient(loss, conv_outputs)

    # Pool the gradients over all the axes except the channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weigh the output feature map with the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Resize heatmap to match the size of the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Ensure the input image is in the correct format (uint8)
    if img.dtype != np.uint8:
        img = np.uint8(255 * img)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img

# Obtain Grad-CAM heatmap for a sample image
sample_image, _ = next(val_generator)
heatmap = get_grad_cam(model, sample_image)

# Convert the sample image from the generator (batch format) to an actual image
sample_image = sample_image[0]

# Convert image from RGB to BGR format (OpenCV uses BGR format)
sample_image_bgr = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)

# Overlay the heatmap on the sample image
superimposed_img = overlay_heatmap_on_image(sample_image_bgr, heatmap)

# Display the original image, heatmap, and superimposed image
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(sample_image_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Heatmap')
plt.imshow(heatmap, cmap='jet')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Superimposed Image')
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()