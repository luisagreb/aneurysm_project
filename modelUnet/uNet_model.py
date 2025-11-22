import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# --- Custom Metrics and Loss (Dice Coefficient and Loss) ---

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for binary segmentation.
    Measures the overlap between the prediction and the ground truth.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice loss (1 - Dice Coefficient). Minimizing this maximizes the Dice Score.
    """
    return 1.0 - dice_coefficient(y_true, y_pred)

# --- U-Net Helper Functions ---

def conv_block_3d(input_tensor, num_filters, kernel_size=(3, 3, 3), padding='same'):
    """Two 3D Convolutional layers followed by Batch Normalization and ReLU activation."""
    # First convolution
    x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    
    # Second convolution
    x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x

# --- 3D U-Net Model Definition ---

def unet_model(input_shape, num_classes):
    """
    Defines the 3D U-Net architecture.
    
    Args:
        input_shape (tuple): (depth, height, width, channels)
        num_classes (int): Number of output classes (1 for binary segmentation).
        
    Returns:
        tf.keras.Model: Compiled 3D U-Net model.
    """
    
    # Encoder (Contracting Path)
    inputs = Input(input_shape)
    
    # Block 1
    conv1 = conv_block_3d(inputs, 32)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    # Block 2
    conv2 = conv_block_3d(pool1, 64)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    # Block 3
    conv3 = conv_block_3d(pool2, 128)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    # Bottom (Bottleneck)
    bottleneck = conv_block_3d(pool3, 256)
    
    # Decoder (Expanding Path)
    
    # Block 4
    up4 = UpSampling3D(size=(2, 2, 2))(bottleneck)
    up4 = concatenate([up4, conv3], axis=-1)
    conv4 = conv_block_3d(up4, 128)
    
    # Block 5
    up5 = UpSampling3D(size=(2, 2, 2))(conv4)
    up5 = concatenate([up5, conv2], axis=-1)
    conv5 = conv_block_3d(up5, 64)

    # Block 6
    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = concatenate([up6, conv1], axis=-1)
    conv6 = conv_block_3d(up6, 32)

    # Output Layer
    # Use sigmoid for binary (num_classes=1) segmentation
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    final_conv = Conv3D(num_classes, kernel_size=(1, 1, 1), activation=activation)(conv6)

    model = Model(inputs=inputs, outputs=final_conv)
    
    # Compile the model
    # We use Adam optimizer, and Dice Loss for robust medical segmentation
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=[dice_coefficient, 'accuracy'])
    
    return model