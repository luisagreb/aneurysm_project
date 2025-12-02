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

# --- NEW: Weighted Dice Loss ---
def weighted_dice_loss(y_true, y_pred, class_weights=None):
    """
    A Dice Loss variant that weights the loss contribution of the positive (1)
    and negative (0) classes. This is critical for highly imbalanced datasets.
    
    If class_weights is None, it defaults to a standard Dice Loss.
    Example class_weights: [0.1, 0.9] -> less weight for background (0),
    more weight for nucleus (1).
    """
    smooth = 1e-7
    
    # Calculate the standard Dice score
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    # Denominators
    numerator = 2. * intersection + smooth
    denominator = K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    
    dice_score = numerator / denominator
    
    # If custom weights are provided, apply them.
    if class_weights is not None:
        # Calculate the contribution of each class to the total Dice calculation
        # This is a simplification of GDL, focusing on the loss gradient.
        # We will use the standard Dice Loss (1 - Dice) but we'll modify the
        # weights in the background to emphasize the foreground.
        # The key is to heavily penalize missing the foreground.
        
        # Calculate the loss for foreground (nucleus) and background
        # Foreground (1):
        loss_fg = 1.0 - (2. * K.sum(y_true_f * y_pred_f) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        
        # Background (0) - using IoU on the inverted mask:
        y_true_inv = 1.0 - y_true_f
        y_pred_inv = 1.0 - y_pred_f
        
        loss_bg = 1.0 - (2. * K.sum(y_true_inv * y_pred_inv) + smooth) / (K.sum(y_true_inv) + K.sum(y_pred_inv) + smooth)

        # Apply weights: class_weights[0] for background, class_weights[1] for foreground
        total_loss = class_weights[0] * loss_bg + class_weights[1] * loss_fg
        return total_loss
    
    # Default to standard Dice Loss if no weights are provided
    return 1.0 - dice_score


# --- U-Net Helper Functions ---

def conv_block_3d(input_tensor, num_filters, kernel_size=(3, 3, 3), padding='same'):
    """Two 3D Convolutional layers followed by Batch Normalization and ReLU activation."""
    # First convolution
    x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second convolution
    x = Conv3D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def unet_model(input_shape, num_classes, learning_rate=1e-4, weights=[0.05, 0.95]):
    """
    Builds and compiles the 3D U-Net model using a weighted loss function.
    
    Args:
        input_shape (tuple): Shape of the input image (D, H, W, C).
        num_classes (int): Number of segmentation classes (usually 1 for binary).
        learning_rate (float): Initial learning rate for the Adam optimizer.
        weights (list): [weight_for_background, weight_for_foreground]
        
    Returns:
        Model: Compiled 3D U-Net model.
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
    # Use sigmoid for binary (num_classes=1)
    # Use softmax for multi-class (num_classes > 1)
    if num_classes == 1:
        activation = 'sigmoid'
        final_conv = Conv3D(num_classes, (1, 1, 1), activation=activation, padding='same')(conv6)
    else:
        activation = 'softmax'
        final_conv = Conv3D(num_classes, (1, 1, 1), activation=activation, padding='same')(conv6)
        
    model = Model(inputs=inputs, outputs=final_conv)
    
    # CRITICAL: Compile with the Weighted Dice Loss and a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=lambda y_true, y_pred: weighted_dice_loss(y_true, y_pred, weights),
        metrics=[dice_coefficient]
    )
    
    return model