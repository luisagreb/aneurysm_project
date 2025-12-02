import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D,
    concatenate, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def weighted_dice_loss(y_true, y_pred,
                       class_weights=(0.2, 0.8),
                       smooth=1e-7):
    """
    Simple weighted Dice:
      loss = 1 - (w0 * Dice_bg + w1 * Dice_fg)

    class_weights: (w_bg, w_fg)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # foreground
    dice_fg_num = 2. * K.sum(y_true_f * y_pred_f) + smooth
    dice_fg_den = K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    dice_fg = dice_fg_num / dice_fg_den

    # background
    y_true_bg = 1.0 - y_true_f
    y_pred_bg = 1.0 - y_pred_f
    dice_bg_num = 2. * K.sum(y_true_bg * y_pred_bg) + smooth
    dice_bg_den = K.sum(y_true_bg) + K.sum(y_pred_bg) + smooth
    dice_bg = dice_bg_num / dice_bg_den

    w_bg, w_fg = class_weights
    dice_total = w_bg * dice_bg + w_fg * dice_fg

    return 1.0 - dice_total


def conv_block_3d(x, filters, kernel_size=(3, 3, 3), padding="same"):
    x = Conv3D(filters, kernel_size, padding=padding,
               kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv3D(filters, kernel_size, padding=padding,
               kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def unet_model(input_shape,
               num_classes=1,
               learning_rate=1e-4,
               class_weights=(0.2, 0.8)):
    """
    3D U-Net for binary segmentation.
    """

    inputs = Input(shape=input_shape)

    # Encoder
    c1 = conv_block_3d(inputs, 16)
    p1 = MaxPooling3D(pool_size=(2, 2, 2))(c1)

    c2 = conv_block_3d(p1, 32)
    p2 = MaxPooling3D(pool_size=(2, 2, 2))(c2)

    c3 = conv_block_3d(p2, 64)
    p3 = MaxPooling3D(pool_size=(2, 2, 2))(c3)

    # Bottleneck
    bn = conv_block_3d(p3, 128)

    # Decoder
    u4 = UpSampling3D(size=(2, 2, 2))(bn)
    u4 = concatenate([u4, c3], axis=-1)
    c4 = conv_block_3d(u4, 64)

    u5 = UpSampling3D(size=(2, 2, 2))(c4)
    u5 = concatenate([u5, c2], axis=-1)
    c5 = conv_block_3d(u5, 32)

    u6 = UpSampling3D(size=(2, 2, 2))(c5)
    u6 = concatenate([u6, c1], axis=-1)
    c6 = conv_block_3d(u6, 16)

    if num_classes == 1:
        out = Conv3D(1, (1, 1, 1), activation="sigmoid", padding="same")(c6)
    else:
        out = Conv3D(num_classes, (1, 1, 1),
                     activation="softmax", padding="same")(c6)

    model = Model(inputs=inputs, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=lambda yt, yp: weighted_dice_loss(
            yt, yp, class_weights=class_weights
        ),
        metrics=[dice_coefficient],
    )

    return model