import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Conv3D, MaxPooling3D, Dropout,
                                     BatchNormalization, GlobalAveragePooling3D, Activation, Add, ReLU, Dropout)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

DATA_DIR = r"C:\Users\LA\Desktop\ESO_reshape"
CLASSES = ['fistula', 'normal']
IMG_SIZE = (128, 128, 128)  # 目标大小 (H, W, D)最佳256，256，256
BATCH_SIZE = 2
EPOCHS = 300

# 加载 NIfTI 图像
def load_nifti_image(file_path):
    nifti = nib.load(file_path)
    img = nifti.get_fdata()

    img = tf.image.resize(img, IMG_SIZE[:2])
    if img.shape[-1] > IMG_SIZE[2]:
        z_center = img.shape[-1] // 2
        z_start = max(z_center - IMG_SIZE[2] // 2, 0)
        z_end = z_start + IMG_SIZE[2]
        img = img[:, :, z_start:z_end]
    else:
        pad_width = IMG_SIZE[2] - img.shape[-1]
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        img = np.pad(img, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant')

    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return np.expand_dims(img, axis=-1)

# 加载数据集
def load_dataset(data_dir):
    images, labels = [], []
    for label, cls in enumerate(CLASSES):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.exists(cls_path):
            print(f"Category {cls} not found at {cls_path}")
            continue
        for file_name in os.listdir(cls_path):
            file_path = os.path.join(cls_path, file_name)
            if file_name.endswith('.nii.gz'):
                try:
                    img = load_nifti_image(file_path)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return np.array(images), np.array(labels)

# 增强图像
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)

    # 添加 3D 随机旋转
    angle = tf.random.uniform([], minval=-15, maxval=15, dtype=tf.float32)  # 角度范围 -15 到 15
    radians = angle * np.pi / 180
    image = tfa.image.rotate(image, radians)

    # 随机缩放
    zoom_factor = tf.random.uniform([], minval=0.8, maxval=1.2)
    transform_matrix = tf.convert_to_tensor([zoom_factor, 0, 0, 0, zoom_factor, 0, 0, 0], dtype=tf.float32)
    image = tfa.image.transform(image, transform_matrix)

    return image

# 创建增强后的训练集
def balance_training_data(train_images, train_labels):
    fistula_images = train_images[train_labels == 0]
    normal_images = train_images[train_labels == 1]

    # 增强有病数据至与没病数据数量相同
    augmented_images = []
    while len(fistula_images) + len(augmented_images) < len(normal_images):
        for img in fistula_images:
            augmented_images.append(augment_image(img))
            if len(fistula_images) + len(augmented_images) >= len(normal_images):
                break

    augmented_images = np.array(augmented_images)
    augmented_labels = np.zeros(len(augmented_images))

    balanced_images = np.concatenate([train_images, augmented_images])
    balanced_labels = np.concatenate([train_labels, augmented_labels])

    print(f"Training data balanced: fistula count = {np.sum(balanced_labels == 0)}, normal count = {np.sum(balanced_labels == 1)}")
    return balanced_images, balanced_labels

# def residual_block(inputs, filters):
#     shortcut = inputs
#
#     # 主路径
#     x = Conv3D(filters, kernel_size=(3, 3, 3), padding="same")(inputs)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = Conv3D(filters, kernel_size=(3, 3, 3), padding="same")(x)
#     x = BatchNormalization()(x)
#
#     # 确保 shortcut 的通道数与 x 一致
#     if shortcut.shape[-1] != x.shape[-1]:
#         shortcut = Conv3D(filters, kernel_size=(1, 1, 1), padding="same")(shortcut)
#
#     # Add 层
#     x = Add()([shortcut, x])
#     x = ReLU()(x)
#
#     return x
# # 构建模型
# def build_model(input_shape):
#     input_layer = Input(shape=input_shape)
#     x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(input_layer)
#     x = BatchNormalization()(x)
#     x = MaxPooling3D(pool_size=(2, 2, 2))(x)
#
#     for filters in [64, 128, 256]:
#         x = residual_block(x, filters)
#         x = MaxPooling3D(pool_size=(2, 2, 2))(x)
#
#     x = GlobalAveragePooling3D()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     output_layer = Dense(1, activation='sigmoid')(x)
#
#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.compile(optimizer=Adam(learning_rate=0.0001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
#     return model
def residual_block(inputs, filters, block_name):
    shortcut = inputs

    # 主路径
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding="same", name=f"{block_name}_conv1")(inputs)
    x = BatchNormalization(name=f"{block_name}_bn1")(x)
    x = ReLU(name=f"{block_name}_relu1")(x)
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding="same", name=f"{block_name}_conv2")(x)
    x = BatchNormalization(name=f"{block_name}_bn2")(x)

    # 确保 shortcut 的通道数与 x 一致
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = Conv3D(filters, kernel_size=(1, 1, 1), padding="same", name=f"{block_name}_shortcut")(shortcut)

    # Add 层
    x = Add(name=f"{block_name}_add")([shortcut, x])
    x = ReLU(name=f"{block_name}_relu2")(x)

    return x

def build_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu', name="initial_conv")(input_layer)
    x = BatchNormalization(name="initial_bn")(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), name="initial_pool")(x)

    # 添加命名的残差块
    for i, filters in enumerate([64, 128, 256]):
        x = residual_block(x, filters, block_name=f"res_block_{i}")
        x = MaxPooling3D(pool_size=(2, 2, 2), name=f"res_block_{i}_pool")(x)

    x = GlobalAveragePooling3D(name="global_avg_pool")(x)
    x = Dense(128, activation='relu', name="dense_128")(x)
    x = Dropout(0.5, name="dropout")(x)
    output_layer = Dense(1, activation='sigmoid', name="output_layer")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def plot_and_save_metrics(history):
    """绘制并保存损失、准确率和 AUC 曲线"""
    metrics = ['accuracy', 'auc', 'loss']
    for metric in metrics:
        plt.figure()
        plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.legend()
        plt.savefig(f'{metric}_curve_name.png')
        plt.close()
# 主函数
def main():
    images, labels = load_dataset(DATA_DIR)
    print(f"images size: {len(images)}")
    print(f"labels size: {len(labels)}")

    if len(images) == 0:
        print("No images found. Please check your dataset structure.")
        return

    # 第一次划分：训练集和临时集（临时集后续再分为验证集和测试集）
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.4, random_state=4)  # 训练集占60%，临时集占40%

    # 第二次划分：临时集划分为验证集和测试集
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=4)  # 临时集分成验证集和测试集，每个占20%

    # 打印数据集大小
    print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}, Test images: {len(test_images)}")

    # 平衡训练数据
    balanced_images, balanced_labels = balance_training_data(train_images, train_labels)

    # 创建训练、验证、测试数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((balanced_images, balanced_labels))
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    class_weights = {0: len(labels) / (2 * np.sum(labels == 0)),
                     1: len(labels) / (2 * np.sum(labels == 1))}

    # 构建模型
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 1))
    model.summary()

    # 定义回调函数
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
    ]

    # 训练模型
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset,
                        class_weight=class_weights, callbacks=callbacks)

    # 保存模型
    model.save('fistula_classification_model_balanced_validation-300-2-128-128-name.h5')
    print("Model saved successfully.")

    # 绘制并保存训练和验证过程的曲线
    plot_and_save_metrics(history)

    # 测试集评估
    y_pred = (model.predict(test_dataset) > 0.5).astype("int32")
    print(classification_report(test_labels, y_pred, target_names=CLASSES))
    cm = confusion_matrix(test_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap='Blues')
    plt.savefig('validation_confusion_matrix_name.png')
    plt.close()

if __name__ == "__main__":
    main()