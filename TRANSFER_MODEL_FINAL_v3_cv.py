import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,roc_curve, auc, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Conv3D, MaxPooling3D, Dropout,
                                     BatchNormalization, GlobalAveragePooling3D, Activation, Add, ReLU, Dropout)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from tensorflow.keras.regularizers import l2
import pandas as pd
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
    fistula_images = train_images[train_labels == 1]
    normal_images = train_images[train_labels == 0]

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

    print(f"Training data balanced: fistula count = {np.sum(balanced_labels == 1)}, normal count = {np.sum(balanced_labels == 0)}")
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

def plot_and_save_metrics(history,fold):
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
        plt.savefig(f'fold_{fold+1}_{metric}_curve_name.png')
        plt.close()
# 主函数
def main():

    images, labels = load_dataset(DATA_DIR)

    if len(images) == 0:
        print("No images found.")
        return

    images = np.array(images)
    labels = np.array(labels)

    outer_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

    results = []

    os.makedirs("cv_models", exist_ok=True)
    os.makedirs("cv_plots", exist_ok=True)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    # =====================================================
    # 外层 CV
    # =====================================================

    for fold, (train_idx, test_idx) in enumerate(
            outer_kfold.split(images, labels)):

        print(f"\n========== Fold {fold+1} ==========")

        # ===== 外层划分 =====
        train_images = images[train_idx]
        train_labels = labels[train_idx]

        test_images = images[test_idx]
        test_labels = labels[test_idx]

        # =====================================================
        # 训练集 SMOTE 平衡（只在 train 上）
        # =====================================================

        train_flat = train_images.reshape(len(train_images), -1)

        smote = SMOTE(random_state=4)
        train_balanced, labels_balanced = smote.fit_resample(
            train_flat,
            train_labels
        )

        train_balanced = train_balanced.reshape(
            -1,
            IMG_SIZE[0],
            IMG_SIZE[1],
            IMG_SIZE[2],
            1
        )

        # =====================================================
        # train / val 划分（在平衡后数据中）
        # =====================================================

        X_train, X_val, y_train, y_val = train_test_split(
            train_balanced,
            labels_balanced,
            test_size=0.2,
            stratify=labels_balanced,
            random_state=4
        )

        # =====================================================
        # dataset pipeline
        # =====================================================

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)
        ).shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)
        ).batch(BATCH_SIZE)

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_images, test_labels)
        ).batch(BATCH_SIZE)

        # =====================================================
        # 模型
        # =====================================================

        model = build_model((IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 1))

        # callbacks = [
        #     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10),
        #     EarlyStopping(monitor='val_loss', patience=30,
        #                   restore_best_weights=True)
        # ]
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
        ]
        model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        # 训练模型
        history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset,
                             callbacks=callbacks)
        # =====================================================
        # 保存模型
        # =====================================================

        model_path = f"cv_models/fold_{fold+1}_best_model.h5"
        model.save(model_path)
        plot_and_save_metrics(history,fold)
        # =====================================================
        # test评估
        # =====================================================

        y_prob = model.predict(test_dataset).ravel()
        y_pred = (y_prob > 0.5).astype(int)

        fpr, tpr, _ = roc_curve(test_labels, y_prob)
        roc_auc = auc(fpr, tpr)
        # ===== 插值用于ROC汇总 =====
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

        # ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.legend()
        plt.title(f"ROC Fold {fold+1}")
        plt.savefig(f"cv_plots/roc_fold_{fold+1}.png")
        plt.close()

        # Confusion matrix
        cm = confusion_matrix(test_labels, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix Fold {fold+1}")
        plt.savefig(f"cv_plots/cm_fold_{fold+1}.png")
        plt.close()

        TN, FP, FN, TP = cm.ravel()

        # ===== 医学指标 =====
        sensitivity = TP / (TP + FN + 1e-8)  # Recall+
        specificity = TN / (TN + FP + 1e-8)
        ppv = TP / (TP + FP + 1e-8)  # Precision+
        npv = TN / (TN + FN + 1e-8)

        # ===== sklearn报告 =====
        report = classification_report(
            test_labels,
            y_pred,
            output_dict=True
        )

        results.append({
            "fold": fold + 1,

            # 基本指标
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc": roc_auc,

            # 医学指标
            "sensitivity": sensitivity,
            "specificity": specificity,
            "PPV": ppv,
            "NPV": npv
        })

    # =====================================================
    # 合并 ROC 曲线
    # =====================================================

    plt.figure(figsize=(8, 8))

    # 每折 ROC
    for i, tpr in enumerate(tprs):
        plt.plot(
            mean_fpr,
            tpr,
            alpha=0.3,
            label=f"Fold {i + 1} AUC={aucs[i]:.3f}"
        )

    # mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    std_tpr = np.std(tprs, axis=0)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color='blue',
        label=f"Mean ROC (AUC={mean_auc:.3f})",
        lw=2
    )

    # 标准差阴影
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.fill_between(
        mean_fpr,
        tpr_lower,
        tpr_upper,
        color='grey',
        alpha=0.2,
        label="±1 std"
    )

    plt.plot([0, 1], [0, 1], '--', color='red')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("5-Fold Cross-Validation ROC")
    plt.legend(loc="lower right")

    plt.savefig("cv_plots/combined_roc.png", dpi=300)
    plt.close()

    # =====================================================
    # 保存CV结果
    # =====================================================

    df = pd.DataFrame(results)
    df.to_csv("cv_metrics.csv", index=False)

    # boxplot
    plt.figure()
    plt.boxplot(df[["accuracy","precision","recall","f1","auc","sensitivity","specificity","PPV","NPV"]])
    plt.xticks(range(1,10),["Acc","Precision","Recall","F1","AUC","Sensitivity","Specificity","PPV","NPV"])
    plt.title("Cross-validation Performance")
    plt.savefig("cv_plots/cv_boxplot.png")
    plt.close()

    print("\n===== CV Summary =====")
    print(df.describe())


if __name__ == "__main__":
    main()