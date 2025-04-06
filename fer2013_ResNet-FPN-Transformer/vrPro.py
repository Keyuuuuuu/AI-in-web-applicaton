import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, applications
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import warnings
from scipy import ndimage

warnings.filterwarnings('ignore')

# Check for GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")


# Enhanced Data loader class using tf.data API instead of ImageDataGenerator
class FER2013Dataset:
    def __init__(self, root_dir, split='train', batch_size=128, img_size=(48, 48), advanced_augmentation=False):
        self.root_dir = os.path.join(root_dir, split)
        self.batch_size = batch_size
        self.img_size = img_size
        self.split = split
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.advanced_augmentation = advanced_augmentation

        # Use tf.data instead of ImageDataGenerator
        self.dataset = self._create_dataset()

    def _parse_function(self, filename, label):
        """Parse a single image file."""
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        image_resized = tf.image.resize(image_decoded, self.img_size)
        image = tf.cast(image_resized, tf.float32) / 255.0
        image = (image - 0.5) * 2  # Normalize to [-1, 1]
        return image, label

    @tf.function
    def _augment(self, image, label):
        """Basic augmentation functions using TensorFlow operations."""
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)

        # Random shift using padding and cropping
        paddings = tf.constant([[4, 4], [4, 4], [0, 0]])
        image = tf.pad(image, paddings, "REFLECT")
        image = tf.image.random_crop(image, size=[self.img_size[0], self.img_size[1], 1])

        # Ensure values stay in valid range
        image = tf.clip_by_value(image, -1.0, 1.0)
        return image, label

    @tf.function
    def _advanced_augment(self, image, label):
        """Advanced augmentation functions that match the original implementation."""
        # Apply basic augmentations first
        image, label = self._augment(image, label)

        # Random noise (20% probability)
        if tf.random.uniform(()) < 0.2:
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
            image = image + noise
            image = tf.clip_by_value(image, -1.0, 1.0)

        # Random occlusion (15% probability)
        if tf.random.uniform(()) < 0.15:
            h, w = self.img_size

            # Create occlusion box with random size
            occlusion_h = tf.cast(h * tf.random.uniform([], 0.1, 0.25), tf.int32)
            occlusion_w = tf.cast(w * tf.random.uniform([], 0.1, 0.25), tf.int32)

            # Random position for occlusion
            y_pos = tf.random.uniform([], 0, h - occlusion_h, dtype=tf.int32)
            x_pos = tf.random.uniform([], 0, w - occlusion_w, dtype=tf.int32)

            # Create mask with zeros at occlusion area
            mask = tf.ones([h, w, 1])
            occlusion = tf.zeros([occlusion_h, occlusion_w, 1])
            paddings = [[y_pos, h - y_pos - occlusion_h], [x_pos, w - x_pos - occlusion_w], [0, 0]]
            occlusion_mask = tf.pad(occlusion, paddings)

            # Apply occlusion by masking
            image = image * (1.0 - occlusion_mask)

        # Simple blur simulation (10% probability)
        # Using fixed blur sizes with conditional execution to make it graph-compatible
        random_val = tf.random.uniform(())

        # Conditional blur with fixed kernel sizes
        def apply_blur_2x2():
            blurred = tf.expand_dims(image, 0)  # Add batch dimension
            blurred = tf.nn.avg_pool2d(blurred, ksize=2, strides=1, padding='SAME')
            return tf.squeeze(blurred, 0)  # Remove batch dimension

        def apply_blur_3x3():
            blurred = tf.expand_dims(image, 0)
            blurred = tf.nn.avg_pool2d(blurred, ksize=3, strides=1, padding='SAME')
            return tf.squeeze(blurred, 0)

        # No blur by default
        image = tf.cond(
            random_val < 0.1,
            lambda: tf.cond(
                tf.random.uniform(()) < 0.5,
                apply_blur_2x2,  # 50% chance of 2x2 blur
                apply_blur_3x3  # 50% chance of 3x3 blur
            ),
            lambda: image  # 90% chance of no blur
        )

        return image, label

    def _create_dataset(self):
        """Create the tf.data.Dataset pipeline with optimized performance."""
        # Get all image paths and labels
        image_paths = []
        image_labels = []

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_file in os.listdir(class_dir):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    image_paths.append(os.path.join(class_dir, img_file))
                    image_labels.append(i)

        # Create dataset from paths and labels
        paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        labels_ds = tf.data.Dataset.from_tensor_slices(image_labels)
        ds = tf.data.Dataset.zip((paths_ds, labels_ds))

        # Parse images in parallel
        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)

        if self.split == 'train':
            # Cache dataset after parsing but before augmentation
            ds = ds.cache()

            # Shuffle first to ensure different augmentations each epoch
            ds = ds.shuffle(buffer_size=min(10000, len(image_paths)))

            # Apply augmentations
            if self.advanced_augmentation:
                ds = ds.map(self._advanced_augment, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

            # Batch and prefetch
            ds = ds.batch(self.batch_size, drop_remainder=False)
            ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            # For validation and test sets, just batch and prefetch
            ds = ds.batch(self.batch_size)
            ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return ds

    def get_dataset(self):
        return self.dataset


# MixupDataGenerator class is kept for reference but not used directly
class MixupDataGenerator:
    """Implements Mixup data augmentation strategy using TensorFlow operations"""

    def __init__(self, dataset, alpha=0.2, num_classes=7):
        self.dataset = dataset
        self.alpha = alpha
        self.num_classes = num_classes

    @tf.function
    def _mixup_batch(self, images, labels):
        """Apply mixup to a batch of images and labels using efficient TF operations."""
        batch_size = tf.shape(images)[0]

        # Generate mixup coefficients from beta distribution
        alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
        gamma = tf.random.gamma(shape=[batch_size], alpha=alpha_tensor, beta=alpha_tensor)
        lam = gamma / (gamma + tf.random.gamma(shape=[batch_size], alpha=alpha_tensor, beta=alpha_tensor))
        lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])

        # Create shuffled indices
        indices = tf.random.shuffle(tf.range(batch_size))

        # Mix images
        mixed_images = lam_x * images + (1 - lam_x) * tf.gather(images, indices)

        # Convert sparse labels to one-hot and mix them
        labels_one_hot = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
        shuffled_labels_one_hot = tf.gather(labels_one_hot, indices)
        lam_y = tf.reshape(lam, [batch_size, 1])
        mixed_labels = lam_y * labels_one_hot + (1 - lam_y) * shuffled_labels_one_hot

        return mixed_images, mixed_labels

    def get_mixup_dataset(self):
        """Returns a dataset with mixup applied to each batch."""
        return self.dataset.map(
            self._mixup_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)


# Focal Loss implementation
class FocalLoss(tf.keras.losses.Loss):
    """实现Focal Loss"""

    def __init__(self, alpha=0.25, gamma=2.0, from_logits=True, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # 将稀疏标签转换为one-hot编码
        if len(tf.shape(y_true)) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])

        # 应用softmax如果y_pred是logits
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # 计算交叉熵
        ce = -y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))

        # 计算Focal权重
        weight = tf.pow(1 - y_pred, self.gamma) * y_true

        # 应用alpha平衡
        if self.alpha > 0:
            alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
            weight = weight * alpha_factor

        # 计算最终损失
        loss = weight * ce
        return tf.reduce_sum(loss, axis=-1)


# Transformer encoder layer
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead, dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation=activation),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        attn_output = self.mha(inputs, inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ResNet FPN backbone
class ResNetFPN(tf.keras.Model):
    def __init__(self):
        super(ResNetFPN, self).__init__()

        # 加载预训练的ResNet50
        base_model = applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(48, 48, 3)  # 必须是3通道
        )

        # 获取各阶段的输出层
        c2_output = base_model.get_layer('conv2_block3_out').output  # 56x56
        c3_output = base_model.get_layer('conv3_block4_out').output  # 28x28
        c4_output = base_model.get_layer('conv4_block6_out').output  # 14x14
        c5_output = base_model.get_layer('conv5_block3_out').output  # 7x7

        # 创建一个输出多个特征的模型
        self.feature_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=[c2_output, c3_output, c4_output, c5_output]
        )

        # FPN层
        self.toplayer = layers.Conv2D(256, 1, padding='same', name='fpn_c5p5')
        self.latlayer1 = layers.Conv2D(256, 1, padding='same', name='fpn_c4p4')
        self.latlayer2 = layers.Conv2D(256, 1, padding='same', name='fpn_c3p3')
        self.latlayer3 = layers.Conv2D(256, 1, padding='same', name='fpn_c2p2')
        self.smooth1 = layers.Conv2D(256, 3, padding='same', name='fpn_p4')
        self.smooth2 = layers.Conv2D(256, 3, padding='same', name='fpn_p3')
        self.smooth3 = layers.Conv2D(256, 3, padding='same', name='fpn_p2')

    def _upsample_add(self, x, y):
        """上采样x到y的尺寸并相加"""
        _, H, W, _ = y.shape
        x_up = tf.image.resize(x, size=(H, W), method='bilinear')
        return x_up + y

    def call(self, inputs, training=None):
        # 将灰度图像转换为3通道（通道复制方法）
        x = tf.concat([inputs, inputs, inputs], axis=-1)  # [batch, 48, 48, 3]

        # 通过特征提取器获取各个阶段的特征
        c2, c3, c4, c5 = self.feature_extractor(x, training=training)

        # FPN自上而下的路径
        p5 = self.toplayer(c5)  # [batch, 7, 7, 256]

        p4 = self._upsample_add(p5, self.latlayer1(c4))  # [batch, 14, 14, 256]
        p4 = self.smooth1(p4)

        p3 = self._upsample_add(p4, self.latlayer2(c3))  # [batch, 28, 28, 256]
        p3 = self.smooth2(p3)

        p2 = self._upsample_add(p3, self.latlayer3(c2))  # [batch, 56, 56, 256]
        p2 = self.smooth3(p2)

        return c5, p2, p3, p4, p5


# 位置编码
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = layers.Dropout(dropout)

        # 创建位置编码
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, axis=0)  # [1, max_len, d_model]

        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x, training=None):
        # 添加位置编码到输入
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :tf.shape(x)[1], :]
        return self.dropout(x, training=training)


# Transformer编码器
class TransformerEncoder(tf.keras.Model):
    def __init__(self, d_model=256, nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # [CLS]令牌嵌入
        self.cls_token = self.add_weight(
            shape=(1, 1, d_model),
            initializer=tf.random_normal_initializer(stddev=0.02),
            trainable=True,
            name="cls_token"
        )

        # Transformer编码器层
        self.encoder_layers = [
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout=dropout, activation='relu'
            ) for _ in range(num_layers)
        ]

    def call(self, x, training=None):
        # x形状: [batch_size, height, width, channels]
        batch_size = tf.shape(x)[0]

        # 将张量重塑为序列
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, [batch_size, h * w, -1])  # [batch_size, h*w, channels]

        # 添加[CLS]令牌
        cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        x = tf.concat([cls_tokens, x], axis=1)

        # 添加位置编码
        x = self.pos_encoder(x, training=training)

        # 通过Transformer编码器层
        for layer in self.encoder_layers:
            x = layer(x, training=training)

        # 提取[CLS]令牌输出
        cls_output = x[:, 0]

        return cls_output, x


# 完整的混合模型
class FERHybridModel(tf.keras.Model):
    def __init__(self, num_classes=7):
        super(FERHybridModel, self).__init__()
        self.resnet_fpn = ResNetFPN()

        # 将Transformer应用于FPN特征P5
        self.transformer = TransformerEncoder(d_model=256, nhead=8, num_layers=3)

        # 主分类头 - 基于Transformer的[CLS]令牌输出
        self.main_classifier = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes)
        ])

        # 辅助分类头1 - 直接使用C5特征
        self.aux_classifier1 = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes)
        ])

        # 添加额外的辅助分类头来使用P2-P4特征
        # 辅助分类头2 - 使用P4特征
        self.aux_classifier2 = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes)
        ])

        # 辅助分类头3 - 使用P3特征
        self.aux_classifier3 = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes)
        ])

        # 辅助分类头4 - 使用P2特征
        self.aux_classifier4 = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes)
        ])

    def call(self, inputs, training=None):
        # 通过ResNet+FPN
        c5, p2, p3, p4, p5 = self.resnet_fpn(inputs, training=training)

        # 通过Transformer处理P5特征
        cls_output, _ = self.transformer(p5, training=training)

        # 主分类头（基于Transformer）
        main_logits = self.main_classifier(cls_output, training=training)

        # 辅助分类头1（基于C5）
        aux1_logits = self.aux_classifier1(c5, training=training)

        # 辅助分类头2-4（基于P2-P4）
        aux2_logits = self.aux_classifier2(p4, training=training)
        aux3_logits = self.aux_classifier3(p3, training=training)
        aux4_logits = self.aux_classifier4(p2, training=training)

        # 训练时返回所有分类头的结果，推理时只返回主分类头
        if training:
            return main_logits, aux1_logits, aux2_logits, aux3_logits, aux4_logits
        else:
            return main_logits, aux1_logits


# 修改后的自定义训练步骤与加权损失
class FERTrainer:
    def __init__(self, model, optimizer, loss_fn, alpha=0.7, use_mixup=False, mixup_alpha=0.2, mixup_prob=0.5):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.alpha = alpha  # 主分类头与辅助分类头之间的权重
        self.beta = (1 - alpha) / 4  # 四个辅助分类头平分剩余权重

        # Mixup相关参数
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha  # Beta分布参数
        self.mixup_prob = mixup_prob  # 应用Mixup的概率
        self.num_classes = 7  # 类别数

        # 指标
        self.train_loss = metrics.Mean(name='train_loss')
        self.train_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.val_loss = metrics.Mean(name='val_loss')
        self.val_accuracy = metrics.SparseCategoricalAccuracy(name='val_accuracy')

        # 用于跟踪当前epoch和step
        self.current_epoch = 0
        self.current_step = 0

    def _apply_mixup(self, images, labels):
        """在训练步骤中应用mixup数据增强"""
        batch_size = tf.shape(images)[0]

        # 生成mixup系数
        alpha = tf.constant(self.mixup_alpha, dtype=tf.float32)
        gamma = tf.random.gamma(shape=[batch_size], alpha=alpha, beta=alpha)
        lam = gamma / (gamma + tf.random.gamma(shape=[batch_size], alpha=alpha, beta=alpha))

        # 将lambda重塑为适合图像和标签的形状
        lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
        lam_y = tf.reshape(lam, [batch_size, 1])

        # 创建混洗索引
        indices = tf.random.shuffle(tf.range(batch_size))

        # 混合图像
        mixed_images = lam_x * images + (1 - lam_x) * tf.gather(images, indices)

        # 对标签进行one-hot编码
        labels_one_hot = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
        shuffled_labels_one_hot = tf.gather(labels_one_hot, indices)

        # 混合one-hot标签
        mixed_labels_one_hot = lam_y * labels_one_hot + (1 - lam_y) * shuffled_labels_one_hot

        return mixed_images, mixed_labels_one_hot

    @tf.function
    def train_step(self, images, labels):
        # 初始化变量，确保在所有条件分支中都存在
        labels_one_hot = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
        use_one_hot_labels = False

        # 如果启用了mixup，随机决定是否应用
        if self.use_mixup and tf.random.uniform(()) < self.mixup_prob:
            images, labels_one_hot = self._apply_mixup(images, labels)
            use_one_hot_labels = True

        with tf.GradientTape() as tape:
            # 前向传播
            outputs = self.model(images, training=True)

            # 检查输出是否为元组或列表
            if isinstance(outputs, (tuple, list)):
                # 处理5个输出的情况
                if len(outputs) == 5:
                    main_logits, aux1_logits, aux2_logits, aux3_logits, aux4_logits = outputs

                    # 根据标签类型计算损失
                    if use_one_hot_labels:
                        main_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels_one_hot, logits=main_logits))
                        aux1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels_one_hot, logits=aux1_logits))
                        aux2_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels_one_hot, logits=aux2_logits))
                        aux3_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels_one_hot, logits=aux3_logits))
                        aux4_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels_one_hot, logits=aux4_logits))
                    else:
                        main_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.cast(labels, tf.int32), logits=main_logits))
                        aux1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.cast(labels, tf.int32), logits=aux1_logits))
                        aux2_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.cast(labels, tf.int32), logits=aux2_logits))
                        aux3_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.cast(labels, tf.int32), logits=aux3_logits))
                        aux4_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.cast(labels, tf.int32), logits=aux4_logits))

                    # 组合所有损失
                    loss = (self.alpha * main_loss +
                            self.beta * aux1_loss +
                            self.beta * aux2_loss +
                            self.beta * aux3_loss +
                            self.beta * aux4_loss)
                # 处理2个输出的情况
                elif len(outputs) == 2:
                    main_logits, aux_logits = outputs

                    if use_one_hot_labels:
                        main_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels_one_hot, logits=main_logits))
                        aux_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels_one_hot, logits=aux_logits))
                    else:
                        main_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.cast(labels, tf.int32), logits=main_logits))
                        aux_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.cast(labels, tf.int32), logits=aux_logits))

                    loss = self.alpha * main_loss + (1 - self.alpha) * aux_loss
                else:
                    raise ValueError(f"Unexpected number of outputs: {len(outputs)}")
            else:
                # 处理单一输出的情况
                main_logits = outputs
                if use_one_hot_labels:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels_one_hot, logits=main_logits))
                else:
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.cast(labels, tf.int32), logits=main_logits))

        # 反向传播
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # 检查哪些变量没有梯度（调试用）
        non_trainable_vars = []
        for grad, var in zip(gradients, self.model.trainable_variables):
            if grad is None:
                non_trainable_vars.append(var.name)

        if non_trainable_vars:
            print(f"Epoch {self.current_epoch}, Step {self.current_step}: 以下变量没有梯度:")
            for var_name in non_trainable_vars:
                print(f"  - {var_name}")

        # 应用梯度更新
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # 更新指标
        self.train_loss.update_state(loss)

        # 确保使用正确的主分类头输出
        if isinstance(outputs, (tuple, list)):
            main_logits = outputs[0]

        # 计算准确率 - 注意处理mixup和非mixup情况
        if use_one_hot_labels:
            # 使用argmax将one-hot标签转换回稀疏索引来计算准确率
            self.train_accuracy.update_state(tf.argmax(labels_one_hot, axis=1), main_logits)
        else:
            self.train_accuracy.update_state(labels, main_logits)

        # 递增步骤计数
        self.current_step += 1

        return loss

    @tf.function
    def val_step(self, images, labels):
        # 前向传播
        outputs = self.model(images, training=False)

        # 处理不同数量的输出
        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 2:
                main_logits, aux_logits = outputs
                main_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(labels, tf.int32), logits=main_logits))
                aux_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(labels, tf.int32), logits=aux_logits))
                loss = self.alpha * main_loss + (1 - self.alpha) * aux_loss
            else:
                # 验证时不期望有5个输出
                main_logits = outputs[0]
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(labels, tf.int32), logits=main_logits))
        else:
            main_logits = outputs
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.int32), logits=main_logits))

        # 更新指标
        self.val_loss.update_state(loss)
        if isinstance(outputs, (tuple, list)):
            main_logits = outputs[0]
        self.val_accuracy.update_state(labels, main_logits)

        return loss

    def train(self, train_dataset, val_dataset, epochs=30, class_weights=None):
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        # 创建检查点管理器
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory='./checkpoints', max_to_keep=3  # 保留更多检查点
        )

        for epoch in range(epochs):
            # 更新当前epoch
            self.current_epoch = epoch + 1
            # 每个epoch重置step计数
            self.current_step = 0

            # 在每个epoch开始时重置指标
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            # 训练阶段
            for images, labels in tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
                self.train_step(images, labels)

            # 验证阶段
            for images, labels in tqdm(val_dataset, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                self.val_step(images, labels)

            # 收集用于绘图的指标
            train_loss = self.train_loss.result().numpy()
            train_acc = self.train_accuracy.result().numpy() * 100
            val_loss = self.val_loss.result().numpy()
            val_acc = self.val_accuracy.result().numpy() * 100

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # 显示结果
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_manager.save()
                print(f'模型已保存，验证准确率: {val_acc:.2f}%')

            # 也定期保存（每5个epoch）
            if (epoch + 1) % 5 == 0:
                checkpoint_manager.save(checkpoint_number=epoch + 1)
                print(f'定期检查点已保存，epoch {epoch + 1}')

        # 绘制训练曲线
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_accs, label='Train Acc')
        plt.plot(range(1, epochs + 1), val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

        return train_losses, val_losses, train_accs, val_accs

    def test(self, test_dataset, classes):
        # 加载最佳模型
        checkpoint = tf.train.Checkpoint(model=self.model)
        latest = tf.train.latest_checkpoint('./checkpoints')
        if latest:
            checkpoint.restore(latest)
            print(f"从 {latest} 恢复模型")

        all_preds = []
        all_targets = []
        class_correct = {i: 0 for i in range(len(classes))}
        class_total = {i: 0 for i in range(len(classes))}

        # 测试阶段
        test_accuracy = metrics.SparseCategoricalAccuracy()

        for images, labels in tqdm(test_dataset, desc='Testing'):
            # 前向传播（只使用主分类头）
            outputs = self.model(images, training=False)
            if isinstance(outputs, (tuple, list)):
                main_logits = outputs[0]
            else:
                main_logits = outputs

            # 更新准确率
            test_accuracy.update_state(labels, main_logits)

            # 收集用于混淆矩阵的预测
            pred = tf.argmax(main_logits, axis=1).numpy()
            all_preds.extend(pred)
            all_targets.extend(labels.numpy())

            # 计算每类准确率
            for i in range(len(pred)):
                label = int(labels[i])
                class_total[label] += 1
                if pred[i] == label:
                    class_correct[label] += 1

        test_acc = test_accuracy.result().numpy() * 100
        print(f'测试准确率: {test_acc:.2f}%')

        # 打印每类准确率
        print("\n每个类别的准确率:")
        for i in range(len(classes)):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                print(f"{classes[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")

        # 计算并绘制混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('预测')
        plt.ylabel('真实')
        plt.title('混淆矩阵')
        plt.savefig('confusion_matrix.png')
        plt.show()

        # 计算标准化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('预测')
        plt.ylabel('真实')
        plt.title('标准化混淆矩阵')
        plt.savefig('normalized_confusion_matrix.png')
        plt.show()

        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(all_targets, all_preds, target_names=classes))

        return test_acc, cm


def main():
    # 设置GPU内存增长
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 数据路径和参数
    data_root = './fer2013'
    batch_size = 128
    img_size = (48, 48)
    num_epochs = 100

    # 创建数据加载器 - 使用优化后的tf.data实现
    print("Creating datasets...")
    train_data = FER2013Dataset(root_dir=data_root, split='train', batch_size=batch_size,
                                img_size=img_size, advanced_augmentation=True)
    val_data = FER2013Dataset(root_dir=data_root, split='valid', batch_size=batch_size, img_size=img_size)
    test_data = FER2013Dataset(root_dir=data_root, split='test', batch_size=batch_size, img_size=img_size)

    # 获取基础数据集 - 不再创建或合并mixup数据集
    train_dataset = train_data.get_dataset()
    val_dataset = val_data.get_dataset()
    test_dataset = test_data.get_dataset()

    # 类别权重处理
    class_counts = [3995, 436, 4097, 7215, 4830, 3171, 4965]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    print(f"类别权重: {[round(w, 2) for w in class_weights]}")

    # 创建模型
    model = FERHybridModel(num_classes=7)

    # 优化器和学习率调度
    lr_schedule = optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-4,
        first_decay_steps=10 * (total_samples // batch_size),
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-6
    )
    optimizer = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    # 损失函数
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    # 创建训练器 - 现在在训练步骤中直接应用mixup
    trainer = FERTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=focal_loss,
        alpha=0.7,
        use_mixup=True,
        mixup_alpha=0.2,
        mixup_prob=0.5
    )

    # 模型编译
    dummy_input = tf.random.normal((1, 48, 48, 1))
    _ = model(dummy_input)
    print("模型编译完成")

    # 训练
    print(f"开始训练，共{num_epochs}个epoch...")
    train_losses, val_losses, train_accs, val_accs = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=num_epochs,
        class_weights=class_weights
    )

    # 测试
    print("测试模型...")
    test_acc, confusion_mat = trainer.test(test_dataset, train_data.classes)
    print(f"最终测试准确率: {test_acc:.2f}%")

    # 保存模型
    model.save('final_complete_model')
    print("训练完成!")


if __name__ == '__main__':
    main()