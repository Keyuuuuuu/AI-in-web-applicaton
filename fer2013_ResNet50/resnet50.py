import os

print("当前工作目录:", os.getcwd())
import datetime
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import time
from tensorflow.keras import mixed_precision
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import gc
import pathlib

# 启用XLA JIT编译以提高性能
tf.config.optimizer.set_jit(True)


# 配置类：集中管理所有参数
class Config:
    def __init__(self):
        # 基本设置
        self.random_seed = 42
        self.use_mixed_precision = True

        # 路径设置
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('../logs', timestamp)
        # 修改为FER2013数据集路径
        self.train_dir = 'fer2013/train'
        self.val_dir = 'fer2013/valid'
        self.test_dir = 'fer2013/test'

        # 图像预处理参数
        self.img_height = 224
        self.img_width = 224
        self.batch_size = 128  # 可以尝试更大的批次大小，如256或512
        self.num_classes = 7

        # 类别标签
        self.class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # 训练参数
        self.learning_rate = 1e-4
        self.epochs = 30
        self.dropout_rate1 = 0.5
        self.dropout_rate2 = 0.4

        # 数据加载参数
        self.buffer_size = 10000  # 用于数据集随机打乱的缓冲区大小
        self.prefetch_buffer = tf.data.AUTOTUNE  # 自动优化预取缓冲区大小

        # 微调参数
        self.freeze_layers = 100  # 冻结部分ResNet50层，提高训练速度和防止过拟合

        # 数据增强参数
        self.rotation_range = 20
        self.width_shift_range = 0.2
        self.height_shift_range = 0.2
        self.horizontal_flip = True


# 设置日志记录
def setup_logging(save_dir):
    """配置日志系统"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'console_output.txt')

    # 获取logger实例
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 清除现有处理器
    if logger.handlers:
        logger.handlers.clear()

    # 创建文件handler并设置格式
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 添加控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# 初始化环境
def initialize_environment(config, logger):
    """初始化训练环境：设置随机种子、GPU、字体等"""
    # 设置随机种子
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    # 配置混合精度
    if config.use_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        logger.info("已启用混合精度训练")

    # GPU设置
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"检测到 {len(gpus)} 个GPU设备，显存配置成功")
        except RuntimeError as e:
            logger.error(e)
    else:
        logger.warning("未检测到GPU设备，将使用CPU训练（速度可能较慢）")

    # 设置通用字体（避免中文字体问题）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)

    # 保存配置信息
    with open(os.path.join(config.save_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        for key, value in config.__dict__.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"配置信息已保存到: {os.path.join(config.save_dir, 'config.txt')}")


# 数据增强函数
def augment_image(image):
    """执行数据增强操作"""
    # 随机水平翻转
    image = tf.image.random_flip_left_right(image)

    # 随机调整亮度
    image = tf.image.random_brightness(image, 0.2)

    # 随机调整对比度
    image = tf.image.random_contrast(image, 0.8, 1.2)

    return image


# 数据准备
def prepare_data_tfdata(config, logger):
    """使用tf.data API准备训练和验证数据集"""

    # 定义处理单个图像的函数
    def process_path(file_path, label):
        # 读取图像
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [config.img_height, config.img_width])

        # 预处理 (与ResNet50预处理函数等效)
        img = tf.keras.applications.resnet50.preprocess_input(img)

        # 转换标签为one-hot编码
        label = tf.one_hot(label, config.num_classes)
        return img, label

    # 定义增强图像的函数
    def augment(image, label):
        # 随机水平翻转
        image = tf.image.random_flip_left_right(image)

        # 随机裁剪和调整大小
        image = tf.image.resize_with_crop_or_pad(image, config.img_height + 30, config.img_width + 30)
        image = tf.image.random_crop(image, [config.img_height, config.img_width, 3])

        # 随机亮度和对比度调整
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)

        return image, label

    # 获取训练数据路径和标签
    train_files = []
    train_labels = []

    for i, class_name in enumerate(config.class_labels):
        class_dir = os.path.join(config.train_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"训练集中未找到类别目录: {class_dir}")
            continue

        class_files = [str(path) for path in pathlib.Path(class_dir).glob('*.[jJ][pP][gG]')]
        class_files.extend([str(path) for path in pathlib.Path(class_dir).glob('*.[jJ][pP][eE][gG]')])
        class_files.extend([str(path) for path in pathlib.Path(class_dir).glob('*.[pP][nN][gG]')])

        if not class_files:
            logger.warning(f"类别 '{class_name}' 中未找到图像文件")
            continue

        train_files.extend(class_files)
        train_labels.extend([i] * len(class_files))

    # 获取验证数据路径和标签
    val_files = []
    val_labels = []

    for i, class_name in enumerate(config.class_labels):
        class_dir = os.path.join(config.val_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"验证集中未找到类别目录: {class_dir}")
            continue

        class_files = [str(path) for path in pathlib.Path(class_dir).glob('*.[jJ][pP][gG]')]
        class_files.extend([str(path) for path in pathlib.Path(class_dir).glob('*.[jJ][pP][eE][gG]')])
        class_files.extend([str(path) for path in pathlib.Path(class_dir).glob('*.[pP][nN][gG]')])

        if not class_files:
            logger.warning(f"类别 '{class_name}' 中未找到图像文件")
            continue

        val_files.extend(class_files)
        val_labels.extend([i] * len(class_files))

    # 获取测试数据路径和标签
    test_files = []
    test_labels = []

    for i, class_name in enumerate(config.class_labels):
        class_dir = os.path.join(config.test_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"测试集中未找到类别目录: {class_dir}")
            continue

        class_files = [str(path) for path in pathlib.Path(class_dir).glob('*.[jJ][pP][gG]')]
        class_files.extend([str(path) for path in pathlib.Path(class_dir).glob('*.[jJ][pP][eE][gG]')])
        class_files.extend([str(path) for path in pathlib.Path(class_dir).glob('*.[pP][nN][gG]')])

        if not class_files:
            logger.warning(f"类别 '{class_name}' 中未找到图像文件")
            continue

        test_files.extend(class_files)
        test_labels.extend([i] * len(class_files))

    # 创建训练数据集
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    # 打乱训练集
    train_ds = train_ds.shuffle(buffer_size=min(config.buffer_size, len(train_files)),
                                reshuffle_each_iteration=True,
                                seed=config.random_seed)
    # 映射处理函数
    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    # 应用数据增强
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    # 批处理
    train_ds = train_ds.batch(config.batch_size)
    # 添加repeat()以确保数据集不会在一个epoch后耗尽
    train_ds = train_ds.repeat()
    # 预取
    train_ds = train_ds.prefetch(config.prefetch_buffer)

    # 创建验证数据集
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(config.batch_size)
    # 验证集也添加repeat()以便多轮评估
    val_ds = val_ds.repeat()
    val_ds = val_ds.prefetch(config.prefetch_buffer)

    # 创建测试数据集
    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(config.batch_size)
    test_ds = test_ds.prefetch(config.prefetch_buffer)

    # 计算类别权重
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # 记录数据集信息
    logger.info(f"训练集样本数: {len(train_files)}")
    logger.info(f"验证集样本数: {len(val_files)}")
    logger.info(f"测试集样本数: {len(test_files)}")
    logger.info(f"类别分布: {dict(class_counts)}")
    logger.info(f"类别权重: {class_weights}")

    # 检查是否所有类别都有样本
    missing_classes = [config.class_labels[i] for i in range(config.num_classes) if i not in class_counts]
    if missing_classes:
        logger.warning(f"以下类别在训练集中没有样本: {missing_classes}")

    # 计算每个epoch的步数
    steps_per_epoch = len(train_files) // config.batch_size
    validation_steps = len(val_files) // config.batch_size
    test_steps = len(test_files) // config.batch_size
    logger.info(f"每个epoch的训练步数: {steps_per_epoch}")
    logger.info(f"每个epoch的验证步数: {validation_steps}")
    logger.info(f"测试的步数: {test_steps}")

    return train_ds, val_ds, test_ds, class_weights, steps_per_epoch, validation_steps, test_steps


# 创建模型
def create_model(config, logger):
    """创建并编译模型"""
    logger.info("创建模型...")
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(config.img_height, config.img_width, 3)
    )

    # 有选择地冻结基础模型层
    if config.freeze_layers > 0:
        for layer in base_model.layers[:config.freeze_layers]:
            layer.trainable = False
        trainable_layers = sum(1 for layer in base_model.layers if layer.trainable)
        logger.info(f"已冻结基础模型的前 {config.freeze_layers} 层，共 {trainable_layers} 层可训练")
    else:
        logger.info("所有层都设置为可训练")

    # 创建模型
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(config.dropout_rate1),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(config.dropout_rate2),
        layers.Dense(config.num_classes, activation='softmax')
    ])

    # 创建优化器
    optimizer = optimizers.Adam(learning_rate=config.learning_rate)
    if config.use_mixed_precision:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # 输出模型摘要
    model.summary(print_fn=logger.info)

    # 尝试进行模型预热
    logger.info("执行模型预热...")
    try:
        # 创建一批假数据
        dummy_input = tf.random.normal([1, config.img_height, config.img_width, 3])
        # 预热模型
        _ = model(dummy_input, training=False)
        logger.info("模型预热完成")
    except Exception as e:
        logger.error(f"模型预热失败: {e}")

    return model


# 自定义回调函数
class MemoryCleanup(Callback):
    """在每个epoch结束时清理内存"""

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


class TrainingTimeLogger(Callback):
    """记录每个epoch的训练时间"""

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        logging.getLogger(__name__).info(f"Epoch {epoch + 1}/{self.params['epochs']} 耗时: {epoch_time:.2f} 秒")


# 创建回调函数
def create_callbacks(config, logger):
    """创建训练回调函数"""
    callbacks = [
        TensorBoard(log_dir=os.path.join(config.save_dir, 'tensorboard_logs')),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            os.path.join(config.save_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TrainingTimeLogger(),
        MemoryCleanup()
    ]

    return callbacks


# 训练模型
def train_model(model, train_ds, val_ds, class_weights, callbacks, config, logger, steps_per_epoch, validation_steps):
    """训练模型并返回训练历史"""
    logger.info("开始训练模型...")
    start_time = time.time()

    try:
        # 使用steps_per_epoch和validation_steps参数控制每个epoch的步数
        # 由于数据集已经添加了repeat()，不会在一个epoch后耗尽
        history = model.fit(
            train_ds,
            epochs=config.epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            class_weight=class_weights
        )

        end_time = time.time()
        train_time = end_time - start_time
        logger.info(f"训练完成，总耗时: {train_time:.2f} 秒")

        # 保存最终模型
        model.save(os.path.join(config.save_dir, 'final_model.keras'), save_format='keras')
        logger.info(f"最终模型已保存到: {os.path.join(config.save_dir, 'final_model.keras')}")

        return history

    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise


# 绘制训练曲线
def plot_training_curves(history, config, logger):
    """绘制训练曲线"""
    try:
        plt.figure(figsize=(16, 8))

        # 准确率曲线
        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # 损失曲线
        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 精确率曲线
        if 'precision' in history.history:
            plt.subplot(2, 2, 3)
            plt.plot(history.history['precision'], label='Training Precision')
            plt.plot(history.history['val_precision'], label='Validation Precision')
            plt.title('Training and Validation Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()

        # 召回率曲线
        if 'recall' in history.history:
            plt.subplot(2, 2, 4)
            plt.plot(history.history['recall'], label='Training Recall')
            plt.plot(history.history['val_recall'], label='Validation Recall')
            plt.title('Training and Validation Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()

        plt.tight_layout()
        curves_path = os.path.join(config.save_dir, 'training_curves.png')
        plt.savefig(curves_path)
        plt.close()

        logger.info(f"训练曲线已保存至: {curves_path}")

    except Exception as e:
        logger.error(f"绘制训练曲线时出错: {str(e)}")


# 评估模型
def evaluate_model(model, test_ds, config, logger, test_steps):
    """评估模型并生成混淆矩阵和分类报告"""
    logger.info("开始评估模型...")

    try:
        # 模型评估
        evaluation = model.evaluate(test_ds, steps=test_steps, verbose=1)
        metrics = model.metrics_names

        logger.info("模型评估结果:")
        for name, value in zip(metrics, evaluation):
            logger.info(f"{name}: {value:.4f}")

        # 获取预测结果和真实标签
        y_true = []
        y_pred = []

        # 迭代测试数据集获取预测
        for batch_x, batch_y in test_ds.take(test_steps):
            batch_pred = model.predict_on_batch(batch_x)

            # 将one-hot标签转换为类别索引
            batch_y_indices = tf.argmax(batch_y, axis=1).numpy()
            batch_pred_indices = tf.argmax(batch_pred, axis=1).numpy()

            y_true.extend(batch_y_indices)
            y_pred.extend(batch_pred_indices)

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 创建混淆矩阵热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=config.class_labels,
                    yticklabels=config.class_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()

        # 保存混淆矩阵图
        confusion_matrix_path = os.path.join(config.save_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        logger.info(f"混淆矩阵已保存至: {confusion_matrix_path}")

        # 生成分类报告
        report = classification_report(y_true, y_pred,
                                       target_names=config.class_labels,
                                       digits=4)

        # 保存分类报告
        report_path = os.path.join(config.save_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"分类报告已保存至: {report_path}")

        # 在日志中显示分类报告
        logger.info("分类报告：\n" + report)

        # 计算每个类别的准确率
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for label, acc in zip(config.class_labels, class_accuracy):
            logger.info(f"{label} 类别准确率: {acc:.4f}")

        # 绘制每类准确率条形图
        plt.figure(figsize=(12, 6))
        plt.bar(config.class_labels, class_accuracy)
        plt.title('Accuracy per Class')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存条形图
        class_acc_path = os.path.join(config.save_dir, 'class_accuracy.png')
        plt.savefig(class_acc_path)
        plt.close()
        logger.info(f"各类别准确率图已保存至: {class_acc_path}")

    except Exception as e:
        logger.error(f"评估模型时出错: {str(e)}")
        logger.exception("详细错误信息:")


# 主函数
def main():
    """主函数"""
    # 创建配置
    config = Config()

    # 设置日志
    logger = setup_logging(config.save_dir)
    logger.info("=" * 50)
    logger.info("面部表情分类模型训练开始")
    logger.info("=" * 50)

    try:
        # 初始化环境
        initialize_environment(config, logger)

        # 数据准备
        train_ds, val_ds, test_ds, class_weights, steps_per_epoch, validation_steps, test_steps = prepare_data_tfdata(
            config, logger)

        # 创建模型
        model = create_model(config, logger)

        # 创建回调函数
        callbacks = create_callbacks(config, logger)

        # 训练模型
        history = train_model(model, train_ds, val_ds, class_weights, callbacks, config, logger, steps_per_epoch,
                              validation_steps)

        # 绘制训练曲线
        plot_training_curves(history, config, logger)

        # 评估模型
        evaluate_model(model, test_ds, config, logger, test_steps)

        logger.info("=" * 50)
        logger.info("面部表情分类模型训练完成")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"程序执行过程中出错: {str(e)}")
        logger.exception("详细错误信息:")


if __name__ == "__main__":
    main()