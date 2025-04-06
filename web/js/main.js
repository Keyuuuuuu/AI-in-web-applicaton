// main.js
import FeatureMapVisualizer from './featureMap.js';
import ResNetVisualizer from './resnetVis.js';


// 全局变量和常量定义
let model = null;
let currentLanguage = 'en';
let latestPredictions = null;
let isModelLoaded = false;
let cameraController = null;
let currentMode = 'image';
let predictionChart = null;
let featureMapVisualizer = null;
let networkVisualizer = null;

const IMG_HEIGHT = 224;
const IMG_WIDTH = 224;
const NUM_CLASSES = 7;

// 语言资源
const languageResources = {
    en: {
        title: "Face Expression Recognition",
        uploadText: "Upload Image",
        chooseFile: "Choose File",
        predictedExpression: "Predicted Expression:",
        unableToDetermine: "Unable to determine expression.",
        confidenceTitle: "Confidence Scores:",
        languageLabel: "Language:",
        loadingModel: "Loading model...",
        modelNotLoaded: "Please wait for model to load",
        modelLoadSuccess: "Model loaded successfully!",
        modelLoadError: "Failed to load model: ",
        invalidImageFile: "Please upload an image file",
        fileReadError: "Failed to read file",
        imageLoadError: "Failed to load image",
        predictionError: "Prediction failed: ",
        startCamera: "Start Camera",
        stopCamera: "Stop Camera",
        realtimeExpression: "Real-time Expression:",
        cameraError: "Camera access error: ",
        noCamera: "No camera found",
        imageMode: "Image Mode",
        cameraMode: "Camera Mode",
        modelStatus: "Model Status: ",
        loaded: "Loaded",
        notLoaded: "Not Loaded",
        expressions: ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger'],
        predictionTime: "Prediction Time:",
        milliseconds: "ms",
        confidenceChartTitle: "Expression Recognition Confidence Distribution",
        confidenceLabel: "Confidence",
        confidenceValue: "Confidence (%)",
        featureMapTitle: "Feature Map Visualization",
        realtimeFeatureMapTitle: "Real-time Feature Maps",
        featureMapLoading: "Loading feature maps...",
        featureMapError: "Failed to load feature maps",
        featureMapLegend: "Activation Intensity",
        playButton: "Play",
        pauseButton: "Pause",
        speedLabel: "Animation Speed:",
        noFeatureMaps: "No feature maps available",
        networkMode: "Network Structure",
        networkTitle: "Network Structure Visualization",
        networkInfo: "Network Information",
        zoomIn: "Zoom In",
        zoomOut: "Zoom Out",
        resetView: "Reset View",
        // 详细信息面板内容
        layerName: "Layer Name",
        layerType: "Layer Type",
        parameters: "Parameters",
        filters: "Filters",
        kernelSize: "Kernel Size",
        strides: "Strides",
        padding: "Padding",
        activation: "Activation Function",
        useBias: "Use Bias",
        inputShape: "Input Shape",
        outputShape: "Output Shape",
        inConnections: "Input Connections",
        outConnections: "Output Connections",
        // 其他配置项
        convConfig: "Convolution Configuration",
        activationConfig: "Activation Configuration",
        shapeInfo: "Shape Information",
        connectionInfo: "Connection Information",
        basicInfo: "Basic Information",
        inputChannels: "Input Channels",
        outputChannels: "Output Channels",
        paddingTypes: {
            valid: "No Padding",
            same: "Same Padding",
            causal: "Causal Padding"
        },
        activationTypes: {
            relu: "ReLU",
            linear: "Linear",
            sigmoid: "Sigmoid",
            tanh: "Tanh"
        },
        useBiasValues: {
            true: "Yes",
            false: "No"
        },
        none: "None",
        retry: "Retry"
    },
    zh: {
        title: "表情识别",
        uploadText: "上传图片",
        chooseFile: "选择文件",
        predictedExpression: "预测的表情：",
        unableToDetermine: "无法识别表情。",
        confidenceTitle: "置信度：",
        languageLabel: "语言：",
        loadingModel: "正在加载模型...",
        modelNotLoaded: "请等待模型加载完成",
        modelLoadSuccess: "模型加载成功！",
        modelLoadError: "模型加载失败：",
        invalidImageFile: "请上传图片文件",
        fileReadError: "文件读取失败",
        imageLoadError: "图片加载失败",
        predictionError: "预测失败：",
        startCamera: "开启摄像头",
        stopCamera: "关闭摄像头",
        realtimeExpression: "实时表情：",
        cameraError: "摄像头访问错误：",
        noCamera: "未找到摄像头",
        imageMode: "图片模式",
        cameraMode: "摄像头模式",
        modelStatus: "模型状态：",
        loaded: "已加载",
        notLoaded: "未加载",
        expressions: ['中性', '开心', '伤心', '惊讶', '恐惧', '厌恶', '生气'],
        predictionTime: "预测用时：",
        milliseconds: "毫秒",
        confidenceChartTitle: "表情识别置信度分布",
        confidenceLabel: "置信度",
        confidenceValue: "置信度 (%)",
        featureMapTitle: "特征图可视化",
        realtimeFeatureMapTitle: "实时特征图",
        featureMapLoading: "正在加载特征图...",
        featureMapError: "特征图加载失败",
        featureMapLegend: "激活强度",
        playButton: "播放",
        pauseButton: "暂停",
        speedLabel: "动画速度：",
        noFeatureMaps: "无可用特征图",
        networkMode: "网络结构",
        networkTitle: "网络结构可视化",
        networkInfo: "网络信息",
        zoomIn: "放大",
        zoomOut: "缩小",
        resetView: "重置视图",
        // 详细信息面板内容
        layerName: "层名称",
        layerType: "层类型",
        parameters: "参数数量",
        filters: "过滤器数量",
        kernelSize: "卷积核",
        strides: "步长",
        padding: "填充方式",
        activation: "激活函数",
        useBias: "是否使用偏置",
        inputShape: "输入形状",
        outputShape: "输出形状",
        inConnections: "输入连接数",
        outConnections: "输出连接数",
        // 其他配置项
        convConfig: "卷积层配置",
        activationConfig: "激活函数配置",
        shapeInfo: "形状信息",
        connectionInfo: "连接信息",
        basicInfo: "基本信息",
        inputChannels: "输入通道",
        outputChannels: "输出通道",
        paddingTypes: {
            valid: "无填充",
            same: "SAME填充",
            causal: "因果填充"
        },
        activationTypes: {
            relu: "ReLU激活",
            linear: "线性激活",
            sigmoid: "Sigmoid激活",
            tanh: "Tanh激活"
        },
        useBiasValues: {
            true: "使用",
            false: "不使用"
        },
        none: "无",
        retry: "重试"
    }
};


// 摄像头控制器类
class CameraController {
    constructor(model, displayCallback) {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('cameraCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.model = model;
        this.displayCallback = displayCallback;
        this.streaming = false;
        this.animationFrame = null;
        this.processCount = 0;
        this.lastProcessTime = 0;
        this.FPS = 15;
        this.minProcessInterval = 1000 / this.FPS;
    }

    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: IMG_WIDTH },
                    height: { ideal: IMG_HEIGHT }
                },
                audio: false
            });

            this.video.srcObject = stream;
            this.video.style.display = 'block';
            this.canvas.style.display = 'block';
            this.streaming = true;

            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.processFrame();
            };

            await this.video.play();

        } catch (error) {
            throw new Error(languageResources[currentLanguage].cameraError + error.message);
        }
    }

    stop() {
        if (this.streaming) {
            const stream = this.video.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            this.video.srcObject = null;
            this.streaming = false;
            this.video.style.display = 'none';
            this.canvas.style.display = 'none';
            if (this.animationFrame) {
                cancelAnimationFrame(this.animationFrame);
            }
        }
    }

    // 摄像头帧处理函数
    async processFrame() {
        if (!this.streaming) return;

        const currentTime = Date.now();
        const timeDiff = currentTime - this.lastProcessTime;

        if (timeDiff >= this.minProcessInterval) {
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

            try {
                const startTime = performance.now();
                const imgTensor = preprocessImage(this.canvas);

                // 同时进行预测和特征图提取
                const [predictions, featureMaps] = await Promise.all([
                    this.model.predict(imgTensor).data(),
                    featureMapVisualizer.extractFeatureMaps(imgTensor)
                ]);

                const endTime = performance.now();
                const predictionTime = Math.round(endTime - startTime);

                // 显示预测结果
                this.displayCallback(Array.from(predictions), true, predictionTime);

                // 更新特征图
                const featureMapContainer = document.getElementById('realtimeFeatureMaps');
                if (featureMapContainer) {
                    await featureMapVisualizer.animateFeatureMaps(featureMaps, featureMapContainer);
                }

                // 更新网络可视化
                if (currentMode === 'network' && networkVisualizer && networkVisualizer.isInitialized) {
                    await updateNetworkVisualization();
                }

                tf.dispose(imgTensor);
                this.lastProcessTime = currentTime;
            } catch (error) {
                console.error('Frame processing error:', error);
            }
        }

        this.animationFrame = requestAnimationFrame(() => this.processFrame());
    }
}

// 修改 loadModel 函数中的网络可视化初始化部分
async function loadModel() {
    const texts = languageResources[currentLanguage];
    toggleLoading(true);

    try {
        console.log('Starting model loading...');
        model = await tf.loadLayersModel('tfjs_model/model.json');

        // 初始化特征图可视化器
        featureMapVisualizer = new FeatureMapVisualizer();
        await featureMapVisualizer.initialize(model);
        featureMapVisualizer.currentLanguage = currentLanguage;

        // 安全地检测和调用方法，绝不会抛出错误
        if (featureMapVisualizer &&
            typeof featureMapVisualizer.setVisibility === 'function') {
            featureMapVisualizer.setVisibility(true);
        } else {
            // 替代操作：直接设置属性或调用其他可用方法
            if (featureMapVisualizer) {
                featureMapVisualizer.isVisible = true;
                // 如果有resume方法，调用它
                if (typeof featureMapVisualizer.resume === 'function') {
                    featureMapVisualizer.resume();
                }
            }
        }

        // 设置模型加载状态
        isModelLoaded = true;
        updateModelStatus(true);

        // 如果当前是网络模式，初始化网络可视化
        if (currentMode === 'network') {
            await handleNetworkMode();
        }

        console.log(texts.modelLoadSuccess);

    } catch (err) {
        console.error('Model loading error:', err);
        updateModelStatus(false);
        throw new Error(texts.modelLoadError + err.message);
    } finally {
        toggleLoading(false);
    }
}


// 图像预处理函数
function preprocessImage(imageElement) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imageElement);
        tensor = tf.image.resizeBilinear(tensor, [IMG_HEIGHT, IMG_WIDTH]);
        tensor = tensor.toFloat();
        tensor = tf.reverse(tensor, -1);
        const means = [103.939, 116.779, 123.68];
        const centered = tf.sub(tensor, means);
        return centered.expandDims(0);
    });
}

// 表情识别函数
// 在 predictExpression 函数中修改特征图处理部分
// predictExpression 函数的正确实现
// 在 predictExpression 函数中添加调试信息
async function predictExpression(imageElement) {
    let imgTensor = null;
    const startTime = performance.now();

    try {
        if (!isModelLoaded || !model) {
            throw new Error(languageResources[currentLanguage].modelNotLoaded);
        }

        console.log('Starting image preprocessing...');
        imgTensor = preprocessImage(imageElement);

        // 分开进行预测和特征图提取
        const predictions = await model.predict(imgTensor).data();
        const endTime = performance.now();
        const predictionTime = Math.round(endTime - startTime);

        // 显示预测结果
        displayResult(Array.from(predictions), false, predictionTime);

        // 更新网络可视化
        if (currentMode === 'network' && networkVisualizer && networkVisualizer.isInitialized) {
            await updateNetworkVisualization();
        }

        // 提取并显示特征图
        if (featureMapVisualizer) {
            const featureMaps = await featureMapVisualizer.extractFeatureMaps(imgTensor);
            if (featureMaps && featureMaps.length > 0) {
                const featureMapContainer = document.getElementById('imageFeatureMaps');
                if (featureMapContainer) {
                    await featureMapVisualizer.animateFeatureMaps(featureMaps, featureMapContainer);
                }
            }
        }

    } catch (err) {
        console.error('Prediction error:', err);
        throw err;
    } finally {
        if (imgTensor) {
            tf.dispose(imgTensor);
        }
    }
}

// 页面加载和初始化函数
window.addEventListener('load', async () => {
    console.log('TensorFlow.js version:', tf.version.tfjs);
    try {
        initializeText();
        setupEventListeners();
        await loadModel();
    } catch (error) {
        console.error('Initialization failed:', error);
        showError(error.message);
    }
});

// 初始化事件监听器
function setupEventListeners() {
    // 语言选择监听
    document.getElementById('languageSelect').addEventListener('change', function (event) {
        currentLanguage = event.target.value;
        initializeText();
        document.documentElement.lang = currentLanguage;
        if (featureMapVisualizer) {
            featureMapVisualizer.currentLanguage = currentLanguage;
        }
        if (latestPredictions) {
            displayResult(latestPredictions, currentMode === 'camera');
        }
    });

    // 模式切换监听
    document.getElementById('imageMode').addEventListener('click', () => switchMode('image'));
    document.getElementById('cameraMode').addEventListener('click', () => switchMode('camera'));

    // 图片上传监听
    document.getElementById('imageUpload').addEventListener('change', handleImageUpload);

    // 摄像头控制监听
    document.getElementById('startCamera').addEventListener('click', async () => {
        try {
            document.getElementById('startCamera').style.display = 'none';
            document.getElementById('stopCamera').style.display = 'inline-block';
            document.getElementById('realtimeResult').style.display = 'block';

            if (!cameraController) {
                cameraController = new CameraController(model, displayResult);
            }
            await cameraController.start();
        } catch (error) {
            showError(error.message);
            document.getElementById('startCamera').style.display = 'inline-block';
            document.getElementById('stopCamera').style.display = 'none';
        }
    });

    document.getElementById('stopCamera').addEventListener('click', () => {
        if (cameraController) {
            cameraController.stop();
        }
        document.getElementById('startCamera').style.display = 'inline-block';
        document.getElementById('stopCamera').style.display = 'none';
        document.getElementById('realtimeResult').style.display = 'none';
    });
    // 添加网络模式按钮监听器
    document.getElementById('networkMode').addEventListener('click', () => switchMode('network'));

    // 添加网络控制按钮监听器
    // document.getElementById('zoomIn')?.addEventListener('click', () => {
    //     if (networkVisualizer) networkVisualizer.zoomIn();
    // });

    // document.getElementById('zoomOut')?.addEventListener('click', () => {
    //     if (networkVisualizer) networkVisualizer.zoomOut();
    // });

    // document.getElementById('resetView')?.addEventListener('click', () => {
    //     if (networkVisualizer) networkVisualizer.resetView();
    // });

    // 语言切换时也要更新网络可视化器的语言
    document.getElementById('languageSelect').addEventListener('change', function (event) {
        currentLanguage = event.target.value;
        initializeText();
        document.documentElement.lang = currentLanguage;
        if (featureMapVisualizer) {
            featureMapVisualizer.currentLanguage = currentLanguage;
        }
        if (networkVisualizer) {
            networkVisualizer.setLanguage(currentLanguage);
        }
        if (latestPredictions) {
            displayResult(latestPredictions, currentMode === 'camera');
        }
    });
}



// 模式切换函数
// 模式切换函数
function switchMode(mode) {
    if (!isModelLoaded && mode === 'network') {
        showError(languageResources[currentLanguage].modelNotLoaded);
        return;
    }

    currentMode = mode;

    // 1. 更新UI状态
    updateModeButtons(mode);
    updateSectionVisibility(mode);
    clearPreviousResults();

    // 2. 处理各模式特定逻辑
    switch (mode) {
        case 'image':
            handleImageMode();
            break;
        case 'camera':
            handleCameraMode();
            break;
        case 'network':
            handleNetworkMode();
            break;
    }
}

// UI状态更新函数
function updateModeButtons(mode) {
    const modes = ['image', 'camera', 'network'];
    modes.forEach(m => {
        document.getElementById(`${m}Mode`).classList.toggle('active', mode === m);
    });
}

function updateSectionVisibility(mode) {
    const sections = ['image', 'camera', 'network'];
    sections.forEach(section => {
        document.getElementById(`${section}Section`).style.display =
            mode === section ? 'block' : 'none';
    });
}

// 图片模式处理
function handleImageMode() {
    // 停止摄像头
    if (cameraController) {
        cameraController.stop();
    }
    // 清除图片预览
    const imagePreview = document.getElementById('imagePreview');
    imagePreview.src = '';
    imagePreview.style.display = 'none';
}

// 摄像头模式处理
function handleCameraMode() {
    // 重置摄像头按钮状态
    document.getElementById('startCamera').style.display = 'inline-block';
    document.getElementById('stopCamera').style.display = 'none';
    document.getElementById('realtimeResult').style.display = 'none';

    // 确保之前的摄像头实例被清理
    if (cameraController) {
        cameraController.stop();
    }
}

// 在 main.js 中
// main.js
async function handleNetworkMode() {
    const networkContainer = document.getElementById('networkVis');
    if (!networkContainer) {
        console.error('Network visualization container not found');
        return;
    }

    try {
        // 添加加载提示
        networkContainer.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">${languageResources[currentLanguage].networkLoading}</div>
        `;

        // 初始化可视化
        if (!networkVisualizer) {
            // 在创建实例时传入语言资源
            networkVisualizer = new ResNetVisualizer(languageResources);
            // 设置当前语言
            networkVisualizer.setLanguage(currentLanguage);
        }

        // 使用模型的 JSON 文件路径
        const success = await networkVisualizer.initialize(
            networkContainer,
            'tfjs_model/model.json'
        );

        if (!success) {
            throw new Error(languageResources[currentLanguage].networkInitError);
        }

    } catch (error) {
        console.error('Failed to initialize network visualizer:', error);
        showError(languageResources[currentLanguage].networkInitError);

        // 显示错误信息
        networkContainer.innerHTML = `
            <div class="error-message">
                <p>${error.message}</p>
                <button onclick="handleNetworkMode()">
                    ${languageResources[currentLanguage].retry || '重试'}
                </button>
            </div>
        `;
    }
}

// 更新网络可视化
async function updateNetworkVisualization() {
    let imgElement = null;

    // 根据当前模式获取正确的图像元素
    if (currentMode === 'image') {
        imgElement = document.getElementById('imagePreview');
    } else if (currentMode === 'camera') {
        imgElement = document.getElementById('cameraCanvas');
    }

    // 确保图像元素可用且可见
    if (imgElement && imgElement.style.display !== 'none') {
        try {
            const imgTensor = preprocessImage(imgElement);
            await networkVisualizer.updateActivations(imgTensor);
            tf.dispose(imgTensor);
        } catch (error) {
            console.error('Failed to update network activations:', error);
            showError(languageResources[currentLanguage].networkUpdateError);
        }
    }
}

// 处理图片上传
async function handleImageUpload(event) {
    const texts = languageResources[currentLanguage];
    clearPreviousResults();
    showError('');

    if (!isModelLoaded) {
        showError(texts.modelNotLoaded);
        return;
    }

    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        showError(texts.invalidImageFile);
        return;
    }

    const reader = new FileReader();
    reader.onerror = () => showError(texts.fileReadError);
    reader.onload = async (e) => {
        const imgElement = new Image();
        imgElement.onerror = () => showError(texts.imageLoadError);
        imgElement.onload = async () => {
            try {
                document.getElementById('imagePreview').src = e.target.result;
                document.getElementById('imagePreview').style.display = 'block';
                await predictExpression(imgElement);
            } catch (err) {
                showError(texts.predictionError + (err.message || ""));
            }
        };
        imgElement.src = e.target.result;
    };
    reader.readAsDataURL(file);
}


// 创建图表函数
function createChart(containerId, predictions, expressions, texts) {
    console.log('Creating chart for container:', containerId);
    const container = document.getElementById(containerId);
    if (!container) {
        console.error('Chart container not found:', containerId);
        return;
    }

    // 确保容器有合适的尺寸
    container.style.height = '300px';
    container.style.width = '100%';

    // 创建画布
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);

    // 销毁现有图表
    if (predictionChart) {
        predictionChart.destroy();
        predictionChart = null;
    }

    // 准备数据
    const data = {
        labels: expressions,
        datasets: [{
            label: texts.confidenceLabel,
            data: predictions.map(p => parseFloat((p * 100).toFixed(2))),
            backgroundColor: [
                'rgba(54, 162, 235, 0.7)',   // 中性
                'rgba(255, 206, 86, 0.7)',   // 开心
                'rgba(75, 192, 192, 0.7)',   // 伤心
                'rgba(153, 102, 255, 0.7)',  // 惊讶
                'rgba(255, 99, 132, 0.7)',   // 恐惧
                'rgba(255, 159, 64, 0.7)',   // 厌恶
                'rgba(199, 199, 199, 0.7)'   // 生气
            ],
            borderColor: [
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 99, 132, 1)',
                'rgba(255, 159, 64, 1)',
                'rgba(199, 199, 199, 1)'
            ],
            borderWidth: 1,
            borderRadius: 4
        }]
    };

    // 图表配置
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                grid: {
                    color: 'rgba(0, 0, 0, 0.05)'
                },
                ticks: {
                    callback: value => `${value}%`
                },
                title: {
                    display: true,
                    text: texts.confidenceValue,
                    font: {
                        size: 14,
                        weight: 'normal'
                    }
                }
            },
            x: {
                grid: {
                    display: false
                }
            }
        },
        plugins: {
            legend: {
                display: false
            },
            title: {
                display: true,
                text: texts.confidenceChartTitle,
                padding: {
                    top: 10,
                    bottom: 20
                },
                font: {
                    size: 16,
                    weight: 'bold'
                }
            },
            tooltip: {
                callbacks: {
                    label: function (context) {
                        return `${texts.confidenceLabel}: ${context.parsed.y.toFixed(1)}%`;
                    }
                }
            }
        },
        animation: {
            duration: 500,
            easing: 'easeOutQuart'
        }
    };

    // 创建新图表
    try {
        console.log('Creating new chart with data:', data);
        predictionChart = new Chart(canvas, {
            type: 'bar',
            data: data,
            options: options
        });
    } catch (error) {
        console.error('Error creating chart:', error);
        container.innerHTML = '<p class="chart-error">创建图表时出错</p>';
    }
}

// 显示预测结果
function displayResult(predictions, isRealtime = false, predictionTime = null) {
    latestPredictions = predictions;
    const texts = languageResources[currentLanguage];
    const expressions = texts.expressions;

    const resultElement = isRealtime ?
        document.getElementById('realtimeResult') :
        document.getElementById('result');

    // 清除当前容器的内容
    resultElement.innerHTML = '';
    resultElement.style.display = 'block';

    // 创建表情结果显示区域
    const expressionDiv = document.createElement('div');
    expressionDiv.className = 'expression-result';
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const predictedExpression = expressions[maxIndex] || texts.unableToDetermine;

    if (isRealtime) {
        expressionDiv.innerHTML = `${texts.realtimeExpression} <strong>${predictedExpression}</strong>`;
    } else {
        expressionDiv.innerHTML = `${texts.predictedExpression} <strong>${predictedExpression}</strong>`;

        if (predictionTime !== null) {
            const timeDiv = document.createElement('div');
            timeDiv.className = 'prediction-time';
            timeDiv.textContent = `${texts.predictionTime} ${predictionTime} ${texts.milliseconds}`;
            resultElement.appendChild(timeDiv);
        }
    }
    resultElement.appendChild(expressionDiv);

    // 创建图表容器
    const chartContainer = document.createElement('div');
    chartContainer.className = 'confidence-chart';
    chartContainer.id = isRealtime ? 'realtimeChart' : 'resultChart';
    resultElement.appendChild(chartContainer);

    // 创建特征图容器
    const featureMapContainer = document.createElement('div');
    featureMapContainer.className = 'feature-map-container';
    featureMapContainer.id = isRealtime ? 'realtimeFeatureMaps' : 'imageFeatureMaps';
    resultElement.appendChild(featureMapContainer);

    // 创建图表
    createChart(chartContainer.id, predictions, expressions, texts);
}

// 清除之前的结果
function clearPreviousResults() {
    if (predictionChart) {
        predictionChart.destroy();
        predictionChart = null;
    }

    if (featureMapVisualizer) {
        featureMapVisualizer.stop();
    }

    const resultElements = {
        'result': ['expression-label', 'prediction-time', 'confidenceList', 'imageFeatureMaps'],
        'realtimeResult': ['realtimeExpression', 'realtimeConfidence', 'realtimeFeatureMaps']
    };

    for (const [container, elements] of Object.entries(resultElements)) {
        const containerElement = document.getElementById(container);
        if (containerElement) {
            containerElement.style.display = 'none';
            containerElement.innerHTML = '';
        }
        elements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.innerHTML = '';
            }
        });
    }

    const imagePreview = document.getElementById('imagePreview');
    if (imagePreview) {
        imagePreview.style.display = 'none';
        imagePreview.src = '';
    }

    showError('');
}

// 更新模型状态显示
function updateModelStatus(loaded) {
    const texts = languageResources[currentLanguage];
    const status = loaded ? texts.loaded : texts.notLoaded;
    document.getElementById('modelStatus').innerText = `${texts.modelStatus}${status}`;
}

// 显示/隐藏加载指示器
function toggleLoading(isLoading) {
    document.getElementById('loading').style.display = isLoading ? 'block' : 'none';
}

// 显示错误信息
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.style.display = message ? 'block' : 'none';
    errorDiv.innerText = message;
}

// 初始化文本
function initializeText() {
    const texts = languageResources[currentLanguage];
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.dataset.i18n;
        if (texts[key]) {
            element.innerText = texts[key];
        }
    });
    updateModelStatus(isModelLoaded);
}

// 清理内存
function cleanupMemory() {
    if (networkVisualizer) {
        try {
            networkVisualizer.dispose();
            networkVisualizer = null;
        } catch (e) {
            console.error('Network visualizer cleanup error:', e);
        }
    }

    if (featureMapVisualizer) {
        try {
            featureMapVisualizer.dispose();
            featureMapVisualizer = null;
        } catch (e) {
            console.error('Feature map visualizer cleanup error:', e);
        }
    }

    if (model) {
        try {
            model.dispose();
            model = null;
        } catch (e) {
            console.error('Model cleanup error:', e);
        }
    }

    if (cameraController) {
        try {
            cameraController.stop();
            cameraController = null;
        } catch (e) {
            console.error('Camera cleanup error:', e);
        }
    }

    if (predictionChart) {
        try {
            predictionChart.destroy();
            predictionChart = null;
        } catch (e) {
            console.error('Chart cleanup error:', e);
        }
    }

    try {
        tf.dispose();
    } catch (e) {
        console.error('TensorFlow cleanup error:', e);
    }
}

// 页面加载和初始化
window.addEventListener('load', async () => {
    console.log('TensorFlow.js version:', tf.version.tfjs);
    try {
        initializeText();
        setupEventListeners();
        await loadModel();
    } catch (error) {
        console.error('Initialization failed:', error);
        showError(error.message);
    }
});