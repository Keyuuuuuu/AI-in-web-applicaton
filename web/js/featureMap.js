class FeatureMapVisualizer {
    constructor() {
        this.layers = [];
        this.currentLayerIndex = 0;
        this.animationId = null;
        this.canvas = document.createElement('canvas');
        this.canvas.width = 300;
        this.canvas.height = 300;
        this.ctx = this.canvas.getContext('2d');
        this.container = null;
        this.intermediateModels = [];
        this.isAnimating = false;
        this.isPaused = false;
        this.currentLanguage = 'zh';
        this.featureMaps = null;
        this.lastFrameTime = 0;
        this.frameInterval = 1000;
        this.isVisible = false;
        this.isInitialized = false;
        this.cachedFeatureMaps = null;

        console.log('FeatureMapVisualizer 初始化:', {
            isInitialized: this.isInitialized,
            currentLanguage: this.currentLanguage
        });
    }

    validateFeatureMap(featureMap) {
        try {
            return featureMap &&
                typeof featureMap.layerName === 'string' &&
                Array.isArray(featureMap.features) &&
                featureMap.features.length > 0 &&
                Array.isArray(featureMap.features[0]) &&
                featureMap.features[0].length > 0 &&
                Array.isArray(featureMap.shape) &&
                typeof featureMap.channelCount === 'number';
        } catch (error) {
            console.error('Error in validateFeatureMap:', error);
            return false;
        }
    }

    validateContainer(container) {
        try {
            return container && container instanceof HTMLElement;
        } catch (error) {
            console.error('Error in validateContainer:', error);
            return false;
        }
    }

    async initialize(model) {
        try {
            console.log('Initializing feature map visualizer...');
            // 检查方法初始化前是否存在
            console.log('initialize开始前检查setVisibility方法:',
                typeof this.setVisibility === 'function',
                '原型链上的所有方法:',
                Object.getOwnPropertyNames(Object.getPrototypeOf(this)));

            const resnetLayer = model.layers.find(layer => layer.name === 'resnet50');
            if (!resnetLayer) {
                console.warn('ResNet50 layer not found');
                return false;
            }

            // 定义要提取的层
            const layerConfigs = [
                { name: 'conv1', patterns: ['conv1', 'bn_conv1', 'activation'] },
                { name: 'conv2', patterns: ['conv2_block1', 'conv2_block2', 'conv2_block3'] },
                { name: 'conv3', patterns: ['conv3_block1', 'conv3_block2', 'conv3_block3', 'conv3_block4'] },
                { name: 'conv4', patterns: ['conv4_block1', 'conv4_block3', 'conv4_block6'] },
                { name: 'conv5', patterns: ['conv5_block1', 'conv5_block2', 'conv5_block3'] }
            ];

            this.layers = [];

            // 收集所有层
            for (const config of layerConfigs) {
                for (const pattern of config.patterns) {
                    const layer = resnetLayer.layers.find(l => l.name.includes(pattern));
                    if (layer) {
                        this.layers.push(layer);
                    }
                }
            }

            console.log('Found layers:', this.layers.map(l => l.name));

            // 创建中间模型
            this.intermediateModels = this.layers.map(layer => {
                return tf.model({
                    inputs: resnetLayer.inputs,
                    outputs: layer.output
                });
            });

            // 检查初始化完成后方法是否存在
            console.log('initialize结束前检查setVisibility方法:',
                typeof this.setVisibility === 'function',
                'this对象上的所有属性:',
                Object.getOwnPropertyNames(this),
                'this.__proto__上的所有属性:',
                Object.getOwnPropertyNames(this.__proto__));

            // 如果setVisibility方法不存在，在这里添加
            if (typeof this.setVisibility !== 'function') {
                console.warn('initialize方法结束前，setVisibility方法不存在，正在添加...');
                // 动态添加方法
                this.setVisibility = function (visible) {
                    console.log('调用动态添加的setVisibility方法:', visible);
                    this.isVisible = visible;
                    if (visible) {
                        if (typeof this.resume === 'function') {
                            this.resume();
                        } else {
                            console.warn('resume方法不存在');
                        }
                    } else {
                        if (typeof this.pause === 'function') {
                            this.pause();
                        } else {
                            console.warn('pause方法不存在');
                        }
                    }
                };
                console.log('动态添加setVisibility方法后再次检查:',
                    typeof this.setVisibility === 'function');
            }

            // 初始化完成标志
            this.isInitialized = true;
            return true;
        } catch (error) {
            console.error('Feature map initializer error:', error);
            return false;
        }
    }

    async extractFeatureMaps(inputTensor) {
        try {
            if (!inputTensor) {
                throw new Error('Invalid input tensor');
            }

            const featureMaps = [];

            for (let i = 0; i < this.intermediateModels.length; i++) {
                try {
                    const model = this.intermediateModels[i];
                    const layerName = this.layers[i].name;
                    const output = model.predict(inputTensor);
                    const features = await output.array();
                    const shape = output.shape;

                    if (!features || !features[0]) {
                        console.warn(`Invalid features for layer ${layerName}`);
                        output.dispose();
                        continue;
                    }

                    const channelsToShow = Math.min(4, shape[3]);
                    const featureChannels = [];

                    for (let c = 0; c < channelsToShow; c++) {
                        const channelIndex = Math.floor(c * (shape[3] / channelsToShow));
                        const featureMap = Array.from({ length: shape[1] }, (_, y) =>
                            Array.from({ length: shape[2] }, (_, x) =>
                                features[0][y][x][channelIndex]
                            )
                        );
                        featureChannels.push(featureMap);
                    }

                    const hasValidData = featureChannels.some(channel =>
                        channel.some(row => row.some(val => val !== 0 && !Number.isNaN(val)))
                    );

                    if (!hasValidData) {
                        console.warn(`No valid data found in layer ${layerName}`);
                        output.dispose();
                        continue;
                    }

                    featureMaps.push({
                        layerName: `${layerName} (${shape[3]} channels)`,
                        features: featureChannels[0], // 使用第一个通道作为主要特征图
                        shape: [shape[1], shape[2]],
                        channelCount: shape[3]
                    });

                    output.dispose();
                } catch (layerError) {
                    console.error(`Error processing layer ${i}:`, layerError);
                }
            }

            if (featureMaps.length === 0) {
                throw new Error('No valid feature maps extracted');
            }

            console.log('Processed feature maps:', featureMaps);
            return featureMaps;

        } catch (error) {
            console.error('Error extracting feature maps:', error);
            return null;
        }
    }

    drawFeatureMap(features) {
        try {
            if (!features || !features.length || !features[0] || !features[0].length) {
                console.warn('Invalid features data for drawing');
                return this.canvas;
            }

            const height = features.length;
            const width = features[0].length;
            const maxSize = 300;
            const scale = Math.min(maxSize / width, maxSize / height);
            const displayWidth = Math.floor(width * scale);
            const displayHeight = Math.floor(height * scale);

            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;
            this.ctx.clearRect(0, 0, displayWidth, displayHeight);

            const flatFeatures = features.flat();
            const min = Math.min(...flatFeatures);
            const max = Math.max(...flatFeatures);
            const range = max - min;

            const imageData = this.ctx.createImageData(displayWidth, displayHeight);
            const data = imageData.data;

            for (let y = 0; y < displayHeight; y++) {
                for (let x = 0; x < displayWidth; x++) {
                    const srcX = Math.min(Math.floor(x / scale), width - 1);
                    const srcY = Math.min(Math.floor(y / scale), height - 1);
                    const value = range === 0 ? 0 : (features[srcY][srcX] - min) / range;
                    const index = (y * displayWidth + x) * 4;

                    if (value < 0.25) {
                        data[index] = 0;
                        data[index + 1] = 0;
                        data[index + 2] = Math.floor(value * 4 * 255);
                    } else if (value < 0.5) {
                        const v = (value - 0.25) * 4;
                        data[index] = 0;
                        data[index + 1] = Math.floor(v * 255);
                        data[index + 2] = Math.floor((1 - v) * 255);
                    } else if (value < 0.75) {
                        const v = (value - 0.5) * 4;
                        data[index] = Math.floor(v * 255);
                        data[index + 1] = 255;
                        data[index + 2] = 0;
                    } else {
                        const v = (value - 0.75) * 4;
                        data[index] = 255;
                        data[index + 1] = Math.floor((1 - v) * 255);
                        data[index + 2] = 0;
                    }
                    data[index + 3] = 255;
                }
            }

            this.ctx.putImageData(imageData, 0, 0);
            this.ctx.strokeStyle = '#666';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(0, 0, displayWidth, displayHeight);

            return this.canvas;
        } catch (error) {
            console.error('Error in drawFeatureMap:', error);
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            return this.canvas;
        }
    }

    drawCurrentFeatureMap(featureMap) {
        try {
            console.log('开始绘制特征图:', {
                layerName: featureMap.layerName,
                featuresValid: !!featureMap.features && featureMap.features.length > 0
            });

            // 基础验证
            if (!this.validateFeatureMap(featureMap) || !this.validateContainer(this.container)) {
                console.warn('无效的特征图或容器');
                return;
            }

            // 1. 确保容器结构正确
            let header = this.container.querySelector('.feature-map-header');
            let layerInfo = this.container.querySelector('.layer-info');
            let controlsContainer = this.container.querySelector('.feature-map-controls');
            let canvasContainer = this.container.querySelector('.feature-map-canvas');
            let legendContainer = this.container.querySelector('.feature-map-legend');

            // 如果容器结构不存在，创建完整结构
            if (!header) {
                // 清空容器
                this.container.innerHTML = '';

                // 创建header
                header = document.createElement('div');
                header.className = 'feature-map-header';

                const title = document.createElement('h3');
                title.className = 'feature-map-title';
                title.textContent = this.currentLanguage === 'zh' ? '神经网络特征图可视化' : 'Neural Network Feature Map Visualization';

                header.appendChild(title);
                this.container.appendChild(header);

                // 创建层信息
                layerInfo = document.createElement('div');
                layerInfo.className = 'layer-info fade-in';
                this.container.appendChild(layerInfo);

                // 创建控制容器
                controlsContainer = document.createElement('div');
                controlsContainer.className = 'feature-map-controls';
                this.container.appendChild(controlsContainer);

                // 创建画布容器
                canvasContainer = document.createElement('div');
                canvasContainer.className = 'feature-map-canvas';
                this.container.appendChild(canvasContainer);

                // 创建图例
                legendContainer = document.createElement('div');
                legendContainer.className = 'feature-map-legend';

                // 创建图例内容
                const legendItems = [
                    { color: '#0000FF', text: this.currentLanguage === 'zh' ? '低活跃度' : 'Low Activation' },
                    { color: '#00FFFF', text: this.currentLanguage === 'zh' ? '中低活跃度' : 'Medium-Low Activation' },
                    { color: '#FFFF00', text: this.currentLanguage === 'zh' ? '中高活跃度' : 'Medium-High Activation' },
                    { color: '#FF0000', text: this.currentLanguage === 'zh' ? '高活跃度' : 'High Activation' }
                ];

                legendItems.forEach(item => {
                    const legendItem = document.createElement('div');
                    legendItem.className = 'legend-item';

                    const colorBox = document.createElement('div');
                    colorBox.className = 'legend-color';
                    colorBox.style.backgroundColor = item.color;

                    const text = document.createElement('span');
                    text.textContent = item.text;

                    legendItem.appendChild(colorBox);
                    legendItem.appendChild(text);
                    legendContainer.appendChild(legendItem);
                });

                this.container.appendChild(legendContainer);

                console.log('创建了完整的特征图容器结构');
            }

            // 2. 更新层信息
            if (layerInfo) {
                layerInfo.textContent = featureMap.layerName;
                layerInfo.classList.add('fade-in');
                setTimeout(() => layerInfo.classList.remove('fade-in'), 500);
            }

            // 3. 更新控制容器
            if (controlsContainer) {
                // 只有在控制容器为空时创建按钮
                if (controlsContainer.children.length === 0) {
                    // 创建暂停/播放按钮
                    const playPauseButton = document.createElement('button');
                    playPauseButton.id = 'playPauseButton';
                    playPauseButton.className = 'control-button';
                    playPauseButton.textContent = this.isPaused ?
                        (this.currentLanguage === 'zh' ? '播放' : 'Play') :
                        (this.currentLanguage === 'zh' ? '暂停' : 'Pause');

                    // 添加事件监听器
                    playPauseButton.addEventListener('click', () => {
                        this.toggleAnimation();
                        playPauseButton.textContent = this.isPaused ?
                            (this.currentLanguage === 'zh' ? '播放' : 'Play') :
                            (this.currentLanguage === 'zh' ? '暂停' : 'Pause');
                    });

                    controlsContainer.appendChild(playPauseButton);

                    // 创建速度控制
                    const speedControl = document.createElement('select');
                    speedControl.className = 'control-button';

                    const options = [
                        { value: 2000, text: this.currentLanguage === 'zh' ? '慢速' : 'Slow' },
                        { value: 1000, text: this.currentLanguage === 'zh' ? '中速' : 'Medium' },
                        { value: 500, text: this.currentLanguage === 'zh' ? '快速' : 'Fast' }
                    ];

                    options.forEach(opt => {
                        const option = document.createElement('option');
                        option.value = opt.value;
                        option.textContent = opt.text;
                        if (parseInt(opt.value) === this.frameInterval) {
                            option.selected = true;
                        }
                        speedControl.appendChild(option);
                    });

                    speedControl.addEventListener('change', (e) => {
                        this.frameInterval = parseInt(e.target.value);
                        console.log('帧间隔更新为:', this.frameInterval);
                    });

                    controlsContainer.appendChild(speedControl);
                }
            }

            // 4. 更新画布容器
            if (canvasContainer) {
                // 清空画布容器
                canvasContainer.innerHTML = '';

                // 创建新画布
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');

                // 添加动画类
                canvas.className = 'animating';

                // 设置初始尺寸
                canvas.width = 300;
                canvas.height = 300;

                // 获取特征图数据
                const features = featureMap.features;
                if (!features || !features.length || !features[0] || !features[0].length) {
                    console.warn('无效的特征数据');
                    return;
                }

                // 计算显示尺寸
                const height = features.length;
                const width = features[0].length;
                const maxSize = 300;
                const scale = Math.min(maxSize / width, maxSize / height);
                const displayWidth = Math.floor(width * scale);
                const displayHeight = Math.floor(height * scale);

                // 设置画布尺寸
                canvas.width = displayWidth;
                canvas.height = displayHeight;

                // 清除画布
                ctx.clearRect(0, 0, displayWidth, displayHeight);

                // 计算特征图值范围
                const flatFeatures = features.flat();
                const min = Math.min(...flatFeatures);
                const max = Math.max(...flatFeatures);
                const range = max - min;

                console.log('特征图数据范围:', { min, max, range });

                // 创建ImageData
                const imageData = ctx.createImageData(displayWidth, displayHeight);
                const data = imageData.data;

                // 填充像素数据
                for (let y = 0; y < displayHeight; y++) {
                    for (let x = 0; x < displayWidth; x++) {
                        const srcX = Math.min(Math.floor(x / scale), width - 1);
                        const srcY = Math.min(Math.floor(y / scale), height - 1);
                        const value = range === 0 ? 0 : (features[srcY][srcX] - min) / range;
                        const index = (y * displayWidth + x) * 4;

                        // 热力图着色
                        if (value < 0.25) {
                            data[index] = 0;
                            data[index + 1] = 0;
                            data[index + 2] = Math.floor(value * 4 * 255);
                        } else if (value < 0.5) {
                            const v = (value - 0.25) * 4;
                            data[index] = 0;
                            data[index + 1] = Math.floor(v * 255);
                            data[index + 2] = Math.floor((1 - v) * 255);
                        } else if (value < 0.75) {
                            const v = (value - 0.5) * 4;
                            data[index] = Math.floor(v * 255);
                            data[index + 1] = 255;
                            data[index + 2] = 0;
                        } else {
                            const v = (value - 0.75) * 4;
                            data[index] = 255;
                            data[index + 1] = Math.floor((1 - v) * 255);
                            data[index + 2] = 0;
                        }

                        // 设置alpha通道
                        data[index + 3] = 255;
                    }
                }

                // 将数据绘制到画布
                ctx.putImageData(imageData, 0, 0);

                // 添加边框
                ctx.strokeStyle = '#666';
                ctx.lineWidth = 1;
                ctx.strokeRect(0, 0, displayWidth, displayHeight);

                // 将画布添加到容器
                canvasContainer.appendChild(canvas);

                // 添加Debug信息
                const debugInfo = document.createElement('div');
                debugInfo.style.cssText = `
                font-size: 11px;
                color: #999;
                margin-top: 10px;
                text-align: center;
            `;
                debugInfo.textContent = `${featureMap.layerName} (${displayWidth}x${displayHeight})`;
                canvasContainer.appendChild(debugInfo);

                // 移除动画类，300ms后
                setTimeout(() => {
                    canvas.classList.remove('animating');
                }, 300);

                console.log('Canvas已添加到DOM:', {
                    canvasSize: `${displayWidth}x${displayHeight}`,
                    added: canvasContainer.contains(canvas)
                });
            }

        } catch (error) {
            console.error('特征图绘制错误:', error);

            // 显示错误信息
            if (this.container) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'feature-map-error';
                errorDiv.textContent = `特征图渲染错误: ${error.message}`;
                this.container.appendChild(errorDiv);
            }
        }
    }

    startAnimation() {
        try {
            if (!this.featureMaps || this.featureMaps.length === 0) {
                console.warn('No feature maps available for animation');
                return;
            }

            console.log('Starting animation with:', {
                featureMapsLength: this.featureMaps.length,
                containerExists: !!this.container,
                containerSize: {
                    width: this.container?.clientWidth,
                    height: this.container?.clientHeight
                }
            });

            // Make sure container is prepared
            if (this.container) {
                this.container.style.position = 'relative';
                this.container.style.minHeight = '300px';
                this.container.style.width = '100%';
                this.container.style.background = '#f8f9fa';
                this.container.style.borderRadius = '8px';
                this.container.style.overflow = 'hidden';
            }

            this.isAnimating = true;
            this.isPaused = false;
            this.currentLayerIndex = 0;
            this.lastFrameTime = 0;

            // Start animation loop
            requestAnimationFrame((t) => this.animateFrame(t));

            // If there's a play/pause button, update its text
            const playPauseButton = this.container?.querySelector('#playPauseButton');
            if (playPauseButton) {
                playPauseButton.textContent = this.currentLanguage === 'zh' ? '暂停' : 'Pause';
            }
        } catch (error) {
            console.error('Error in startAnimation:', error);
        }
    }

    animateFrame(timestamp) {
        try {
            // Only continue if animation is active and not paused
            if (!this.isAnimating || this.isPaused) return;

            // Setup for timing
            if (!this.lastFrameTime) this.lastFrameTime = timestamp;
            const elapsed = timestamp - this.lastFrameTime;

            if (elapsed >= this.frameInterval) {
                // Get the current feature map to display
                const currentFeatures = this.featureMaps[this.currentLayerIndex];

                console.log('Animating feature map:', {
                    index: this.currentLayerIndex,
                    layerName: currentFeatures?.layerName,
                    containerVisible: this.isVisible
                });

                // Validate and draw the feature map
                if (this.validateFeatureMap(currentFeatures) && this.isVisible) {
                    this.drawCurrentFeatureMap(currentFeatures);
                }

                // Move to next feature map
                this.currentLayerIndex = (this.currentLayerIndex + 1) % this.featureMaps.length;
                this.lastFrameTime = timestamp;
            }

            // Continue animation loop
            requestAnimationFrame((t) => this.animateFrame(t));
        } catch (error) {
            console.error('Error in animateFrame:', error);
            this.stop();
        }
    }

    async animateFeatureMaps(featureMaps, container) {
        console.log('Starting feature map animation:', {
            mapsCount: featureMaps?.length,
            containerExists: !!container,
            containerID: container?.id
        });

        try {
            // Validate inputs
            if (!Array.isArray(featureMaps) || !featureMaps.length) {
                console.warn('Invalid feature maps array');
                return;
            }

            if (!this.validateContainer(container)) {
                console.warn('Invalid container element');
                return;
            }

            // Cache feature maps and set container
            this.cachedFeatureMaps = featureMaps;
            this.container = container;

            // Clear the container before adding new elements
            while (container.firstChild) {
                container.removeChild(container.firstChild);
            }

            // Create heading for feature maps
            const heading = document.createElement('h3');
            heading.textContent = this.currentLanguage === 'zh' ? '神经网络特征图可视化' : 'Neural Network Feature Map Visualization';
            heading.style.cssText = `
            text-align: center;
            margin: 10px 0;
            padding: 5px;
            font-size: 16px;
        `;
            container.appendChild(heading);

            // Ensure visibility and start animation
            this.isVisible = true;
            this.featureMaps = featureMaps;
            this.stop(); // Stop any previous animation
            this.startAnimation();

            console.log('Feature map animation started successfully');

            return true;
        } catch (error) {
            console.error('Error in animateFeatureMaps:', error);
            return false;
        }
    }



    // 1. 修改 pause() 方法，移除导致特征图消失的代码
    pause() {
        // 只设置暂停状态，不改变可见性或清除画布
        this.isPaused = true;

        // 更新按钮文本
        const playPauseButton = this.container?.querySelector('#playPauseButton');
        if (playPauseButton) {
            playPauseButton.textContent = this.currentLanguage === 'zh' ? '播放' : 'Play';
        }
    }

    // 2. 改进 resume() 方法，确保恢复动画时一切正常
    resume() {
        if (this.cachedFeatureMaps && this.container) {
            this.isVisible = true;
            this.featureMaps = this.cachedFeatureMaps;
            this.isPaused = false;
            this.lastFrameTime = 0;

            // 更新按钮文本
            const playPauseButton = this.container?.querySelector('#playPauseButton');
            if (playPauseButton) {
                playPauseButton.textContent = this.currentLanguage === 'zh' ? '暂停' : 'Pause';
            }

            // 启动动画
            requestAnimationFrame((t) => this.animateFrame(t));
        }
    }


    // 新增 toggleAnimation 方法，用于播放/暂停切换
    toggleAnimation() {
        if (this.isPaused) {
            this.resume();
            // Update button text
            const playPauseButton = this.container?.querySelector('#playPauseButton');
            if (playPauseButton) {
                playPauseButton.textContent = this.currentLanguage === 'zh' ? '暂停' : 'Pause';
            }
        } else {
            this.pause();
            // Update button text
            const playPauseButton = this.container?.querySelector('#playPauseButton');
            if (playPauseButton) {
                playPauseButton.textContent = this.currentLanguage === 'zh' ? '播放' : 'Play';
            }
        }
    }

    stop() {
        try {
            this.isAnimating = false;
            this.isPaused = false;
            this.currentLayerIndex = 0;
            this.lastFrameTime = 0;

            // 仅清除特征图画布，保留控制元素
            const canvas = this.container?.querySelector('.feature-map-canvas');
            if (canvas) {
                canvas.innerHTML = '';
            }
        } catch (error) {
            console.error('Error in stop:', error);
        }
    }

    dispose() {
        try {
            this.stop();
            this.cachedFeatureMaps = null;
            this.isVisible = false;
            if (this.intermediateModels) {
                this.intermediateModels.forEach(model => {
                    try {
                        model.dispose();
                    } catch (error) {
                        console.warn('Error disposing model:', error);
                    }
                });
            }
            this.intermediateModels = [];
            this.layers = [];
            this.container = null;
            this.featureMaps = null;

            if (this.ctx) {
                try {
                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                } catch (error) {
                    console.warn('Error clearing canvas:', error);
                }
            }
        } catch (error) {
            console.error('Error in dispose:', error);
        }
    }

    setVisibility(visible) {
        this.isVisible = visible;
        if (visible) {
            this.resume();
        } else {
            this.pause();
        }
    }

    isFeatureMapVisible() {
        return this.isVisible;
    }
}

export default FeatureMapVisualizer;
