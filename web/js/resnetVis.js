// resnetVis.js
class ResNetVisualizer {
    // resnetVis.js 开头的构造函数中
    constructor(languageResources) {  // 添加参数
        // 基础属性
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.labelRenderer = null;
        this.composer = null;

        // 容器和模型
        this.container = null;
        this.model = null;

        // 数据存储
        this.nodes = new Map();
        this.connections = new Map();

        // 交互属性
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.selectedNode = null;

        // 状态控制
        this.isInitialized = false;

        // 添加新的状态管理属性
        this.hoveredNode = null;
        this.selectedNode = null;
        this.detailPanel = null;
        this.hoveredLabel = null;

        // 语言相关
        this.languageResources = languageResources;  // 保存传入的语言资源
        this.currentLanguage = 'en';  // 默认英文
    }


    // 语言切换方法
    setLanguage(lang) {
        if (this.languageResources[lang]) {
            this.currentLanguage = lang;
            // 如果当前有选中的节点，更新信息面板
            if (this.selectedNode) {
                this.showDetailPanel(this.selectedNode.userData);
            }
        }
    }


    // 在 ResNetVisualizer 类中修改 initialize 方法
    async initialize(container, modelPath) {
        if (!container) {
            throw new Error('Container not provided');
        }

        try {
            console.log('Starting initialization process...');

            // 1. 容器设置（保留）
            this.container = container;
            this.container.style.position = 'relative';
            this.container.style.overflow = 'hidden';

            this.updateInitializationStatus('正在初始化基础组件...');

            // 2. 初始化场景（保留）
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0xffffff);
            if (!this.scene) {
                throw new Error('Scene initialization failed');
            }

            // 3. 初始化材质（移动到这里）
            this.updateInitializationStatus('正在初始化材质系统...');
            await this.initMaterials();

            // 4. 初始化场景和光照（移动到这里）
            this.updateInitializationStatus('正在设置场景光照...');
            await this.initScene();

            // 5. 相机设置（保留优化）
            this.updateInitializationStatus('正在设置相机...');
            const width = container.clientWidth;
            const height = container.clientHeight;
            const aspect = width / height;

            this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 10000);
            const distance = 2700;
            this.camera.position.set(
                -distance * 0.8,
                distance * 0.3,
                distance * 0.8
            );
            this.camera.lookAt(0, 0, 0);

            // 6. 渲染器初始化（保留优化）
            this.updateInitializationStatus('正在初始化渲染器...');
            this.renderer = new THREE.WebGLRenderer({
                antialias: true,
                alpha: false,
                preserveDrawingBuffer: true
            });

            // 渲染器设置（保留）
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.setSize(width, height);
            this.renderer.setClearColor(0xffffff, 1);

            // 渲染器DOM设置（保留）
            this.renderer.domElement.style.position = 'absolute';
            this.renderer.domElement.style.top = '0';
            this.renderer.domElement.style.left = '0';
            this.renderer.domElement.style.width = '100%';
            this.renderer.domElement.style.height = '100%';
            container.appendChild(this.renderer.domElement);

            // 7. 标签渲染器（保留优化）
            this.updateInitializationStatus('正在初始化标签系统...');
            this.labelRenderer = new THREE.CSS2DRenderer();
            this.labelRenderer.setSize(width, height);
            this.labelRenderer.domElement.style.position = 'absolute';
            this.labelRenderer.domElement.style.top = '0';
            this.labelRenderer.domElement.style.left = '0';
            this.labelRenderer.domElement.style.width = '100%';
            this.labelRenderer.domElement.style.height = '100%';
            this.labelRenderer.domElement.style.pointerEvents = 'none';
            container.appendChild(this.labelRenderer.domElement);

            // 8. 控制器设置（保留优化）
            this.updateInitializationStatus('正在设置控制器...');
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.minDistance = 500;
            this.controls.maxDistance = 5000;
            this.controls.target.set(0, 0, 0);
            this.controls.update();

            // 9. 详情面板初始化（移动到这里）
            await this.initializeDetailPanel();

            // 10. 加载模型数据
            this.updateInitializationStatus('正在加载模型数据...');
            const response = await fetch(modelPath);
            if (!response.ok) {
                throw new Error(`Failed to load model JSON: ${response.statusText}`);
            }
            const modelJson = await response.json();
            const networkData = await this.parseModelJson(modelJson);

            // 11. 事件监听和可视化创建
            this.addEventListeners();
            await this.createNetworkVisualization(networkData);

            // 12. 启动渲染循环
            this.isInitialized = true;
            this.animate();
            this.render();

            this.removeInitializationStatus();
            console.log('Initialization completed successfully');

            return true;
        } catch (error) {
            console.error('Initialization error:', error);
            this.handleInitializationError(error);
            throw error;
        }
    }

    async initBasicComponents() {
        try {
            // 初始化场景
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0xffffff);

            // 初始化相机
            const aspect = this.container.clientWidth / this.container.clientHeight;
            this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 10000);
            this.camera.position.set(-800, 400, 800);
            this.camera.lookAt(0, 0, 0);

            // 初始化渲染器
            this.renderer = new THREE.WebGLRenderer({
                antialias: true,
                alpha: true
            });
            this.renderer.setPixelRatio(window.devicePixelRatio);
            this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
            this.container.appendChild(this.renderer.domElement);

            // 初始化控制器
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.update();

            // 初始化标签渲染器
            this.labelRenderer = new THREE.CSS2DRenderer();
            this.labelRenderer.setSize(this.container.clientWidth, this.container.clientHeight);
            this.labelRenderer.domElement.style.position = 'absolute';
            this.labelRenderer.domElement.style.top = '0';
            this.labelRenderer.domElement.style.pointerEvents = 'none';
            this.container.appendChild(this.labelRenderer.domElement);

            return true;
        } catch (error) {
            console.error('Failed to initialize basic components:', error);
            throw error;
        }
    }

    updateInitializationStatus(message) {
        // 先移除现有的状态显示
        this.removeInitializationStatus();

        // 创建新的状态显示
        const statusElement = document.createElement('div');
        statusElement.className = 'initialization-status';
        statusElement.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        z-index: 1000;
    `;
        statusElement.textContent = message;
        this.container.appendChild(statusElement);
    }

    removeInitializationStatus() {
        const existingStatus = this.container.querySelector('.initialization-status');
        if (existingStatus) {
            existingStatus.remove();
        }
    }

    // 添加一个专门处理初始化错误的方法
    handleInitializationError(error) {
        this.removeInitializationStatus();

        const errorMessage = document.createElement('div');
        errorMessage.className = 'visualization-error';
        errorMessage.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        z-index: 1000;
    `;

        errorMessage.innerHTML = `
        <h3 style="color: #e74c3c; margin: 0 0 10px 0">初始化错误</h3>
        <p style="margin: 0 0 15px 0">${error.message}</p>
        <button onclick="location.reload()" style="
            padding: 8px 16px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        ">重试</button>
    `;

        this.container.appendChild(errorMessage);
    }

    // 1. 修改场景初始化方法，增强光照
    initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff); // 保持白色背景

        // 降低环境光强度
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);  // 从0.4降至0.3
        this.scene.add(ambientLight);

        // 调整平行光
        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.6);  // 从0.8降至0.6
        directionalLight1.position.set(1, 1, 1);
        this.scene.add(directionalLight1);

        // 调整补充光源，保持柔和
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);  // 从0.4降至0.3
        directionalLight2.position.set(-1, -1, -1);
        this.scene.add(directionalLight2);

        // 减弱点光源
        const pointLight = new THREE.PointLight(0xffffff, 0.3);  // 从0.4降至0.3
        pointLight.position.set(0, 200, 200);
        this.scene.add(pointLight);
    }

    // // 2. 修改相机设置
    // initCamera() {
    //     const aspect = window.innerWidth / window.innerHeight;
    //     this.camera = new THREE.PerspectiveCamera(
    //         60,                // 保持视场角不变
    //         aspect,
    //         0.1,              // 保持近裁剪面
    //         20000             // 保持远裁剪面
    //     );

    //     // 将z轴距离从5000减小到1500，让视角更近
    //     this.camera.position.set(0, 0, 5000);
    //     this.camera.lookAt(0, 0, 0);
    // }

    // 1. 修改渲染器初始化
    initRenderer() {
        if (this.renderer) this.renderer.domElement.remove();

        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true, // 启用透明
            preserveDrawingBuffer: true // 保留绘图缓冲
        });

        const container = this.container;
        const width = container.clientWidth;
        const height = container.clientHeight;

        // 设置渲染器尺寸和像素比
        this.renderer.setSize(width, height, false); // false参数防止自动设置canvas样式
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // 限制最大像素比
        this.renderer.setClearColor(0xffffff, 1); // 设置背景色和透明度

        // 确保容器样式正确
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';

        // 添加到DOM
        this.container.appendChild(this.renderer.domElement);

        // 更新标签渲染器
        if (this.labelRenderer) this.labelRenderer.domElement.remove();
        this.labelRenderer = new THREE.CSS2DRenderer();
        this.labelRenderer.setSize(width, height);
        this.labelRenderer.domElement.style.position = 'absolute';
        this.labelRenderer.domElement.style.top = '0';
        this.labelRenderer.domElement.style.left = '0';
        this.labelRenderer.domElement.style.pointerEvents = 'none';
        this.container.appendChild(this.labelRenderer.domElement);
    }

    // 2. 修改相机初始化
    // initCamera() {
    //     const container = this.container;
    //     const aspect = container.clientWidth / container.clientHeight;

    //     this.camera = new THREE.PerspectiveCamera(
    //         45, // 降低FOV以获得更好的视野
    //         aspect,
    //         1,
    //         10000
    //     );

    //     // 调整相机位置以获得更好的视角
    //     const distance = 2000;
    //     this.camera.position.set(
    //         -distance * 0.7,
    //         distance * 0.4,
    //         distance * 0.7
    //     );
    //     this.camera.lookAt(0, 0, 0);
    // }

    // 3. 添加窗口大小变化处理
    handleResize() {
        if (!this.container || !this.camera || !this.renderer) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        // 更新相机
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        // 更新渲染器
        this.renderer.setSize(width, height, false);
        this.labelRenderer.setSize(width, height);

        // 强制重新渲染
        this.renderer.render(this.scene, this.camera);
        this.labelRenderer.render(this.scene, this.camera);
    }


    // 6. 添加辅助线以便于调试
    addHelpers() {
        // 添加网格辅助线
        const gridHelper = new THREE.GridHelper(1000, 20, 0xcccccc, 0xcccccc);
        gridHelper.position.y = -100;
        this.scene.add(gridHelper);

        // 添加坐标轴辅助线
        const axesHelper = new THREE.AxesHelper(500);
        this.scene.add(axesHelper);
    }

    // initLights() {
    //     const ambientLight = new THREE.AmbientLight(0x404040);
    //     this.scene.add(ambientLight);

    //     const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    //     directionalLight.position.set(1, 1, 1);
    //     this.scene.add(directionalLight);

    //     const pointLight = new THREE.PointLight(0xffffff, 0.5);
    //     pointLight.position.set(0, 200, 200);
    //     this.scene.add(pointLight);
    // }

    toggleInfo() {
        const networkInfo = document.querySelector('.network-info');
        const toggleInfoBtn = document.getElementById('toggleInfo');

        if (!networkInfo || !toggleInfoBtn) {
            console.warn('Network info panel or button not found');
            return;
        }

        const isCurrentlyVisible = networkInfo.style.display === 'block';

        // 切换显示状态
        networkInfo.style.display = isCurrentlyVisible ? 'none' : 'block';

        // 更新按钮状态
        if (isCurrentlyVisible) {
            toggleInfoBtn.classList.remove('active');
            toggleInfoBtn.setAttribute('aria-pressed', 'false');
        } else {
            toggleInfoBtn.classList.add('active');
            toggleInfoBtn.setAttribute('aria-pressed', 'true');
        }

        console.log('Info panel visibility:', !isCurrentlyVisible);
    }

    // 3. 修改 initControls 方法中的事件绑定部分
    initControls() {
        if (!this.camera || !this.renderer) {
            console.error('Camera or renderer not initialized');
            return;
        }

        // 确保创建新的控制器之前，先处理现有的控制器
        if (this.controls) {
            this.controls.dispose();
        }

        // 创建并配置控制器
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);

        // 设置控制器的基本参数
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.enableZoom = true;
        this.controls.enableRotate = true;
        this.controls.enablePan = true;

        // 设置缩放限制
        this.controls.minDistance = 500;
        this.controls.maxDistance = 5000;

        // 更新控制器
        this.controls.update();
    }

    // 2. 创建简略标签和详细信息面板
    initializeDetailPanel() {
        this.detailPanel = document.querySelector('.network-info');
        if (!this.detailPanel) {
            console.warn('Network info panel not found in DOM');
            return;
        }

        // 初始化关闭按钮事件
        const closeButton = this.detailPanel.querySelector('.close-button');
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                this.detailPanel.classList.remove('visible');
            });
        }
    }

    setRenderQuality(quality) {
        if (!this.renderer || !this.container) return;

        switch (quality) {
            case 'high':
                this.renderer.setPixelRatio(window.devicePixelRatio);
                this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
                break;
            case 'medium':
                this.renderer.setPixelRatio(1);
                this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
                break;
            case 'low':
                this.renderer.setPixelRatio(0.75);
                this.renderer.setSize(
                    this.container.clientWidth * 0.75,
                    this.container.clientHeight * 0.75,
                    true
                );
                break;
        }
    }


    initLabelRenderer() {
        this.labelRenderer = new THREE.CSS2DRenderer();
        this.labelRenderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.labelRenderer.domElement.style.position = 'absolute';
        this.labelRenderer.domElement.style.top = '0';
        this.labelRenderer.domElement.style.pointerEvents = 'none';
        this.container.appendChild(this.labelRenderer.domElement);
    }



    initMaterials() {
        this.materials = {
            node: {
                // 卷积层：使用渐变蓝色，专业且富有层次
                conv2d: {
                    early: new THREE.MeshPhongMaterial({
                        color: 0x40A9FF,      // 亮蓝色
                        shininess: 65,
                        specular: 0x40A9FF,
                        transparent: true,
                        opacity: 0.9
                    }),
                    middle: new THREE.MeshPhongMaterial({
                        color: 0x1890FF,      // 中蓝色
                        shininess: 65,
                        specular: 0x1890FF,
                        transparent: true,
                        opacity: 0.9
                    }),
                    deep: new THREE.MeshPhongMaterial({
                        color: 0x0050B3,      // 深蓝色
                        shininess: 65,
                        specular: 0x0050B3,
                        transparent: true,
                        opacity: 0.9
                    })
                },
                // 批归一化层：使用清新的绿色
                batchNorm: new THREE.MeshPhongMaterial({
                    color: 0x52C41A,
                    shininess: 60,
                    specular: 0x52C41A,
                    transparent: true,
                    opacity: 0.85
                }),
                // 池化层：使用优雅的紫色
                pooling: new THREE.MeshPhongMaterial({
                    color: 0x722ED1,
                    shininess: 60,
                    specular: 0x722ED1,
                    transparent: true,
                    opacity: 0.85
                }),
                // Add层：使用醒目的琥珀色
                add: new THREE.MeshPhongMaterial({
                    color: 0xFAAD14,
                    shininess: 70,
                    specular: 0xFAAD14,
                    transparent: true,
                    opacity: 0.9
                })
            },
            connection: {
                // 前向连接：深色但不失细节
                forward: new THREE.LineBasicMaterial({
                    color: 0x595959,
                    linewidth: 2,
                    transparent: true,
                    opacity: 0.7
                }),
                // 残差连接：金色，呼应Add层
                residual: new THREE.LineDashedMaterial({
                    color: 0xFAAD14,
                    dashSize: 5,
                    gapSize: 3,
                    linewidth: 2,
                    transparent: true,
                    opacity: 0.8
                })
            }
        };

        // 增强的材质检查
        try {
            // 检查卷积层材质
            console.log('Conv2D materials check:', {
                early: this.materials.node.conv2d.early instanceof THREE.Material,
                middle: this.materials.node.conv2d.middle instanceof THREE.Material,
                deep: this.materials.node.conv2d.deep instanceof THREE.Material
            });

            // 检查其他节点材质
            console.log('Other materials check:', {
                batchNorm: this.materials.node.batchNorm instanceof THREE.Material,
                pooling: this.materials.node.pooling instanceof THREE.Material,
                add: this.materials.node.add instanceof THREE.Material
            });

            // 检查连接材质
            console.log('Connection materials check:', {
                forward: this.materials.connection.forward instanceof THREE.Material,
                residual: this.materials.connection.residual instanceof THREE.Material
            });

            // 验证材质克隆功能
            const testClone = {
                conv2dEarly: this.materials.node.conv2d.early.clone() instanceof THREE.Material,
                batchNorm: this.materials.node.batchNorm.clone() instanceof THREE.Material,
                add: this.materials.node.add.clone() instanceof THREE.Material,
                forward: this.materials.connection.forward.clone() instanceof THREE.Material,
                residual: this.materials.connection.residual.clone() instanceof THREE.Material
            };
            console.log('Clone functionality check:', testClone);

        } catch (error) {
            console.error('Error during materials initialization:', error);
            throw new Error('Materials initialization failed');
        }

        return true;
    }

    parseModelJson(modelJson) {
        try {
            const resnetLayers = modelJson.modelTopology.model_config.config.layers[1].config.layers;

            if (!resnetLayers) {
                console.error('Model structure:', modelJson);
                throw new Error('Could not find ResNet layers in model JSON');
            }

            console.log('\n==== Starting Model Parsing ====');
            console.log('Total layers found:', resnetLayers.length);

            const nodes = [];
            const links = [];
            const layerMap = new Map();

            resnetLayers.forEach((layer, index) => {
                console.log('\n==== Processing Layer ====');
                console.log('Layer Info:', {
                    name: layer.name,
                    class: layer.class_name,
                    index: index,
                    inboundNodes: layer.inbound_nodes
                });

                // 验证层配置
                if (!layer.config) {
                    console.warn(`Missing config for layer: ${layer.name}`);
                }

                // 只跳过真正的辅助层
                if (['InputLayer', 'Activation'].includes(layer.class_name)) {
                    console.log('Skipping helper layer:', layer.name);
                    return;
                }

                // 创建基础节点结构
                const node = {
                    id: index,
                    name: layer.name,
                    className: layer.class_name,
                    inbound: layer.inbound_nodes?.[0] || [],
                    outbound: [],

                    // 基础配置
                    config: {
                        // 卷积层配置
                        filters: layer.config?.filters,
                        kernelSize: layer.config?.kernel_size,
                        strides: layer.config?.strides,
                        padding: layer.config?.padding,

                        // 常用配置
                        activation: layer.config?.activation,
                        useBias: layer.config?.use_bias,

                        // 密集层配置
                        units: layer.config?.units,

                        // 池化层配置
                        poolSize: layer.config?.pool_size,
                        poolStrides: layer.config?.pool_strides,

                        // 形状信息初始化为 null
                        inputShape: null,
                        outputShape: null
                    }
                };

                console.log('Created node structure:', {
                    name: node.name,
                    config: node.config
                });

                // 添加层特定的额外属性
                switch (layer.class_name) {
                    case 'Conv2D':
                        node.layerType = 'convolution';
                        node.visualConfig = {
                            filters: layer.config?.filters,
                            kernelSize: layer.config?.kernel_size
                        };
                        break;

                    case 'MaxPooling2D':
                        node.layerType = 'pooling';
                        node.visualConfig = {
                            poolSize: layer.config?.pool_size
                        };
                        break;

                    case 'Dense':
                        node.layerType = 'dense';
                        node.visualConfig = {
                            units: layer.config?.units
                        };
                        break;

                    case 'BatchNormalization':
                        node.layerType = 'normalization';
                        node.visualConfig = {
                            momentum: layer.config?.momentum,
                            epsilon: layer.config?.epsilon
                        };
                        break;

                    case 'Add':
                        node.layerType = 'add';
                        node.visualConfig = {};
                        break;

                    default:
                        node.layerType = 'general';
                        break;
                }

                // 形状计算验证
                console.log('\n---- Shape Calculation ----');
                console.log(`Starting shape calculation for: ${node.name}`);

                // 计算输入形状
                console.log('1. Input Shape Calculation');
                const inputShape = this.calculateInputShape(layer, nodes);
                console.log('Input Shape Result:', {
                    layer: node.name,
                    calculatedShape: inputShape,
                    isArray: Array.isArray(inputShape)
                });
                node.config.inputShape = inputShape;

                // 验证输入形状
                if (!Array.isArray(inputShape)) {
                    console.warn(`Invalid input shape for ${node.name}:`, inputShape);
                }

                // 计算输出形状
                console.log('2. Output Shape Calculation');
                const outputShape = this.calculateOutputShape(layer, inputShape);
                console.log('Output Shape Result:', {
                    layer: node.name,
                    calculatedShape: outputShape,
                    isArray: Array.isArray(outputShape)
                });
                node.config.outputShape = outputShape;

                // 验证输出形状
                if (!Array.isArray(outputShape)) {
                    console.warn(`Invalid output shape for ${node.name}:`, outputShape);
                }

                // 最终形状验证
                console.log('\nShape Verification:', {
                    layer: node.name,
                    inputShape: node.config.inputShape,
                    outputShape: node.config.outputShape,
                    inputValid: Array.isArray(node.config.inputShape),
                    outputValid: Array.isArray(node.config.outputShape)
                });

                // 添加到集合
                nodes.push(node);
                layerMap.set(node.name, node);

                // 节点存储验证
                const storedNode = nodes[nodes.length - 1];
                console.log('\nStored Node Verification:', {
                    name: storedNode.name,
                    hasInputShape: Array.isArray(storedNode.config.inputShape),
                    hasOutputShape: Array.isArray(storedNode.config.outputShape),
                    inputShape: storedNode.config.inputShape,
                    outputShape: storedNode.config.outputShape
                });
            });

            // 处理层之间的连接
            console.log('\n==== Processing Connections ====');
            nodes.forEach(node => {
                if (Array.isArray(node.inbound)) {
                    node.inbound.forEach(conn => {
                        if (Array.isArray(conn)) {
                            const sourceNode = layerMap.get(conn[0]);
                            if (sourceNode) {
                                const connectionType = this.determineConnectionType(sourceNode, node);
                                links.push({
                                    source: sourceNode.id,
                                    target: node.id,
                                    type: connectionType,
                                    metadata: {
                                        sourceLayer: sourceNode.className,
                                        targetLayer: node.className,
                                        connectionType: connectionType
                                    }
                                });
                                sourceNode.outbound.push(node.id);
                            }
                        }
                    });
                }
            });

            // 最终验证
            console.log('\n==== Final Verification ====');
            console.log('Total nodes created:', nodes.length);
            console.log('Total connections created:', links.length);

            // 验证每个节点的形状信息
            nodes.forEach(node => {
                console.log(`Node ${node.name} final shapes:`, {
                    inputShape: node.config.inputShape,
                    outputShape: node.config.outputShape,
                    hasValidShapes: Array.isArray(node.config.inputShape) && Array.isArray(node.config.outputShape)
                });
            });

            return { nodes, links };
        } catch (error) {
            console.error('Error parsing model JSON:', error);
            throw error;
        }
    }

    calculateInputShape(layer, nodes) {
        console.log('\n==== Input Shape Calculation ====');
        console.log('Layer:', layer);
        console.log('Inbound nodes:', layer.inbound_nodes);
        console.log('First inbound node:', layer.inbound_nodes?.[0]);
        console.log('First connection:', layer.inbound_nodes?.[0]?.[0]);

        // 处理第一层
        if (layer.name === 'conv1_pad') {
            return [224, 224, 3];
        }

        // 处理没有输入节点的层
        if (!layer.inbound_nodes || layer.inbound_nodes.length === 0) {
            return [224, 224, 3];
        }

        // 获取连接信息
        const connection = layer.inbound_nodes[0][0];
        let prevLayerName;

        // 处理四元组格式 [layerName, nodeIndex, tensorIndex, {}]
        if (Array.isArray(connection)) {
            prevLayerName = connection[0];
        } else {
            prevLayerName = connection;
        }

        console.log('Previous layer name:', prevLayerName);

        // 查找前一层节点
        let prevNode = nodes.find(n => n.name === prevLayerName);

        // 处理block间的连接
        if (!prevNode && prevLayerName.endsWith('_out')) {
            // 从输出层回溯到对应的add层
            const blockName = prevLayerName.replace('_out', '_add');
            prevNode = nodes.find(n => n.name === blockName);
        }

        // 处理ReLU层
        if (!prevNode && prevLayerName.endsWith('_relu')) {
            // 从ReLU层回溯到对应的bn层
            const baseName = prevLayerName.replace('_relu', '_bn');
            prevNode = nodes.find(n => n.name === baseName);
        }

        if (!prevNode) {
            console.warn(`Cannot find previous layer: ${prevLayerName}`);
            return null;
        }

        if (!Array.isArray(prevNode.config.outputShape)) {
            console.warn(`Invalid output shape from previous layer:`, prevNode.config.outputShape);
            return null;
        }

        // 返回形状的深拷贝
        return [...prevNode.config.outputShape];
    }

    calculateOutputShape(layer, inputShape) {
        console.log('\n==== Output Shape Calculation ====');
        console.log(`Calculating output shape for: ${layer.name} (${layer.class_name})`);
        console.log('Input shape:', inputShape);
        console.log('Layer configuration:', layer.config);

        if (!Array.isArray(inputShape)) {
            console.warn(`Invalid input shape for ${layer.name}:`, inputShape);
            return null;
        }

        const [height, width, channels] = inputShape;
        let outputShape;

        try {
            switch (layer.class_name) {
                case 'ZeroPadding2D': {
                    const padding = layer.config?.padding || [[1, 1], [1, 1]];
                    console.log('ZeroPadding2D parameters:', padding);
                    outputShape = [
                        height + padding[0][0] + padding[0][1],
                        width + padding[1][0] + padding[1][1],
                        channels
                    ];
                    console.log('ZeroPadding2D output:', outputShape);
                    break;
                }

                case 'Conv2D': {
                    const kernelSize = layer.config?.kernel_size || [1, 1];
                    const strides = layer.config?.strides || [1, 1];
                    const padding = layer.config?.padding || 'valid';
                    const filters = layer.config?.filters;

                    console.log('Conv2D parameters:', {
                        kernelSize,
                        strides,
                        padding,
                        filters
                    });

                    let outputHeight, outputWidth;
                    if (padding === 'valid') {
                        outputHeight = Math.floor((height - kernelSize[0] + 1) / strides[0]);
                        outputWidth = Math.floor((width - kernelSize[1] + 1) / strides[1]);
                    } else if (padding === 'same') {
                        outputHeight = Math.ceil(height / strides[0]);
                        outputWidth = Math.ceil(width / strides[1]);
                    }

                    outputShape = [outputHeight, outputWidth, filters];
                    console.log('Conv2D output:', outputShape);
                    break;
                }

                case 'MaxPooling2D': {
                    const poolSize = layer.config?.pool_size || [2, 2];
                    const strides = layer.config?.strides || poolSize;
                    const padding = layer.config?.padding || 'valid';

                    console.log('MaxPooling2D parameters:', {
                        poolSize,
                        strides,
                        padding
                    });

                    let outputHeight, outputWidth;
                    if (padding === 'valid') {
                        outputHeight = Math.floor((height - poolSize[0] + 1) / strides[0]);
                        outputWidth = Math.floor((width - poolSize[1] + 1) / strides[1]);
                    } else if (padding === 'same') {
                        outputHeight = Math.ceil(height / strides[0]);
                        outputWidth = Math.ceil(width / strides[1]);
                    }

                    outputShape = [outputHeight, outputWidth, channels];
                    console.log('MaxPooling2D output:', outputShape);
                    break;
                }

                case 'BatchNormalization':
                case 'Activation':
                case 'Add': {
                    console.log(`${layer.class_name}: Maintaining input shape`);
                    outputShape = [...inputShape];
                    break;
                }

                default: {
                    console.log(`Default case: Maintaining input shape for ${layer.class_name}`);
                    outputShape = [...inputShape];
                }
            }

            if (!outputShape || !Array.isArray(outputShape) || outputShape.some(dim => !Number.isFinite(dim))) {
                console.error('Invalid output shape calculated:', outputShape);
                return null;
            }

            console.log(`Final output shape for ${layer.name}:`, outputShape);
            return outputShape;

        } catch (error) {
            console.error(`Error calculating output shape for ${layer.name}:`, error);
            return null;
        }
    }

    createEnhancedLabel(nodeData) {
        const div = document.createElement('div');
        div.className = 'node-label';

        div.style.cssText = `
        color: #2c3e50;
        background: rgba(255, 255, 255, 0.95);
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        opacity: 0;
        transition: all 0.3s ease;
        pointer-events: none;
        white-space: nowrap;
    `;

        // 根据节点类型添加详细信息
        const name = this.getShortLayerName(nodeData.className);
        let details = '';

        // 根据不同类型的层添加特定信息
        if (nodeData.className === 'Conv2D') {
            const filters = nodeData.config?.filters || '';
            const kernelSize = nodeData.config?.kernelSize || [];
            details = `(${filters}f ${kernelSize[0]}×${kernelSize[1]})`;
        }

        div.textContent = `${name} ${details}`;

        const label = new THREE.CSS2DObject(div);
        label.position.set(0, 25, 0);
        return label;
    }

    async createNetworkVisualization(networkData) {
        try {
            console.log('Creating network visualization with data:', networkData);

            if (!this.materials || !this.materials.node) {
                throw new Error('Materials not properly initialized');
            }

            const positions = this.calculateNodePositions(networkData);

            // 创建节点
            networkData.nodes.forEach((nodeData, index) => {
                const geometry = this.getNodeGeometry(nodeData);
                const material = this.getNodeMaterial(nodeData);

                // 创建主网格并添加基础属性
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.copy(positions[index]);
                mesh.userData = nodeData;

                // 添加节点发光效果
                const glowMesh = this.createGlowEffect(mesh);
                mesh.add(glowMesh);

                // 为特定类型的节点添加动画效果
                if (nodeData.className === 'Add') {
                    const glow = this.createGlowEffect(mesh, 0xf1c40f);
                    mesh.add(glow);
                    this.addPulseAnimation(mesh);
                }

                // 添加节点到场景
                this.nodes.set(nodeData.name, mesh);
                this.scene.add(mesh);

                // 创建增强的标签系统
                const label = this.createLabel(nodeData);
                if (label) {
                    mesh.add(label);
                }
            });

            // 创建连接线
            await this.createConnections(networkData);

            // 计算场景边界以获取中心点
            const box = new THREE.Box3().setFromObject(this.scene);
            const center = box.getCenter(new THREE.Vector3());

            // 只调整控制器的目标点到场景中心
            this.controls.target.copy(center);
            this.controls.update();

            // 强制渲染
            this.renderer.render(this.scene, this.camera);
            if (this.labelRenderer) {
                this.labelRenderer.render(this.scene, this.camera);
            }

            // 保存初始相机状态
            this.initialCameraState = {
                position: this.camera.position.clone(),
                target: this.controls.target.clone()
            };

            return true;
        } catch (error) {
            console.error('Error creating network visualization:', error);
            throw error;
        }
    }

    // 创建发光效果
    createGlowEffect(mesh) {
        const glowMaterial = new THREE.MeshPhongMaterial({
            color: mesh.material.color,
            transparent: true,
            opacity: 0.3,
            side: THREE.BackSide
        });

        const glowMesh = new THREE.Mesh(mesh.geometry.clone(), glowMaterial);
        glowMesh.scale.multiplyScalar(1.2);
        glowMesh.isGlow = true;  // 添加标记以便识别
        return glowMesh;
    }

    addNodeAnimation(mesh) {
        const originalScale = mesh.scale.clone();
        const pulseScale = originalScale.multiplyScalar(1.1);
        let increasing = true;

        const animate = () => {
            if (increasing) {
                mesh.scale.lerp(pulseScale, 0.1);
                if (mesh.scale.x >= pulseScale.x - 0.01) {
                    increasing = false;
                }
            } else {
                mesh.scale.lerp(originalScale, 0.1);
                if (mesh.scale.x <= originalScale.x + 0.01) {
                    increasing = true;
                }
            }
        };

        mesh.animation = animate;
        return animate;
    }

    // 创建增强的连接方法
    async createConnections(networkData) {
        const promises = networkData.links.map(async (link) => {
            const sourceNode = this.nodes.get(link.source);
            const targetNode = this.nodes.get(link.target);

            if (sourceNode && targetNode) {
                const connection = this.createEnhancedConnection(sourceNode, targetNode, link.type);
                const key = `${link.source}->${link.target}`;
                this.connections.set(key, connection);
                this.scene.add(connection);
            }
        });

        await Promise.all(promises);
    }

    // 创建增强的连接线
    createEnhancedConnection(sourceNode, targetNode, type) {
        const points = this.calculateConnectionPoints(sourceNode, targetNode, type);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        let material;
        if (type === 'residual') {
            material = new THREE.LineDashedMaterial({
                color: 0xE67E22,
                dashSize: 3,
                gapSize: 1,
                linewidth: 1,
            });
            const line = new THREE.Line(geometry, material);
            line.computeLineDistances(); // 关键：必须调用此方法才能显示虚线
            return line;
        } else {
            material = new THREE.LineBasicMaterial({
                color: 0x34495e,
                linewidth: 1,
                transparent: true,
                opacity: 0.7
            });
            return new THREE.Line(geometry, material);
        }
    }

    // 获取节点材质
    getNodeMaterial(nodeData) {
        console.log('Getting material for node:', {
            name: nodeData.name,
            class: nodeData.className,
            type: nodeData.layerType
        });

        try {
            // 检查必要的参数
            if (!nodeData || !nodeData.className) {
                console.error('Invalid node data:', nodeData);
                throw new Error('Invalid node data provided to getNodeMaterial');
            }

            // 主要的材质选择逻辑
            switch (nodeData.className) {
                case 'Conv2D': {
                    const layerDepth = this.getLayerDepth(nodeData);
                    console.log('Conv2D layer depth:', layerDepth);

                    // 验证材质是否存在
                    let material;
                    if (layerDepth < 3) {
                        material = this.materials.node.conv2d.early;
                    } else if (layerDepth < 6) {
                        material = this.materials.node.conv2d.middle;
                    } else {
                        material = this.materials.node.conv2d.deep;
                    }

                    if (!material) {
                        console.error('Failed to get Conv2D material for depth:', layerDepth);
                        return this.materials.node.conv2d.early; // 默认材质
                    }
                    return material;
                }

                case 'BatchNormalization': {
                    const material = this.materials.node.batchNorm;
                    if (!material) {
                        console.error('BatchNormalization material not found');
                        return this.materials.node.conv2d.early; // 默认材质
                    }
                    return material;
                }

                case 'MaxPooling2D': {
                    const material = this.materials.node.pooling;
                    if (!material) {
                        console.error('Pooling material not found');
                        return this.materials.node.conv2d.early; // 默认材质
                    }
                    return material;
                }

                case 'Add': {
                    const material = this.materials.node.add;
                    if (!material) {
                        console.error('Add material not found');
                        return this.materials.node.conv2d.early; // 默认材质
                    }
                    return material;
                }

                default: {
                    console.warn('Unknown layer type:', nodeData.className);
                    return this.materials.node.conv2d.early; // 默认材质
                }
            }
        } catch (error) {
            console.error('Error in getNodeMaterial:', error);
            // 确保即使发生错误也返回一个有效的材质
            return this.materials.node.conv2d.early;
        }
    }

    // 辅助方法：获取层的深度
    getLayerDepth(nodeData) {
        const match = nodeData.name.match(/conv(\d+)/);
        return match ? parseInt(match[1]) : 1;
    }

    // 添加节点光晕效果
    addNodeGlow(mesh, nodeData) {
        // 创建光晕几何体
        const glowGeometry = mesh.geometry.clone();
        const glowMaterial = new THREE.MeshPhongMaterial({
            color: mesh.material.color,
            transparent: true,
            opacity: 0.3,
            side: THREE.BackSide
        });

        const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
        glowMesh.scale.multiplyScalar(1.2);
        mesh.add(glowMesh);

        // 为特定类型的节点添加脉冲动画
        if (nodeData.className === 'Add' || nodeData.className === 'Conv2D') {
            this.addPulseAnimation(glowMesh);
        }
    }

    // 添加脉冲动画
    addPulseAnimation(mesh) {
        const animate = () => {
            const time = Date.now() * 0.001;
            mesh.material.opacity = 0.2 + Math.sin(time * 2) * 0.1;
            requestAnimationFrame(animate);
        };
        animate();
    }


    // 2. 修改标签创建方法，统一使用 opacity 控制可见性
    createLabel(nodeData) {
        const div = document.createElement('div');
        div.className = 'node-label';

        div.style.cssText = `
        color: #2c3e50;
        background: rgba(255, 255, 255, 0.95);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        opacity: 0;
        transition: opacity 0.2s ease;
        pointer-events: none;
        white-space: nowrap;
    `;

        // 直接使用原始的层类型名称
        div.textContent = `${nodeData.className}: ${nodeData.name}`;

        const label = new THREE.CSS2DObject(div);
        label.position.set(0, 25, 0);
        return label;
    }

    extractNetworkStructure() {
        if (!this.model || !this.model.layers) {
            throw new Error('Invalid model structure');
        }

        const nodes = [];
        const links = [];
        const layerMap = new Map();

        // 处理每一层
        this.model.layers.forEach((layer, index) => {
            const node = {
                id: index,
                name: layer.name,
                type: layer.getClassName(),
                config: layer.getConfig(),
                params: layer.countParams(),
                shape: layer.outputShape
            };
            nodes.push(node);
            layerMap.set(layer.name, node);
        });

        // 处理层之间的连接
        this.model.layers.forEach((layer, index) => {
            if (layer.inboundNodes && layer.inboundNodes.length > 0) {
                layer.inboundNodes.forEach(node => {
                    if (node.inboundLayers) {
                        node.inboundLayers.forEach(inboundLayer => {
                            const sourceNode = layerMap.get(inboundLayer.name);
                            if (sourceNode) {
                                links.push({
                                    source: sourceNode.id,
                                    target: index,
                                    type: this.determineConnectionType(inboundLayer, layer)
                                });
                            }
                        });
                    }
                });
            }
        });

        return { nodes, links };
    }



    determineConnectionType(sourceNode, targetNode) {
        console.log('Determining connection type:', {
            source: {
                name: sourceNode.name,
                class: sourceNode.className
            },
            target: {
                name: targetNode.name,
                class: targetNode.className
            }
        });

        if (targetNode.className === 'Add') {
            // 使用名称匹配来判断残差连接
            const sourceBlock = sourceNode.name.match(/block(\d+)/);
            const targetBlock = targetNode.name.match(/block(\d+)/);

            if (sourceBlock && targetBlock) {
                const blockDiff = Math.abs(parseInt(sourceBlock[1]) - parseInt(targetBlock[1]));
                if (blockDiff === 0) {
                    console.log('Detected residual connection in block:', sourceBlock[1]);
                    return 'residual';
                }
            }
        }
        return 'forward';
    }


    getNodeGeometry(nodeData) {
        const baseSize = 20;

        // 解析节点在网络中的位置和类型
        const stageName = nodeData.name.match(/conv(\d+)/);
        const stageNum = stageName ? parseInt(stageName[1]) : 1;
        const blockMatch = nodeData.name.match(/block(\d+)/);
        // const blockNum = blockMatch ? parseInt(blockMatch[1]) : 1;

        // 计算深度缩放因子
        const depthScale = stageNum ? (stageNum / 5) : 1; // 随着阶段增加而增加深度

        // 输入处理阶段的几何体设计
        if (nodeData.name === 'input_1') {
            // 输入层使用菱形几何体
            return new THREE.OctahedronGeometry(baseSize * 1.2, 0);
        }

        if (nodeData.name.includes('pad')) {
            // ZeroPadding2D层使用扁平六边形
            return new THREE.CylinderGeometry(
                baseSize * 0.9 * depthScale,    // 顶部半径
                baseSize * 0.9 * depthScale,    // 底部半径
                baseSize * 0.3,                 // 高度
                6,                              // 六边形
                1,                              // 高度分段
                false                           // 封闭几何体
            );
        }

        switch (nodeData.className) {
            case 'Conv2D': {
                const kernelSize = nodeData.config?.kernelSize || [3, 3];
                const filters = nodeData.config?.filters || 64;

                // 首个卷积层
                if (nodeData.name === 'conv1_conv') {
                    return new THREE.BoxGeometry(
                        baseSize * 1.8,
                        baseSize * 1.8,
                        baseSize * (filters / 32) * depthScale
                    );
                }

                // 1x1卷积层
                if (kernelSize[0] === 1) {
                    return new THREE.BoxGeometry(
                        baseSize * 0.6,
                        baseSize * 0.6,
                        baseSize * (filters / 64) * depthScale // 调整深度系数
                    );
                }

                // 3x3卷积层
                const depth = this.getLayerDepth(nodeData);
                const lengthScale = Math.min(1 + (depth * 0.5), 4); // 随深度增加而增长，但有上限

                return new THREE.BoxGeometry(
                    baseSize,
                    baseSize,
                    baseSize * lengthScale * (filters / 64) * depthScale
                );
            }

            case 'BatchNormalization': {
                // 创建一个BufferGeometry作为基础
                const geometry = new THREE.BufferGeometry();

                // 定义一个复杂的形状来代表批归一化层
                // 上部圆环的顶点
                const upperRingVertices = [];
                const ringSegments = 32;
                const ringRadius = baseSize * 2.0 * depthScale;    // 从1.2增加到2.0
                const tubeRadius = baseSize * 0.4 * depthScale;    // 从0.2增加到0.4

                for (let i = 0; i <= ringSegments; i++) {
                    const theta = (i / ringSegments) * Math.PI * 2;
                    const phi = (i / ringSegments) * Math.PI * 2;

                    // 主环
                    const x = Math.cos(theta) * ringRadius;
                    const y = baseSize * 0.8;  // 从0.4增加到0.8，上移更多
                    const z = Math.sin(theta) * ringRadius;

                    // 添加环的横截面
                    for (let j = 0; j < 8; j++) {
                        const angle = (j / 8) * Math.PI * 2;
                        const dx = Math.cos(angle) * tubeRadius;
                        const dy = Math.sin(angle) * tubeRadius;

                        upperRingVertices.push(
                            x + dx,
                            y + dy,
                            z
                        );
                    }
                }

                // 创建顶点属性
                geometry.setAttribute(
                    'position',
                    new THREE.Float32BufferAttribute(upperRingVertices, 3)
                );

                // 计算法线
                geometry.computeVertexNormals();

                // 创建网格
                const material = new THREE.MeshPhongMaterial({
                    color: 0x2F855A,
                    shininess: 60,
                    transparent: true,
                    opacity: 0.85
                });

                // 返回最终的网格
                return geometry;
            }

            case 'MaxPooling2D': {
                // 池化层使用棱锥体
                return new THREE.ConeGeometry(
                    baseSize * 0.8 * depthScale,    // 底部半径
                    baseSize * 1.2,                 // 高度
                    8                               // 底部分段数
                );
            }

            case 'Add': {
                // Add层使用八面体表示信息汇合
                const size = baseSize * 1.2 * depthScale;
                return new THREE.OctahedronGeometry(size);
            }

            case 'Activation': {
                // ReLU激活层使用小型圆柱体
                return new THREE.CylinderGeometry(
                    baseSize * 0.4 * depthScale,    // 顶部半径
                    baseSize * 0.4 * depthScale,    // 底部半径
                    baseSize * 0.3,                 // 高度
                    8                               // 分段数
                );
            }

            default: {
                // 其他层使用默认的球体
                return new THREE.SphereGeometry(baseSize * 0.5 * depthScale);
            }
        }
    }

    createConnection(sourceNode, targetNode, type) {
        // 获取连接线的点集
        const points = this.calculateConnectionPoints(
            sourceNode.position,
            targetNode.position,
            type,
            sourceNode.userData,
            targetNode.userData
        );

        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        // 使用预定义的材质而不是每次创建新的
        const material = type === 'residual' ?
            this.materials.connection.residual :
            this.materials.connection.forward;

        const line = new THREE.Line(geometry, material);

        if (type === 'residual') {
            line.computeLineDistances();  // 确保虚线效果生效
            this.addResidualConnectionEffect(line);
        }

        line.userData = {
            source: sourceNode.userData.id,
            target: targetNode.userData.id,
            type: type
        };

        return line;
    }

    calculateConnectionPoints(start, end, type, sourceData, targetData) {
        const startPoint = new THREE.Vector3().copy(start);
        const endPoint = new THREE.Vector3().copy(end);

        if (type === 'residual') {
            return this.createResidualCurve(startPoint, endPoint);
        } else {
            return this.createForwardCurve(startPoint, endPoint, sourceData, targetData);
        }
    }

    createResidualCurve(start, end) {
        // 优化残差连接的曲线控制点计算
        const distance = start.distanceTo(end);
        const midPoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
        const heightOffset = distance * 0.3;  // 减小高度偏移使曲线更自然

        // 调整控制点位置
        const control1 = new THREE.Vector3(
            start.x + (midPoint.x - start.x) * 0.5,
            midPoint.y + heightOffset,
            start.z + (midPoint.z - start.z) * 0.5
        );

        const control2 = new THREE.Vector3(
            midPoint.x + (end.x - midPoint.x) * 0.5,
            midPoint.y + heightOffset,
            midPoint.z + (end.z - midPoint.z) * 0.5
        );

        const curve = new THREE.CubicBezierCurve3(start, control1, control2, end);
        return curve.getPoints(30);  // 减少点数以提高性能
    }

    createForwardCurve(start, end, sourceData, targetData) {
        // 简化前向连接的曲线
        const points = [];
        points.push(start);

        // 仅对较长的连接添加中间点
        if (start.distanceTo(end) > 100) {
            const midPoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
            const yOffset = this.getConnectionYOffset(sourceData.className, targetData.className);
            midPoint.y += yOffset;
            points.push(midPoint);
        }

        points.push(end);
        return points;
    }

    addResidualConnectionEffect(line) {
        // 确保连接线有可用的材质
        if (!line.material || !(line.material instanceof THREE.LineDashedMaterial)) {
            console.warn('Invalid line material for residual connection effect');
            return;
        }

        // 保存原始的材质属性
        const originalOpacity = line.material.opacity;
        const originalDashSize = line.material.dashSize;
        const originalGapSize = line.material.gapSize;

        // 创建动画函数
        const animate = () => {
            const time = Date.now() * 0.001; // 当前时间（秒）

            if (line && line.material) {
                // 添加呼吸效果（透明度变化）
                line.material.opacity = originalOpacity * (0.6 + Math.sin(time * 2) * 0.2);

                // 添加流动效果（虚线偏移）
                line.material.dashOffset = -time * 5;

                // 可选：添加虚线大小的变化效果
                const dashScale = 1 + Math.sin(time * 3) * 0.2;
                line.material.dashSize = originalDashSize * dashScale;
                line.material.gapSize = originalGapSize * dashScale;

                // 更新材质
                line.material.needsUpdate = true;
            }

            // 继续动画循环
            requestAnimationFrame(animate);
        };

        // 启动动画
        animate();

        // 返回动画函数以便后续可能的清理
        return animate;
    }

    // 获取连接的Y轴偏移量
    getConnectionYOffset(sourceType, targetType) {
        // 根据层类型返回适当的偏移量
        const baseOffset = 20;

        if (sourceType === 'Conv2D' && targetType === 'BatchNormalization') {
            return baseOffset;
        } else if (sourceType === 'BatchNormalization' && targetType === 'Activation') {
            return baseOffset * 1.5;
        }

        return baseOffset * 0.5;
    }

    calculateNodePositions(networkData) {
        const positions = [];
        // 增加间距以适应新的垂直布局
        const stageSpacing = 350;    // 增加阶段间距
        const blockSpacing = 180;    // 增加块间距
        const layerSpacing = 80;    // 增加层间距

        // 创建阶段分组映射
        const stageMap = new Map();

        networkData.nodes.forEach(node => {
            // 从节点名称中提取阶段信息（如conv2_block1, conv3_block1等）
            const stageName = node.name.match(/conv(\d+)_block/);
            if (stageName) {
                const stageNum = parseInt(stageName[1]);
                if (!stageMap.has(stageNum)) {
                    stageMap.set(stageNum, []);
                }
                stageMap.get(stageNum).push(node);
            } else {
                // 处理不属于任何阶段的节点
                if (!stageMap.has(0)) {
                    stageMap.set(0, []);
                }
                stageMap.get(0).push(node);
            }
        });

        // 计算每个节点的位置
        let currentZ = 0;
        stageMap.forEach((stageNodes, stageNum) => {
            // 将每个阶段的节点按块分组
            const blockGroups = new Map();
            stageNodes.forEach(node => {
                const blockMatch = node.name.match(/block(\d+)/);
                const blockNum = blockMatch ? parseInt(blockMatch[1]) : 0;
                if (!blockGroups.has(blockNum)) {
                    blockGroups.set(blockNum, []);
                }
                blockGroups.get(blockNum).push(node);
            });

            // 计算每个块内节点的位置
            let currentX = -((blockGroups.size - 1) * blockSpacing) / 2;
            blockGroups.forEach((blockNodes, blockNum) => {
                // 在Y轴上错开不同类型的层
                blockNodes.forEach((node, index) => {
                    const nodeIndex = networkData.nodes.indexOf(node);
                    const y = this.calculateNodeYPosition(node);

                    // 为相邻的BatchNormalization和Conv2D层添加额外的间距调整
                    let zOffset = 0;
                    if (node.className === 'BatchNormalization') {
                        // 让BatchNormalization层更靠近其对应的Conv2D层
                        zOffset = -layerSpacing * 0.3;
                    }

                    positions[nodeIndex] = new THREE.Vector3(
                        currentX,
                        y,
                        currentZ + (index * layerSpacing) + zOffset
                    );
                });
                currentX += blockSpacing;
            });

            currentZ += stageSpacing;
        });

        return positions;
    }

    // 添加新的辅助方法来计算节点的Y轴位置
    calculateNodeYPosition(node) {
        // 基础高度单位
        const baseHeight = 50;

        switch (node.className) {
            case 'Conv2D': {
                // 卷积层作为基准层
                return 0;
            }
            case 'BatchNormalization': {
                // 批归一化层略高于卷积层
                return baseHeight * 0.8;
            }
            case 'MaxPooling2D': {
                // 池化层位于较低位置
                return -baseHeight * 1.2;
            }
            case 'Add': {
                // Add层位于最高位置
                return baseHeight * 2;
            }
            case 'Activation': {
                // 激活层位于中等高度
                return baseHeight * 0.4;
            }
            default: {
                // 其他层保持在基准位置
                return 0;
            }
        }
    }

    getNodeLevel(node) {
        if (!node.inbound || node.inbound.length === 0) return 0;

        const maxParentLevel = Math.max(
            ...node.inbound.map(conn => {
                // 从当前节点集合中查找父节点
                const parentNode = this.nodes.get(conn[0]);
                if (!parentNode) return 0;
                return this.getNodeLevel(parentNode);
            })
        );

        return maxParentLevel + 1;
    }
    // ------------------------
    // 事件处理方法
    // ------------------------

    // 1. 修改事件监听器注册
    addEventListeners() {
        if (!this.renderer || !this.renderer.domElement) {
            console.error('Renderer not initialized');
            return;
        }

        // 移除现有的事件监听器
        this.removeEventListeners();

        // 窗口大小改变事件
        this._handleResize = () => {
            if (!this.container) return;

            const width = this.container.clientWidth;
            const height = this.container.clientHeight;

            // 更新相机
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();

            // 更新渲染器尺寸
            this.renderer.setSize(width, height, false);
            if (this.labelRenderer) {
                this.labelRenderer.setSize(width, height);
            }

            // 强制重新渲染
            this.render();
        };

        // 鼠标移动事件
        this._handleMouseMove = (event) => {
            event.preventDefault();
            const rect = this.renderer.domElement.getBoundingClientRect();
            this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            this.handleHover();
        };

        // 点击事件
        this._handleClick = (event) => {
            event.preventDefault();
            this.handleClick();
        };

        // 双击事件
        this._handleDoubleClick = (event) => {
            event.preventDefault();

            // 如果有保存的初始相机状态，则恢复
            if (this.initialCameraState) {
                this.camera.position.copy(this.initialCameraState.position);
                this.controls.target.copy(this.initialCameraState.target);
            } else {
                // 使用与 initialize 方法中相同的参数（作为备用）
                const distance = 3000;
                this.camera.position.set(
                    -distance * 0.8,
                    distance * 0.3,
                    distance * 0.8
                );
                this.controls.target.set(0, 0, 0);
            }

            this.camera.lookAt(this.controls.target);
            this.controls.update();
            this.render();
        };

        // 添加事件监听器
        window.addEventListener('resize', this._handleResize);
        this.renderer.domElement.addEventListener('mousemove', this._handleMouseMove);
        this.renderer.domElement.addEventListener('click', this._handleClick);
        this.renderer.domElement.addEventListener('dblclick', this._handleDoubleClick);
    }

    removeEventListeners() {
        if (this._handleResize) {
            window.removeEventListener('resize', this._handleResize);
        }
        if (this._handleMouseMove) {
            this.renderer.domElement.removeEventListener('mousemove', this._handleMouseMove);
        }
        if (this._handleClick) {
            this.renderer.domElement.removeEventListener('click', this._handleClick);
        }
        if (this._handleOutsideClick) {
            document.removeEventListener('click', this._handleOutsideClick);
        }
        if (this._handleDoubleClick) {
            this.renderer.domElement.removeEventListener('dblclick', this._handleDoubleClick);
        }
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        console.log(`Window resized: ${width} x ${height}`); // 输出调试信息

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
        this.labelRenderer.setSize(width, height);

        if (this.composer) {
            this.composer.setSize(width, height);
        }
    }


    onMouseMove(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(Array.from(this.nodes.values()));
        this.renderer.domElement.style.cursor = intersects.length > 0 ? 'pointer' : 'default';
    }



    // 修改点击事件处理
    onClick(event) {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(Array.from(this.nodes.values()));

        // 隐藏所有节点的标签
        this.nodes.forEach(node => {
            const label = node.children.find(child => child instanceof THREE.CSS2DObject);
            if (label) {
                label.element.style.display = 'none';
            }
        });

        if (intersects.length > 0) {
            const clickedNode = intersects[0].object;
            this.selectNode(clickedNode);  // 选择节点时调用 selectNode
            const label = clickedNode.children.find(child => child instanceof THREE.CSS2DObject);
            if (label) {
                label.element.style.display = 'block';
            }
        } else {
            this.clearSelection();
        }
    }


    resetNodeColors() {
        // 遍历所有节点，恢复原始颜色
        this.nodes.forEach(node => {
            if (node.userData.originalMaterial) {
                node.material = node.userData.originalMaterial;
                delete node.userData.originalMaterial;  // 清除保存的原始材质
            }
        });
    }


    // 选择和高亮方法
    selectNode(node) {
        console.log('selectNode called:', node); // 添加日志
        if (this.selectedNode === node) return;

        this.resetNodeColors();  // 重置其他节点的颜色

        this.selectedNode = node;

        // 保存当前材质（如果没有保存过）
        if (!node.userData.originalMaterial) {
            node.userData.originalMaterial = node.material;
        }

        // 为节点设置新的材质
        const highlightMaterial = new THREE.MeshPhongMaterial({
            color: node.userData.originalMaterial.color,
            transparent: true,
            opacity: 0.9,
            shininess: 50,
            specular: new THREE.Color(0x444444),
            emissive: new THREE.Color(0x2980b9),
            emissiveIntensity: 0.5
        });

        node.material = highlightMaterial;

        // 显示该节点的详细信息
        console.log('Showing detail panel for node:', node.userData); // 添加日志
        this.showDetailPanel(node.userData);
    }


    // 清除选择状态
    clearSelection() {
        if (this.selectedNode) {
            try {
                const nodeData = this.selectedNode.userData;
                let material;

                console.log('clearSelection - Node type:', nodeData.className);
                console.log('clearSelection - Original material color:', this.selectedNode.material.color);
                console.log('clearSelection - Node depth:', nodeData.name.match(/conv(\d+)/)?.[1]);

                // 如果有保存的原始材质，直接使用
                if (this.selectedNode.userData.originalMaterial) {
                    this.selectedNode.material = this.selectedNode.userData.originalMaterial;
                    delete this.selectedNode.userData.originalMaterial;  // 清除保存的原始材质
                } else {
                    // 如果没有保存的原始材质，使用预定义材质
                    switch (nodeData.className) {
                        case 'Conv2D': {
                            const depth = nodeData.name.match(/conv(\d+)/)
                                ? parseInt(nodeData.name.match(/conv(\d+)/)[1])
                                : 1;

                            console.log('clearSelection - Conv2D depth:', depth);

                            // 使用预定义材质的引用而不是克隆
                            if (depth <= 2) {
                                material = this.materials.node.conv2d.early;
                            } else if (depth <= 4) {
                                material = this.materials.node.conv2d.middle;
                            } else {
                                material = this.materials.node.conv2d.deep;
                            }

                            console.log('clearSelection - Selected Conv2D material:', material.color);
                            break;
                        }
                        case 'BatchNormalization':
                            material = this.materials.node.batchNorm;
                            break;
                        case 'MaxPooling2D':
                            material = this.materials.node.pooling;
                            break;
                        case 'Add':
                            material = this.materials.node.add;
                            console.log('clearSelection - Add node material color:', this.materials.node.add.color);
                            break;
                        default:
                            material = this.materials.node.conv2d.early;
                    }

                    if (material) {
                        // 使用预定义材质，确保材质保持一致
                        this.selectedNode.material = material;
                    }
                }

                // 处理标签可见性：将标签隐藏
                const label = this.selectedNode.children.find(child =>
                    child instanceof THREE.CSS2DObject
                );
                if (label && label.element) {
                    label.element.style.opacity = '0';
                }

                // 清除选中状态
                this.selectedNode = null;

            } catch (error) {
                console.error('Error in clearSelection:', error, {
                    nodeData: this.selectedNode?.userData,
                    materials: this.materials,
                    materialSystem: JSON.stringify(this.materials, null, 2)
                });
            }
        }

        // 隐藏详情面板
        if (this.detailPanel) {
            this.detailPanel.classList.remove('visible');
        }
    }


    highlightResidualBlock(startNode) {
        const addNode = Array.from(this.nodes.values()).find(node =>
            node.userData.type === 'Add' &&
            this.isConnectedToAdd(startNode, node)
        );

        if (addNode) {
            const mainPath = this.getMainPath(startNode, addNode);
            const skipConnection = this.getSkipConnection(startNode, addNode);

            mainPath.forEach(node => {
                node.material = new THREE.MeshPhongMaterial({
                    color: 0x00ff00,
                    emissive: 0x44ff44,
                    transparent: true,
                    opacity: 0.8
                });
            });

            if (skipConnection) {
                skipConnection.material = new THREE.LineBasicMaterial({
                    color: 0xff0000,
                    linewidth: 2,
                    transparent: true,
                    opacity: 0.8
                });
            }

            addNode.material = new THREE.MeshPhongMaterial({
                color: 0xffff00,
                emissive: 0xffff44,
                transparent: true,
                opacity: 0.8
            });
        }
    }

    // 动画和效果方法

    // 新增 render 方法
    render() {
        if (!this.isInitialized || !this.scene || !this.camera || !this.renderer) return;

        // 更新控制器
        if (this.controls) {
            this.controls.update();
        }

        // 更新动画
        this.updateAnimations();

        // 渲染场景
        this.renderer.render(this.scene, this.camera);

        // 渲染标签
        if (this.labelRenderer) {
            this.labelRenderer.render(this.scene, this.camera);
        }
    }

    // 修改 animate 方法
    animate = () => {
        if (!this.isInitialized) return;

        // 调用render方法进行渲染
        this.render();

        // 继续动画循环
        requestAnimationFrame(this.animate);
    };

    // 修改 updateAnimations 方法
    updateAnimations() {
        if (!this.isInitialized) return;

        // 更新节点动画
        this.nodes?.forEach(node => {
            if (node.animation) {
                node.animation();
            }

            // 更新发光效果
            node.children?.forEach(child => {
                if (child.isGlow && child.material) {
                    child.material.opacity = 0.3 + Math.sin(Date.now() * 0.001) * 0.1;
                    child.material.needsUpdate = true; // 确保材质更新
                }
            });
        });

        // 更新连接动画
        this.connections?.forEach(connection => {
            if (connection.animation) {
                connection.animation();
            }
        });
    }

    addConnectionAnimation(connection) {
        const material = connection.material;
        const startOpacity = material.opacity;

        // 创建辉光效果
        const glowMaterial = material.clone();
        glowMaterial.opacity = 0.3;
        glowMaterial.transparent = true;

        const glowGeometry = connection.geometry.clone();
        const glowLine = new THREE.Line(glowGeometry, glowMaterial);
        this.scene.add(glowLine);

        // 动画参数
        let time = 0;
        const animate = () => {
            time += 0.01;
            material.opacity = startOpacity + Math.sin(time * 5) * 0.2;
            glowMaterial.opacity = 0.3 + Math.sin(time * 3) * 0.1;

            glowLine.position.copy(connection.position);
            glowLine.rotation.copy(connection.rotation);
            glowLine.scale.copy(connection.scale);
        };

        return { animate, glowLine };
    }

    addNodePulse(node) {
        const originalScale = node.scale.clone();
        const pulseScale = originalScale.multiplyScalar(1.2);
        let pulsing = false;

        const pulse = () => {
            if (pulsing) {
                node.scale.lerp(pulseScale, 0.1);
            } else {
                node.scale.lerp(originalScale, 0.1);
            }

            if (node.scale.distanceTo(pulsing ? pulseScale : originalScale) < 0.01) {
                pulsing = !pulsing;
            }
        };

        return pulse;
    }

    addGlowEffect() {
        try {
            // 确保所需的着色器已加载
            if (!THREE.LuminosityHighPassShader) {
                console.warn('LuminosityHighPassShader not loaded, skipping glow effect');
                return;
            }

            const renderScene = new THREE.RenderPass(this.scene, this.camera);
            const bloomPass = new THREE.UnrealBloomPass(
                new THREE.Vector2(window.innerWidth, window.innerHeight),
                1.0,  // 强度
                0.4,  // 半径
                0.85  // 阈值
            );

            // 设置辉光参数
            bloomPass.threshold = 0.3;
            bloomPass.strength = 0.8;
            bloomPass.radius = 0.5;

            this.composer = new THREE.EffectComposer(this.renderer);
            this.composer.addPass(renderScene);
            this.composer.addPass(bloomPass);

            // 设置渲染尺寸
            this.composer.setSize(
                this.container.clientWidth,
                this.container.clientHeight
            );
        } catch (error) {
            console.warn('Failed to add glow effect:', error);
            // 如果添加辉光效果失败，继续使用基本渲染
            this.composer = null;
        }
    }

    // 连接分析和处理方法
    getLayerDepth(layer) {
        let depth = 0;
        let currentLayer = layer;

        while (currentLayer.inboundNodes &&
            currentLayer.inboundNodes[0] &&
            currentLayer.inboundNodes[0].inboundLayers &&
            currentLayer.inboundNodes[0].inboundLayers[0]) {
            currentLayer = currentLayer.inboundNodes[0].inboundLayers[0];
            depth++;
        }

        return depth;
    }

    getLayerPath(layer) {
        let path = 0;
        let currentLayer = layer;

        while (currentLayer.inboundNodes &&
            currentLayer.inboundNodes[0] &&
            currentLayer.inboundNodes[0].inboundLayers &&
            currentLayer.inboundNodes[0].inboundLayers[0]) {
            currentLayer = currentLayer.inboundNodes[0].inboundLayers[0];
            path++;
        }

        return path;
    }

    isConnectedToAdd(node, addNode) {
        const connectionKey = `${node.userData.id}->${addNode.userData.id}`;
        return this.connections.has(connectionKey);
    }

    getMainPath(startNode, endNode) {
        const path = [];
        let currentNode = startNode;

        while (currentNode && currentNode !== endNode) {
            path.push(currentNode);
            const nextNode = this.getNextNode(currentNode);
            if (!nextNode || nextNode === endNode) break;
            currentNode = nextNode;
        }

        return path;
    }

    getSkipConnection(startNode, endNode) {
        const connectionKey = `${startNode.userData.id}->${endNode.userData.id}`;
        return this.connections.get(connectionKey);
    }

    getNextNode(node) {
        const connections = Array.from(this.connections.values())
            .filter(conn => conn.userData.source === node.userData.id);

        if (connections.length > 0) {
            return this.nodes.get(connections[0].userData.target);
        }
        return null;
    }


    // UI 更新和信息显示方法
    // 在 ResNetVisualizer 类中修改 updateLayerInfo 方法
    updateLayerInfo(nodeData) {
        console.log('完整的节点数据:', nodeData);
        console.log('节点配置信息:', nodeData.config);

        const config = nodeData.config || {};
        const unknown = 'Unknown';
        const none = 'None';

        // 基本信息
        document.getElementById('layer-name').textContent = nodeData.name;
        document.getElementById('layer-type').textContent = nodeData.className;
        document.getElementById('layer-params').textContent = this.calculateParams(nodeData);

        // 处理卷积层配置
        if (nodeData.className === 'Conv2D') {
            // 过滤器信息
            document.getElementById('layer-filters').textContent =
                `${config.batchInputShape?.[3] || unknown}, ${config.filters || unknown}`;

            // 卷积核信息
            document.getElementById('layer-kernel-size').textContent =
                `${config.kernelSize ? config.kernelSize.join(' × ') : unknown}`;

            // 步长信息
            document.getElementById('layer-strides').textContent =
                `${config.strides ? config.strides.join(' × ') : unknown}`;

            // 填充信息
            document.getElementById('layer-padding').textContent =
                `${config.padding === 'valid' ? 'Valid' :
                    config.padding === 'same' ? 'Same' :
                        config.padding || unknown}`;
        }

        // 激活函数配置
        document.getElementById('layer-activation').textContent =
            `${config.activation === 'linear' ? 'Linear' :
                config.activation === 'relu' ? 'ReLU' :
                    config.activation || unknown}`;

        // 偏置信息
        document.getElementById('layer-use-bias').textContent =
            `${config.useBias ? 'Yes' : 'No'}`;

        // 形状信息 - 修改这部分
        document.getElementById('layer-input-shape').textContent =
            Array.isArray(config.inputShape) ?
                `[${config.inputShape.join(', ')}]` :
                'N/A';

        document.getElementById('layer-output-shape').textContent =
            Array.isArray(config.outputShape) ?
                `[${config.outputShape.join(', ')}]` :
                'N/A';

        // 连接信息
        const inConnections = nodeData.inbound?.length > 0
            ? `${nodeData.inbound.length} (${nodeData.inbound.map(conn => conn[0]).join(', ')})`
            : `0 (${none})`;

        const outConnections = nodeData.outbound?.length > 0
            ? `${nodeData.outbound.length} (${nodeData.outbound.join(', ')})`
            : `0 (${none})`;

        document.getElementById('layer-in-connections').textContent = inConnections;
        document.getElementById('layer-out-connections').textContent = outConnections;
    }

    getNodeColor(type) {
        const colorMap = {
            'Conv2D': 0x81ecec,
            'BatchNormalization': 0x74b9ff,
            'Activation': 0xa29bfe,
            'Add': 0xff7675,
            'MaxPooling2D': 0xffeaa7,
            'Dense': 0x55efc4,
            'Dropout': 0xfab1a0,
            'GlobalAveragePooling2D': 0xe17055,
            'ZeroPadding2D': 0x6c5ce7
        };
        return colorMap[type] || 0xdfe6e9;
    }

    getShortLayerName(type) {
        // 使用英文缩写
        const nameMap = {
            'Conv2D': 'Conv',
            'BatchNormalization': 'BN',
            'Activation': 'Act',
            'MaxPooling2D': 'Pool',
            'Dense': 'Dense',
            'Dropout': 'Drop',
            'GlobalAveragePooling2D': 'GAP',
            'ZeroPadding2D': 'ZeroPad',
            'Add': 'Add'
        };
        return nameMap[type] || type;
    }

    // ------------------------
    // 性能优化方法
    // ------------------------

    optimizeRendering() {
        const frustum = new THREE.Frustum();
        const matrix = new THREE.Matrix4().multiplyMatrices(
            this.camera.projectionMatrix,
            this.camera.matrixWorldInverse
        );
        frustum.setFromProjectionMatrix(matrix);

        this.nodes.forEach(node => {
            if (frustum.containsPoint(node.position)) {
                node.visible = true;
                if (node.children.length > 0) {
                    const label = node.children[0];
                    label.visible = true;
                }
            } else {
                node.visible = false;
                if (node.children.length > 0) {
                    const label = node.children[0];
                    label.visible = false;
                }
            }
        });

        this.connections.forEach(connection => {
            const sourceNode = this.nodes.get(connection.userData.source);
            const targetNode = this.nodes.get(connection.userData.target);

            if (sourceNode && targetNode) {
                connection.visible = sourceNode.visible || targetNode.visible;
            }
        });
    }

    setRenderQuality(quality) {
        switch (quality) {
            case 'high':
                this.renderer.setPixelRatio(window.devicePixelRatio);
                this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
                break;
            case 'medium':
                this.renderer.setPixelRatio(1);
                this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
                break;
            case 'low':
                this.renderer.setPixelRatio(0.75);
                this.renderer.setSize(
                    this.container.clientWidth * 0.75,
                    this.container.clientHeight * 0.75,
                    true
                );
                break;
        }
    }

    // 处理鼠标悬停
    handleHover() {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(Array.from(this.nodes.values()));

        // 更新鼠标样式
        this.renderer.domElement.style.cursor = intersects.length > 0 ? 'pointer' : 'default';

        // 处理之前悬停的节点
        if (this.hoveredNode && this.hoveredNode !== this.selectedNode) {
            const label = this.hoveredNode.children[0];
            if (label && label.element && label.element.style) {
                label.element.style.opacity = '0';
            }
        }

        // 处理新的悬停节点，但不影响已选中的节点
        if (intersects.length > 0) {
            const newHoveredNode = intersects[0].object;
            if (newHoveredNode !== this.selectedNode) {
                this.hoveredNode = newHoveredNode;
                const label = this.hoveredNode.children[0];
                if (label && label.element && label.element.style) {
                    label.element.style.opacity = '1';
                    // 确保悬停标签只显示简略信息
                    if (label.element.textContent !== undefined) {
                        label.element.textContent = this.getShortLayerName(this.hoveredNode.userData.className);
                    }
                }
            }
        } else {
            this.hoveredNode = null;
        }
    }

    // 处理点击事件
    handleClick() {
        console.log('Click event triggered'); // 添加日志
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(Array.from(this.nodes.values()));

        if (intersects.length > 0) {
            const clickedNode = intersects[0].object;
            console.log('Clicked node:', clickedNode);
            console.log('Clicked node userData:', clickedNode.userData);
            this.selectNode(clickedNode);
        } else {
            this.clearSelection();
        }
    }

    // 更新标签显示
    updateHoverLabel(label, nodeData) {
        const labelElement = label.element;
        labelElement.innerHTML = `
        <div style="font-weight: bold;">${this.getShortLayerName(nodeData.className)}</div>
        <div style="font-size: 10px;">${nodeData.name}</div>
    `;
    }


    showDetailPanel(nodeData) {
        console.log('showDetailPanel called with:', nodeData); // 添加日志
        console.log('Detail panel element:', this.detailPanel); // 添加日志

        if (!this.detailPanel) {
            console.warn('Detail panel is not initialized!'); // 添加警告
            return;
        }

        // 首先完全重置面板
        this.resetDetailPanel();

        // 只有在有有效节点数据时才更新显示
        if (nodeData && nodeData.className) {
            this.updateLayerInfo(nodeData);
        }

        this.detailPanel.classList.add('visible');
    }

    // 辅助函数：获取嵌套的翻译文本
    getText(key, nestedKey = null) {
        const texts = this.languageResources[this.currentLanguage];
        if (!texts) return key;

        if (nestedKey) {
            return texts[key]?.[nestedKey] || key;
        }
        return texts[key] || key;
    }


    // updateBasicInfo(nodeData) {
    //     const valueElement = document.getElementById('layer-name');
    //     valueElement.textContent = nodeData.name;
    //     document.getElementById('layer-type').textContent = nodeData.className;
    //     document.getElementById('layer-params').textContent = this.calculateParams(nodeData);
    // }

    // updateConvConfig(nodeData) {
    //     const config = nodeData.config || {};
    //     const translations = {
    //         inputChannels: this.getText('inputChannels'),
    //         outputChannels: this.getText('outputChannels'),
    //         kernelSize: this.getText('kernelSize'),
    //         strides: this.getText('strides'),
    //         padding: this.getText('padding'),
    //         unknown: this.getText('unknown')
    //     };

    //     const fields = {
    //         'layer-filters': {
    //             label: translations.inputChannels,
    //             value: `${config.batchInputShape?.[3] || translations.unknown}, ${config.filters || translations.unknown}`
    //         },
    //         'layer-kernel-size': {
    //             label: translations.kernelSize,
    //             value: config.kernelSize ? `${config.kernelSize[0]} × ${config.kernelSize[1]}` : 'N/A'
    //         },
    //         'layer-strides': {
    //             label: translations.strides,
    //             value: config.strides ? `${config.strides[0]} × ${config.strides[1]}` : 'N/A'
    //         },
    //         'layer-padding': {
    //             label: translations.padding,
    //             value: this.getText('paddingTypes', config.padding)
    //         }
    //     };

    //     Object.entries(fields).forEach(([id, { label, value }]) => {
    //         const element = document.getElementById(id);
    //         if (element) {
    //             element.textContent = `${label}: ${value}`;
    //         }
    //     });
    // }

    // updateActivationConfig(nodeData) {
    //     const config = nodeData.config || {};
    //     document.getElementById('layer-activation').textContent =
    //         `${this.getText('activation')}: ${this.getText('activationTypes', config.activation)}`;
    //     document.getElementById('layer-use-bias').textContent =
    //         `${this.getText('useBias')}: ${this.getText('useBiasValues', config.useBias)}`;
    // }

    // updateConnectionInfo(nodeData) {
    //     const inbound = nodeData.inbound?.length > 0 ?
    //         `${nodeData.inbound.length} (${nodeData.inbound.map(conn => conn[0]).join(', ')})` :
    //         `0 (${this.getText('none')})`;
    //     const outbound = nodeData.outbound?.length > 0 ?
    //         `${nodeData.outbound.length} (${nodeData.outbound.join(', ')})` :
    //         `0 (${this.getText('none')})`;

    //     document.getElementById('layer-in-connections').textContent = inbound;
    //     document.getElementById('layer-out-connections').textContent = outbound;
    // }


    resetDetailPanel() {
        // 获取所有信息字段并设置为默认值
        const defaultValues = {
            'layer-name': 'N/A',
            'layer-type': 'N/A',
            'layer-params': '0',
            'layer-filters': 'N/A',
            'layer-kernel-size': 'N/A',
            'layer-strides': 'N/A',
            'layer-padding': 'N/A',
            'layer-activation': 'Activation Function: N/A',
            'layer-use-bias': 'Use Bias: No',
            'layer-input-shape': 'Input Shape: N/A',
            'layer-output-shape': 'Output Shape: N/A',
            'layer-in-connections': 'Input Connections: 0 (None)',
            'layer-out-connections': 'Output Connections: 0 (None)'
        };

        // 重置所有字段为默认值
        Object.entries(defaultValues).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }

    // 更新基本信息
    updateBasicInfo(nodeData) {
        const nameElement = document.getElementById('layer-name');
        const typeElement = document.getElementById('layer-type');
        const paramsElement = document.getElementById('layer-params');

        if (nameElement) nameElement.textContent = nodeData.name;
        if (typeElement) typeElement.textContent = nodeData.className;
        if (paramsElement) paramsElement.textContent = this.calculateParams(nodeData);
    }

    // 更新卷积配置信息
    updateConvConfig(nodeData) {
        const config = nodeData.config || {};
        const filtersElement = document.getElementById('layer-filters');
        const kernelElement = document.getElementById('layer-kernel-size');
        const stridesElement = document.getElementById('layer-strides');
        const paddingElement = document.getElementById('layer-padding');

        if (filtersElement && config.filters) {
            // 添加输入输出通道的完整信息
            const inputChannels = config.batchInputShape?.[3] || '未知';
            filtersElement.textContent = `输入通道: ${inputChannels}, 输出通道(过滤器): ${config.filters}`;
        }

        if (kernelElement && config.kernelSize) {
            const size = Array.isArray(config.kernelSize) ?
                `${config.kernelSize[0]} × ${config.kernelSize[1]}` :
                config.kernelSize;
            kernelElement.textContent = `卷积核: ${size}`;
        }

        // 步长和填充策略
        if (stridesElement && config.strides) {
            const stride = Array.isArray(config.strides) ?
                `${config.strides[0]} × ${config.strides[1]}` :
                config.strides;
            stridesElement.textContent = `步长: ${stride}`;
        }

        if (paddingElement) {
            const paddingTypes = {
                'valid': '无填充',
                'same': 'SAME填充(保持输出大小)',
                'causal': '因果填充'
            };
            paddingElement.textContent = `填充: ${paddingTypes[config.padding] || config.padding}`;
        }
    }

    // 更新激活函数配置
    updateActivationConfig(nodeData) {
        const config = nodeData.config || {};
        const activationElement = document.getElementById('layer-activation');
        const biasElement = document.getElementById('layer-use-bias');

        if (activationElement) {
            const activations = {
                'relu': 'ReLU激活',
                'linear': '线性激活',
                'sigmoid': 'Sigmoid激活',
                'tanh': 'Tanh激活'
            };
            activationElement.textContent = `激活函数: ${activations[config.activation] || config.activation}`;
        }

        // 批归一化参数
        if (nodeData.className === 'BatchNormalization') {
            const bnElement = document.getElementById('layer-bn-params');
            if (bnElement) {
                bnElement.textContent = `动量: ${config.momentum}, Epsilon: ${config.epsilon}`;
            }
        }

        if (biasElement) {
            biasElement.textContent = `偏置: ${config.useBias ? '使用' : '不使用'}`;
        }
    }

    // 更新形状信息
    // updateShapeInfo(nodeData) {
    //     const config = nodeData.config || {};

    //     const inputShapeElement = document.getElementById('layer-input-shape');
    //     if (inputShapeElement && config.inputShape) {
    //         inputShapeElement.textContent = JSON.stringify(config.inputShape);
    //     }

    //     const outputShapeElement = document.getElementById('layer-output-shape');
    //     if (outputShapeElement && config.outputShape) {
    //         outputShapeElement.textContent = JSON.stringify(config.outputShape);
    //     }
    // }

    // 更新连接信息
    updateConnectionInfo(nodeData) {
        const inElement = document.getElementById('layer-in-connections');
        const outElement = document.getElementById('layer-out-connections');

        if (inElement) {
            const inCount = nodeData.inbound?.length || 0;
            inElement.textContent = `${inCount} (${nodeData.inbound?.map(conn => conn[0]).join(', ') || '无'})`;
        }

        if (outElement) {
            const outCount = nodeData.outbound?.length || 0;
            outElement.textContent = `${outCount} (${nodeData.outbound?.join(', ') || '无'})`;
        }
    }

    // 计算参数数量的辅助方法
    calculateParams(nodeData) {
        const config = nodeData.config || {};
        let params = 0;

        if (nodeData.className === 'Conv2D') {
            const filters = config.filters || 0;
            const kernelSize = config.kernelSize || [1, 1];
            const inputChannels = config.batchInputShape ? config.batchInputShape[3] : 1;
            params = filters * (kernelSize[0] * kernelSize[1] * inputChannels + (config.useBias ? 1 : 0));
        }

        return params.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }

    // 8. 清除选择状态
    clearSelection() {
        if (this.selectedNode) {
            try {
                const nodeData = this.selectedNode.userData;
                let material;

                // 直接使用原始材质引用，不再克隆
                switch (nodeData.className) {
                    case 'Conv2D': {
                        const depth = nodeData.name.match(/conv(\d+)/)
                            ? parseInt(nodeData.name.match(/conv(\d+)/)[1])
                            : 1;

                        if (depth <= 2) {
                            material = this.materials.node.conv2d.early;
                        } else if (depth <= 4) {
                            material = this.materials.node.conv2d.middle;
                        } else {
                            material = this.materials.node.conv2d.deep;
                        }
                        break;
                    }
                    case 'BatchNormalization':
                        material = this.materials.node.batchNorm;
                        break;
                    case 'MaxPooling2D':
                        material = this.materials.node.pooling;
                        break;
                    case 'Add':
                        material = this.materials.node.add;
                        break;
                    default:
                        material = this.materials.node.conv2d.early;
                }

                // 更新选中节点的材质
                if (material) {
                    this.selectedNode.material = material;
                }

                // 处理标签可见性
                const label = this.selectedNode.children.find(child =>
                    child instanceof THREE.CSS2DObject
                );
                if (label && label.element) {
                    label.element.style.opacity = '0';
                }

                this.selectedNode = null;

            } catch (error) {
                console.error('Error in clearSelection:', error);
            }
        }

        if (this.detailPanel) {
            this.detailPanel.classList.remove('visible');
        }
    }


    // 错误处理和资源管理方法
    handleError(error) {
        console.error('Visualization error:', error);

        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `
            <p>An error occurred: ${error.message}</p>
            <button onclick="this.resetVisualization()">Reset Visualization</button>
        `;
        this.container.appendChild(errorDiv);

        this.resetVisualization();
    }

    resetVisualization() {
        this.dispose();

        try {
            this.initialize(this.container, this.model);
        } catch (error) {
            console.error('Failed to reset visualization:', error);
        }
    }

    dispose() {
        if (!this.isInitialized) return;

        try {
            // 移除事件监听器
            if (this._handleResize) {
                window.removeEventListener('resize', this._handleResize);
            }

            if (this.renderer && this.renderer.domElement) {
                if (this._handleMouseMove) {
                    this.renderer.domElement.removeEventListener('mousemove', this._handleMouseMove);
                }
                if (this._handleClick) {
                    this.renderer.domElement.removeEventListener('click', this._handleClick);
                }
                if (this._handleDoubleClick) {
                    this.renderer.domElement.removeEventListener('dblclick', this._handleDoubleClick);
                }
            }

            if (this._handleOutsideClick) {
                document.removeEventListener('click', this._handleOutsideClick);
            }

            // 清理场景对象
            if (this.scene) {
                this.scene.traverse((object) => {
                    if (object.geometry) {
                        object.geometry.dispose();
                    }
                    if (object.material) {
                        if (Array.isArray(object.material)) {
                            object.material.forEach(material => material.dispose());
                        } else {
                            object.material.dispose();
                        }
                    }
                });
            }

            // 清理渲染器
            if (this.renderer) {
                this.renderer.dispose();
                this.renderer.forceContextLoss();
                if (this.renderer.domElement && this.renderer.domElement.parentNode) {
                    this.renderer.domElement.remove();
                }
            }

            // 清理标签渲染器
            if (this.labelRenderer && this.labelRenderer.domElement) {
                if (this.labelRenderer.domElement.parentNode) {
                    this.labelRenderer.domElement.remove();
                }
            }

            // 清理后期处理
            if (this.composer) {
                if (this.composer.passes) {
                    this.composer.passes.forEach(pass => {
                        if (pass && typeof pass.dispose === 'function') {
                            pass.dispose();
                        }
                    });
                }

                if (this.composer.renderTarget1?.dispose) {
                    this.composer.renderTarget1.dispose();
                }
                if (this.composer.renderTarget2?.dispose) {
                    this.composer.renderTarget2.dispose();
                }

                this.composer = null;
            }

            // 清理节点和连接
            if (this.nodes) {
                this.nodes.forEach(node => {
                    if (node.animation) node.animation = null;
                });
                this.nodes.clear();
            }

            if (this.connections) {
                this.connections.forEach(connection => {
                    if (connection.animation) connection.animation = null;
                });
                this.connections.clear();
            }

            // 清空引用
            this.scene = null;
            this.camera = null;
            this.renderer = null;
            this.controls = null;
            this.labelRenderer = null;
            this.composer = null;

            // 清除事件处理器引用
            this._handleResize = null;
            this._handleMouseMove = null;
            this._handleClick = null;
            this._handleOutsideClick = null;
            this._handleDoubleClick = null;

            this.isInitialized = false;
        } catch (error) {
            console.warn('Error during cleanup:', error);
        }
    }
}

export default ResNetVisualizer;