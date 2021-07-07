#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "calibrator.h"
#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>

// #define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32

#define DEVICE 0
#define CHECK(status)   \
  do{ \
      auto rst = (status);  \
      if(rst){   \
          std::cerr << "Cuda Failure, with cudaEror = " << rst << std::endl;  \
          abort();    \
      }   \
  }while(0)


const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const int INPUT_CHANNEL = 3;
const int INPUT_HEIGHT = 512;
const int INPUT_WIDTH = 512;
const int CLASSES = 3;
static Logger gLogger;

static const char* engineFile = "../unet_fp32.engine";
static const int batchsize = 1;
static const std::string image = "../roi.png";
static const std::string detImg = "../roi_det_int8.png";

using namespace nvinfer1;


// wights
std::ofstream wtscount("../weightsCount.txt");
void write(std::string context){
    wtscount << context << std::endl;

}


// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file){
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    // std::cout << "len " << len << std::endl;
    write(lname + ".weight");
    write(lname + ".bias");
    write(lname + ".running_mean");
    write(lname + ".running_var");
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

// 构建DoubleConv网络层
IActivationLayer* addDoubleConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap,std::string lname, ITensor& input, int outpCh, int middle = 0, int k = 3, int s = 1, int p = 1){
    // (卷积层 + bn层 + relu层) * 2
    if(!middle){
        middle = outpCh;
    }
    // 添加卷积层
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, middle, DimsHW{k,k}, weightMap[lname + ".0" + ".weight"], weightMap[lname + ".0" +  ".bias"]);
   // weights
    write(lname + ".0" + ".weight"); 
    write(lname + ".0" + ".bias");
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    // 添加BN层
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-3);
    assert(bn1);
    // 添加ReLU网络层
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    // 添加卷积层 
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outpCh, DimsHW{k,k}, weightMap[lname + ".3" + ".weight"], weightMap[lname + ".3" + ".bias"]); 
    write(lname + ".3" + ".weight"); 
    write(lname + ".3" + ".bias");
    assert(conv2);
    conv2->setStrideNd(DimsHW{s, s});
    conv2->setPaddingNd(DimsHW{p, p});
    // 添加BN层
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".4", 1e-3);
    assert(bn2);
    // 添加ReLU网络层
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}

// 构建下采样模块(最大池化 + doubleconv)
IActivationLayer* addDownConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap,std::string lname, ITensor& input,  int outpCh, int middle = 0){
    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{2,2});
    assert(pool1);
    IActivationLayer* doubleConv1 = addDoubleConv(network, weightMap, lname, *pool1->getOutput(0), outpCh);
    assert(doubleConv1);
    return doubleConv1;
}

// 构建上采样模块（线性插值 + doubleconv)
IActivationLayer* addUpConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap,std::string lname, ITensor& input, ITensor& preTensor ,int outpCh, int middle = 0,  bool bilinear = true){
    IResizeLayer* resizeLayer = network->addResize(input);
    assert(resizeLayer);
    resizeLayer->setResizeMode(ResizeMode::kLINEAR);
    resizeLayer->setAlignCorners(true);
    float scales[] = {1.0, 2.0, 2.0};
    resizeLayer->setScales(scales, 3);
    int diffW = preTensor.getDimensions().d[2] - resizeLayer->getOutput(0)->getDimensions().d[2];
    int diffH = preTensor.getDimensions().d[1] - resizeLayer->getOutput(0)->getDimensions().d[1];
    IPaddingLayer* pad1 = network->addPaddingNd(*resizeLayer->getOutput(0), DimsHW(diffH / 2, diffW / 2), DimsHW(diffH - diffH / 2, diffW - diffW / 2));
    ITensor* tensors[] = {&preTensor, pad1->getOutput(0)};
    assert(tensors);
    IConcatenationLayer* concat1 = network->addConcatenation(tensors, 2);
    assert(concat1);
    IActivationLayer* doubleConv1 = addDoubleConv(network, weightMap, lname, *concat1->getOutput(0), outpCh, middle);
    assert(doubleConv1);
    return doubleConv1;
}

// 创建network模型
INetworkDefinition* createNetwork(IBuilder* builder, const int batch_size, std::map<std::string, Weights> weightMap ,bool bilinear = true){
    std::cout <<"begine to build network..."<<std::endl;
    INetworkDefinition* network = builder->createNetworkV2(0U);
    assert(network);
    // 设置编码层
    ITensor* data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims3{INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH });
    assert(data);
    IActivationLayer* inc = addDoubleConv(network, weightMap, "inc.double_conv", *data, 64);
    assert(inc);
    IActivationLayer* down1 = addDownConv(network, weightMap, "down1.maxpool_conv.1.double_conv", *inc->getOutput(0), 128);
    assert(down1);
    IActivationLayer* down2 = addDownConv(network, weightMap, "down2.maxpool_conv.1.double_conv", *down1->getOutput(0), 256);
    assert(down2);
    IActivationLayer* down3 = addDownConv(network, weightMap, "down3.maxpool_conv.1.double_conv", *down2->getOutput(0),  512);
    assert(down3);
    int factor = 1;
    if(bilinear){
        factor = 2;
    }
    IActivationLayer* down4 = addDownConv(network, weightMap, "down4.maxpool_conv.1.double_conv", *down3->getOutput(0), 1024 / factor);
    assert(down4);
    // 设置解码层
    IActivationLayer* up1 = addUpConv(network, weightMap, "up1.conv.double_conv", *down4->getOutput(0), *down3->getOutput(0), 512 / factor, 512);
    assert(up1);
    IActivationLayer* up2 = addUpConv(network, weightMap, "up2.conv.double_conv", *up1->getOutput(0), *down2->getOutput(0), 256 / factor, 256);
    assert(up2);
    IActivationLayer* up3 = addUpConv(network, weightMap, "up3.conv.double_conv", *up2->getOutput(0), *down1->getOutput(0), 128 / factor, 128);
    assert(up3); 
    IActivationLayer* up4 = addUpConv(network, weightMap, "up4.conv.double_conv", *up3->getOutput(0), *inc->getOutput(0), 64, 64);
    assert(up4);
    IConvolutionLayer* prob = network->addConvolutionNd(*up4->getOutput(0), CLASSES, DimsHW{1,1}, weightMap["outc.conv.weight"], weightMap["outc.conv.bias"]);
    assert(prob);
    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME); 
    network->markOutput(*prob->getOutput(0));
    std::cout << "[ok] finish building network!" << std::endl;
    return network;
}


// 创建engine模型
ICudaEngine* createCudaEngine(IBuilder* builder, const int batch_size){
    std::map<std::string, Weights> weightMap = loadWeights("../weights/unet.wts");
    INetworkDefinition* network = createNetwork(builder, batch_size, weightMap);
    assert(network && "network is nullptr!");
    std::cout << "layers = " << std::to_string(network->getNbLayers()) << std::endl;
    // for(int i = 0; i < network->getNbLayers(); ++i){
	// std::cout << "layer: " << std::to_string(i) <<" name = " <<network->getLayer(i)->getName()<< std::endl; 
	// auto dims = network->getLayer(i)->getOutput(0)->getDimensions();
	// std::cout << "channel = " << std::to_string(dims.d[0]) << std::endl;
	// std::cout << "height = " << std::to_string(dims.d[1]) << std::endl;
	// std::cout << "width = " << std::to_string(dims.d[2]) << std::endl;

    // }
    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config && "config is nullptr!");
    builder->setMaxBatchSize(batch_size);
    config->setMaxWorkspaceSize(1<<20);
    // 使用fp16半精度进行推理
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "Yes!" : "No!") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calib = new Int8EntropyCalibrator2(1, INPUT_WIDTH, INPUT_HEIGHT, "./coco_calib/", "../unet_int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calib);
#endif

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine && "engine is nullptr!");
   for(auto& mem: weightMap){
        if(mem.second.values){ 
          free((void*)(mem.second.values));
        }
    }
    network->destroy();
    config->destroy(); 
    return engine;
}

// 序列化模型
void serializeModel(const int batch_size, const char* engineFile){
    IBuilder* builder = createInferBuilder(gLogger);
    ICudaEngine* engine = createCudaEngine(builder, batch_size);
    IHostMemory* modelStream = engine->serialize();
    assert(modelStream);
    std::cout << "[ok] finish sirializing!" <<std::endl;
    std::cout << "begin to write engineFile... " << std::endl;
    // 将modeStream保存到文件中
    std::ofstream f(engineFile);
    assert(f.is_open() && "falid to open engineFile!");
    f.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    f.close();
    std::cout << "[ok] finish writing!" << std::endl;
    builder->destroy();
    engine->destroy();
}

// 进行推理
void inference(IExecutionContext& context, float* input, float* output, const int batchsize){
    // 为输入输出分配显存
    std::cout << "inferencing... " << std::endl;
    const ICudaEngine& engine = context.getEngine();
    std::cout << "[ok] get engine!" << std::endl;
    // std::cout << engine.getName() << std::endl;
    int nBindings = engine.getNbBindings();
    assert(nBindings == 2);
    // 输入输出显存数组的指针
    void* buffers[2];
    const int inputIdx = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIdx = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    // 为输入输出数组分配显存
    int memSize = batchsize * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
    //int opmem = batchsize * 256*64*64 * sizeof(float);
    assert(memSize > 0);
    std::cout << "begine allocate mem ... " << std::endl;
    CHECK(cudaMalloc(&buffers[inputIdx], memSize));
    CHECK(cudaMalloc(&buffers[outputIdx], memSize));
    //CHECK(cudaMalloc(&buffers[outputIdx], opmem));
    std::cout << "[ok] finish cudaMalloc!" << std::endl;
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 将数据从host复制到device上
    std::cout << "copy memory from host to device ..." << std::endl;
    CHECK(cudaMemcpyAsync(buffers[inputIdx], input, memSize, cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
    std::cout << "context enqueue ..." << std::endl;
    context.enqueue(batchsize, buffers, stream, nullptr);
    // 将计算结果从device复制到host上
    std::cout << "copy results from device to host ..." << std::endl;
    CHECK(cudaMemcpyAsync(output, buffers[outputIdx], memSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));

    //CHECK(cudaMemcpyAsync(output, buffers[outputIdx], memSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
    // 同步，直到所有的stream都运行完
    CHECK(cudaStreamSynchronize(stream));
    
    // 释放stream和显存
    std::cout << "free stream and memory.." <<std::endl;
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(buffers[inputIdx]));
    CHECK(cudaFree(buffers[outputIdx]));
    std::cout << "[ok] finish inference!" << std::endl;
}

// 反序列化模型
char* deserializeModel(const char* engineFile, int& engineSize){
    std::cout << "begine deserializeModel ... " << std::endl;
    std::ifstream f(engineFile);
    assert(f.is_open() && "fail to open engineFile ");
    f.seekg(0, std::ios::end);
    engineSize = f.tellg();
    f.seekg(0, std::ios::beg);
    assert(engineSize > 0 && "engine is empty!");
    std::cout << "size = " << std::to_string(engineSize) << std::endl;
    char* engineStream =  new char[engineSize];
    f.read(engineStream, engineSize);
    f.close();
    assert(engineStream && "engineStream is nullptr!");
    std::cout << "[ok] finish deserializeModel!" << std::endl;
    return engineStream;
}

// 读取图像
float* readImage(){
    cv::Mat img = cv::imread(image);
    cv::resize(img, img, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
	if (!img.data) {
		std::cout << "fail to open img file" << std::endl;
		abort();
	}
	std::cout << "cols = " << std::to_string(img.cols) << std::endl;
	std::cout << "rows = " << std::to_string(img.rows) << std::endl;
	int cols = img.cols * img.channels();
	int rows = img.rows;
	if (img.isContinuous()) {
	 	std::cout << "image is continuous!" << std::endl;
		cols *= rows;
		rows = 1;
	}
	// 将图像的数据保存到一维数组中[红色，绿色，蓝色]
	int offset = 0;
    std::cout << "resized img cols = " << std::to_string(img.cols) << std::endl;
	std::cout << "resized img rows = " << std::to_string(img.rows) << std::endl;
	static float data[INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH];
	// 遍历每一行数据
	for (int i = 0; i < rows; ++i) {
		// 得到该行的指针
		uchar* pointer = img.ptr<uchar>(i);
		// 每一列数据（每三步处理一次）
		for (int j = 0; j < cols; j += img.channels()) {
			// 红色数据放到[0, H * W -1]的位置，绿色放到[H*W , 2 * H * W-1]的位置，蓝色放到[2*W*H, 3*W*H-1]的位置
			data[offset] =( static_cast<float>(pointer[j + 2]) / 255.0 - 0.5) / 0.5;
			data[offset + INPUT_HEIGHT * INPUT_WIDTH] = (static_cast<float>(pointer[j + 1]) / 255.0 - 0.5) / 0.5;
			data[offset + 2 * INPUT_HEIGHT *INPUT_WIDTH] = (static_cast<float>(pointer[j]) / 255.0 - 0.5) / 0.5;
			++offset;
		}
	}
    std::cout << "[ok] finish read image!" << std::endl;
    return data;
}

void writeIptImg(float *input, const char* file){
    std::cout << "start write image" << std::endl;
    std::ofstream opt(file);
    assert(opt.is_open());
    int count = INPUT_HEIGHT * INPUT_WIDTH;
    for(int i = 0; i < 3 * count; ++i){
        opt << std::to_string(input[i]) << std::endl;
    }
    std::cout << "[ok] finish writing image!" << std::endl;
}

// 保存检测图像
void saveImg(float* data){
    // int rows = img.rows;
	// int cols = img.cols * img.channels();
    std::cout << "start write image..." << std::endl;
	cv::Mat dstMat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(0,0,0));
	int offset = 0;
	// 遍历data数组将其乘以255.f，然后再转成uchar
	for (int i = 0; i < INPUT_HEIGHT; ++i) {
		uchar* pointer = dstMat.ptr<uchar>(i);
		for (int j = 0; j < INPUT_WIDTH * INPUT_CHANNEL; j += 3) {
			// RGB -> BGR
			// 蓝色通道
			pointer[j] = static_cast<uchar>((data[2 * INPUT_HEIGHT * INPUT_WIDTH + offset] * 0.5 + 0.5) * 255.0);
			pointer[j + 1] = static_cast<uchar>((data[INPUT_HEIGHT * INPUT_WIDTH + offset] * 0.5 + 0.5) * 255.0);
			pointer[j + 2] = static_cast<uchar>((data[offset] * 0.5 + 0.5 ) * 255.0);
			++offset;
		}
	}
	if (!dstMat.data) {
		std::cout << "detection is nullptr!" << std::endl;
		abort();
	}
	cv::imwrite(detImg, dstMat);
    std::cout << "[ok] finish writing image!" << std::endl;
}

int main(int argc, char** argv){
    cudaSetDevice(DEVICE);
    // 从终端中输入两个参数
    if(argc != 2){
        std::cerr << "Please input two parameters!" << std::endl;
        std::cerr << "./unet -s: serialize model." << std::endl;
        std::cerr <<"./unet -d: deserialize model and inference."<< std::endl;
        return -1;
    }
    // [1]: 对模型进行序列化，以二进制的形式保存在硬盘上
    if(std::string(argv[1]) == "-s"){
        std::cout << "begine serialize..." << std::endl;
        serializeModel(batchsize, engineFile); 
        std::cout << "[ok] finish serialize!" << std::endl;
        return 0;
    }else if(std::string(argv[1]) != "-d"){
        std::cerr << "Please input two parameters!" << std::endl;
        std::cerr << "./unet -s: serialize model." << std::endl;
        std::cerr <<"./unet -d: deserialize model and inference."<< std::endl;
        return -1;
    }
    // [2]: 进行反序列化并进行推理
    int SIZE = batchsize * INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH;
    int engineSize = 0;
    float* input = readImage();
   // writeIptImg(input, "../input.txt");
    float *output = new float[SIZE];
    //long long  sizeO = 64*512*512;
    //float *output = new float[256*64*64];
    assert(input != nullptr && "image is nullptr");
    assert(output != nullptr && "output is nullptr");
    // 反序列化读取engine二进制文件
    // std::cout << "engineeeeee" << std::endl;
    char* engineStream = deserializeModel(engineFile, engineSize);
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineStream, engineSize, nullptr);
    assert(engine); 
    std::cout << "layers = " << std::to_string(engine->getNbLayers()) << std::endl;
    IExecutionContext* context = engine->createExecutionContext();
    assert(context);
    // 运行1次
    int COUNT = 5;
    for(int i = 0; i < COUNT; ++i){
        std::cout << std::to_string(i) << " : inferencing ..." << std::endl;
        auto start = std::chrono::system_clock::now();
        inference(*context, input, output, batchsize);
        auto end = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << std::to_string(i) << ": consuming time is " << time << "ms." << std::endl;
    }
    // 将图像保存到本地中
    assert(output && "ouput data is nullptr!");
    // writeIptImg(output, "../output.txt");
    // 将图片写入到指定的文件中
    saveImg(output);
    // 销毁不用的变量
    delete []output;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
