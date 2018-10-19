#### Demo
- apk

[TensorFlow官方版本， 一个预训练的apk](https://www.tensorflow.org/lite/demo_android)
[Source App](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/demo_android.md)

- 官方实例实现poents classification，`实验成功`

[1. 先training模型](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0)

[2. 部署到手机中](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#1)

- 第三方mnist实现，`未测试`

[1. tf.keras mnist的实现](https://heartbeat.fritz.ai/intro-to-machine-learning-on-android-how-to-convert-a-custom-model-to-tensorflow-lite-e07d2d9d50e3)

[2. android模块](https://heartbeat.fritz.ai/introduction-to-machine-learning-on-android-part-2-building-an-app-to-recognize-handwritten-d58ebc01950)

- [Android Object Detection](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193)

#### Problem
- TensorFlow Lite操作数类型
默认只支持， float32和uint8的类型，这样就是意味着在放入到Android中，`BYTE_SIZE_OF_FLOAT = 4 OR 1(float32, uint8, 32/8, 8/8)`
- 支持操作[函数](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/tf_ops_compatibility.md)
- Android SDK over than 23
- TensorFlow over than 1.8
- 出现`java.nio.BufferOverflowException`

```java
E/AndroidRuntime: FATAL EXCEPTION: CameraBackground
    Process: android.example.com.tflitecamerademo, PID: 7971
    java.nio.BufferOverflowException
        at java.nio.DirectByteBuffer.putFloat(DirectByteBuffer.java:459)
        at com.example.android.tflitecamerademo.ImageClassifier.convertBitmapToByteBuffer(ImageClassifier.java:207)

// 该问题是ByteBuffer.allocateDirect的大小问题，注意确认好，以下三个变量的大小
 ByteBuffer.allocateDirect(
            4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
```
- 出现`java.lang.IllegalArgumentException: Cannot convert between a TensorFlowLite buffer with 270000 bytes and a ByteBuffer with 2700000 bytes.`
该问题一般是由于`DIM_IMG_SIZE_X， DIM_IMG_SIZE_Y， 或者是DIM_PIXEL_SIZE没有设置正确`
- 使用tflite或者是.h5文件在server端正确，到android端出现错误，无法正确识别
一个很重要的问题就是**输入**和**输出**，输入一定要按照训练的时候来输入，如果你做了augmentation，`均一化`数据，这样在android中也需要均一化。
输出一般是softmax对应assert/.txt文件，顺序和训练时一定要对应，并且不能有多余换行，否则会报输出的错
- Android Bitmap出现NullPointException问题,可以按照以下comment进行更改

```java
bitmap = Bitmap.createScaledBitmap(bitmap,getImageSizeX(),getImageSizeY(), true);
//        bitmap.getPixels(intValues, 0, getImageSizeX(), 0, 0, getImageSizeX(), getImageSizeY());
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
```



#### Tools
- 所有的转换方法都在，`tf.contrib.lite`这个tensorflow类下面
示例转换代码, [bash version](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/convert/cmdline_examples.md)，转换到tflite， tensorflow r1.11， [python version](https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/lite/toco/g3doc/python_api.md#interpreter-file)

```bash
# pb --> tflite, 注意指定好input and output layer
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --graph_def_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1
```
```bash
# floating-point --> tflite, 直接在tensorflow里面保存的checkpoint即可以完成变换
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --saved_model_dir=/tmp/saved_model
```
```bash
# tf.keras --> tflite, 注意这个是tf下面的keras而不是原生Keras，原生需要通过.h5 --> .pb --> .tflite
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --keras_model_file=/tmp/keras_model.h5
```
```bash
#一些多输入和多输出
tflite_convert \
  --graph_def_file=/tmp/inception_v1_2016_08_28_frozen.pb \
  --output_file=/tmp/foo.tflite \
  --input_shapes=1,28,28,96:1,28,28,16:1,28,28,192:1,28,28,64 \
  --input_arrays=InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_3/MaxPool_0a_3x3/MaxPool,InceptionV1/InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/Relu \
  --output_arrays=InceptionV1/Logits/Predictions/Reshape_1
  
tflite_convert \
  --graph_def_file=/tmp/inception_v1_2016_08_28_frozen.pb \
  --output_file=/tmp/foo.tflite \
  --input_arrays=input \
  --output_arrays=InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu
```


- 核心帮助文档:[https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/lite/g3doc/](https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/lite/g3doc/)
- 模型quantization，[压缩](https://www.tensorflow.org/performance/model_optimization)
- Keras To TensorFlow 工具，原始的不能够显示Input，Output Layer名字，我稍微修改了一下，原始[代码](https://github.com/amir-abdi/keras_to_tensorflow), 修改[部分](https://github.com/kehuantiantang/AndroidTensorflow/blob/master/keras_to_tensorflow.py#L115-L120)
- 代码中转换和具体训练的模型，看[这里](https://github.com/kehuantiantang/AndroidTensorflow/blob/master/Keras%20To%20TensorFlow%20Lite.ipynb)

#### Useful Website
[https://www.tensorflow.org/lite/](https://www.tensorflow.org/lite/)

[cpu and gpu performance in android](https://hackernoon.com/building-an-insanely-fast-image-classifier-on-android-with-mobilenets-in-tensorflow-dc3e0c4410d4)

