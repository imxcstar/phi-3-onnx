## 使用(Use)

### 1.克隆此项目(Clone Project)

git clone --recursive https://github.com/imxcstar/phi-3-onnx

### 2.模型下载(Model Download)

https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4

### 3.替换源码里的模型目录位置(Replace the model directory location in the source code)

``` csharp
static AIChat aiChat = new AIChat("./phi3-mini-4k-cpu");
```

## 预览(Preview)

![1.png](/Preview/1.png)

## 感谢(Thanks)

[\[Gradio.Net\]](https://github.com/feiyun0112/Gradio.Net/)

[\[onnxruntime-genai\]](https://github.com/microsoft/onnxruntime-genai/)