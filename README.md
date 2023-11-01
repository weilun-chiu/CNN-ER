# ODML_Lab2

clone from https://github.com/eric0708/ODML_Lab2.git, only leave the inference data and model here.

Log on Jetson Nano (baseline model and its AMP mode)
```
root@linux:/jetson-voice/CNN-ER# python model_inference.py 
Validation Accuracy: 45.54%
Average Inference Latency: 0.0004002889349626807
Average Inference Latency(b1): 0.023833829464193164
Validation Accuracy: 45.65%
Average Inference Latency: 0.0003558070121261363
Average Inference Latency(b1): 0.023009400394375764
```

follow this guide to export onnx model
https://pytorch.org/docs/stable/onnx_dynamo.html
