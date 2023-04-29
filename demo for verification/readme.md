支持动态输入的onnx to trt 

cd /usr/src/tensorrt/bin

./trtexec --onnx=youronnx.onnx --saveEngine=yourengine.trt --minShapes=inputs:1x1x32x32 --optShapes=inputs:1x1x32x32 --maxShapes=inputs:10x1x32x32 --fp16

**the shape need to be modified according to your own situation.**









