# -ROBOMATSER_LIF-2023-



./trtexec --onnx=/home/rcclub/MVSDK/demo/infer/2Binary_classification.onnx --saveEngine=/home/rcclub/MVSDK/demo/infer/2Binary_classification.trt --minShapes=inputs:1x1x32x32 --optShapes=inputs:1x1x32x32 --maxShapes=inputs:10x1x32x32 --fp16 



./trtexec --onnx=/home/rcclub/MVSDK/demo/infer/11Binary_classification.onnx --saveEngine=/home/rcclub/MVSDK/demo/infer/11Binary_classification.trt --minShapes=inputs:1x1x32x32 --optShapes=inputs:1x1x32x32 --maxShapes=inputs:10x1x32x32 --fp16
