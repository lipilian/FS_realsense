# First test Fast FoundationStereo Inferencing.

## 0.1 Install CUDA 13.2
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda-repo-ubuntu2404-13-2-local_13.2.0-595.45.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-13-2-local_13.2.0-595.45.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-2
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```
## 0.2 Install TensorRT 10.16.1
- Download nv-tensorrt-local-repo-ubuntu2404-10.16.1-cuda-13.2_1.0-1_amd64.deb
```
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2404-10.16.1-cuda-13.2_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2404-10.16.1-cuda-13.2/nv-tensorrt-local-78CF83ED-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```
## 0. Install pytorch 2.13 xformer with cuda 13.2
```
conda create -n ffs python=3.12 && conda activate ffs
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python3 -m pip install --upgrade ninja
TORCH_CUDA_ARCH_LIST="12.0" \
python3 -m pip install -v --no-build-isolation \
  git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
pip install -r requirements.txt
```
## 0.4 Test the python inference (have to sucessfully run the python inference before running the C++ inference)
```
python scripts/run_demo.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output/ --remove_invisible 0 --denoise_cloud 1  --scale 1 --get_pc 1 --valid_iters 8 --max_disp 192 --zfar 100
```
## 1.1 Build plug in 
```
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
  -DTENSORRT_ROOT=/usr
cmake --build build -j4
```
```
python scripts/make_plugin_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output/ffs_plugin_1280x800 \
  --height 800 \
  --width 1280 \
  --valid_iters 4 \
  --max_disp 192
```
```
cpp/build/ffs_build_single_engine \
  output/ffs_plugin_1280x800/fast_foundationstereo_plugin.onnx \
  output/ffs_plugin_1280x800/fast_foundationstereo.engine
```

## 1.2 Move engine and onnx.yaml to models/ffs_1280x800
## 1.3 Move thrid party dependencies to thrid_party/fast_foundation_stereo_runtime for inferencing call.

## 1.4 Install opencv
```
sudo apt-get install libopencv-dev
```

## 2.1 Build the program
```
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
  -DFFS_TENSORRT_ROOT=/usr
```
```
cmake --build build --parallel
```

## 2.2 Run first Program: extract information from db3 file
- single frame check
```
./build/ffs_offline_validate \
  --input data/offline_test/raw/test_D455.db3 \
  --frames 1 \
  --infer-frame 1 \
  --engine-dir models/ffs_1280x800 \
  --max-depth-m 1 \
  --output data/offline_test/infer_frame_000001
```
- long sequence benchmark
```
./build/ffs_offline_validate \
  --input data/offline_test/raw/test_D455.db3 \
  --frames 1 \
  --warmup-frames 20 \
  --benchmark-frames 100 \
  --engine-dir models/ffs_1280x800 \
  --output data/offline_test/benchmark_100
```
## 2.3 No UI , 15 hz 1280x800 inferencing with realtime D455 camera.
```
./build/ffs_live_infer \
  --engine-dir models/ffs_1280x800 \
  --max-depth-m 1
```
## 2.4 With UI test
```
cmake --build build --parallel

./build/ffs_live_viewer \
  --engine-dir models/ffs_1280x800 \
  --point-step 4 \
  --max-depth-m 1
```

## 2.4 Add imGUI to repo as third party dependency.

```
git clone --branch docking --depth 1 \
  https://github.com/ocornut/imgui.git third_party/imgui
```


## 3. Start working on high resolution and high iteration FoundationStereo.

- use 23-51-11 Vit Large to seek highest performance and quality for single snapshot inferencing.
- 3GB model weights. 
- create conda environment with fs.
```
conda create -n fs python=3.12 && conda activate fs
```
- install torch & xformers
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python3 -m pip install --upgrade ninja
TORCH_CUDA_ARCH_LIST="12.0" \
python3 -m pip install -v --no-build-isolation \
  git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```
- install other dependencies
```
pip install -r requirements.txt
```
- test demo
chaneg run_demo.py line 63 to 
```
ckpt = torch.load(ckpt_dir, weights_only=False)
```
```
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./weights/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/
```
- test customized input (realsense D455 input)
```
python scripts/run_demo.py --left_file ../data/high_stereo_pair_test/left_rgb.png --right_file ../data/high_stereo_pair_test/right_rgb.png --ckpt_dir ./weights/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/
```
- 1280x800 inferencing time is 14.85s 
Based on descirption of this paper, The model performs better for image width size <1000. You can run with smaller scale, e.g. --scale 0.5 to downsize input image, then upsize the output depth to your need with nearest neighbor interpolation.
I will scale image to 960x600 to minimize the cost, then add 8 pixels padding to make it 960x608 for inferencing.

## 4. Make the onnx model with 960x608
- First change cfg.yaml in weights/23-51-11/cfg.yaml 
```
max_disp: 192
```
- use [python script](python_onnx_maker/make_onnx_liu.py) to make onnx model by set dynamo to False. 
- convert onnx to tensorrt engine. 
```
trtexec --onnx=/home/liu4000/Desktop/FS_realsense/models/fs_960x608_iters32/fs.onnx --saveEngine=/home/liu4000/Desktop/FS_realsense/models/fs_960x608_iters32/fs.engine
```
- test the speed 
```
trtexec \
  --loadEngine=models/fs_960x608_iters32/fs.engine \
  --warmUp=1000 \
  --duration=10 \
  --iterations=100 \
  --useCudaGraph \
  --noDataTransfers
```
