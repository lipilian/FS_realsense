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
