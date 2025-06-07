# conda create -n OASeg python=3.9
# conda activate OASeg
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

pip install "git+https://github.com/facebookresearch/pytorch3d.git"

root_dir=$(pwd)

cd third_party
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd $root_dir

cd third_party
git clone https://github.com/qqlu/Entity.git
cp -r Entity/Entityv2/CropFormer detectron2/projects
cd detectron2/projects/CropFormer/entity_api/PythonAPI
make


cd ../..
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
pip install -U openmim
mim install mmcv
cd $root_dir

cp scripts/mask_predict/* third_party/detectron2/projects/CropFormer/demo_cropformer

# download pretrained models
# TODO

cd third_party
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

# replace 'future_fstrings' with 'utf-8'
# TODO