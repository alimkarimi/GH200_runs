# GH200_runs
This repo has code to run vision transformers on a GH200 GPU.

To get started, you need to make sure you have the right version of Nvidia Toolkit, Driver, and PyTorch.

This is so that hardware and software are able to communicate with eachother. This repo will run with:

- Nvidia Toolkit versioning - 12.4
- Nvidia Driver versioning - 550.54.15

You will also need PyTorch 2.4 and build it from wheel using:

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

Once you have the relevant packages installed, you can run the following commands to get the images downloaded:

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip annotation_trainval2014.zip

python3 get_coco_images.py -train=True -val=True

Once images are downloaded, you can train and test the model using:

python3 models.py

# Video demo of training and testing model: 
https://www.youtube.com/watch?v=68EcNX74C1Y

