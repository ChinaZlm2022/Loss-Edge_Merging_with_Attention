# Enhancing Point Cloud Completion with Fine-Grained Geometric Perception
Submitted to the Visual Computer Journal

This repository implements the FGGP-PCC network.
By integrating a collaborative hierarchical attention architecture,
the network deeply fuses local geometric features with global contextual information, achieving high‑fidelity reconstruction of complex missing structures. To evaluate the effectiveness of the proposed method, we compare it with other state‑of‑the‑art approaches on benchmark datasets. The method achieves leading results in both subjective visual quality and objective quantitative metrics, while also demonstrating significant advantages in lightweight design and time efficiency.

This repository is still under constructions.

If you have any questions about the code, please email me. Thanks!

This is the Pytorch implement of Enhancing Point Cloud Completion with Fine-Grained Geometric Perception.


##0) Environment
Pytorch 1.0.1
Python 3.7.4

##1) Dataset
```
  Shapenet：https://www.shapenet.org/
  Completion3D：https://completion3d.stanford.edu/
  PCN:https://gateway.infinitescript.com/s/ShapeNetCompletion
  modelnet40：http://modelnet.cs.princeton.edu/#
```
##2) Train
```
python Train_main.py 
```
Change ‘crop_point_num’ to control the number of missing points.
Change ‘point_scales_list ’to control different input resolutions.
Change ‘D_choose’to control without using D-net.

##3) Evaluate the Performance on ShapeNet
```
python show_recon.py
```
Show the completion results, the program will generate txt files in 'test-examples'.
```
python show_CD.py
```
Show the Chamfer Distances and two metrics in our paper.

##4) Visualization of csv File

We provide some incomplete point cloud in file 'test_one'. Use the following code to complete a incomplete point cloud of csv file:
```
python Test_csv.py
```
change ‘infile’and  ‘infile_real’to select different incomplete point cloud in ‘test_one’

##5) Visualization of Examples

Using Meshlab to visualize  the txt files.
