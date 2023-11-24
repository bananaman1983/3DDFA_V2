# before import, make sure FaceBoxes and Sim3DR are built successfully, e.g.,
import sys
from subprocess import call
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import string

torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Solvay_conference_1927.jpg/1400px-Solvay_conference_1927.jpg', 'solvay.jpg')

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except Exception as e:
        print(f"Errorrrrr: {e}!")
        
print(os.getcwd())
#os.chdir("/FaceBoxes/utils")
print(os.getcwd())
#run_cmd("python build.py build_ext --inplace")
#os.chdir("/Sim3DR")
print(os.getcwd())
#run_cmd("python setup.py build_ext --inplace")
print(os.getcwd())
#os.chdir("/utils/asset")
print(os.getcwd())
#run_cmd("gcc -shared -Wall -O3 render.c -o render.so -fPIC")
#os.chdir("/app")
print(os.getcwd())


import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool


import matplotlib.pyplot as plt
from skimage import io
import gradio as gr

# load config
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Init FaceBoxes and TDDFA, recommend using onnx flag
onnx_flag = True  # or True to use ONNX to speed up
if onnx_flag:    
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
else:
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=False, **cfg)
    


def inference (img, radio_option):
    #set dense_flag from the radiobox option
    dense_flag = radio_option in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
    new_suffix = f'.{radio_option}' if radio_option in ('ply', 'obj') else '.jpg'
    wfp = f'examples/results/output_{radio_option}_'+''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) + new_suffix

    # face detection
    boxes = face_boxes(img)
    # regress 3DMM params
    param_lst, roi_box_lst = tddfa(img, boxes)
    # reconstruct vertices 
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)  
    #RGB to BGR
    imgBGR= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #decide output according to radiobox arguments
    if radio_option == '2d_sparse':
     draw_landmarks(imgBGR, ver_lst, show_flag=False, dense_flag=dense_flag, wfp=wfp)    
    elif radio_option == '2d_dense':
     draw_landmarks(imgBGR, ver_lst, show_flag=False, dense_flag=dense_flag, wfp=wfp)
    elif radio_option == '3d':
     render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=False, wfp=wfp)
    elif radio_option == 'depth':
     # if `with_bf_flag` is False, the background is black
     depth(imgBGR, ver_lst, tddfa.tri, show_flag=False, wfp=wfp, with_bg_flag=True)
    elif radio_option == 'pncc':
     pncc(imgBGR, ver_lst, tddfa.tri, show_flag=False, wfp=wfp, with_bg_flag=True)
    elif radio_option == 'uv_tex':
     uv_tex(imgBGR, ver_lst, tddfa.tri, show_flag=False, wfp=wfp)
    elif radio_option == 'pose':
     viz_pose(imgBGR, param_lst, ver_lst, show_flag=False,wfp=wfp)
    elif radio_option == 'ply':
     ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    elif radio_option == 'obj':
     ser_to_obj(imgBGR, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    else:
     raise ValueError(f'Unknown opt {radio_option}')

    #fetch processed image
    if radio_option in ('2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex'):
     outimg = cv2.imread(wfp)
     resized_img = cv2.resize(outimg, (img.shape[1],img.shape[0]))
     resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    else:
     #no output image(.obj or .ply)
     resized_img = img
    #if dense_flag is false recalc ver_list with dense flag up
    if dense_flag == False:
     ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)  
    return render(resized_img, ver_lst, tddfa.tri, alpha=0.6, show_flag=False);



title = "3DDFA V2"
description = "demo for 3DDFA V2. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2009.09960'>Towards Fast, Accurate and Stable 3D Dense Face Alignment</a> | <a href='https://github.com/cleardusk/3DDFA_V2'>Github Repo</a></p>"
examples = [
    ['solvay.jpg'], 
['examples/inputs/emma.jpg'], 
['examples/inputs/JianzhuGuo.jpg'], 
['examples/inputs/trump_hillary.jpg']
]
gr.Interface(
    inference, 
    [
     gr.Image(type="numpy", label="Input"), 
     gr.Radio(['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj'], value='2d_sparse')
     ], 
    gr.Image(type="numpy", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=examples
    ).launch()
