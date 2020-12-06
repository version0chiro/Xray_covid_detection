
from ctypes import *
import cv2
import numpy as np
import runner
import os
import math
import threading
import time
import argparse
import json
import xir.graph 
import xir.subgraph
import pathlib
# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def get_subgraph (g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.children
    if s.metadata.get_attr_str ("device") == "DPU"]
    return sub

'''
run CNN with batch
dpu: dpu runner
img: imagelist to be run
'''
def runDPU(id,start,dpu,img,listImage):

    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    outputHeight = outputTensors[0].dims[1]
    outputWidth = outputTensors[0].dims[2]
    outputChannel = outputTensors[0].dims[3]
    # tensorformat = dpu.get_tensor_format()
    # if tensorformat == dpu.TensorFormat.NCHW:
    #     outputHeight = outputTensors[0].dims[2]
    #     outputWidth = outputTensors[0].dims[3]
    #     outputChannel = outputTensors[0].dims[1]
    # elif tensorformat == dpu.TensorFormat.NHWC:
    #     outputHeight = outputTensors[0].dims[1]
    #     outputWidth = outputTensors[0].dims[2]
    #     outputChannel = outputTensors[0].dims[3]
    # else:
    #     exit("Format error")
    outputSize = outputHeight*outputWidth*outputChannel
    #softmax = np.empty(outputSize)

    batchSize = inputTensors[0].dims[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    while count < n_of_images:
        # print(listImage[count])
        if (count+batchSize<=n_of_images):
            runSize=batchSize
        else:
            runSize=n_of_images-count

 
        shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])
        
        """ prepare batch input/output """
        outputData = []
        inputData = []
        outputData.append(np.empty((runSize,outputHeight,outputWidth,outputChannel), dtype = np.float32, order = 'C'))
        inputData.append(np.empty((shapeIn), dtype = np.float32, order = 'C'))
        
        """ init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j,...] = img[(count+j)% n_of_images].reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])

        """ run with batch """
        job_id = dpu.execute_async(inputData,outputData)
        
        dpu.wait(job_id)

        predictions = outputData[0][0]
        print(predictions)
        predictions = predictions[0][0]
        predictions = softmax(predictions)
        # print("predictions shape: ",predictions.shape)
        y = np.argmax(predictions)
        classes = {
            0 : 'covid',
            1 : 'normal'
         }
        print("detected object is : "+str(classes[y]))
        count = count + runSize
        
        
    return



def runApp(batchSize, threads, image_dir,model):


    listImage=os.listdir(image_dir)

    runTotal = len(listImage)
    
    global out_q
    out_q = [None] * runTotal
    g = xir.graph.Graph.deserialize(pathlib.Path(model))
    subgraphs = get_subgraph(g)
    assert len(subgraphs) == 1
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(runner.Runner(subgraphs[0],"run"))
    

    """ pre-process all images """
    img = []
    print(listImage)
    for i in range(len(listImage)):
        
        image = cv2.imread(os.path.join(image_dir,listImage[i]))
        cv2.imshow('test',image)
        cv2.waitKey()
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image,200,200)
        image = image.reshape(-1,200,200,3).astype('float32')
        image = image/255.0
        img.append(image)


    """run with batch """
    threadAll = []
   
    start = 0
    
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start + (len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i],in_q,listImage))
        threadAll.append(t1)
        start = end
    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1
    
    fps = float(runTotal/timetotal)
    print("FPS=%.2f, total frames = %.0f , time = %.4f seconds" %(fps,runTotal,timetotal))
    
    return


def main():

    # command line arguments
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-m','--model', type=str,
                    default='/home/root/DPU_Covid_19_detection_target/dpu_densenetx_0.elf'
                    )
    
    ap.add_argument('-i', '--image_dir',
                    type=str,
                    default='images',
  	                help='Path of images folder. Default is ./images')
    ap.add_argument('-t', '--threads',
                    type=int,
                    default=1,
  	                help='Number of threads. Default is 1')
    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=1,
  	                help='Input batchsize. Default is 1')
    args = ap.parse_args()


    runApp(args.batchsize, args.threads, args.image_dir,args.model)

    
if __name__ == '__main__':
    main()