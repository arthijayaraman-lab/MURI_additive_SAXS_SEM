import cv2
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
import scipy.ndimage as spim
from skimage.transform import radon
import sklearn.cluster._kmeans as kmeans
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import json 

def scatter_plot_from_dict(data_dict, clst_folder, file_name): 
  plt.boxplot([data_dict[label] for label in data_dict], labels=list(data_dict.keys()))
  plt.xlabel("Number of clusters")
  plt.ylabel("Area of grain um")
  plt.title("Cluster number vs Grain size smaple - {}".format(file_name ))
  plt.savefig(os.path.join(clst_folder ,'scatter_plot_{}.png'.format(file_name)))     
  plt.clf()
  return None

def box_plot_from_dict(data_dict, clst_folder, file_name):
  plt.boxplot([data_dict[label] for label in data_dict], labels=list(data_dict.keys()))
  plt.xlabel("Number of clusters")
  plt.ylabel("Area of grain um")
  #plt.ylim((0,0.7))
  plt.title("Cluster number vs Grain size smaple")
  plt.savefig(os.path.join(clst_folder ,'scatter_plot_{}.png'.format(file_name)))       
  plt.clf()
  return None

def overlay_images(img1, img2, alpha=0.5):
  if img1.shape != img2.shape:
    raise ValueError("Images must have the same dimensions for overlay.")
  img1 = img1.astype(np.float32)
  img2 = img2.astype(np.float32)
  overlayed_img = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
  overlayed_img = overlayed_img.astype(np.uint8)
  return overlayed_img
  
if __name__ == "__main__":
    """
    Post processing and grain size analysis 
    """
    win_siz = 128 # tile size in pixels
    analysis_folder="/home/p51pro/UD/jayraman_lab/MURI_Additive/grin_detection/dataset_analyse_paper/analysis_folder/"
    for smaple_name in os.listdir(analysis_folder):
        cnt_area_unt = 6.2e-6 # pixel to distance measure in microns 
        smaple_folder = os.path.join(analysis_folder, smaple_name)
        output_folder = os.path.join(smaple_folder,"output_best")
        smaple_area_ratios = dict()
        
        if "5.67MPa" in smaple_name: # smaples in 5.67MPa have different resolution 
            cnt_area_unt = 1.55e-6
        
        # measurement level
        for msmt_folder_name in sorted(os.listdir(output_folder)):
            msmt_folder = os.path.join(output_folder, msmt_folder_name)
            raw_inp_img = cv2.imread(os.path.join(smaple_folder, msmt_folder_name),0)
            
            # Offset resolution lost in tiling 
            oft = win_siz//2
            raw_inp_img = raw_inp_img[oft:-(oft), oft:-(oft)]
            
            cls_area_ratios = dict()
            # sample level
            for clst_folder_name in sorted(os.listdir(msmt_folder)):
                clst_folder = os.path.join(msmt_folder, clst_folder_name)
                no_cls = int(clst_folder_name.split("cls_")[-1])
                all_masks = np.zeros((raw_inp_img.shape[0],raw_inp_img.shape[1])).astype(np.float64)
    
                area_ratio = [] 
                area_ratio_smaple_level = []
                
                # cluster level 
                for i in range(no_cls):
                    try:
                        cls_img = cv2.imread(os.path.join(clst_folder,"out_mask_{}.0.png".format(i)), 0)
                        # resixe predictions to original SEM image size
                        cls_img = cv2.resize(cls_img, (cls_img.shape[1]*2, cls_img.shape[0]*2), interpolation = cv2.INTER_LINEAR)
                    except:
                        continue
                    ####### DENOISING ########
                    # kernels for denoising
                    ker_3 = np.ones((3,3))
                    ker_5 = np.ones((5,5))
                    ker_7 = np.ones((7,7))
                    ker_9 = np.ones((9,9))
                    
                    if "5.67MPa" in smaple_name:
                        cls_img = cv2.erode(cls_img, ker_5, iterations=2)
                        cls_img = cv2.dilate(cls_img, ker_7, iterations=2)
                        for ct in range(2):
                            cls_img = cv2.blur(cls_img, (19,19)) 
                    else:
                        cls_img = cv2.erode(cls_img, ker_3, iterations=2)
                        cls_img = cv2.dilate(cls_img, ker_5, iterations=2)
                        for ct in range(2):
                            cls_img = cv2.blur(cls_img, (11,11)) 

                    _, cls_img = cv2.threshold(cls_img, 150, 255, cv2.THRESH_BINARY)


                    """
                    if "5.67MPa" in smaple_name:
                        kernel = np.ones((9, 9), np.uint8)     
                    if "5.67MPa" in smaple_name:
                        ksize = (19, 19)
                    """

                    # check for connected regions and seperate disconnected grains 
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cls_img, connectivity=4)
                    binary_masks = []
                    if num_labels>1:
                        for label in range(1, num_labels):  # Start from 1 to exclude the background label (0)
                            mask = (labels == label).astype(np.uint8) * 255
                            binary_masks.append(mask)
                    else:
                        binary_masks.append((cls_img==255).astype(np.uint8) * 255)

                    # measure area and make boundary 
                    for mi, mask in enumerate(binary_masks):
                        plt.imshow(mask)
                        plt.savefig(os.path.join(clst_folder, 'grain_{}_{}.png'.format(i,mi)))
                        mask = ((np.array(mask)//255).astype(np.float64))
                        mask = cv2.resize(mask, (raw_inp_img.shape[1], raw_inp_img.shape[0]), interpolation = cv2.INTER_LINEAR)
                        
                        edg_mask = cv2.Canny((mask*255).astype(np.uint8), threshold1=100, threshold2=200)
                        all_masks+=edg_mask
                        
                        area_ratio.append(np.count_nonzero(mask>0)*cnt_area_unt) 
                        area_ratio_smaple_level.append(np.count_nonzero(mask>0)*cnt_area_unt) 
                        
                cls_area_ratios[no_cls]=area_ratio
                if no_cls not in smaple_area_ratios.keys():
                    smaple_area_ratios[no_cls] = np.array(area_ratio_smaple_level)
                else:
                    smaple_area_ratios[no_cls] = np.append(smaple_area_ratios[no_cls], np.array(area_ratio_smaple_level), axis=0)

                overlay_img = overlay_images(raw_inp_img, all_masks, alpha=0.6)
                
                plt.imshow(np.array(overlay_img))
                plt.savefig(os.path.join(clst_folder ,'overlay_predictions.png'))
                plt.clf()
                
            
            cls_area_ratios = dict(sorted(cls_area_ratios.items()))
            scatter_plot_from_dict(cls_area_ratios, smaple_folder, msmt_folder_name)
        
        smaple_area_ratios = dict(sorted(smaple_area_ratios.items()))
        consol_values = smaple_area_ratios.values()
        consol_values = [x for xs in consol_values for x in xs]

        new_cons = dict()
        new_cons["all"] = list(consol_values)
        box_plot_from_dict(new_cons, smaple_folder, "all_consolidated")
            
        print("-"*50)
        print("Name of sample - ",smaple_name)
        print("Stats")
        print("Mean grain size - ", np.mean(np.array(consol_values)))
        print("Median grain size - ", np.median(np.array(consol_values)))
        print("Std grain size - ", np.std(np.array(consol_values)))     

                
        # make image with grain boundaries overlay 
        weights = np.ones_like(consol_values)/len(consol_values)

        consol_values = np.array(list(consol_values))
        #consol_values[consol_values>=0.0005]

        hist, bins = np.histogram(list(consol_values), bins=[0.005,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4], weights=weights)
        #hist, bins = np.histogram(list(consol_values), bins="auto")#, weights=weights)

        np.save("{}_data.npz".format(smaple_name), np.array([(hist),(bins[:-1])]))
        
