
# coding: utf-8


import numpy as np
import scipy.io as sio
import scipy
import scipy.spatial
from scipy.linalg import norm
np.set_printoptions(threshold=100)

mat_contents = sio.loadmat('umist_cropped.mat')
facedat = mat_contents['facedat']

people = facedat[0]
people1 = facedat[0][0]
people2 = facedat[0][1]
people1_face1 = facedat[0][0][:,:,0]
people2_face2 = facedat[0][1][:,:,1]

vec_people1_face1 = people1_face1.flatten()
vec_people2_face2 = people2_face2.flatten()
pixels=112*92


people1_pic_size = facedat[0][0][:][91][1].size
print people1_pic_size
mat_people1 = np.zeros((people1_pic_size,pixels))
for i in range(0,people1_pic_size):
        mat_people1[i]= facedat[0][0][:,:,i].flatten()# mat_people1 is matrix describe the the class1 matrix(size,112*92) 
vec_people1 = mat_people1.flatten()
#for test
#print mat_people1[0].reshape(112,92)

people2_pic_size = facedat[0][1][:][91][1].size
print people2_pic_size
mat_people2 = np.zeros((people2_pic_size,pixels))
for i in range(0,people2_pic_size):
        mat_people2[i]= facedat[0][1][:,:,i].flatten()# mat_people2 is matrix describe the the class2 matrix(size,112*92)
vec_people2 = mat_people2.flatten()
#for test
#print mat_people2[0].reshape(112,92)

def cal_distance_mat_people12(mat_peoplei,mat_peoplej):
    sizei = mat_peoplei.shape[0]
    sizej = mat_peoplej.shape[0]
    print sizei,sizej
    size = sizei+sizej
    print size
    mat_dist = np.zeros((size,size),dtype='f') 
    mat_people12=np.row_stack((mat_peoplei,mat_peoplej))
    for i in range(0,size):
        for j in range(0,size):
            mat_dist[i][j]= scipy.spatial.distance.euclidean(mat_people12[i],mat_people12[j])
    return mat_dist # the output is the distance matrix of the people1 and people2

mat_dist=cal_distance_mat_people12(mat_people1,mat_people2)
print mat_dist
print mat_dist[0][35]


#it calculate the people ith' class distance
# return the sum of the class distance and the vector of the class distance

def cal_mat_class_distance(mat_peoplei):  
    rows = mat_peoplei.shape[0]
    dist = 0
    num = 0
    vec_class_distance = np.zeros((rows*(rows-1)/2),dtype='f')
    for i in range(0,rows-1):
        for j in range(i+1,rows):
            vec_class_distance[num] = scipy.spatial.distance.euclidean(mat_peoplei[i],mat_peoplei[j])
            dist += scipy.spatial.distance.euclidean(mat_peoplei[i],mat_peoplei[j])
            num += 1          
    return dist,vec_class_distance


print cal_mat_class_distance(mat_people1)

#calculate the dataset's average class distance
def cal_av_mat_class_distance (mat_people):
    dist = 0
    num = 0
    for item in mat_people:
        people_item_pic_size = item[:][91][1].size
        num += people_item_pic_size * (people_item_pic_size-1)/2
        # mat_people_item is the matrix describe the item th class
        mat_people_item = item.flatten().reshape(people_item_pic_size,112*92)
        vec_people_item = item.flatten()
        result_dist,result_vetor =  cal_mat_class_distance(mat_people_item)
        dist += result_dist
    av_dist = dist / num
    return av_dist

av_mat_class_distance = cal_av_mat_class_distance(people) 
print av_mat_class_distance


dist1,vec_class_distance_people1 = cal_mat_class_distance(mat_people1) 
std1 = np.std(vec_class_distance_people1)# std of the people1
dist2,vec_class_distance_people2 = cal_mat_class_distance(mat_people2) 
std2 = np.std(vec_class_distance_people2) # std of the people2
std12 = (std1+std2)/2 # average std between people1 and people2
print std12


mat_W = np.exp(-mat_dist/std12) #mat_W is the similarity matrix , the kernal function is  exp(-d/std)
print mat_W

#only for test
print mat_W.shape


import python_ncut_lib as nc # import the normalized cut 
#unlimited display
#np.set_printoptions(threshehold = np.nan)
nbEigen = 3
eigen_value,vector=nc.ncut(mat_W,nbEigen)
vec_dis = nc.discretisation(vector)
print eigen_value
print vec_dis
