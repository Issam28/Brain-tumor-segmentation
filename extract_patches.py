import random
#from skimage import io
import numpy as np
from glob import glob
import SimpleITK as sitk
from keras.utils import np_utils

class Pipeline(object):

    
    def __init__(self, list_train ,Normalize=True):
        self.scans_train = list_train
        self.train_im=self.read_scans(Normalize)

        
    def read_scans(self,Normalize):

        train_im=[]
        for i in range(len( self.scans_train)):
            if i%10==0:
                print('iteration [{}]'.format(i))

            flair = glob( self.scans_train[i] + '/*_flair.nii.gz')
            t2 = glob( self.scans_train[i] + '/*_t2.nii.gz')
            gt = glob( self.scans_train[i] + '/*_seg.nii.gz')
            t1 = glob( self.scans_train[i] + '/*_t1.nii.gz')
            t1c = glob( self.scans_train[i] + '/*_t1ce.nii.gz')

            t1s=[scan for scan in t1 if scan not in t1c]

            if (len(flair)+len(t2)+len(gt)+len(t1s)+len(t1c))<5:
                print("there is a problem here!!! the problem lies in this patient :", self.scans_train[i])
                continue
            scans = [flair[0], t1s[0], t1c[0], t2[0], gt[0]]
            
            #read a volume composed of 4 modalities
            tmp = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))]

            #crop each volume to have a size of (146,192,152) to discard some unwanted background and thus save some computational power ;)
            z0=1
            y0=29
            x0=42
            z1=147
            y1=221  
            x1=194  
            tmp=np.array(tmp)
            tmp=tmp[:,z0:z1,y0:y1,x0:x1]

            #normalize each slice
            if Normalize==True:
                tmp=self.norm_slices(tmp)

            train_im.append(tmp)
            del tmp    
        return  np.array(train_im)
    
    
    def sample_patches_randomly(self, num_patches, d , h , w ):

        '''
        INPUT:
        num_patches : the total number of samled patches
        d : this correspnds to the number of channels which is ,in our case, 4 MRI modalities
        h : height of the patch
        w : width of the patch
        OUTPUT:
        patches : np array containing the randomly sampled patches
        labels : np array containing the corresping target patches
        '''
        patches, labels = [], []
        count = 0

        #swap axes to make axis 0 represents the modality and axis 1 represents the slice. take the ground truth
        gt_im = np.swapaxes(self.train_im, 0, 1)[4]   

        #take flair image as mask
        msk = np.swapaxes(self.train_im, 0, 1)[0]
        #save the shape of the grounf truth to use it afterwards
        tmp_shp = gt_im.shape

        #reshape the mask and the ground truth to 1D array
        gt_im = gt_im.reshape(-1).astype(np.uint8)
        msk = msk.reshape(-1).astype(np.float32)

        # maintain list of 1D indices while discarding 0 intensities
        indices = np.squeeze(np.argwhere((msk!=-9.0) & (msk!=0.0)))
        del msk

        # shuffle the list of indices of the class
        np.random.shuffle(indices)

        #reshape gt_im
        gt_im = gt_im.reshape(tmp_shp)

        #a loop to sample the patches from the images
        i = 0
        pix = len(indices)
        while (count<num_patches) and (pix>i):
            #randomly choose an index
            ind = indices[i]
            i+= 1
            #reshape ind to 3D index
            ind = np.unravel_index(ind, tmp_shp)
            # get the patient and the slice id
            patient_id = ind[0]
            slice_idx=ind[1]
            p = ind[2:]
            #construct the patch by defining the coordinates
            p_y = (p[0] - (h)/2, p[0] + (h)/2)
            p_x = (p[1] - (w)/2, p[1] + (w)/2)
            p_x=list(map(int,p_x))
            p_y=list(map(int,p_y))
            
            #take patches from all modalities and group them together
            tmp = self.train_im[patient_id][0:4, slice_idx,p_y[0]:p_y[1], p_x[0]:p_x[1]]
            #take the coresponding label patch
            lbl=gt_im[patient_id,slice_idx,p_y[0]:p_y[1], p_x[0]:p_x[1]]

            #keep only paches that have the desired size
            if tmp.shape != (d, h, w) :
                continue
            patches.append(tmp)
            labels.append(lbl)
            count+=1
        patches = np.array(patches)
        labels=np.array(labels)
        return patches, labels
        
        

    def norm_slices(self,slice_not): 
        '''
            normalizes each slice , excluding gt
            subtracts mean and div by std dev for each slice
            clips top and bottom one percent of pixel intensities
        '''
        normed_slices = np.zeros(( 5,146, 192, 152)).astype(np.float32)
        for slice_ix in range(4):
            normed_slices[slice_ix] = slice_not[slice_ix]
            for mode_ix in range(146):
                normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])
        normed_slices[-1]=slice_not[-1]

        return normed_slices    
   


    def _normalize(self,slice):
        '''
            input: unnormalized slice 
            OUTPUT: normalized clipped slice
        '''
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]
        if np.std(slice)==0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp= (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            #since the range of intensities is between 0 and 5000 ,the min in the normalized slice corresponds to 0 intensity in unnormalized slice
            #the min is replaced with -9 just to keep track of 0 intensities so that we can discard those intensities afterwards when sampling random patches
            tmp[tmp==tmp.min()]=-9
            return tmp


'''
def save_image_png (img,output_file="img.png"):
    """
    save 2d image to disk in a png format
    """
    img=np.array(img).astype(np.float32)
    if np.max(img) != 0:
        img /= np.max(img)   # set values < 1                  
    if np.min(img) <= -1: # set values > -1
        img /= abs(np.min(img))
    io.imsave(output_file, img)
'''


    
def concatenate ():

    '''
    concatenate two parts into one dataset
    this can be avoided if there is enough RAM as we can directly from the whole dataset
    '''
    Y_labels_2=np.load("y_dataset_second_part.npy").astype(np.uint8)
    X_patches_2=np.load("x_dataset_second_part.npy").astype(np.float32)
    Y_labels_1=np.load("y_dataset_first_part.npy").astype(np.uint8)
    X_patches_1=np.load("x_dataset_first_part.npy").astype(np.float32)

    #concatenate both parts
    X_patches=np.concatenate((X_patches_1, X_patches_2), axis=0)
    Y_labels=np.concatenate((Y_labels_1, Y_labels_2), axis=0)
    del Y_labels_2,X_patches_2,Y_labels_1,X_patches_1

    #shuffle the whole dataset
    shuffle = list(zip(X_patches, Y_labels))
    np.random.seed(138)
    np.random.shuffle(shuffle)
    X_patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
    Y_labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
    del shuffle


    np.save( "x_training",X_patches.astype(np.float32) )
    np.save( "y_training",Y_labels.astype(np.uint8))
    #np.save( "x_valid",X_patches_valid.astype(np.float32) )
    #np.save( "y_valid",Y_labels_valid.astype(np.uint8))


if __name__ == '__main__':
    
    #Paths for Brats2017 dataset
    path_HGG = glob('Brats2017/Brats17TrainingData/HGG/**')
    path_LGG = glob('Brats2017/Brats17TrainingData/LGG/**')
    path_all=path_HGG+path_LGG

    #shuffle the dataset
    np.random.seed(2022)
    np.random.shuffle(path_all)

    np.random.seed(1555)
    start=0
    end=20
    #set the total number of patches
    #this formula extracts approximately 3 patches per slice
    num_patches=146*(end-start)*3
    #define the size of a patch
    h=128
    w=128 
    d=4 

    pipe=Pipeline(list_train=path_all[start:end],Normalize=True)
    Patches,Y_labels=pipe.sample_patches_randomly(num_patches,d, h, w)

    #transform the data to channels_last keras format
    Patches=np.transpose(Patches,(0,2,3,1)).astype(np.float32)

    # since the brats2017 dataset has only 4 labels,namely 0,1,2 and 4 as opposed to previous datasets 
    # this transormation is done so that we will have 4 classes when we one-hot encode the targets
    Y_labels[Y_labels==4]=3

    #transform y to one_hot enconding for keras  
    shp=Y_labels.shape[0]
    Y_labels=Y_labels.reshape(-1)
    Y_labels = np_utils.to_categorical(Y_labels).astype(np.uint8)
    Y_labels=Y_labels.reshape(shp,h,w,4)

    #shuffle the whole dataset
    shuffle = list(zip(Patches, Y_labels))
    np.random.seed(180)
    np.random.shuffle(shuffle)
    Patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
    Y_labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
    del shuffle
    
    print("Size of the patches : ",Patches.shape)
    print("Size of their correponding targets : ",Y_labels.shape)

    #save to disk as npy files
    #np.save( "x_dataset_first_part",Patches )
    #np.save( "y_dataset_first_part",Y_labels)


