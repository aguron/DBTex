from math import log
import numpy as np
from skimage import filters
from skimage.morphology import remove_small_objects, remove_small_holes, skeletonize, reconstruction, extrema, convex_hull_image
from skimage.morphology import opening, closing, square, dilation,disk, binary_erosion,binary_dilation, binary_opening, erosion
from skimage.transform import pyramid_reduce, pyramid_expand
from skimage.measure import regionprops
from skimage.feature import greycomatrix,greycoprops
from skimage.segmentation import watershed, find_boundaries
from skimage.util import img_as_float
from skimage.feature.blob import _prune_blobs
from skimage.feature import peak_local_max
from skimage import exposure
from scipy import ndimage
from scipy import optimize
from sklearn.cluster import DBSCAN
import collections
import cv2

def bad_image_filter(image):
    '''
    Recognizes bad mammograms based on Otsu's thresholding.
    Method:
    1. Apply Otsu's thresholding to binarize image
    2. Denoise image, remove small holes/objects
    3. Find largest two connected components 
    Assumption: These correspond to background and breast tissue
    4. Find which of these two components occupies the most area
    in the lowest quartile.
    Assumption: This component is the background which means the 
    other is the breast tissue
    5. Select the breast tissue component
    6. Find if the component does not touch the left/right boundary
    Assumption: In this case the MG image is bad

    Args:
        image (ndarray): grayscale image as ndarray
    Returns:
        True if image bad
        False, side of the tissue otherwise. side='L' or 'R'
    '''

    # Otsu's thresholding and corresponding mask
    thr = filters.threshold_otsu(image)
    image = image>thr
    # Denoising
    image = remove_small_objects(remove_small_holes(image))
    #Extract connected components
    image, nlabels = ndimage.label(image)
    # Find size of each component
    labels, size = np.unique(image, return_counts=True)
    labels_by_size = np.argsort(-size) # Descending order
    # Assumption: The two largest components correspond to background
    # and breast tissue 
    # Choosing breast:
    # Assumption: More background rather than breast in lowest quartile
    lowest_quartile_point = int(image.shape[0]*3/4)
    slicing = image[lowest_quartile_point:,:]
    object0_size = (slicing == labels_by_size[0]).sum()
    object1_size = (slicing == labels_by_size[1]).sum()
    if object0_size>object1_size:
        breast_label = labels_by_size[1]
    else:
        breast_label = labels_by_size[0]
    
    # Breast tissue mask
    image = image == breast_label
    # Declare bad image if breast_component is not 
    # touching the left/right boundary
    slice_x, slice_y = ndimage.find_objects(image)[0]
    if not(slice_y.start==0 or slice_y.stop== image.shape[1]):
        return True, 'U' # Side is unknown

    if slice_y.start==0:
        side = 'L'
    else:
        side = 'R'
    return False, side

def delineate_breast(image, otsu_percent=0.2, bdry_remove=0.0, erosion=0, return_slice = False):
    '''Given a mammogram image, it outputs the delineation mask
    Args:
        image (np.array, 2-dim): breast image
        otsu_percent (float, optional): thresholding factor (threshold = otsu_percent*otsu_threshold)
        bdry_remove (float, optional): percentage of dimension to be removed
        return_slice (bool, optional): if true performs erosion of the object
    Returns: 
        binary mask of breast (and optionally slices describing the bounding box)
    '''

    image_original = image
    thr = filters.threshold_otsu(image)
    image = image>otsu_percent*thr
    # Denoising
    image = remove_small_objects(remove_small_holes(image))
    image, nlabels = ndimage.label(image)
    # Find size of each component
    labels, size = np.unique(image, return_counts=True)
    labels_by_size = np.argsort(-size) # Descending order
    # Assumption: The two largest components correspond to background
    # and breast tissue
    # Assumption: Out of the two the component with the highest median 
    # intensity is the breast
    c0 = image_original[image==labels_by_size[0]]
    c1 = image_original[image==labels_by_size[1]]
    c0_brightness = np.median(c0)
    c1_brightness = np.median(c1)
    if c1_brightness>c0_brightness:
        mask = (image==labels_by_size[1])
    else:
        mask = (image==labels_by_size[0])
    # Fill holes within structure
    mask = ndimage.binary_fill_holes(mask)

    # if gives all image fails
    #if np.sum(mask) == mask.shape[0]*mask.shape[1]:
    #    return None 

    # remove mask on the image boundaries
    h, w = mask.shape
    h_remove, w_remove = int(bdry_remove*h), int(bdry_remove*w)

    mask[0:h_remove,:]=0
    mask[h-h_remove:h,:]=0
    mask[:,0:w_remove]=0
    mask[:,w-w_remove:w]=0

    mask = binary_erosion(mask,disk(erosion))

    if return_slice:       
        return mask, ndimage.find_objects(mask)[0]
    else:
        return mask

def detect_calcifications(img, thr=10.):
    """Morphological calcification detector, using pyramid scheme
    Args:
        img (np.array, 2-dim): breast image
        thr (float, 0.0-256.0, optional): for thresholding the filters
    Returns:
        mask: binary np.array
    """
    #pyramid
    img1 = opening(closing(img, square(3)),square(3))
    img1 = pyramid_reduce(img1, multichannel=False)
    img2 = opening(closing(img1, square(3)),square(3))
    img2 = pyramid_reduce(img2, multichannel=False)
    # filters
    filter1 = img-np.minimum(pyramid_expand(img1, multichannel=False), img)
    filter2 = img-np.minimum(pyramid_expand(img2,upscale=4, multichannel=False), img)
    #thresholding
    filter1 = filter1>(thr/256.)
    filter2 = filter2>(thr/256.)
    #combine
    filter_comb = np.logical_or(filter1,filter2)

    mask = remove_small_objects(filter_comb, min_size=3)
    mask = dilation(mask,disk(1))
    return mask

def emax(image, h=5):
    """Finds extended maximum
    """
    image = (image*255).astype(int)
    hmax = reconstruction(image-h, image, method='dilation')
    rmax =  hmax+1-reconstruction(hmax, hmax+1, method='dilation')
    return rmax

def rmin(image):
    image = (image*255).astype(int)
    return reconstruction(image+1, image, method='erosion')-image

def grad(image):
    return dilation(image,square(3))-erosion(image,square(3))

def min_imposition(image, marker):
    fm = (~marker)*image.max()
    pm = np.minimum(fm, image) #pointwise minimum
    return reconstruction(fm, pm, method='erosion')

def detect_calcifications_Ciecholewski(img, thr=10., h=5,):
    """Morphological calcification detector, using pyramid scheme
    Args:
        img (np.array, 2-dim): breast image
        thr (float, 0.0-256.0, optional): for thresholding the filters
    Returns:
        mask: binary np.array
    """
    ### Stage 1
    #### Step 1
    i2 = img*(img>21/255)
    #### Step 2
    #pyramid
    img1 = opening(closing(img, square(3)),square(3))
    img1 = pyramid_reduce(img1, multichannel=False)
    img2 = opening(closing(img1, square(3)),square(3))
    img2 = pyramid_reduce(img2, multichannel=False)
    # filters
    filter1 = img-np.minimum(pyramid_expand(img1, multichannel=False), img)
    filter2 = img-np.minimum(pyramid_expand(img2,upscale=4, multichannel=False), img)
    #thresholding
    filter1 = filter1>(thr/256.)
    filter1 = remove_small_objects(filter1,3)
    filter2 = filter2>(thr/256.)
    filter2 = remove_small_objects(filter2,3)
    #### Step 3
    # extended max
    iemax = emax(i2,h)

    #### Step 4
    marker1 = np.logical_and(filter1,iemax)

    marker2 = np.logical_and(filter2,iemax)

    result1 = reconstruction(marker1, iemax, method='dilation')
    result2 = reconstruction(marker2, iemax, method='dilation')
    result = np.logical_or(result1,result2)
    #### Filter objects
#     result = remove_small_objects(result,5)
#     small_objects = remove_small_objects(result,100)
#     result = np.logical_xor(result,small_objects)
    ### Stage 2
    # CO filtering
    img_co = opening(closing(i2, square(3)),square(3))
    # Inversion
    img_co = 1.-img_co

    img_min = rmin(img_co)

    # Internal marker
    int_marker = np.logical_and(img_min,result) 
    # Gradient of inverted image
    img_grad = grad(img_co) 
    # External marker with watershed
    ws_markers_bool = extrema.local_minima(img_co)
    ws_markers = ndimage.label(ws_markers_bool)[0]
    ext_marker = find_boundaries(watershed(img_co, markers=ws_markers), mode='outer')
    # Dilating external marker
    ext_marker_dil = binary_dilation(ext_marker,square(3))
    # Finding non-intersecting part of internal marker with external dilated marker
    intersection = np.logical_and(int_marker,ext_marker_dil)
    int_marker2 = np.logical_xor(int_marker,intersection)
    # Combine markers
    marker = np.logical_or(int_marker2,ext_marker)
    
    # Minimum imposition on gradient
    img_min_impo = min_imposition(img_grad,marker)
    # Watershed segmentation
    img_watershed = watershed(img_min_impo, markers=ndimage.label(int_marker2)[0])
    max_area=5000
    label_list = list()
    for rprop in regionprops(img_watershed):
        if rprop.area<=max_area:
            label_list.append(rprop.label)
    final_mask = np.isin(img_watershed,label_list)
    return final_mask

def create_patches(img, patch_size):
    """Creating patches, padding appropriately the image first. 
    TO GET DEPRECIATED
    """
    # add padding
    h, w = img.shape
    h_patch, w_patch = patch_size
    h_pad, w_pad = h_patch-np.mod(h,h_patch), w_patch-np.mod(w,w_patch)
    img_pad = np.zeros((h+h_pad, w+w_pad))
    img_pad[0:h,0:w] = img
    # create patches
    n_rows = int((h+h_pad)/h_patch)
    n_columns = int((w+w_pad)/w_patch)
    #initialize
    patches = np.zeros((n_rows,n_columns,patch_size[0], patch_size[1]))
    for i in range(0,n_rows):
        for j in range(0,n_columns):
            top = i*patch_size[0]
            bottom = (i+1)*patch_size[0]
            left = j*patch_size[1]
            right = (j+1)*patch_size[1] 
            patches[i,j] = img_pad[top:bottom,left:right]
    return patches, (h_pad, w_pad)

def image_from_patches(patches, padding):
    """Gluing patches to output whole image. 
    TO GET DEPRECIATED
    """
    h_patch, w_patch = patches.shape[2], patches.shape[3]
    n_rows, n_columns = patches.shape[0], patches.shape[1]
    h_pad, w_pad = padding
    # initialize image
    img = np.zeros((n_rows*h_patch, n_columns*w_patch))
    # combine patches
    for i in range(0,n_rows):
        for j in range(0,n_columns):
            top = i*h_patch
            bottom = (i+1)*h_patch
            left = j*w_patch
            right = (j+1)*w_patch
            img[top:bottom,left:right] = patches[i,j]
    # remove added padding
    h, w = img.shape
    img = img[0:h-h_pad,0:w-w_pad]
    
    return img

def detect_calcifications_whole_image(img, thr=10., erosion=10, method='pyramid'):
    """Morphological segmentation of calcifications
    TO GET DEPRECIATED
    """
    patches, padding = create_patches(img, (500,500))
    mask_patches = np.zeros_like(patches,dtype=int) 
    n_rows, n_columns = patches.shape[0], patches.shape[1]
    for i in range(0,n_rows):
        for j in range(0,n_columns):
            if method=='pyramid':
                mask_patches[i,j]=detect_calcifications(patches[i,j], thr=thr)
            else:
                mask_patches[i,j]=detect_calcifications_Ciecholewski(patches[i,j], thr=thr)
    #recostruct image
    mask_calc = image_from_patches(mask_patches, padding)
    mask_breast = delineate_breast(img, bdry_remove=0.)
    mask_breast = binary_erosion(mask_breast,disk(erosion))
    mask = np.logical_and(mask_calc,mask_breast)
    mask = ndimage.binary_fill_holes(mask)
    return mask
    

def make_exponential_mask(img, locations, radius, alpha, INbreast=False):
    """Creating exponential proximity function mask.
    Args:
        img (np.array, 2-dim): the image, only it's size is important
        locations (np.array, 2-dim): array should be (n_locs x 2) in size and 
            each row should correspond to a location [x,y]. Don't need to be integer,
            truncation is applied.
            NOTICE [x,y] where x is row number (distance from top) and y column number
            (distance from left)
        radius (int): radius of the exponential pattern
        alpha (float): decay rate
        INbreast (bool, optional): Not needed anymore, handled when parsing INbreast dataset
    Returns:
        mask (np.array, 0.0-1.0): Exponential proximity function
    """
    # create kernel which we will be adding at locations
    # Kernel has radial exponential decay form
    kernel = np.zeros((2*radius+1,2*radius+1))
    for i in range(0, kernel.shape[0]):
        for j in range(0, kernel.shape[1]):
            d = np.sqrt((i-radius)**2+(j-radius)**2)
            if d<= radius:
                kernel[i,j]=(np.exp(alpha*(1-d/radius))-1)/(np.exp(alpha)-1)
                
    # pad original img to avoid out of bounds errors
    img  = np.pad(img, radius+1, 'constant').astype(float)

    # update locations
    locations = np.array(locations)+radius+1
    locations = np.round(locations).astype(int)

    # initialize mask
    mask = np.zeros_like(img)    

    for location in locations:
        if INbreast:
            y, x = location
        else:
            x, y = location
        # add kernel
        mask[x-radius:x+radius+1, y-radius:y+radius+1] =np.maximum(mask[x-radius:x+radius+1, y-radius:y+radius+1],kernel)
        
    # unpad
    mask  = mask[radius+1:-radius-1,radius+1:-radius-1]
    
    return mask

def make_exponential_mask_from_binary_mask(mask, radius, alpha):
    """Creating exponential proximity function mask given the binary mask.
    Args:
        mask (np.array, 2-dim): binary mask
        radius (int): radius of the exponential pattern
        alpha (float): decay rate
    Returns:
        mask (np.array, 0.0-1.0): Exponential proximity function
    """
    locations = np.array(np.where(mask)).T
    mask = make_exponential_mask(mask, locations, radius, alpha, INbreast=False)
    
    return mask


def make_exponential_mask_from_binary_mask_inwards(mask, radius, alpha):
    """Creating exponential proximity function mask given the binary mask.
    Exponential pattern starts from the interior of large objects and end on it's boundary
    For the small objects pattern starts from the boundary and goes outwards.
    Args:
        mask (np.array, 2-dim): binary mask
        radius (int): radius of the exponential pattern
        alpha (float): decay rate
    Returns:
        mask (np.array, 0.0-1.0): Exponential proximity function
    """
    mask = mask.astype(bool)
    large_object_mask = remove_small_objects(mask,(2*radius)**2+1)
    small_object_mask = np.logical_xor(mask,large_object_mask)
    large_object_mask = binary_erosion(large_object_mask, disk(radius))
    mask = np.logical_or(large_object_mask,small_object_mask)
    locations = np.array(np.where(mask)).T
    mask = make_exponential_mask(mask, locations, radius, alpha, INbreast=False)
    
    return mask

def make_exponential_mask_from_binary_mask_positive_negative(mask, radius, alpha):
    """Creating exponential proximity function mask given the binary mask.
    Exponential pattern starts from the interior of large objects and end on it's boundary
    For the small objects pattern starts from the boundary and goes outwards.
    Args:
        mask (np.array, 2-dim): binary mask
        radius (int): radius of the exponential pattern
        alpha (float): decay rate
    Returns:
        mask (np.array, 0.0-1.0): Exponential proximity function
    """
    mask = mask.astype(bool)
    large_object_mask = remove_small_objects(mask,2)
    small_object_mask = np.logical_xor(mask,large_object_mask)
    locations = np.array(np.where(large_object_mask)).T
    large_object_mask = make_exponential_mask(large_object_mask, locations, radius, alpha, INbreast=False)
    locations = np.array(np.where(small_object_mask)).T
    small_object_mask = make_exponential_mask(small_object_mask, locations, radius, alpha, INbreast=False)
    mask = large_object_mask-small_object_mask
    return mask


def make_exponential_mask_from_binary_mask_points(mask, radius, alpha):
    """Creating exponential proximity function mask given the binary mask.
    Exponential pattern starts from the interior of large objects and end on it's boundary
    For the small objects pattern starts from the boundary and goes outwards.
    Args:
        mask (np.array, 2-dim): binary mask
        radius (int): radius of the exponential pattern
        alpha (float): decay rate
    Returns:
        mask (np.array, 0.0-1.0): Exponential proximity function
    """
    mask = mask.astype(bool)
    label, _ = ndimage.label(mask)
    locations = list()
    for rg in regionprops(label):
        locations.append(rg.centroid)
    locations = np.array(locations).astype(int)
    mask = make_exponential_mask(mask, locations, radius, alpha, INbreast=False)
    return mask


def make_exponential_mask_from_binary_mask_points_only(mask, radius, alpha):
    """Creating exponential proximity function mask given the binary mask.
    Exponential pattern starts from the interior of large objects and end on it's boundary
    For the small objects pattern starts from the boundary and goes outwards.
    Args:
        mask (np.array, 2-dim): binary mask
        radius (int): radius of the exponential pattern
        alpha (float): decay rate
    Returns:
        mask (np.array, 0.0-1.0): Exponential proximity function
    """
    mask = mask.astype(bool)
    large_object_mask = remove_small_objects(mask,2)
    small_object_mask = np.logical_xor(mask,large_object_mask)
    locations = np.array(np.where(small_object_mask)).T
    small_object_mask = make_exponential_mask(small_object_mask, locations, radius, alpha, INbreast=False)
    return small_object_mask


def inverse_exponential_transform(img, radius, alpha):
    """Apply inverse transformation on img array
    """
    N = np.exp(alpha)-1
    r = radius*(1-(1/alpha)*np.log(np.abs(1+img*N)))
    return r

def get_bounding_boxes(img, kernel_size, stride):
    """Gives all bounding boxes covering an image with a sliding window
    Args:
        img (np.array or torch.tensor)
        kernel_size (int): size of the kernel, kernel is a square
        stride (int): stride of the sliding window motion
    Returns:
        bounding boxes list (list of (x, xp,y, yp)), convention of bounding box
        (top, bottom, left, right)
    Notice: If not all image covered by sliding window, additional
    bounding boxes are created ending at the end of each dimension
    """
    h, w = img.shape
    h_p, w_p = kernel_size, kernel_size
    s = stride

    # All x locations
    x = [i*s for i in range(0, int((h-h_p)/s)+1)]
    # Add one more point if didn't cover all range
    if x[-1]+h_p!=h:
        x.append(h-h_p)
    # All y locations
    y = [j*s for j in range(0, int((w-w_p)/s)+1)]
    # Add one more point if didn't cover all range
    if y[-1]+w_p!=w:
        y.append(w-w_p)

    # All bounding boxes in the form (x,xp,y,yp)
    # x,y: the top left corner/ xp,yp: the bottom right corner
    bbList = [(xi,xi+h_p, yi, yi+w_p) for xi in x for yi in y]

    return bbList

def mask_filter_bounding_boxes(mask, bbList):
    """Keeps only bounding boxes that have overlap with mask
    Args:
        mask (np.array, 2-dim): binary mask
        bbList (list of (x,xp,y,yp))
    Returns:
        reduced bounding boxes list (list of (x, xp,y, yp)), convention of bounding box
        (top, bottom, left, right)
    """
    bbListFiltered = []
    for bb in bbList:
        mask_patch = mask[bb[0]:bb[1],bb[2]:bb[3]]
        if (mask_patch).sum() > 0:  
            bbListFiltered.append(bb)
    return bbListFiltered



def delineate_paddle(img, side = 'L'):

    img = exposure.equalize_adapthist(img)
    img = filters.median(img)
    thr = filters.threshold_otsu(img)
    img = img>thr
    # Denoise
    img = binary_erosion(img,disk(10))
    img = binary_erosion(img,disk(10))
    img = binary_erosion(img,disk(10))
    img = binary_erosion(img,disk(10))
    img = binary_erosion(img,disk(10))
    img = binary_erosion(img,disk(10))
    img = binary_erosion(img,disk(10))
    img = remove_small_objects(img, min_size=200)
    img = binary_opening(img,disk(10))
    img = binary_opening(img,disk(10))
    img = binary_opening(img,disk(10))
    img = binary_opening(img,disk(10))
    # Skeletonize to get lines of the objects
    img = skeletonize(img)
    # Remove irrelevant parts
    # Upper 10%, Lower 10%, Middle 20%, Left or right 30% (depending on side)
    h = img.shape[0]
    w = img.shape[1]
    h_rem = int(0.1*h)
    w_rem =int(0.3*w)
    h_cent = int(h/2)
    img[:h_rem]=False
    img[-h_rem:]=False
    img[h_cent-h_rem:h_cent+h_rem]=False
    if side == 'L':
        img[:,-w_rem:]=False
    else:
        img[:,:w_rem]=False
    # Fitting circle on remaining skeleton
    locs = np.argwhere(img)
    x, y = locs[:,0], locs[:,1]
    x_m, y_m = x.mean(), y.mean()
    tupl_initial = (x_m, y_m, 1000.)
    tupl_final, success = optimize.leastsq(funcCircle, tupl_initial, args=(x,y))

    return tupl_final

def funcCircle(tupl, x, y):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return (x-tupl[0])**2+(y-tupl[1])**2-tupl[2]**2


def dbscan_clustering(mask, eps=72, min_samples=3, return_cluster_centroids=False):
    '''Uses DBSCAN to map a binary mask into clusters
    Args:
        mask (2d np.array): binary mask
        eps (float, optional): dbscan parameter, default is ~0.5cm in INbreast dataset
    Returns:
        cluster_label (2d np.array): multi-valued array representing clusters 
        n_clusters: number of clusters found
        n_noise: number of objects without cluster assignment
    '''
    if mask.sum()==0.:
        return mask,0,0
    else:
        mask = mask.astype(bool)
    points = []
    bboxes = []
    orig_labels = []

    label,_ = ndimage.label(mask)
    regions = regionprops(label)
    for rg in regions:
        x,y = rg.centroid
        points.append([x,y])
        bb = rg.bbox
        bb = (bb[0],bb[2],bb[1],bb[3])
        bboxes.append(bb)
        orig_labels.append(rg.label)
    points = np.array(points)

    X= points
    db=DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    n_noise = np.sum(db.labels_==-1)
    n_clusters = np.sum(np.unique(db.labels_)>=0)
    new_labels=db.labels_+2
    
    if return_cluster_centroids:
        cluster_label = np.zeros_like(mask)
        for i in range(0,n_clusters):
            points = np.array(points)
            p =np.round(np.mean(points[db.labels_==i], axis=0)).astype(int)
            cluster_label[p[0],p[1]] = 1
        return cluster_label, n_clusters, n_noise
    cluster_label=label.copy()
    for bb, orig_lbl, new_lbl in zip(bboxes,orig_labels,new_labels):
        patch = cluster_label[bb[0]:bb[1],bb[2]:bb[3]]
        patch[patch==orig_lbl]=new_lbl

    return cluster_label, n_clusters, n_noise

def cluster_feature_extraction(cluster_label, n_clusters, n_noise, stats=False, greyscale_img = None, 
                               breast_mask = None):
    features = dict()
    features['n_clusters'] = n_clusters
    features['n_noise'] = n_noise
    if breast_mask is not None:
        for prop in regionprops(breast_mask.astype(int)):
            xbr,ybr,xbr2,ybr2=prop.bbox
            hbr,wbr=xbr2-xbr,ybr2-ybr
    if n_clusters>0 or n_noise>0:
        for prop in regionprops(cluster_label):
            x1,y1,x2,y2=prop.bbox
            patch=cluster_label[x1:x2,y1:y2]==prop.label
            patch_label, n_objects = ndimage.label(patch)
            if greyscale_img is not None:
                img_patch = greyscale_img[x1:x2,y1:y2]
            cl = 'cl{}'.format(prop.label)
            # cluster global features
            features[cl]=dict()
            features[cl]['major_axis_length'] = prop.major_axis_length
            try:
                features[cl]['minor_axis_length'] = prop.minor_axis_length
            except:
                features[cl]['minor_axis_length'] = 1
            features[cl]['convex_area'] = prop.convex_area
            features[cl]['area'] = prop.area
            features[cl]['solidity']= prop.solidity
            features[cl]['inertia_1'] = prop.inertia_tensor_eigvals[0]
            features[cl]['inertia_2'] = prop.inertia_tensor_eigvals[1]
            features[cl]['orientation'] = prop.orientation
            # other moments
            moments = prop.moments_hu.ravel()
            for i_mom,mom in enumerate(moments):
                features[cl]['moments_{}'.format(i_mom)] = mom
            features[cl]['n_objects'] = n_objects
            # convex hull properties
            convexhull = convex_hull_image(patch)
            for prop_hull in regionprops(convexhull.astype(int)):
                features[cl]['eccentricity'] = prop_hull.eccentricity
            # relative location to breast
            if breast_mask is not None:
                x,y=prop.centroid
                features[cl]['breast_loc_x'], features[cl]['breast_loc_y'] = (x-xbr)/hbr, (y-ybr)/wbr
            # individual object features
            obj_features = ['major_axis_length', 'minor_axis_length', 'eccentricity', 'area']
            if n_objects>0:
                features[cl]['mcs'] = dict()
                if greyscale_img is None:
                    mc_regions = regionprops(patch_label)
                else:
                    mc_regions = regionprops(patch_label, img_patch)
                    obj_features = obj_features+['min_intensity', 'max_intensity', 'mean_intensity']
                for f in obj_features:
                    features[cl]['mcs'][f]=list()
                for prop_obj in mc_regions:
                    for f in obj_features:
                        features[cl]['mcs'][f].append(prop_obj[f])
    if stats:
        features = flatten(features)
        features = process_features(features)
    return features


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def process_features(d):
    d = flatten(d)
    d2 = d.copy()
    # finding statistics for mcs
    for k,v in d.items():
        if isinstance(v,list):
            d2.pop(k)
            d2[k+'_mean']=np.mean(v)
            d2[k+'_std']=np.std(v)
    return d2


def greylevel_feature_extraction(img, mask=None):
    features = dict()
    img = (img*(2**8-1)).astype(np.uint8)
    distances = [5]
    angles = [0]
    glcm=greycomatrix(img, distances, angles, 2**8, symmetric=True, normed=True)
    grey_level_features = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    for f in grey_level_features:
        values = greycoprops(glcm,f).ravel()
        for i,v in enumerate(values):
            features[f+str(i)]=v
    if mask is not None:
        img = img*mask
        glcm=greycomatrix(img, [5], [0], 2**8, symmetric=True, normed=True)
        for f in grey_level_features:
            values = greycoprops(glcm,f).ravel()
            for i,v in enumerate(values):
                features[f+'_masked'+str(i)]=v
    return features



def blob_detector_hdog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=2.0,
             overlap=.5, hessian_thr=None):
    """Segments blobs using DoG and Hessian Analysis
    """
 
    image = img_as_float(image)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    gaussian_images = [ndimage.gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                  * sigma_list[i]/(sigma_list[i+1]-sigma_list[i]) 
                  for i in range(k)]

    image_cube = np.stack(dog_images, axis=-1)

    # local_maxima = get_local_maxima(image_cube, threshold)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3,) * (image.ndim + 1)),
                                  threshold_rel=0.0,
                                  exclude_border=False)
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    
    # Hessian for segmentation
    if hessian_thr is None:
        hessian_images = [hessian_negative_definite(dog_image) for dog_image in dog_images]
    else:
        hessian_images = [hessian_criterion(dog_image, hessian_thr) for dog_image in dog_images]

    # Denoise
    #hessian_images = [remove_small_objects(hessian_images[i], min_size = 2*sigma_list[i]**2) for i in range(k)]
    #hessian_images = [remove_large_objects(hessian_images[i], min_size = 3*sigma_list[i]**2) for i in range(k)]
    blobs = _prune_blobs(lm, overlap)
    
    hessian_max_sup = np.zeros_like(image)
    for i in range(k):
        lbl, _ = ndimage.label(hessian_images[i])
        rprops = regionprops(lbl)
        s = sigma_list[i]
        for blob in blobs:
            if blob[2]==s:
                x,y = int(blob[0]), int(blob[1])
                # If we hit the background we cannot count it
                lbl_id = lbl[x,y]
                if lbl_id == 0:
                    continue
                x1,y1,x2,y2 = rprops[lbl_id-1].bbox
                hessian_max_sup[x1:x2,y1:y2] = np.logical_or(hessian_max_sup[x1:x2,y1:y2], lbl[x1:x2,y1:y2]==lbl_id)
    
    return hessian_max_sup

def blob_detector_hdog2(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=2.0,
             overlap=.5, hessian_thr=None):
    """Segments blobs using DoG and Hessian Analysis
    """
 
    image = img_as_float(image)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    gaussian_images = [ndimage.gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                  * sigma_list[i]
                  for i in range(k)]

    image_cube = np.stack(dog_images, axis=-1)

    # local_maxima = get_local_maxima(image_cube, threshold)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3,) * (image.ndim + 1)),
                                  threshold_rel=0.0,
                                  exclude_border=False)
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    
    # Hessian for segmentation
    if hessian_thr is None:
        hessian_images = [hessian_negative_definite(dog_image) for dog_image in dog_images]
    else:
        hessian_images = [hessian_criterion(dog_image, hessian_thr) for dog_image in dog_images]

    # Denoise
    #hessian_images = [remove_small_objects(hessian_images[i], min_size = 2*sigma_list[i]**2) for i in range(k)]
    #hessian_images = [remove_large_objects(hessian_images[i], min_size = 3*sigma_list[i]**2) for i in range(k)]
    blobs = _prune_blobs(lm, overlap)
    
    hessian_max_sup = np.zeros_like(image)
    for i in range(k):
        lbl, _ = ndimage.label(hessian_images[i])
        rprops = regionprops(lbl)
        s = sigma_list[i]
        for blob in blobs:
            if blob[2]==s:
                x,y = int(blob[0]), int(blob[1])
                # If we hit the background we cannot count it
                lbl_id = lbl[x,y]
                if lbl_id == 0:
                    continue
                x1,y1,x2,y2 = rprops[lbl_id-1].bbox
                hessian_max_sup[x1:x2,y1:y2] = np.logical_or(hessian_max_sup[x1:x2,y1:y2], lbl[x1:x2,y1:y2]==lbl_id)
    
    return hessian_max_sup



def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

def hessian_criterion(img_f, thr):
    img_hes = hessian(img_f)
    img_hes_det = img_hes[0,0,:,:]*img_hes[1,1,:,:]-img_hes[1,0,:,:]*img_hes[0,1,:,:]
    img_hes_trace = img_hes[0,0,:,:]+img_hes[1,1,:,:]
    img_hes_neg = np.logical_and(img_hes_det>0,img_hes_trace<0)
    img_hes_other = np.abs(img_hes_det)/np.power(img_hes_trace+1e-20,2)
    #print(img_hes_other.max(),img_hes_other.min(), np.median(img_hes_other))
    img_hes_other = np.logical_and(img_hes_other<thr,img_hes_trace<0)
    #img_hes_other = np.logical_and(img_hes_det<0,img_hes_trace<0)
    img_hes_neg = np.logical_or(img_hes_neg, img_hes_other)
    return img_hes_neg


def hessian_negative_definite(img_f):
    img_hes = hessian(img_f)
    img_hes_det = img_hes[0,0,:,:]*img_hes[1,1,:,:]-img_hes[1,0,:,:]*img_hes[0,1,:,:]
    img_hes_trace = img_hes[0,0,:,:]+img_hes[1,1,:,:]
    img_hes_neg = np.logical_and(img_hes_det>0,img_hes_trace<0)
    return img_hes_neg


def overlap_based_combining(region_mask, object_mask, thr=0.1):
    '''Finds objects in the object_mask that have a percentage
    of ovelap with the region_mask and retains only these.
    '''
    if region_mask.sum() == 0:
        return region_mask
    final_mask = np.zeros_like(region_mask)
    object_label, n_objects = ndimage.label(object_mask)
    props = regionprops(object_label)
    for prop in props:
        x1,y1,x2,y2 = prop.bbox
        bb = (x1,x2,y1,y2)
        lbl = prop.label
        obj_mask = object_label[x1:x2,y1:y2]==lbl
        region_patch = region_mask[x1:x2,y1:y2]
        overlap = np.logical_and(region_patch,obj_mask).sum()
        overlap /= obj_mask.sum()
        if overlap>thr:
            final_mask[x1:x2,y1:y2] += obj_mask
    return final_mask


def hybrid_approach(pred_mask,breast_mask,blob_mask,hybrid_combining='multiplication',
    hybrid_combining_overlap=0.3,
    radius=None, alpha=None, extra_mask=None):
    '''Takes prediction, breast mask and blob mask,
    applies correspodning thresholding and  combining to give
    a final prediction mask
    '''

    # for removing structures at the boundary
    #breast_mask = binary_erosion(breast_mask,square(45))
    breast_mask = cv2.erode(breast_mask.astype(np.float),cv2.getStructuringElement(cv2.MORPH_RECT,(45,45)))
    pred_mask = pred_mask*breast_mask
    blob_mask = blob_mask*breast_mask
    if extra_mask is not None:
        pred_mask = pred_mask*extra_mask
        blob_mask = blob_mask*extra_mask        
    if hybrid_combining=='multiplication':
        pred_mask = blob_mask*pred_mask
    elif hybrid_combining=='none':
        pred_mask = pred_mask
    elif hybrid_combining=='hdog':
        pred_mask = blob_mask
    else:
        pred_mask = overlap_based_combining(pred_mask, blob_mask, 
            thr=hybrid_combining_overlap)
        pred_mask = ndimage.binary_fill_holes(pred_mask)
    return pred_mask