# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:20:34 2024

@author: BRIJB
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:42:09 2024

@author: BRIJB
"""

# -*- coding: utf-8 -*-

import SimpleITK as sitk
import pydicom
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import sys
import matplotlib.pyplot as plt
import time
import csv
import pandas as pd
import SimpleITK as sitk
import pydicom
import numpy as np
import cv2
from skimage import exposure
import re
from scipy.fft import fft2, fftshift
from scipy import ndimage
import matplotlib.pyplot as plt
import glob


def get_timepoints_from_ctp_series(dicom_directory):
    timepoints = []

    # Iterate over all DICOM files in the directory
    for filename in os.listdir(dicom_directory):
        if filename.endswith('.dcm'):
            dicom_path = os.path.join(dicom_directory, filename)
            dicom_data = pydicom.dcmread(dicom_path)
            
            # Extract the AcquisitionTime or ContentTime
            if 'AcquisitionTime' in dicom_data:
                time_str = dicom_data.AcquisitionTime
            elif 'ContentTime' in dicom_data:
                time_str = dicom_data.ContentTime
            else:
                continue
            
            # Convert time_str to a readable format if needed
            timepoints.append(time_str)

    return timepoints


def file_plane(IOP):
   
    IOP_round = [round(x) for x in IOP]
    #print(IOP_round)
    plane = np.cross(IOP_round[0:3], IOP_round[3:6])
    plane = [abs(x) for x in plane]
    if plane[0] == 1:
        return "Sagittal"
    elif plane[1] == 1:
        return "Coronal"
    elif plane[2] == 1:
        return "Axial"
def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel = np.max(image1)
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr
def normalize_image(image):
    image = sitk.GetImageFromArray(image)
    image = sitk.Cast(image, sitk.sitkFloat32)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    mean = stats.GetMean()
    stddev = stats.GetSigma()
    normalized = sitk.ShiftScale(image, shift=-mean, scale=1.0/stddev)
    #image_viewer.Execute(image)
    #sys.exit()
    return sitk.GetArrayFromImage(normalized)

def view_image(image):
    image = sitk.GetImageFromArray(image)
    image = sitk.Cast(image, sitk.sitkFloat32)
    image_viewer.Execute(image)   



def register_images(fixed_image, moving_image):
    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.001, numberOfIterations=500)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(initial_transform)
    
    final_transform = registration_method.Execute(fixed_image, moving_image)
    parameters = final_transform.GetParameters()
    
    # Extract the rotation angles (in radians) and translations (in mm)
    rotation_angles = parameters[:3]
    translations = parameters[3:6]
    # Convert rotation angles to degrees
    rotation_angles_degrees = np.degrees(rotation_angles)
    resampled_moving_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    '''if rotation_angles_degrees[1] > 3:
        print(rotation_angles_degrees)
        image_viewer.Execute(moving_image)
        image_viewer.Execute(resampled_moving_image)
        sys.exit()'''
    return sitk.GetArrayFromImage(resampled_moving_image), final_transform,rotation_angles_degrees,translations

def compute_similarity_metrics(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    data_range = image1.max() - image1.min()
    ssim_index, _ = ssim(image1, image2, data_range=data_range, full=True)
 
    return mse, ssim_index

image_viewer = sitk.ImageViewer()
image_viewer.SetApplication("C:\\Users\\BRIJB\\OneDrive\\Desktop\\fiji-win64\\Fiji.app\\ImageJ-win64.exe")### use FIJI as image viewer 
file_reader = sitk.ImageFileReader()
#source = 'E:\\patient_motion_correction\\002_NCCT' 
#source = 'E:\\patient_motion_correction\\20_SERIES'            ###INPUT TEST DATA path

#source = 'E:\\patient_motion_correction\\004_NCCT_SRC'    ### checked slice 3 and 4 are High degree of Artifact
#source = 'E:\\patient_motion_correction\\695a_2221077_004_NCCT_SRC_case_2' #### checked slice 3,4 pair and 23,24 pair high degree of artifact

#source = 'E:\\patient_motion_correction\\695a_2142124_004_NCCT_SRC_case_3'
#source = 'E:\\patient_motion_correction\\695a_2268512_004_NCCT_SRC_case_4'
#source = 'E:\\patient_motion_correction\\695a_2558925_004_NCCT_SRC_case_5_a'
#source = 'E:\\patient_motion_correction\\695a_2558925_007_NCCT_SRC_case_5_b'
#source = 'E:\\patient_motion_correction\\695a_2649719_004_NCCT_SRC_case_6'
#source = 'E:\\patient_motion_correction\\695a_2856962_004_NCCT_SRC_case_7'
#source = 'E:\\patient_motion_correction\\695a_3194883_004_NCCT_SRC_case_8'
#source="E:\\patient_motion_correction\\raw\HA2138\\CT\\20110126T050821\\20_SERIES"
source="E:\\patient_motion_correction\\CTP_cases\\5_SERIES_a"
#source="E:\\patient_motion_correction\\CTP_cases\\5_SERIES_b"
#source='E:\\patient_motion_correction\\CTA_cases\\6_CT_CT_Angiography_CTA'##did not work for CTA
#SOURCE=source
target='E:\\patient_motion_correction\\Results'
out_file_csv = target+"\\"+(source.split('\\'))[-1]+".csv"
#sys.exit(0)
os.chdir(source)
for f in os.listdir(source):
    new = f.replace('_', '.')
    os.rename(f,new)
    if f.split('.')[-1]!='dcm':
        os.remove(f)
#sys.exit()
study = (source).split('\\')
file_reader.SetImageIO("GDCMImageIO")
files_dicom = os.listdir(source)
if len(files_dicom) == 0:
    print('No dicom files found')
    print('Exiting now')
    exit(0)
else:
    file_name = files_dicom[0]
file_reader.SetFileName(os.path.join(source,file_name))
file_reader.ReadImageInformation()
try: 
    series = file_reader.GetMetaData('0008|103e')
except:
    print('Series description tag does not exist')
    exit(0)  
series_ID = file_reader.GetMetaData('0020|000e')
study = study[-1]
try:
    rows = file_reader.GetMetaData('0028|0010')
except Exception as e:   
    rows = 0
try:
    cols = file_reader.GetMetaData('0028|0011')
except Exception as e:   
    cols = 0

try:
    slice_thickness = file_reader.GetMetaData('0018|0050')
except Exception as e:   
    slice_thickness=0
    print('slice_thickness tag does not exist')
    #print(e)

try:
    pixel_spacing=file_reader.GetMetaData('0028|0030')
except Exception as e:   
    print('pixel_spacing tag does not exist')
    #print(e)
row_spacing = pixel_spacing.split('\\')[0]
col_spacing = pixel_spacing.split('\\')[1]
sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(source, series_ID)


image = file_reader.Execute()
patient_orientation = file_reader.GetMetaData('0020|0037')
print('Image Size:' + str(image.GetSize()))
print('Image Origin:' + str(image.GetOrigin()))
print('Image Spacing:' + str(image.GetSpacing()))
print('Image Direction:' + str(image.GetDirection()))
print('patient orientation:' + str(patient_orientation))
sorted_files =  list(sorted_file_names)
# Calculate the orientation vectors
ds=pydicom.read_file(source+'/'+file_name)
patient_orientation = np.array(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
row_cosines = patient_orientation[0, :]
col_cosines = patient_orientation[1, :]

# Calculate the Field of View (FoV)
rows = int(ds.Rows)
columns = int(ds.Columns)
pixel_spacing = np.array(ds.PixelSpacing, dtype=float)
fov_x = pixel_spacing[1] * columns
fov_y = pixel_spacing[0] * rows
# Calculate the corner points of the scan
image_position_patient = np.array(ds.ImagePositionPatient, dtype=float)
top_left = image_position_patient
top_right = top_left + fov_x * row_cosines
bottom_left = top_left + fov_y * col_cosines
bottom_right = top_left + fov_x * row_cosines + fov_y * col_cosines

# Print the corner points
print("Top Left Corner:", np.round(top_left,2))
print("Top Right Corner:", np.round(top_right,2))
print("Bottom Left Corner:", np.round(bottom_left,2))
print("Bottom Right Corner:", np.round(bottom_right,2))

# Calculate the center of the scan volume
center_x = (top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) / 4
center_y = (top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) / 4
center_z = (top_left[2] + top_right[2] + bottom_left[2] + bottom_right[2]) / 4

# Print the center coordinates
print(f"Center of the Scan Volume: {center_x:.2f},{center_y:.2f},{center_z:.2f}")

# Determine the orientation
orientations = {
    "x": "L" if row_cosines[0] < 0 else "R",
    "y": "P" if row_cosines[1] < 0 else "A",
    "z": "I" if col_cosines[2] < 0 else "S"
     }
orientation_code = orientations["x"] + orientations["y"] + orientations["z"]
try:
    print("\n\t\tPatient position: " + str(ds.PatientPosition))
except:
    print("\t\tPatient position: NA")
print("\t\tOrientation of Patient:" + str(orientation_code))


if len(sorted_files)==0:
    sorted_files=os.listdir(source)
    sorted_file_names=sorted_files
    #sys.exit()

a=pydicom.read_file(source+'/'+file_name)
IOP = a.ImageOrientationPatient
plane = file_plane(IOP)
print('\nPlane of Dicom file is: ',plane)
#sys.exit()
# Example usage

timepoints = get_timepoints_from_ctp_series(source)

#for idx, timepoint in enumerate(timepoints):
    #print(f"Slice {idx}: Timepoint {timepoint}")
same_time_group=[]
same_time_group.append(timepoints[0])
for i in range(len(timepoints)-1):
    if timepoints[0]==timepoints[i+1]:
        same_time_group.append(timepoints[0])
group_slices=[]
for i in range(int(len(timepoints)/len(same_time_group))):
    temp=[]
    for j in range(len(same_time_group)):
        temp.append(i*len(same_time_group)+j)
    group_slices.append(temp)
#sys.exit()

print(f"Number of Unique timepoints/slices: {len(timepoints)} ")
print(f"Number of Slices with same timepoint in each Group:  {len(same_time_group)}")
print(f"Number of Groups: {len(group_slices)}")
Rotation_angles_degrees_list=[]
Translation_after_reg_list=[]
#sys.exit()
#Slices_data_array=[]
Group_mse=[]
Group_ssim=[]
Group_psnr=[]
for i in range(int(len(timepoints)/len(same_time_group))-1):

    #dicom_path1 = 'E:\\patient_motion_correction\\20_SERIES\\HA2138.NA.PID.CT.20110126T050821.625000.20.3517.8243.6174.13.52.53.87..00199.dcm'
    #dicom_path2 = 'E:\\patient_motion_correction\\20_SERIES\\HA2138.NA.PID.CT.20110126T050821.625000.20.3517.8243.6174.13.52.53.87..00200.dcm'
    MSE_list=[]
    SSIM_list=[]
    PSNR_list=[]
    for j in range(len(group_slices[0])):
        dicom_path1 = sorted_files[group_slices[i][j]]
        dicom_path2 = sorted_files[group_slices[i+1][j]]
        #sys.exit()
        dicom1 = pydicom.dcmread(dicom_path1)
        dicom2 = pydicom.dcmread(dicom_path2)
        
        # Extract pixel data
        image1 = dicom1.pixel_array
        image2 = dicom2.pixel_array
        #view_image(image1)
        #view_image(image2)
        #sys.exit()
        # Normalize images
        image1_normalized = normalize_image(image1)
        image2_normalized = normalize_image(image2)
    
        # Register images
        image2_registered, transform, rotation_angles_degrees,translations = register_images(image1_normalized, image2_normalized)
        Rotation_angles_degrees_list.append(rotation_angles_degrees)
        Translation_after_reg_list.append(translations)
        #sys.exit(0)
        # Compute similarity metrics
        mse, ssim_index = compute_similarity_metrics(image1_normalized, image2_registered)
        psnratio = psnr(image1_normalized, image2_normalized)
        MSE_list.append(mse)
        SSIM_list.append(ssim_index)
        PSNR_list.append(psnratio)
        print('\n')
        print(f"Slice Group is {i}::"+ str(group_slices[i]) +"\nSlice ID:" + str(i*len(same_time_group)+j))
        plt.imshow(image1,cmap='gray')
        plt.show()
        #time.sleep(5)
        print(f"Slice Group is {i+1}::"+ str(group_slices[i+1]) +"\nSlice ID:" + str((i+1)*len(same_time_group)+j))
        plt.imshow(image2,cmap='gray')
        plt.show()
        print(f"Mean Squared Error: {mse}")
        print(f"Structural Similarity Index: {ssim_index}")
        print(f"Peak signal to Noise Ratio: {psnratio}")
        #sys.exit()
    #time.sleep(10)
    Group_mse.append(MSE_list)
    Group_ssim.append(SSIM_list)
    Group_psnr.append(PSNR_list)

plt.hist(Group_mse[0],bins=500)
plt.title('MSE')
plt.show()

plt.hist(Group_ssim[0],bins=500)
plt.title('SSIM')
plt.show()

plt.hist(Group_psnr[0],bins=500)
plt.title('PSNR')
plt.show()


## find mean of each of 24 groups
Mean_SSIM_groups=[]
Min_SSIM_groups=[]
Max_SSIM_groups=[]
for i in range(len(Group_ssim)):
    Mean_SSIM_groups.append(np.mean(Group_ssim[i]))
    Min_SSIM_groups.append(np.min(Group_ssim[i]))
    Max_SSIM_groups.append(np.max(Group_ssim[i]))

MEAN_SSIM_all_groups = np.mean(Mean_SSIM_groups)
Low_SSIM_group_id=[]
for i in range(len(Mean_SSIM_groups)):
   if Mean_SSIM_groups[i] < MEAN_SSIM_all_groups:
       Low_SSIM_group_id.append(i)
All_Slices_ID_low_SSIM=[]
for i in Low_SSIM_group_id:
    for j in range(len(same_time_group)):
        All_Slices_ID_low_SSIM.append(len(same_time_group)*i+j)
Result_SSIM_list=np.zeros(len(timepoints))
for i in All_Slices_ID_low_SSIM:
    Result_SSIM_list[i]=1



## find mean of each of 24 groups
Mean_PSNR_groups=[]
Min_PSNR_groups=[]
Max_PSNR_groups=[]
for i in range(len(Group_psnr)):
    Mean_PSNR_groups.append(np.mean(Group_psnr[i]))
    Min_PSNR_groups.append(np.min(Group_psnr[i]))
    Max_PSNR_groups.append(np.max(Group_psnr[i]))

MEAN_PSNR_all_groups = np.mean(Mean_PSNR_groups)
Low_PSNR_group_id=[]
for i in range(len(Mean_PSNR_groups)):
   if Mean_PSNR_groups[i] < MEAN_PSNR_all_groups:
       Low_PSNR_group_id.append(i)
All_Slices_ID_low_PSNR=[]
for i in Low_PSNR_group_id:
    for j in range(len(same_time_group)):
        All_Slices_ID_low_PSNR.append(len(same_time_group)*i+j)
Result_PSNR_list=np.zeros(len(timepoints))
for i in All_Slices_ID_low_PSNR:
    Result_PSNR_list[i]=1
  

  
## find mean of each of 24 groups
Mean_MSE_groups=[]
Min_MSE_groups=[]
Max_MSE_groups=[]
for i in range(len(Group_mse)):
    Mean_MSE_groups.append(np.mean(Group_mse[i]))
    Min_MSE_groups.append(np.min(Group_mse[i]))
    Max_MSE_groups.append(np.max(Group_mse[i]))

MEAN_MSE_all_groups = np.mean(Mean_MSE_groups)
Low_MSE_group_id=[]
for i in range(len(Mean_MSE_groups)):
   if Mean_MSE_groups[i] < MEAN_MSE_all_groups:
       Low_MSE_group_id.append(i)
All_Slices_ID_low_MSE=[]
for i in Low_MSE_group_id:
    for j in range(len(same_time_group)):
        All_Slices_ID_low_MSE.append(len(same_time_group)*i+j)
Result_MSE_list=np.zeros(len(timepoints))
for i in All_Slices_ID_low_MSE:
    Result_MSE_list[i]=1



df = pd.DataFrame() 
df['Slice_ID']   = np.arange(1,len(timepoints)+1)  
df['SSIM'] = Result_SSIM_list
df['PSNR']= Result_PSNR_list
df['MSE'] = Result_MSE_list
df.to_csv(out_file_csv, index=False)
#image_viewer.Execute(resampled_image)   

def moran_peak_ratio(image):
    # Perform Fourier transform
    f_transform = fftshift(fft2(image))
    # Calculate power spectrum
    power_spectrum = np.abs(f_transform) ** 2 
    # Find peak location
    peak_index = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
    #print(peak_index)
    # Calculate average intensity of surrounding frequencies
    neighborhood = power_spectrum[max(peak_index[0] - 1, 0):min(peak_index[0] + 2, power_spectrum.shape[0]),
                                   max(peak_index[1] - 1, 0):min(peak_index[1] + 2, power_spectrum.shape[1])]
    avg_intensity = np.mean(neighborhood)

    # Calculate Moran peak ratio
    peak_value = power_spectrum[peak_index]
    moran_ratio = peak_value / avg_intensity

    return moran_ratio


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

# Step 1: Read the DICOM image
'''dicom_path = 'E:\\patient_motion_correction\\005_AX_SOFT_NCCT//anon_CT.1.3.12.2.1107.5.1.4.54407.30000022102012425976500002871.dcm'
dicom_data = pydicom.dcmread(dicom_path)

# Step 2: Extract the image data
image = dicom_data.pixel_array'''

#dicom_path = 'E:\\patient_motion_correction\\005_AX_SOFT_NCCT'   ### First case
#dicom_path = 'E:\\patient_motion_correction\\002_NCCT'   ### Second case
#dicom_path = 'E:\\patient_motion_correction\\5' 
#dicom_path = 'E:\\patient_motion_correction\\20_SERIES'
#dicom_path = 'E:\\patient_motion_correction\\004_NCCT_SRC'
'''dicom_path =source #'E:\\patient_motion_correction\\004_NCCT_SRC_case_3'

images = os.listdir(dicom_path)
dicom_arrays = []
for image in images:
    
    dicom_data = pydicom.dcmread(dicom_path+'\\'+image,force=True)
    data_array = dicom_data.pixel_array
    #data_array =cv2.GaussianBlur(data_array, (3, 3), 10)
    dicom_arrays.append(data_array)'''
dicom_arrays=[]
for i in range(len(sorted_files)):
    dicom_data = pydicom.dcmread(sorted_files[i])
    data_array = dicom_data.pixel_array
    dicom_arrays.append(data_array)
mpr_list=[]
mpr_new_list=[]
lap_list=[]
lap_new_list=[]
# Step 3: Compute the 2D FFT
i=0
for image in dicom_arrays:
        #image = dicom_arrays[344]
        fft_image = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft_image)
        
        # Step 4: Compute the frequency bins
        rows, cols = image.shape
        row_freq = np.fft.fftfreq(rows)
        col_freq = np.fft.fftfreq(cols)
        row_freq_shifted = np.fft.fftshift(row_freq)
        col_freq_shifted = np.fft.fftshift(col_freq)
        
        # Step 5: Visualize the magnitude of the frequency components
        magnitude_spectrum = np.abs(fft_shifted)
        psd = magnitude_spectrum ** 2             ### Compute the magnitude squared (Power Spectral Density)
        
                                        # Find peak location
                           ##peak_index = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
        #threshold = np.percentile(magnitude_spectrum,10) ### top 1 % freq
        threshold = 0.01 * np.max(magnitude_spectrum)   ### 1percent of highest freq
        #sys.exit(0)
        low_freq_mask = magnitude_spectrum < threshold
        low_freq_image = fft_shifted * low_freq_mask
        low_freq_image = np.fft.ifft2(np.fft.ifftshift(low_freq_image)).real
        low_freq_image_normalized = exposure.rescale_intensity(low_freq_image, in_range='image', out_range=(0, 255)).astype(np.uint8)
        std = np.std(image)
        var = np.var(image)
        
         ####normalized_array = (array - min_val) / (max_val - min_val)
        low_fft_image = np.fft.fft2(low_freq_image)
        low_fft_shifted = np.fft.fftshift(low_fft_image)
        low_magnitude_spectrum = np.abs(low_fft_shifted)
        low_psd = low_magnitude_spectrum ** 2  
        psd_log = np.log(psd+1)
        low_psd_log = np.log(low_psd+1)
        mpr = moran_peak_ratio(image)
        mpr_new = moran_peak_ratio(low_freq_image)
        lap = variance_of_laplacian(image)
        lap_new=variance_of_laplacian(low_freq_image)
        mpr_list.append(mpr)
        mpr_new_list.append(mpr_new)
        lap_list.append(lap)
        lap_new_list.append(lap_new)

        plt.imshow(image,cmap='gray')
        plt.title('Original image')
        plt.colorbar()
        plt.show()
        print(f"Slice is: {i}")
        i+=1
        print('Mean of low_freq_image: '+str(np.mean(low_freq_image)))
       
        print('SD of low_freq_image: ' + str(np.std(low_freq_image)))
     
        print('Moran peak ratio of original Slice: ' +str(mpr))
     
        print('Moran peak ratio of low_freq_image:' + str(mpr_new))
      
        print('Variance of Laplacian of original slice: '+str(lap))
        
        print('Variance of Laplacian of low_freq_image: '+str(lap_new))
       
        plt.imshow(low_freq_image,cmap='gray')
        plt.title('Low_1%_intensities')
        plt.colorbar()
        plt.show()
        #plt.hist(np.log(psd),bins=100)
        #plt.title('Power Spectrum density of original slice on log scale')
        #plt.show()
        #plt.hist(low_psd_log,bins=100)
        #plt.title('Power Spectrum density of Low Freq slice on log scale')
        #plt.show()
        '''im_blur = ndimage.gaussian_filter(low_freq_image, 5)
        plt.imshow(im_blur,cmap='gray')
        plt.title('Blurred_version_of_artifact')
        plt.colorbar()
        plt.show()'''
        #sys.exit(0)

Result_list=[]
diff_lap=[]
for i in range(len(lap_list)):
    diff_lap.append(abs(lap_list[i]-lap_new_list[i]))
    
Result_mpr_list=[]
diff_mpr=[]
for i in range(len(mpr_list)):
    diff_mpr.append(abs(mpr_list[i]-mpr_new_list[i]))

for i in diff_lap:
    if i < 100:
        Result_list.append(1)
    else:
        Result_list.append(0)
for i in diff_mpr:
    if i > 1:
        Result_mpr_list.append(1)
    else:
        Result_mpr_list.append(0)
df= pd.read_csv(out_file_csv)    
    
df['Laplacian'] = Result_list
df['MPR'] = Result_mpr_list
df.to_csv(out_file_csv, index=False)

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:26:24 2024

@author: BRIJB
"""



# Load the DICOM file
#dicom_path = 'E:\\patient_motion_correction\\004_NCCT_SRC\\anon.CT.1.2.840.113619.2.437.3.531367228.355.1700305518.90.1.dcm' 
#dicom = pydicom.dcmread(dicom_path)
def is_image_blurry(image, threshold=6.5):
    # Read the image
    #image = cv2.imread(image_path)
    
    # Convert to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Calculate the variance of the Laplacian
    variance = np.var(laplacian)
    
    # Check if the variance is below the threshold
    is_blurry = (variance > threshold) ## and (variance < 10)
    
    return is_blurry, variance, image, laplacian
#source = 'E:\\patient_motion_correction\\004_NCCT_SRC'  ## new NCCT case by Niruta on 24 /06/2024
#source = 'E:\\patient_motion_correction\\004_NCCT_SRC_case_2'
#source = 'E:\\patient_motion_correction\\004_NCCT_SRC_case_3'
'''os.chdir(source)

files_dicom = os.listdir(source)
if len(files_dicom) == 0:
    print('No dicom files found')
    print('Exiting now')
    exit(0)
else:
    file_name = files_dicom[0]'''

Largest_contours_area_all=[]
Blurry_list=[]
Variance_list=[]
for image in dicom_arrays:
    '''dicom_path = os.path.join(source,file)
    dicom = pydicom.dcmread(dicom_path)'''
    # Extract the image data as a NumPy array
    #image = dicom.pixel_array
    
    #sys.exit(0)
    # Normalize the pixel values to 0-255 (8-bit grayscale)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply Gaussian smoothing
    sigma = 1.5  # Adjust sigma as needed
    smoothed_image = cv2.GaussianBlur(image, (5, 5), sigma)
    
    blurry, variance, image, laplacian=is_image_blurry(smoothed_image)
    Blurry_list.append(blurry)
    Variance_list.append(variance)
    # Apply binary threshold or edge detection
    _, binary_image = cv2.threshold(smoothed_image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    '''contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # find the biggest contour (c) by the area
    c = max(contours, key = cv2.contourArea)'''
    
    
    largest_contour = max(contours, key=cv2.contourArea)
    # Calculate the area of the largest contour
    largest_contour_area = cv2.contourArea(largest_contour)
    Largest_contours_area_all.append(largest_contour_area)
    
    # Draw the largest contour on the original image
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
    plt.figure(figsize=(15, 5))
    plt.imshow(smoothed_image, cmap='gray')
    plt.title('SomoothedImage')
    plt.axis('off')
    plt.show()
    
    
    
    plt.imshow(binary_image, cmap='gray')
    plt.title('Original DICOM Image')
    plt.axis('off')
    plt.show()
    
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.axis('off')
    plt.show()
    print(f"Area of the largest contour: {largest_contour_area} square pixels")
    if blurry:
        print(f"BLurry with variance:{variance}")
    else:
        print(f"Variance: {variance}")

plt.hist(Largest_contours_area_all,bins=10)
plt.show()
middle= int(len(Largest_contours_area_all)/2)
#print("Result of Area Artifact:")

##find Average of Mean,Min and Max areas from all CTP  cases
source_path = 'E:\\patient_motion_correction\\CTP_cases'
source_list = cases_list=os.listdir(source_path)
All_mean_top_area=[]
All_mean_bottom_area=[]
ALL_CTP_cases_Mean=[]
EACH_case_ALL_groups_area_list=[]

for source_data_folder in source_list:
#sys.exit(0)
    timepoints=[]
    timepoints = get_timepoints_from_ctp_series(source_path+'\\'+source_data_folder)

    #for idx, timepoint in enumerate(timepoints):
        #print(f"Slice {idx}: Timepoint {timepoint}")
    same_time_group=[]
    same_time_group.append(timepoints[0])  ###This gets how many timepoints are same for each group
    for i in range(len(timepoints)-1):
        if timepoints[0]==timepoints[i+1]:
            same_time_group.append(timepoints[0])
    group_slices=[]
    for i in range(int(len(timepoints)/len(same_time_group))):
        temp=[]
        for j in range(len(same_time_group)):
            temp.append(i*len(same_time_group)+j)
        group_slices.append(temp)
    file_reader = sitk.ImageFileReader()
    file_reader.SetImageIO("GDCMImageIO")
    files_dicom = os.listdir(source_path+'\\'+source_data_folder)
    if len(files_dicom) == 0:
        print('No dicom files found')
        print('Exiting now')
        exit(0)
    else:
        file_name = files_dicom[1]
    file_reader.SetFileName(os.path.join(source_path+'\\'+source_data_folder,file_name))
    file_reader.ReadImageInformation()
    series_ID = file_reader.GetMetaData('0020|000e')
    Sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(source_path+'\\'+source_data_folder, series_ID)
    image = file_reader.Execute()
    # List of file paths to the brain images
    Largest_contours_area_all=[]
    ALL_groups_areas_list=[]
    for j in range(int(len(timepoints)/len(same_time_group))):
        Group_area_list=[]
        for i in group_slices[j]:
            dicom = pydicom.dcmread(Sorted_file_names[i])
            image = dicom.pixel_array
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sigma = 1.5  # Adjust sigma as needed
            smoothed_image = cv2.GaussianBlur(image, (5, 5), sigma)
             # Apply binary threshold or edge detection
            _, binary_image = cv2.threshold(smoothed_image, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on the original image
            '''contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            
            # find the biggest contour (c) by the area
            c = max(contours, key = cv2.contourArea)'''
            largest_contour = max(contours, key=cv2.contourArea)
            # Calculate the area of the largest contour
            largest_contour_area = cv2.contourArea(largest_contour)
            #print(largest_contour_area)
            Group_area_list.append(largest_contour_area)
        ALL_groups_areas_list.append((Group_area_list))
    EACH_case_ALL_groups_area_list.append((ALL_groups_areas_list))
    #sys.exit(0)
    #All_mean_top_area.append(np.mean(Largest_contours_area_all[0:int(len(Sorted_file_names)/2)]))
    #All_mean_bottom_area.append(np.mean(Largest_contours_area_all[int(len(Sorted_file_names)/2):]))                                                          
Artifact_dict={}
for i in range(len(EACH_case_ALL_groups_area_list)):
    groups = EACH_case_ALL_groups_area_list[i]
    Mean_area=[]
    for group in groups:
        Mean_area.append(np.mean(group))
    mean=np.mean(Mean_area)
    Artifact_groups=[]
    for j in range(len(Mean_area)):
        if Mean_area[j] <mean:
            #print(f"Group {j} of case {i} has Area artifact" )
            Artifact_groups.append(j)
    Artifact_dict[i]=Artifact_groups      
            

Result_area_groups =[]
for key in Artifact_dict.keys():
    Result_area_groups.append(Artifact_dict[key])
ALL_cases_RESULT_area_final_list=[]
for i in range(len(source_list)):
    RESULT_area_final_list=np.zeros(len(os.listdir(source_path+'\\'+source_list[i])))
    timepoints=get_timepoints_from_ctp_series(source_path+'\\'+source_list[i])
    same_time_group=[]
    same_time_group.append(timepoints[0])  ###This gets how many timepoints are same for each group
    for j in range(len(timepoints)-1):
        if timepoints[0]==timepoints[j+1]:
            same_time_group.append(timepoints[0])
    for group_id in Result_area_groups[i]:
        for k in range(len(same_time_group)):
            RESULT_area_final_list[group_id*len(same_time_group)+k]=1
    ALL_cases_RESULT_area_final_list.append(RESULT_area_final_list)
#sys.exit()
#print("\nBlurry Artifact :")
#for i in range(len(Blurry_list)):
    #if Blurry_list[i]:
        #print(f"Slice {i}")
#source="E:\\patient_motion_correction\\raw\HA2138\\CT\\20110126T050821\\20_SERIES"
Result_list=np.zeros(len(os.listdir(source)))
for i in range(len(Blurry_list)):
    if i:
        Result_list[i]=2
    else:
        Result_list[i]=0
df= pd.read_csv(out_file_csv)    

df['Blurry_index'] = Result_list
for i in range(len(source_list)):
    if source_list[i] == source.split('\\')[-1]:
        df['Area'] = ALL_cases_RESULT_area_final_list[i] 
df['Total'] = df['SSIM']+df['PSNR']+df['MSE']+df['Laplacian']+df['MPR']+df['Blurry_index']+df['Area']
df.to_csv(out_file_csv, index=False)




