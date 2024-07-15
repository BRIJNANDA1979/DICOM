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
def file_plane(IOP):
    IOP_round = [round(x) for x in IOP]
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
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=10.0, minStep=0.001, numberOfIterations=500)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(initial_transform)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    parameters = final_transform.GetParameters()
    rotation_angles = parameters[:3]
    translations = parameters[3:6]
    rotation_angles_degrees = np.degrees(rotation_angles)
    resampled_moving_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    return sitk.GetArrayFromImage(resampled_moving_image), final_transform,rotation_angles_degrees,translations
def compute_similarity_metrics(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    data_range = image1.max() - image1.min()
    ssim_index, _ = ssim(image1, image2, data_range=data_range, full=True)
    return mse, ssim_index
image_viewer = sitk.ImageViewer()
image_viewer.SetApplication("C:\\Users\\BRIJB\\OneDrive\\Desktop\\fiji-win64\\Fiji.app\\ImageJ-win64.exe")### use FIJI as image viewer 
file_reader = sitk.ImageFileReader()
source = 'E:\\patient_motion_correction\\15_CT_Non-Contrast_CT_NCCT_thin_slice_Hermes_103_001'
#source = 'E:\\patient_motion_correction\\THIN_NCCT_cases\\16_CT_Non-Contrast_CT_NCCT_thin_slice_Hermes_103_005'
target='E:\\patient_motion_correction\\Results'
out_file_csv = target+"\\"+(source.split('\\'))[-1]+".csv"
os.chdir(source)
for f in os.listdir(source):
    new = f.replace('_', '.')
    os.rename(f,new)
study = (source).split('\\')
file_reader.SetImageIO("GDCMImageIO")
files_dicom = os.listdir(source)
if len(files_dicom) == 0:
    print('No dicom files found')
    print('Exiting now')
    exit(0)
else:
    file_name = files_dicom[1]
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
try:
    pixel_spacing=file_reader.GetMetaData('0028|0030')
except Exception as e:   
    print('pixel_spacing tag does not exist')
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
if len(sorted_files)==0:
    sorted_files=os.listdir(source)
    sorted_file_names=sorted_files
a=pydicom.read_file(source+'/'+file_name)
IOP = a.ImageOrientationPatient
plane = file_plane(IOP)
print('Plane of Dicom file is: ',plane)
MSE_list=[]
SSIM_list=[]
PSNR_list=[]
Rotation_angles_degrees_list=[]
Translation_after_reg_list=[]
for i in range(len(sorted_files)-1):
    dicom_path1 = sorted_files[i]
    dicom_path2 = sorted_files[i+1]
    dicom1 = pydicom.dcmread(dicom_path1)
    dicom2 = pydicom.dcmread(dicom_path2)
    image1 = dicom1.pixel_array
    image2 = dicom2.pixel_array
    image1_normalized = normalize_image(image1)
    image2_normalized = normalize_image(image2)
    image2_registered, transform, rotation_angles_degrees,translations = register_images(image1_normalized, image2_normalized)
    Rotation_angles_degrees_list.append(rotation_angles_degrees)
    Translation_after_reg_list.append(translations)
    mse, ssim_index = compute_similarity_metrics(image1_normalized, image2_registered)
    psnratio = psnr(image1_normalized, image2_normalized)
    MSE_list.append(mse)
    SSIM_list.append(ssim_index)
    PSNR_list.append(psnratio)
    print('\n')
    print("Slice ID::-->" + str(i))
    plt.imshow(image1,cmap='gray')
    plt.show()
    print("Slice ID is::-->" + str(i+1))
    plt.imshow(image2,cmap='gray')
    plt.show()
    print(f"Mean Squared Error: {mse}")
    print(f"Structural Similarity Index: {ssim_index}")
    print(f"Peak signal to Noise Ratio: {psnratio}")
    plt.hist(MSE_list,bins=500)
plt.title('MSE')
plt.show()
plt.hist(SSIM_list,bins=500)
plt.title('SSIM')
plt.show()
plt.hist(PSNR_list,bins=500)
plt.title('PSNR')
plt.show()
High_SSIM = []
Mod_SSIM=[]
Low_SSIM=[]
Poor_SSIM=[]
Low_similarity_slices=[]
for i in range(len(SSIM_list)):
    if SSIM_list[i] >= 0.90:
        High_SSIM.append(i)
    if SSIM_list[i] >= 0.80 and SSIM_list[i] < 0.90:
        Mod_SSIM.append(i)
        print(f"Slice {i} and {i+1} has Moderate Motion Artifact")
    if SSIM_list[i] >= 0.50 and SSIM_list[i] <0.80:
        Low_SSIM.append(i)
        Low_similarity_slices.append(sorted_files[i])
        Low_similarity_slices.append(sorted_files[i+1])
        print(f"\nLowest SSIM means Highest degree artifact is found in slice {i} and {i+1}\n")
    if SSIM_list[i] < 0.50:
        print(f"\n Poor artifact is found in slice {i} and {i+1}")
        Poor_SSIM.append(i)
for i in range(len(Low_SSIM)):
    dicom_path1 = sorted_files[i]
    dicom_path2 = sorted_files[i+1]
    dicom1 = pydicom.dcmread(dicom_path1)
    dicom2 = pydicom.dcmread(dicom_path2)
    image1 = dicom1.pixel_array
    image2 = dicom2.pixel_array
    print('lowest SSIM value for Slice: ' + str(Low_SSIM[i]) +' and ' + str(Low_SSIM[i]+1)+ ' is::'+str(SSIM_list[Low_SSIM[i]]))
    #view_image(image1)
    #view_image(image2)
    #time.sleep(10)
Result_list_ssim=np.zeros(len(sorted_file_names))
for i in Low_SSIM:
    Result_list_ssim[i]=1
    Result_list_ssim[i+1]=1
High_PSNR = []
Mod_PSNR=[]
Low_PSNR=[]
Poor_PSNR=[]    
Result_list_psnr=np.zeros(len(sorted_file_names))
for i in range(len(PSNR_list)):
    if PSNR_list[i] >= 40:
        High_PSNR.append(i)
    if PSNR_list[i] >= 30 and PSNR_list[i] < 40:
        Mod_PSNR.append(i)
    if PSNR_list[i] >= 20 and PSNR_list[i] <30:
        Low_PSNR.append(i)
    if PSNR_list[i] < 20:
        Poor_PSNR.append(i)
for i in Low_PSNR:
    Result_list_psnr[i]=1
    Result_list_psnr[i+1]=1        
High_MSE = []
Mod_MSE=[]
Low_MSE=[]
Poor_MSE=[]      
Result_list_mse=np.zeros(len(sorted_file_names))
for i in range(len(MSE_list)):
    if MSE_list[i] <= 0.05 :
        Low_MSE.append(i)
    if MSE_list[i] > 0.05 and MSE_list[i] < 0.10:
        Mod_MSE.append(i)
    if MSE_list[i] >= 0.10:
        High_MSE.append(i)
for i in High_MSE:
    Result_list_mse[i]=1
    Result_list_mse[i+1]=1   
df = pd.DataFrame() 
df['Slice_ID']   = np.arange(1,len(sorted_file_names)+1)  
df['SSIM'] = Result_list_ssim
df['PSNR']= Result_list_psnr
df['MSE'] = Result_list_mse
df.to_csv(out_file_csv, index=False)
def moran_peak_ratio(image):
    f_transform = fftshift(fft2(image))
    power_spectrum = np.abs(f_transform) ** 2 
    peak_index = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
    neighborhood = power_spectrum[max(peak_index[0] - 1, 0):min(peak_index[0] + 2, power_spectrum.shape[0]),
                                   max(peak_index[1] - 1, 0):min(peak_index[1] + 2, power_spectrum.shape[1])]
    avg_intensity = np.mean(neighborhood)
    peak_value = power_spectrum[peak_index]
    moran_ratio = peak_value / avg_intensity
    return moran_ratio
def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()
dicom_arrays=[]
for i in range(len(sorted_files)):
    dicom_data = pydicom.dcmread(sorted_files[i])
    data_array = dicom_data.pixel_array
    dicom_arrays.append(data_array)
mpr_list=[]
mpr_new_list=[]
lap_list=[]
lap_new_list=[]
i=0
for image in dicom_arrays:
        fft_image = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft_image)
        rows, cols = image.shape
        row_freq = np.fft.fftfreq(rows)
        col_freq = np.fft.fftfreq(cols)
        row_freq_shifted = np.fft.fftshift(row_freq)
        col_freq_shifted = np.fft.fftshift(col_freq)
        magnitude_spectrum = np.abs(fft_shifted)
        psd = magnitude_spectrum ** 2             ### Compute the magnitude squared (Power Spectral Density)
        threshold = 0.01 * np.max(magnitude_spectrum)   ### 1percent of highest freq
        low_freq_mask = magnitude_spectrum < threshold
        low_freq_image = fft_shifted * low_freq_mask
        low_freq_image = np.fft.ifft2(np.fft.ifftshift(low_freq_image)).real
        low_freq_image_normalized = exposure.rescale_intensity(low_freq_image, in_range='image', out_range=(0, 255)).astype(np.uint8)
        std = np.std(image)
        var = np.var(image)
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
def area_of_image(image):
    gray=image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_outermost_boundary_pixels = np.count_nonzero(edges)
    plt.imshow(edges,cmap='gray')
    plt.show()
    return num_outermost_boundary_pixels,edges
def is_image_blurry(image, threshold=6.5):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = np.var(laplacian)
    is_blurry = (variance > threshold) and (variance < 10)    
    return is_blurry, variance, image, laplacian
Largest_contours_area_all=[]
Blurry_list=[]
Variance_list=[]
for image in dicom_arrays:
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sigma =0# Adjust sigma as needed
    smoothed_image = cv2.GaussianBlur(image, (5,5), sigma)
    blurry, variance, image, laplacian=is_image_blurry(smoothed_image)
    Blurry_list.append(blurry)
    Variance_list.append(variance)
    largest_contour_area,filtered_image = area_of_image(image)
    Largest_contours_area_all.append(largest_contour_area)
    print(f"Area of the largest contour: {largest_contour_area} square pixels")
    if blurry:
        print(f"BLurry with variance:{variance}")
    else:
        print(f"Variance: {variance}")
plt.hist(Largest_contours_area_all,bins=10)
plt.show()
middle= int(len(Largest_contours_area_all)/2)
print("Result of Area Artifact:")
source_path = 'E:\\patient_motion_correction\\THIN_NCCT_cases'
source_list = cases_list=os.listdir(source_path)
All_mean_top_area=[]
All_mean_bottom_area=[]
for Source in source_list:
    file_reader = sitk.ImageFileReader()
    file_reader.SetImageIO("GDCMImageIO")
    files_dicom = os.listdir(source_path+'\\'+Source)
    if len(files_dicom) == 0:
        print('No dicom files found')
        print('Exiting now')
        exit(0)
    else:
        file_name = files_dicom[1]
    file_reader.SetFileName(os.path.join(source_path+'\\'+Source,file_name))
    file_reader.ReadImageInformation()
    series_ID = file_reader.GetMetaData('0020|000e')
    Sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(source_path+'\\'+Source, series_ID)
    image = file_reader.Execute()
    Largest_contours_area_all=[]
    for i in range(len(Sorted_file_names)):
        dicom = pydicom.dcmread(Sorted_file_names[i])
        image = dicom.pixel_array
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        sigma = 1.5  # Adjust sigma as needed
        smoothed_image = cv2.GaussianBlur(image, (5, 5), sigma)
        _, binary_image = cv2.threshold(smoothed_image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_area = cv2.contourArea(largest_contour)
        Largest_contours_area_all.append(largest_contour_area)
    All_mean_top_area.append(np.mean(Largest_contours_area_all[0:int(len(Sorted_file_names)/2)]))
    All_mean_bottom_area.append(np.mean(Largest_contours_area_all[int(len(Sorted_file_names)/2):]))                                                          
Mean_area_top = np.mean(All_mean_top_area)
Mean_area_bottom = np.mean(All_mean_bottom_area)
Max_area_top =  np.max(All_mean_top_area)
Max_area_bottom = np.max(All_mean_bottom_area)
Min_area_top =  np.max(All_mean_top_area)
Min_area_bottom = np.max(All_mean_bottom_area)
Result_area_list=np.zeros(len(sorted_file_names))
for i in range(len(sorted_file_names)) :
    if i <=middle:
        if Largest_contours_area_all[i] < Min_area_top:#0.98 * np.mean(Largest_contours_area_all[0:middle]):
            print(f"Slice ID: {i}")
            Result_area_list[i]=1
        else:
            Result_area_list[i]=0
    else:
        if Largest_contours_area_all[i] <  Min_area_bottom: #2 * np.mean(Largest_contours_area_all):
            print(f"Slice ID: {i}")
            Result_area_list[i]=1
        else:
            Result_area_list[i]=0
print("\nBlurry Artifact :")
for i in range(len(Blurry_list)):
    if Blurry_list[i]:
        print(f"Slice {i}")
Result_list=np.zeros(len(sorted_file_names))
for i in range(len(Blurry_list)):
    if i:
        Result_list[i]=2
    else:
        Result_list[i]=0
df= pd.read_csv(out_file_csv)    
df['Blurry_index'] = Result_list
df['Area'] = Result_area_list
df['Total'] = df['SSIM']+df['PSNR']+df['MSE']+df['Laplacian']+df['MPR']+df['Blurry_index']+df['Area']
df.to_csv(out_file_csv, index=False)




