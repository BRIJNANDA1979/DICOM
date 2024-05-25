# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:00:05 2024_version_16

@author: BRIJ
"""
"""    Readme    (Gantry_shift check)
This code is used to get metadata of dicom images and check whetether Slice thickness is uniform or not using 
Z-parameter of axial plane"""

"""
This function takes IOP of an image and returns its plane (Sagittal, Coronal, Transverse/Axial)
"""

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

def montage(all_images_array,patient_id,series_ID):
        ### convert all Hu to range 40-80
        window_array=[]
        for image in all_images_array:
            image[image<40]=40
            image[image>80]=80
            window_array.append(image)
        plt.figure(figsize=(10, 10))
        columns = 5
        rows = 8
        step_size = int(len(window_array)/40)
        if(step_size<1):
            step_size=1
        for i in range(1, columns*rows+1):
            plt.style.use('dark_background')
            plt.subplot(rows, columns, i,aspect='equal')
            plt.imshow(window_array[i*step_size-1][0,:,:])
            plt.subplots_adjust(
                    wspace=0.0, 
                    hspace=0.0)
            plt.axis('off')
            plt.set_cmap(plt.gray())
            if i> len(window_array)-1:
                break
        plt.savefig(args.target + '/' + str(patient_id.replace(' ','')) +'_'+str(series_ID) + '_montage.png') 
        plt.show()
if __name__ == '__main__':                                       # import required libraries
    try:
        import SimpleITK as sitk
        import os
        from scipy import stats
        import numpy as np
        import matplotlib.pyplot as plt
        import argparse
        import pydicom
        import re
        import math
    except ImportError as e:
        print('Libraries missing')
        print(e)
        exit(0)
    except Exception as e:
        print("Libraries are present. Other Error")
        print(e)
        exit(0)
    
    image_viewer = sitk.ImageViewer()

    image_viewer.SetApplication("C:\\Users\\BRIJB\\OneDrive\\Desktop\\fiji-win64\\Fiji.app\\ImageJ-win64.exe")### use FIJI as image viewer                                                  # A file name that belongs to the series we want to read
    parser = argparse.ArgumentParser(description='Input')
    parser.add_argument('-s', '--source', dest='source', help='source data',required=True)
    parser.add_argument('-d', '--target', dest='target',help='target destination',required=True)
    args = parser.parse_args()
    # Read the file's meta-information without reading bulk pixel data
    
    file_reader = sitk.ImageFileReader()
    
    study = (args.source).split('\\')
    file_reader.SetImageIO("GDCMImageIO")
    files_dicom = os.listdir(args.source)
    if len(files_dicom) == 0:
        print('No dicom files found')
        print('Exiting now')
        exit(0)
    else:
        file_name = files_dicom[0]
    file_reader.SetFileName(os.path.join(args.source,file_name))
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
    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(args.source, series_ID)
    image = file_reader.Execute()
    patient_orientation = file_reader.GetMetaData('0020|0037')
    print('Image Size:' + str(image.GetSize()))
    print('Image Origin:' + str(image.GetOrigin()))
    print('Image Spacing:' + str(image.GetSpacing()))
    print('Image Direction:' + str(image.GetDirection()))
    print('patient orientation:' + str(patient_orientation))
    sorted_files =  list(sorted_file_names)
    
    a=pydicom.read_file(args.source+'/'+file_name)
    IOP = a.ImageOrientationPatient
    plane = file_plane(IOP)
    print('Plane of Dicom file is: ',plane)
    if (plane == 'Axial'):                  ##### for Axial plane , use z-values as depth of scan  (Z-plane)
        x_all =[]
        y_all=[]
        all_images_array=[]
        image_position_all =[]
        for file in sorted_files:
            file_reader = sitk.ImageFileReader()
            file_reader.SetImageIO('GDCMImageIO')
            file_reader.SetFileName(os.path.join(args.source,file))
            try:
                file_reader.ReadImageInformation()########### 
            except Exception as e:
                print('Could not open file')
                exit(0)
            series_ID = file_reader.GetMetaData('0020|000e')
            image_position = file_reader.GetMetaData('0020|0032')   #### ImagePatientPosition
            image_position_all.append(image_position)
            x_all.append(image_position.split('\\')[0])
            y_all.append(image_position.split('\\')[1])
            image = file_reader.Execute()
            array = sitk.GetArrayFromImage(image)
            all_images_array.append(array)
        diff_x=[]
        for i in range(len(x_all)-2):
            diff_x.append(float(x_all[i+1]) - float(x_all[i]))
        diff_y=[]
        for i in range(len(y_all)-2):
            diff_y.append(float(y_all[i+1]) - float(y_all[i]))
           
        rms_xy=[]
        for i in range(len(diff_x)-1):
            rms_xy.append(math.sqrt(diff_x[i]*diff_x[i] + diff_y[i]*diff_y[i]))

        mean_diff_x = abs(np.mean(diff_x))
        mean_diff_y = abs(np.mean(diff_y))
        print(mean_diff_x)
        print(mean_diff_y)
        if (mean_diff_x > float(row_spacing)):
            x_shift = mean_diff_x
        else:
            x_shift = 0
        if (mean_diff_y > float(col_spacing)):
            y_shift = mean_diff_y
        else:
            y_shift = 0
        if(x_shift ==0) or (y_shift == 0):
            gantry_shift_status = 'False'
        else:
            gantry_shift_status = 'True'
        
        '''if(abs(max(diff_x)) > float(row_spacing)) and (abs(max(diff_x)) > float(col_spacing)):
            shift_x = abs(max(diff_x))
            shift_y = abs(max(diff_y))
            gantry_shift_status = 'True'
        else:
            shift_x=0
            shift_y=0
            gantry_shift_status = 'False'
        
            '''
        print('gantry_shift_status::' + str(gantry_shift_status))
        #exit(0)
        if not os.path.exists(args.target):
            os.makedirs(args.target)
        for i in range(len(sorted_files)):
            
            itk_image = sitk.ReadImage(os.path.join(args.source,sorted_files[i]))
            s = sorted_files[i].split('_')[-1]
            #print(s)
            #exit(0)
            size = list(itk_image.GetSize())
            size[2] = 0
            
            index = [0, 0, 0]
            
            Extractor = sitk.ExtractImageFilter()
            Extractor.SetSize(size)
            Extractor.SetIndex(index)
            
            slice1 = Extractor.Execute(itk_image)
            dimension = 2
            translation = sitk.TranslationTransform(dimension)
            translation.SetParameters((-x_shift, -y_shift))
            interpolator = sitk.sitkLinear
            default_value = 0
            shifted_slice1 = sitk.Resample(slice1, translation, interpolator,default_value)
   
            #image_viewer.Execute(slice1)
            #image_viewer.Execute(shifted_slice1)
            tmpimg = sitk.JoinSeries(shifted_slice1)
    
            paster = sitk.PasteImageFilter()
            paster.SetDestinationIndex((0,0,0))
            paster.SetSourceIndex((0,0,0))
            paster.SetSourceSize(tmpimg.GetSize())
            sampled_itk_image= paster.Execute(itk_image, tmpimg)
            #image_viewer.Execute(sampled_itk_image)
            origin = itk_image.GetOrigin()
            sampled_origin = [0,0,0]
            sampled_origin[0] = origin[0] + x_shift
            sampled_origin[1] = origin[1] + y_shift
            sampled_origin[2] = origin[2]
            sampled_itk_image.SetOrigin(sampled_origin)
            sitk.WriteImage(sampled_itk_image, args.target +'\\'+'sampled_'+str(s)+'_.dcm');
            #exit(0)
    
if (plane == 'Sagittal'):                 # X-plane
    z_all =[]
    y_all=[]
    all_images_array=[]
    image_position_all =[]
    for file in sorted_files:
        file_reader = sitk.ImageFileReader()
        file_reader.SetImageIO('GDCMImageIO')
        file_reader.SetFileName(os.path.join(args.source,file))
        try:
            file_reader.ReadImageInformation()########### 
        except Exception as e:
            print('Could not open file')
            exit(0)
        series_ID = file_reader.GetMetaData('0020|000e')
        image_position = file_reader.GetMetaData('0020|0032')   #### ImagePatientPosition
        image_position_all.append(image_position)
        z_all.append(image_position.split('\\')[2])
        y_all.append(image_position.split('\\')[1])
        image = file_reader.Execute()
        array = sitk.GetArrayFromImage(image)
        all_images_array.append(array)
    diff_z=[]
    for i in range(len(z_all)-2):
        diff_z.append(float(x_all[i+1]) - float(x_all[i]))
    diff_y=[]
    for i in range(len(y_all)-2):
        diff_y.append(float(y_all[i+1]) - float(y_all[i]))
       
    mean_diff_z = abs(np.mean(diff_z))
    mean_diff_y = abs(np.mean(diff_y))
    print(mean_diff_z)
    print(mean_diff_y)
    if (mean_diff_z > float(row_spacing)):
        z_shift = mean_diff_z
    else:
        z_shift = 0
    if (mean_diff_y > float(col_spacing)):
        y_shift = mean_diff_y
    else:
        y_shift = 0
    if(z_shift ==0) or (y_shift == 0):
        gantry_shift_status = 'False'
    else:
        gantry_shift_status = 'True'
    
    '''if(abs(max(diff_x)) > float(row_spacing)) and (abs(max(diff_x)) > float(col_spacing)):
        shift_x = abs(max(diff_x))
        shift_y = abs(max(diff_y))
        gantry_shift_status = 'True'
    else:
        shift_x=0
        shift_y=0
        gantry_shift_status = 'False'
    
        '''
    print('gantry_shift_status::' + str(gantry_shift_status))
    #exit(0)
    if not os.path.exists(args.target):
        os.makedirs(args.target)
    for i in range(len(sorted_files)):
        
        itk_image = sitk.ReadImage(os.path.join(args.source,sorted_files[i]))
        s = sorted_files[i].split('_')[-1]
        #print(s)
        #exit(0)
        size = list(itk_image.GetSize())
        size[2] = 0
        
        index = [0, 0, 0]
        
        Extractor = sitk.ExtractImageFilter()
        Extractor.SetSize(size)
        Extractor.SetIndex(index)
        
        slice1 = Extractor.Execute(itk_image)
        dimension = 2
        translation = sitk.TranslationTransform(dimension)
        translation.SetParameters((-y_shift, -z_shift))
        interpolator = sitk.sitkLinear
        default_value = 0
        shifted_slice1 = sitk.Resample(slice1, translation, interpolator,default_value)

        #image_viewer.Execute(slice1)
        #image_viewer.Execute(shifted_slice1)
        tmpimg = sitk.JoinSeries(shifted_slice1)

        paster = sitk.PasteImageFilter()
        paster.SetDestinationIndex((0,0,0))
        paster.SetSourceIndex((0,0,0))
        paster.SetSourceSize(tmpimg.GetSize())
        sampled_itk_image= paster.Execute(itk_image, tmpimg)
        #image_viewer.Execute(sampled_itk_image)
        origin = itk_image.GetOrigin()
        sampled_origin = [0,0,0]
        sampled_origin[2] = origin[2] + z_shift
        sampled_origin[1] = origin[1] + y_shift
        sampled_origin[2] = origin[0]
        sampled_itk_image.SetOrigin(sampled_origin)
        sitk.WriteImage(sampled_itk_image, args.target +'\\'+'sampled_'+str(s)+'_.dcm');
        #exit(0)
        
if (plane == 'Coronal'):         #Y-plane         
    x_all =[]
    z_all=[]
    all_images_array=[]
    image_position_all =[]
    for file in sorted_files:
        file_reader = sitk.ImageFileReader()
        file_reader.SetImageIO('GDCMImageIO')
        file_reader.SetFileName(os.path.join(args.source,file))
        try:
            file_reader.ReadImageInformation()########### 
        except Exception as e:
            print('Could not open file')
            exit(0)
        series_ID = file_reader.GetMetaData('0020|000e')
        image_position = file_reader.GetMetaData('0020|0032')   #### ImagePatientPosition
        image_position_all.append(image_position)
        x_all.append(image_position.split('\\')[0])
        z_all.append(image_position.split('\\')[2])
        image = file_reader.Execute()
        array = sitk.GetArrayFromImage(image)
        all_images_array.append(array)
    diff_x=[]
    for i in range(len(x_all)-2):
        diff_x.append(float(x_all[i+1]) - float(x_all[i]))
    diff_z=[]
    for i in range(len(z_all)-2):
        diff_z.append(float(z_all[i+1]) - float(z_all[i]))

    mean_diff_x = abs(np.mean(diff_x))
    mean_diff_z = abs(np.mean(diff_z))
    print(mean_diff_x)
    print(mean_diff_z)
    if (mean_diff_x > float(row_spacing)):
        x_shift = mean_diff_x
    else:
        x_shift = 0
    if (mean_diff_z > float(col_spacing)):
        z_shift = mean_diff_z
    else:
        z_shift = 0
    if(x_shift ==0) or (z_shift == 0):
        gantry_shift_status = 'False'
    else:
        gantry_shift_status = 'True'
    
    '''if(abs(max(diff_x)) > float(row_spacing)) and (abs(max(diff_x)) > float(col_spacing)):
        shift_x = abs(max(diff_x))
        shift_y = abs(max(diff_y))
        gantry_shift_status = 'True'
    else:
        shift_x=0
        shift_y=0
        gantry_shift_status = 'False'
    
        '''
    print('gantry_shift_status::' + str(gantry_shift_status))
    #exit(0)
    if not os.path.exists(args.target):
        os.makedirs(args.target)
    for i in range(len(sorted_files)):
        
        itk_image = sitk.ReadImage(os.path.join(args.source,sorted_files[i]))
        s = sorted_files[i].split('_')[-1]
        #print(s)
        #exit(0)
        size = list(itk_image.GetSize())
        size[2] = 0
        
        index = [0, 0, 0]
        
        Extractor = sitk.ExtractImageFilter()
        Extractor.SetSize(size)
        Extractor.SetIndex(index)
        
        slice1 = Extractor.Execute(itk_image)
        dimension = 2
        translation = sitk.TranslationTransform(dimension)
        translation.SetParameters((-x_shift, -y_shift))
        interpolator = sitk.sitkLinear
        default_value = 0
        shifted_slice1 = sitk.Resample(slice1, translation, interpolator,default_value)

        #image_viewer.Execute(slice1)
        #image_viewer.Execute(shifted_slice1)
        tmpimg = sitk.JoinSeries(shifted_slice1)

        paster = sitk.PasteImageFilter()
        paster.SetDestinationIndex((0,0,0))
        paster.SetSourceIndex((0,0,0))
        paster.SetSourceSize(tmpimg.GetSize())
        sampled_itk_image= paster.Execute(itk_image, tmpimg)
        #image_viewer.Execute(sampled_itk_image)
        origin = itk_image.GetOrigin()
        sampled_origin = [0,0,0]
        sampled_origin[0] = origin[0] + x_shift
        sampled_origin[2] = origin[2] + z_shift
        sampled_origin[1] = origin[1]
        sampled_itk_image.SetOrigin(sampled_origin)
        sitk.WriteImage(sampled_itk_image, args.target +'\\'+'sampled_'+str(s)+'_.dcm');
        #exit(0)