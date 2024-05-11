# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:00:05 2024_version_18

@author: BRIJ
"""
"""    Readme    (Slice thickness error checking)
This code is used to get metadata of dicom images and check whetether Slice thickness is uniform or not using 
Z-parameter of axial plane"""

"""
This function takes IOP of an image and returns its plane (Sagittal, Coronal, Transverse/Axial)
"""
def find_modality(study):
    search_list = ['NON_CONTRAST','Head_Non_Con','AX_BRAIN','Brain_NonCon']
    flag=0
    for i in search_list:
        if re.findall(i,study,flags=re.IGNORECASE):
            modality = 'NCCT'
            flag=1
            return modality
    search_list =  ['AXIAL','de AXIAL','DE_CoW','CTA','ANGIO','CoW']   
    for i in search_list:
        if re.findall(i,study,flags=re.IGNORECASE):
            modality = 'CTA'
            flag=1
            return modality
    search_list = ['Perfusion','CTP','Perf']
    for i in search_list:
        if re.findall(i,study,flags=re.IGNORECASE):
            modality = 'CTP'
            flag=1
            return modality
    search_list = ['PWI','DWI','Diffusion','Diff','T1','T2','Flair']
    for i in search_list:
        if re.findall(i,study,flags=re.IGNORECASE):
            modality = 'MR'
            flag=1
            return modality
    if flag==0:
        modality='None'
        return modality
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

def window(all_images_array,intercept,slope):
    hu_array=[]
    for arr in all_images_array:
        arr = arr.astype(np.float64)
        slope = int(slope)
        intercept=int(intercept)
        arr = slope * arr + intercept   #Rescaling using intercept and slope to get HU values
        hu_array.append(arr)
    return hu_array

    
def montage(all_images_array,patient_id,series_ID):
        ### convert all Hu to range 40-80
        window_array=[]
        for image in all_images_array:
            image[image<40]=40
            image[image>80]=80
            window_array.append(image)
        plt.figure(figsize=(12, 12))
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
    except ImportError as e:
        print('Libraries missing')
        print(e)
        exit(0)
    except Exception as e:
        print("Libraries are present. Other Error")
        print(e)
        exit(0)
    
                                                       # A file name that belongs to the series we want to read
    parser = argparse.ArgumentParser(description='Input')
    parser.add_argument('-s', '--source', dest='source', help='source data',required=True)
    parser.add_argument('-d', '--target', dest='target',help='target destination',required=True)
    args = parser.parse_args()
    # Read the file's meta-information without reading bulk pixel data
    try:
        file_reader = sitk.ImageFileReader()
    except:
        print('File can not be read')
        exit(0)
    print('Please find output HTML path:  ' + str(args.target))
    study = (args.source).split('\\')
    #print(study)
    file_reader.SetImageIO("GDCMImageIO")
    all_files = os.listdir(args.source)
    files_dicom = list(filter(lambda f: f.endswith('.dcm'), all_files))
    files_dicom.sort()
    #print(files_dicom)
    #exit(0)
    if len(files_dicom) == 0:
        print('No dicom files found')
        print('Exiting now')
        exit(0)
    else:
        file_name = files_dicom[0]
    file_reader.SetFileName(os.path.join(args.source,file_name))
    file_reader.ReadImageInformation()
    series_ID = file_reader.GetMetaData('0020|000e')
    try:
        manufacturer = file_reader.GetMetaData('0008|0070')
        print('Manufacturer is : ' + str(manufacturer))
    except:
        manufacturer = 'NA'
    print('Series_ID is :' + series_ID)
    try: 
        series = file_reader.GetMetaData('0008|103e')
    except:
        print('Series description tag does not exist')
  
    study = study[-1]
    modality = find_modality(study)
    if(modality=='None') and (manufacturer!='TOSHIBA'):
        modality =  file_reader.GetMetaData('0008|0060')
    else:
        print('SOP Class UID:'+str(file_reader.GetMetaData('0008|0016')))
        try:
            SOP_class_uid = file_reader.GetMetaData('0008|0016')
            if (SOP_class_uid == '1.2.840.10008.5.1.4.1.1.2.1') or  (SOP_class_uid == '1.2.840.10008.5.1.4.1.1.2') :
                modality='CT'
        except:
                print('Manufacturer is TOSHIBA and modality is Unknown')
                modality='NA' 
    print('Modality is:' + str(modality))
    patient_id = file_reader.GetMetaData('0010|0020')
    patient_name = file_reader.GetMetaData('0010|0010')
    try:
        institution_name = file_reader.GetMetaData('0008|0080')
        institution_name_status='True'
    except Exception as e:
        institution_name_status='False'
    try:
        accession_number = file_reader.GetMetaData('0008|0050')
        accession_number_status='True'
    except Exception as e:
        accession_number_status='False'
    try:
        series_instance_uid = file_reader.GetMetaData('0020|000E')
        series_instance_uid_status='True'
    except Exception as e:
        series_instance_uid_status='False'
    try:
        acquisition_date = file_reader.GetMetaData('0008|0022')
        acquisition_date_status='True'
    except Exception as e:
        acquisition_date_status='False'
    try:
        acquisition_time = file_reader.GetMetaData('0008|0032')
        acquisition_time_status='True'
    except Exception as e:
        acquisition_time_status='False'
    try:
        study_date = file_reader.GetMetaData('0008|0020')
        study_date_status='True'
    except Exception as e:
        study_date_status='False'
    try:
        study_time = file_reader.GetMetaData('0008|0030')
        study_time_status='True'
    except Exception as e:
        study_time_status='False'
    try:
        series_date = file_reader.GetMetaData('0008|0021')
        series_date_status='True'
    except Exception as e:
        series_date_status='False'
    try:
        series_time = file_reader.GetMetaData('0008|0031')
        series_time_status='True'
    except Exception as e:
        series_time_status='False'
    try:
        rows = file_reader.GetMetaData('0028|0010')
    except Exception as e:   
        rows = 0
    try:
        cols = file_reader.GetMetaData('0028|0011')
    except Exception as e:   
        cols = 0
    if(rows!=cols):
        mismatch = 'True'
    else:
        mismatch='False'
    try:
        body_part = file_reader.GetMetaData('0018|0015')
    except:
        body_part ='NA'
    try:
        slice_thickness = file_reader.GetMetaData('0018|0050')
    except Exception as e:   
        slice_thickness=0
        print('slice_thickness tag does not exist')
        #print(e)
    try:
        intercept = file_reader.GetMetaData('0028|1052')
    except Exception as e:
        intercept=0
        print('intercept tag does not exist')
    try:
        slope = file_reader.GetMetaData('0028|1053')
    except Exception as e:   
        slope=0
        print('slope tag does not exist')
    image_date = file_reader.GetMetaData('0008|0023')
    image_time = file_reader.GetMetaData('0008|0033')
    try:
        pixel_spacing=file_reader.GetMetaData(('0028|0030'))
        ss = float(slice_thickness)
        ps = pixel_spacing.split('\\') 
        ax_aspect = (float(ps[1]))/(float(ps[0]))
        sag_aspect = (float(ps[1]))/ss
        cor_aspect = ss/(float(ps[0]))
    except Exception as e:   
        pixel_spacing=0
        print('pixel_spacing tag does not exist')

    number_of_slices = len(files_dicom)
    list_of_IDs =[]
    target_list=[]
    sorted_files = files_dicom
    for file in sorted_files:
        s = file.split('_')
        ss = s[-1]
        list_of_IDs.append(int(ss.replace('.dcm','')))        
    Max = max(list_of_IDs)
    target_list =list(range(1, Max+1,1))
    missing_IDs= list(set(target_list).difference(set(list_of_IDs)))
    if len(missing_IDs)!=0:
        print('Missing Dicom files IDs are :' + str(missing_IDs))
    number_of_missed_slices = len(missing_IDs)
    print('Number of missing Dicom files are :'+str(number_of_missed_slices))                                                                                                # To check plane of image
    if(modality=='XA'):
        slice_thickness=0
        thickness=0
        plane='NA'
        dimensions='NA'
        scan_depth=0
        Max_timepoints=0
        result = 'NA'
        status='NA'
        body_part = str(body_part)
        body_part='NA'
        all_images_array=[]
        for file in sorted_files:
            file_reader = sitk.ImageFileReader()
            file_reader.SetImageIO('GDCMImageIO')
            file_reader.SetFileName(os.path.join(args.source
                                                 , file))
            file_reader.ReadImageInformation()
            series_ID = file_reader.GetMetaData('0020|000e')
            image = file_reader.Execute()
            array = sitk.GetArrayFromImage(image)
            all_images_array.append(array)
        montage(all_images_array,patient_id,series_ID)

    else:
       
        try:
            IOP = file_reader.GetMetaData('0020|0037')#a.ImageOrientationPatient
            plane = file_plane(IOP)
            print('Plane of Dicom file is: ',plane)
        except Exception as e:
            print('ImageOrientationPatient does not exist')
            plane='NA'
        
    if (plane == 'Axial'):                  ##### for Axial plane , use z-values as depth of scan  (Z-plane)
        z_all =[]
        thickness_list=[]
        all_images_array=[]
        patient_position_all =[]
        timepoints=[]
        for file in sorted_files:
            file_reader = sitk.ImageFileReader()
            file_reader.SetImageIO('GDCMImageIO')
            file_reader.SetFileName(os.path.join(args.source
                                                 , file))
            file_reader.ReadImageInformation()
            series_ID = file_reader.GetMetaData('0020|000e')
            patient_position = file_reader.GetMetaData('0020|0032')   #### ImagePatientPosition
            patient_position_all.append(patient_position)
            z_all.append(patient_position.split('\\')[2])
            image = file_reader.Execute()
            array = sitk.GetArrayFromImage(image)
            all_images_array.append(array)

        count=1
        z_all = [float(i) for i in z_all]
        z_all = sorted(z_all, key = lambda x:float(x))
        sorted_z_all = z_all
        for i in range(len(z_all) -2):
            if (float(z_all[i]) == float(z_all[i+1])):
                count+=1
            else:
                timepoints.append(count)
                count=1
        if(modality =='CTP'):
            print("Timepoints are :")
            print(timepoints)
            dimensions = str(str(rows) +' X ' +str(cols)+' X '+str(number_of_slices) +' X '+str(timepoints[0]))
        else:
            dimensions = str(str(rows) +' X ' +str(cols)+' X '+str(number_of_slices))
        Max_timepoints = np.max(timepoints)
        new_list = []
        [new_list.append(i) for i in z_all if i not in new_list]  ### remove duplicate timepoints in case of CTA and CTP
        z_all = new_list
        thickness  = (float(z_all[1]) - float(z_all[0]))
        thickness_list.append(thickness)
        flag=0
        zfirst = float(z_all[0])
        zlast =  float(z_all[len(z_all)-1])
        scan_depth = abs(zlast - zfirst)
        print('Scan_depth is:' + str(scan_depth))
        print('Scan_thickness is:' + str(thickness))
        print('Number_of_slices are :'+str(number_of_slices))
        for i in range(1,len(z_all)-1):
            new = round((float(z_all[i+1]) - float(z_all[i])),2)
            thickness_list.append(new)
            if new == round(thickness,2):   ### use round off function here 2.9==3
                continue
            else:
                flag=1
                print('Result of QC script is :variable thickness')
                result='Variable Thickness'
       
        if flag == 0:
            print('Result of QC script is : Uniform Thickness')
            result='Uniform Thickness'
            if (round(scan_depth,2) - round((number_of_slices * round(thickness,2)),2)) < thickness:
                print('No Scans are missing')
                status ='False'
            else:
                print('Scans are missing')
                status='True'
        else:
            print('Result of QC script is : Variable Thickness')
            window(all_images_array,intercept,slope)
        montage(all_images_array,patient_id,series_ID)
    if plane == 'Sagittal':    ### X-plane
        x_all =[]
        thickness_list=[]
        all_images_array=[]
        patient_position_all =[]
        timepoints=[]
        for file in sorted_files:
            file_reader = sitk.ImageFileReader()
            file_reader.SetImageIO('GDCMImageIO')
            file_reader.SetFileName(os.path.join(args.source
                                                 , file))
            file_reader.ReadImageInformation()
            series_ID = file_reader.GetMetaData('0020|000e')
            patient_position = file_reader.GetMetaData('0020|0032')
            patient_position_all.append(patient_position)
            x_all.append(patient_position.split('\\')[0])
            image = file_reader.Execute()
            array = sitk.GetArrayFromImage(image)
            all_images_array.append(array)
        count=1
        x_all = [float(i) for i in x_all]
        x_all = sorted(x_all, key = lambda x:float(x))
        sorted_x_all = x_all
        #print(z_all)
        for i in range(len(x_all) -2):
            if (float(x_all[i]) == float(x_all[i+1])):
                count+=1
            else:
                timepoints.append(count)
                count=1
        if(modality =='CTP'):
            print("Timepoints are :")
            print(timepoints)
            dimensions = str(str(rows) +' X ' +str(cols)+' X '+str(number_of_slices) +' X '+str(timepoints[0]))
        else:
            dimensions = str(str(rows) +' X ' +str(cols)+' X '+str(number_of_slices))
        Max_timepoints = np.max(timepoints)
        new_list = []
        [new_list.append(i) for i in x_all if i not in new_list]  ### remove duplicate timepoints in case of CTA and CTP
        x_all = new_list
        thickness  = (float(x_all[1]) - float(x_all[0]))
        thickness_list.append(thickness)
        flag=0
        xfirst = float(x_all[0])
        xlast =  float(x_all[len(x_all)-1])
        scan_depth = abs(xlast - xfirst)
        number_of_slices = len(x_all)
        print('Scan_depth is:' + str(scan_depth))
        print('Scan_thickness is:' + str(thickness))
        print('Number_of_slices are :'+str(number_of_slices))
        for i in range(1,len(x_all)-1):
            new = round((float(x_all[i+1]) - float(x_all[i])),2)
            thickness_list.append(new)
            if new == round(thickness,2):   ### use round off function here 2.9==3
                continue
            else:
                flag=1
                print('Result of QC script is :variable thickness')
                result='Variable Thickness'
                status='NA'
                break
        if flag == 0:
            print('Result of QC script is : Uniform Thickness')
            result='Uniform Thickness'
            if (round(scan_depth,2) - round((number_of_slices * round(thickness,2)),2)) < thickness:
                print('No Scans are missing')
                status ='False'
            else:
                print('Scans are missing')
                status='True'
        else:
            print('Result of QC script is : Variable Thickness')
                  
        window(all_images_array,intercept,slope)
        montage(all_images_array,patient_id,series_ID)
    if plane == 'Coronal':    ### Y-plane
            y_all =[]
            thickness_list=[]
            all_images_array=[]
            patient_position_all =[]
            timepoints=[]
            for file in sorted_files:
                file_reader = sitk.ImageFileReader()
                file_reader.SetImageIO('GDCMImageIO')
                file_reader.SetFileName(os.path.join(args.source
                                                     , file))
                file_reader.ReadImageInformation()
                series_ID = file_reader.GetMetaData('0020|000e')
                patient_position = file_reader.GetMetaData('0020|0032')
                y_all.append(patient_position.split('\\')[1])
                image = file_reader.Execute()
                array = sitk.GetArrayFromImage(image)
                all_images_array.append(array)
            count=1
            y_all = [float(i) for i in y_all]
            y_all = sorted(y_all, key = lambda x:float(x))
            sorted_y_all = y_all
            #print(z_all)
            for i in range(len(y_all) -2):
                if (float(y_all[i]) == float(y_all[i+1])):
                    count+=1
                else:
                    timepoints.append(count)
                    count=1

            if(modality =='CTP'):
                print("Timepoints are :")
                print(timepoints)
                dimensions = str(str(rows) +' X ' +str(cols)+' X '+str(number_of_slices) +' X '+str(timepoints[0]))
            else:
                dimensions = str(str(rows) +' X ' +str(cols)+' X '+str(number_of_slices))
            Max_timepoints = np.max(timepoints)
            new_list = []
            [new_list.append(i) for i in z_all if i not in new_list]  ### remove duplicate timepoints in case of CTA and CTP
            print(new_list)
            y_all = new_list
            thickness  = (float(y_all[1]) - float(y_all[0]))
            thickness_list.append(thickness)
            flag=0
            yfirst = float(y_all[0])
            ylast =  float(y_all[len(y_all)-1])
            scan_depth = abs(ylast - yfirst)
            number_of_slices = len(y_all)
            print('Scan_depth is:' + str(scan_depth))
            print('Scan_thickness is:' + str(thickness))
            print('Number_of_slices are :'+str(number_of_slices))
            for i in range(1,len(y_all)-1):
                new = round((float(y_all[i+1]) - float(y_all[i])),2)
                thickness_list.append(new)
                if new == round(thickness,2):   ### use round off function here 2.9==3
                    continue
                else:
                    flag=1
                    print('Result of QC script is :variable thickness')
                    result='Variable Thickness'
                    break
            if flag == 0:
                print('Result of QC script is : Uniform Thickness')
                result='Uniform Thickness'
                if (round(scan_depth,2) - round((number_of_slices * round(thickness,2)),2)) < thickness:
                    print('No Scans are missing')
                    status ='False'
                else:
                    print('Scans are missing')
                    status='True'
            else:
                print('Result of QC script is : Variable Thickness')

            all_images_array = window(all_images_array,intercept,slope)  ### Conversion to HU values of all scans/images
            montage(all_images_array,patient_id,series_ID)
           
    if (plane=='NA'):
        scan_depth=0
        thickness=0
        Max_timepoints=0
        dimensions=0
        result=0
        status='NA'
        all_images_array=[]
        patient_position_all =[]
        for file in sorted_files:
            file_reader = sitk.ImageFileReader()
            file_reader.SetImageIO('GDCMImageIO')
            file_reader.SetFileName(os.path.join(args.source
                                                 , file))
            file_reader.ReadImageInformation()
            series_ID = file_reader.GetMetaData('0020|000e')
            image = file_reader.Execute()
            array = sitk.GetArrayFromImage(image)
            all_images_array.append(array)
        montage(all_images_array,patient_id,series_ID)
    
    try:
        if(len(all_images_array)!=0):
            pixel_data_status = 'True'
    except:
            pixel_data_status = 'False'
    if not os.path.exists(args.target):
        os.makedirs(args.target)
    #os.chdir('c:/QC')
    out=args.target + '\\' + str(patient_id.replace(' ','')) +'_'+str(series_ID) + '_montage.png'
    #print(out)   
    file_one = open(args.target + '/' + str(patient_id) +'_'+str(series_ID) + '_output.html', "w")
    href_path =args.target + '\\style.css'#"c:\\QC\\out\\style.css"
    
    html_display = f""" <!DOCTYPE html>
    <html lang="en">

    <head>
        <link href='style.css' rel='stylesheet' type='text/css'>
        <link rel="stylesheet" href="//maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">
        <meta charset="utf-8">
    </head>

    <body>
        <div class="summary-header">
            <h2>Summary</h2>
        </div>
        <div class="summary-body">
            <div class="summary-awards">
                <ul class="summary-row summary-header">
                    <li class="summary-star"><span class="star goldstar"></span></li>
                    <div class="summary-title">Patient ID: <b>{patient_id}</b></div>
                    </ul>
                <ul class="summary-row summary-row-even">
                    <li class="summary-header-date summary-header-title">PatientName: </li>
                    <li class="summary-header-track">{patient_name}
                        <div class="summary-subtitle">{patient_id}</div>
                    </li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Acquisition Date: </li>
                    <li class="summary-header-date">{acquisition_date_status}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Acquisition Time: </li>
                    <li class="summary-header-date">{acquisition_time_status}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Study Date: </li>
                    <li class="summary-header-date">{study_date_status}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Study Time: </li>
                    <li class="summary-header-date">{study_time_status}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Series Date: </li>
                    <li class="summary-header-date">{series_date_status}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Series Time: </li>
                    <li class="summary-header-date">{series_time_status}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Pixel data: </li>
                    <li class="summary-header-date">{pixel_data_status}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Series description: </li>
                    <li class="summary-header-date">{series}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Series Instance UID: </li>
                    <li class="summary-header-date">{series_instance_uid_status}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Accession number: </li>
                    <li class="summary-header-date">{accession_number_status}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Institution Name: </li>
                    <li class="summary-header-date">{institution_name_status}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Series_ID: </li>
                    <li class="summary-header-date">{series_ID}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Modality: </li>
                    <li class="summary-header-date">{modality}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Body part examined: </li>
                    <li class="summary-header-date">{body_part}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Plane of Dicom file: </li>
                    <li class="summary-header-date">{plane}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Scan depth in mm: </li>
                    <li class="summary-header-date">{scan_depth}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Scan thickness in mm: </li>
                    <li class="summary-header-date">{thickness}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Number of slices: </li>
                    <li class="summary-header-date">{number_of_slices}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Timepoints: </li>
                    <li class="summary-header-date">{Max_timepoints}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Dimensions of each scan: </li>
                    <li class="summary-header-date">{dimensions}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Dimensions mismatch if any: </li>
                    <li class="summary-header-date">{mismatch}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Thickness status (uniform/variable): </li>
                    <li class="summary-header-date">{result}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Number of missing dicom files: </li>
                    <li class="summary-header-date">{number_of_missed_slices}</li>
                </ul>
                <ul class="summary-row summary-header">
                    <li class="summary-header-date">Scans missing: </li>
                    <li class="summary-header-date">{status}</li>
                </ul>
                
                
                
            
            </div>
        </div>
                


    <h2 style="text-align:center"> Visual Aspect </h2>
    <center><img src={out} style="background-color: black; padding: 10px;"></center>
    </body>

    </html>"""

    file_one.write(html_display)
    file_one.close()
    exit(0)
    
    
    
    
    
    
        