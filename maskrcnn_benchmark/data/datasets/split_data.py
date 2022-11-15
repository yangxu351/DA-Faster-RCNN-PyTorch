import os
import random
import argparse
import glob
import sys
sys.path.append('.')
BASE_DIR= '/data/users/yang'

def get_arg(cmt='syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40', syn=True, workbase_data_dir='./real_syn_wdt_vockit'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_base_dir", type=str,
                        help="base path of synthetic data",
                        default=f'{BASE_DIR}/data/synthetic_data_wdt')

    parser.add_argument("--syn_data_dir", type=str, default='{}/{}',
                        help="Path to folder containing synthetic images and annos \{syn_base_dir\}/{cmt}")

    parser.add_argument("--syn_data_imgs_dir", type=str, default='{}/{}_images',
                        help="Path to folder containing synthetic images .jpg \{cmt\}/{cmt}_images")   

    parser.add_argument("--syn_data_segs_dir", type=str, default='{}/{}_annos_dilated',
                        help="Path to folder containing synthetic SegmentationClass .jpg \{cmt\}/{cmt}_annos_dilated")  

    parser.add_argument("--syn_voc_annos_dir", type=str, default='{}/{}_xml_annos/minr{}_linkr{}_px{}whr{}_all_xml_annos',
                        help="syn annos in voc format .xml \{syn_base_dir\}/{cmt}_xml_annos/minr{}_linkr{}_px{}whr{}_all_annos_with_bbox")     
    
    parser.add_argument("--real_base_dir", type=str,default=f'{BASE_DIR}/data/wind_turbine', help="base path of synthetic data")
    parser.add_argument("--real_imgs_dir", type=str, default='{}/{}_crop', help="Path to folder containing real images")
    parser.add_argument("--real_yolo_annos_dir", type=str, default='{}/{}_crop_label_xcycwh', help="Path to folder containing real annos of yolo format")
    parser.add_argument("--real_voc_annos_dir", type=str, default='{}/{}_crop_label_xml_annos', help="Path to folder containing real annos of yolo format")
    # tag: yang adds
    parser.add_argument("--real_data_dir", type=str,default=f'{BASE_DIR}/data/real_data_wdt', help="base path of real  data wdt")
    parser.add_argument("--real_img_aug_dir", type=str, default='{}/{}/JPEGImages_aug', help="Path to folder containing real images")
    parser.add_argument("--real_voc_annos_aug_dir", type=str, default='{}/{}/Annotations_aug', help="Path to folder containing real annos of voc format")
    parser.add_argument("--real_main_dir", type=str, default='{}/{}/ImageSets/Main',
                        help="\{real_data_dir\}/\{cmt\}/ImageSets/Main ")
    
    parser.add_argument("--workdir_data", type=str, default='{}/{}',
                        help="workdir data base synwdt ./real_syn_wdt_vockit/\{cmt\}")
    # parser.add_argument("--workdir_imgsets", type=str, default='{}/ImageSets',
    #                     help="ImageSets folder")
    parser.add_argument("--workdir_main", type=str, default='{}/Main',
                        help="\{workdir_data\}/Main ")                
    parser.add_argument("--min_region", type=int, default=10, help="the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=10,  help="the #pixels between two connected components to be grouped")
    parser.add_argument("--px_thres", type=int, default=12, help="the smallest #pixels to form an edge")
    parser.add_argument("--whr_thres", type=int, default=5, help="ratio threshold of w/h or h/w")                        
    args = parser.parse_args()
    if syn:
        args.syn_data_dir = args.syn_data_dir.format(args.syn_base_dir, cmt)
        args.syn_data_imgs_dir = args.syn_data_imgs_dir.format(args.syn_data_dir, cmt)
        args.syn_data_segs_dir = args.syn_data_segs_dir.format(args.syn_data_dir, cmt)
        args.syn_voc_annos_dir = args.syn_voc_annos_dir.format(args.syn_base_dir, cmt, args.link_r, args.min_region, args.px_thres, args.whr_thres)
        
    else:
        args.real_imgs_dir = args.real_imgs_dir.format(args.real_base_dir, cmt)
        args.real_yolo_annos_dir = args.real_yolo_annos_dir.format(args.real_base_dir, cmt)
        args.real_voc_annos_dir = args.real_voc_annos_dir.format(args.real_base_dir, cmt)
        # tag: 
        args.real_img_aug_dir = args.real_img_aug_dir.format(args.real_data_dir, cmt)
        args.real_voc_annos_aug_dir = args.real_voc_annos_aug_dir.format(args.real_data_dir, cmt)
        args.real_main_dir = args.real_main_dir.format(args.real_data_dir, cmt)

    
    args.workdir_data = args.workdir_data.format(workbase_data_dir, cmt)
    # args.workdir_imgsets = args.workdir_imgsets.format(args.workdir_data)
    args.workdir_main = args.workdir_main.format(args.workdir_data)

    return args

if __name__ == '__main__':
    data_seed = 0
    # data_seed = 1
    # data_seed = 2
    
    

    # cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    # syn = True
    # val_rate = 0.3
     
    cmt = 'xilin_wdt'
    syn = False
    val_rate = 0.3
    # aug_val = False
    aug_val = True
    

    # cmt = 'DJI_wdt'
    # syn = False
    # val_rate = 1

    args = get_arg(cmt, syn)
    if syn:
        files = glob.glob(os.path.join(args.syn_voc_annos_dir, '*.xml'))
    else:
        files = glob.glob(os.path.join(args.real_voc_annos_dir, '*.xml'))
        if aug_val:
            aug_files = glob.glob(os.path.join(args.real_voc_annos_aug_dir, '*.xml'))
            aug_names = [os.path.basename(f).split(".")[0] for f in aug_files]
    #tag: for validation
    # for f in files:
    #     file_name = os.path.basename(f)
    #     print(file_name)
    files_name = sorted([os.path.basename(f).split(".")[0] for f in files]) 
    files_num = len(files_name)
    random.seed(data_seed)  # 设置随机种子，保证随机结果可复现
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    # print(val_index)
    train_file_names = []
    val_file_names = []
    for index, f in enumerate(files):
        file_name = os.path.basename(f).split(".")[0]
        if index in val_index:
            val_file_names.append(file_name)
        else:
            train_file_names.append(file_name)
        if aug_val and file_name in aug_names:
            aug_names.remove(file_name)
    # tag: for validation
    print('val_file_names', val_file_names)

    if aug_val and len(aug_names):
        val_file_names.extend(aug_names)
    
    print('len val files', len(val_file_names))
    try:
        if aug_val:
            train_f = open(os.path.join(args.real_main_dir, f"train_seed{data_seed}.txt"), "w")
            eval_f = open(os.path.join(args.real_main_dir, f"val_aug_seed{data_seed}.txt"), "w")
            train_f.write("\n".join(train_file_names))
            eval_f.write("\n".join(val_file_names))
            
        else:
            all_f = open(os.path.join(args.workdir_main, "all.txt"), "w")
            train_f = open(os.path.join(args.workdir_main, f"train_seed{data_seed}.txt"), "w")
            eval_f = open(os.path.join(args.workdir_main, f"val_seed{data_seed}.txt"), "w")
            train_f.write("\n".join(train_file_names))
            eval_f.write("\n".join(val_file_names))
            all_f.write("\n".join(train_file_names+val_file_names))
            all_f.close()
        train_f.close()
        eval_f.close()
    except FileExistsError as e:
        print(e)
        exit(1)


