import os
import shutil
input_dir = "/media/andivanov/DATA/dynamic_backgrounds_captured/real_test_videos"
phas_dir = "/media/andivanov/DATA/results/"
report_dir = "/media/andivanov/DATA/report/real_videos_results/"

groupname = "cinema"
clipname = "francesca_cinema_longwalk"
frame = "0176"

outdir = os.path.join(report_dir, clipname)
os.makedirs(outdir, exist_ok=True)

bgr_path = os.path.join(input_dir, groupname, clipname, "bgr", f"{frame}.png")
bgr_path_out = os.path.join(outdir,  f"{clipname}_{frame}_bgr.png")
shutil.copyfile(bgr_path, bgr_path_out)

person_path = os.path.join(input_dir, groupname, clipname, "person", f"{frame}.png")
person_path_out = os.path.join(outdir,  f"{clipname}_{frame}_person.png")
shutil.copyfile(person_path, person_path_out)

attention_pha_path = os.path.join(phas_dir, groupname, clipname,
                                  'attention_f3_offset10_from_scratch', 'epoch-38', "pha", f"{frame}.png")
attention_pha_path_out = os.path.join(outdir,  f"{clipname}_{frame}_pha_attention.png")
shutil.copyfile(attention_pha_path, attention_pha_path_out)

rvm_path_path = os.path.join(phas_dir, groupname, clipname,
                                  'rvm_fromscratch', 'epoch39', "pha", f"{frame}.png")
rvm_pha_path_out = os.path.join(outdir,  f"{clipname}_{frame}_pha_rvm.png")
shutil.copyfile(rvm_path_path, rvm_pha_path_out)