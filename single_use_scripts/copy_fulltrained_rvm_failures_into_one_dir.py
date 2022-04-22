import os
import shutil
input_dir = "/media/andivanov/DATA/dynamic_backgrounds_captured/real_test_videos"
phas_dir = "/media/andivanov/DATA/results/"
report_dir = "/media/andivanov/DATA/report/rvm_failures/"

groupname = "office"
clipname = "francesca_office_walk1"
frame = "0080"

outdir = os.path.join(report_dir, clipname)
os.makedirs(outdir, exist_ok=True)

src_path = os.path.join(input_dir, groupname, clipname, "person", f"{frame}.png")
src_path_out = os.path.join(outdir,  f"{clipname}_{frame}_src.png")
shutil.copyfile(src_path, src_path_out)

rvm_path_path = os.path.join(phas_dir, groupname, clipname,
                                  'rvm_fully_trained', "pha", f"{frame}.png")
rvm_pha_path_out = os.path.join(outdir,  f"{clipname}_{frame}_pha_rvm.png")
shutil.copyfile(rvm_path_path, rvm_pha_path_out)
