"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""


RVM_DATA_PATHS = {
    'videomatte': {
        'train': '/media/andivanov/DATA/VideoMatte240K_JPEG_HD_dummy_train/train',
        'valid': '/media/andivanov/DATA/VideoMatte240K_JPEG_HD_dummy_train/valid',
    },
    'background_videos': {
        'train': '/media/andivanov/DATA/DVM_JPEG/train',
        'valid': '/media/andivanov/DATA/DVM_JPEG/valid',
    }
}

SPECIALIZED_DATA_PATHS = {
    'videomatte': {
        'train': '/media/andivanov/DATA/VideoMatte240K_JPEG_SD/train',
        'valid': '/media/andivanov/DATA/VideoMatte240K_JPEG_SD/valid',
    },
    # Currently we pass a directory of videos. Make sure this directory contains a single video folder
    'background_video': {
        'dvm0013': {
            'train': '/media/andivanov/DATA/training_datasets/specialized_iteration_1/bgr/train',
            'valid': '/media/andivanov/DATA/training_datasets/specialized_iteration_1/bgr/valid'
        },
        'room': {
            'train': '/media/andivanov/DATA/training_datasets/specialized_iteration_1/room/train',
            'valid': '/media/andivanov/DATA/training_datasets/specialized_iteration_1/room/valid'
        }
    },
    'DVM': {
        'train': '/media/andivanov/DATA/DVM_JPEG_nohuman/train',
        'valid': '/media/andivanov/DATA/DVM_JPEG_nohuman/valid',
    }
}
