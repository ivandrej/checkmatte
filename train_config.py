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


DATA_PATHS = {
    
    'videomatte': {
        'train': '/media/andivanov/DATA/VideoMatte240K_JPEG_HD_dummy_train/train',
        'valid': '/media/andivanov/DATA/VideoMatte240K_JPEG_HD_dummy_train/valid',
    },
    'background_videos': {
        'train': '/media/andivanov/DATA/DVM_JPEG/train',
        'valid': '/media/andivanov/DATA/DVM_JPEG/valid',
    }
}
