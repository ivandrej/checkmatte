# checkmatte

## Training
The attention model is trained with `train_attention.py`. The documentation in the python file
itself gives examples of how to run training.

## Inference
Examples for how to run inference in `single_video_inference.py`

## Contents

#### /model
Contains: 1. all variants of the attention model 2. The modules that are part of RVM, such as ASPP, the encoder, 
the decoder and the refiner.

The main model discussed in the report is the `model_attention_f3.py`. This model is suitable
for training on resolution 227 x 128.

The second model that trains successfully is `model_attention_f4.py`. This model trains successfully with resolution 512 x 288 and 341 x 192. 
