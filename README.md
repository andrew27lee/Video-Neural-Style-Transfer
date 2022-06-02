# Neural Style Transfer for Videos
[![Run on Ainize](https://ainize.ai/static/images/run_on_ainize_button.svg)](https://ainize.ai/andrew27lee/Video-Neural-Style-Transfer?branch=main)

This code extends the [neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer) image processing technique to video by generating smooth transitions between a sequence of reference style images across video frames. The generated output video is a highly altered, artistic representation of the input video consisting of constantly changing abstract patterns and colors that emulate the original content of the video. The user's choice of style reference images and style sequence order allow for infinite user experimentation and the creation of an endless range of artistically interesting videos.

## Deploy Using Docker
```
$ docker build -t <CUSTOM_IMAGE_NAME> .
$ docker run -d -p 5000:5000 <CUSTOM_IMAGE_NAME>
```

## Examples
### Input Video
![file](/static/videos/input/example.gif)
### Style Image Sequence
![file](/static/images/misc/example_sequence.png)
#### Output Video
![file](/static/videos/output/example.gif)
#### Output Video with Preserved Colors
![file](/static/videos/output/example_pc.gif)
