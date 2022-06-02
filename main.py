import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import glob
import cv2
from config import Config
from flask import Flask, request, render_template

CONTENT_PATH = 'static/videos/input/content.mp4'
STYLE_1_PATH = 'static/images/style/style1.png'
STYLE_2_PATH = 'static/images/style/style2.png'
STYLE_3_PATH = 'static/images/style/style3.png'

app = Flask(__name__)


class StyleFrame:

    MAX_CHANNEL_INTENSITY = 255.0

    def __init__(self, conf=Config):
        self.conf = conf
        os.environ['TFHUB_CACHE_DIR'] = self.conf.TENSORFLOW_CACHE_DIRECTORY
        self.hub_module = hub.load(self.conf.TENSORFLOW_HUB_HANDLE)
        self.input_frame_directory = glob.glob(f'{self.conf.INPUT_FRAME_DIRECTORY}/*')
        self.output_frame_directory = glob.glob(f'{self.conf.OUTPUT_FRAME_DIRECTORY}/*')
        self.style_directory = glob.glob(f'{self.conf.STYLE_REF_DIRECTORY}/*')
        self.ref_count = len(self.conf.STYLE_SEQUENCE)

        files_to_be_cleared = self.output_frame_directory
        if self.conf.CLEAR_INPUT_FRAME_CACHE:
            files_to_be_cleared += self.input_frame_directory
        
        for file in files_to_be_cleared:
            os.remove(file)
        
        self.input_frame_directory = glob.glob(f'{self.conf.INPUT_FRAME_DIRECTORY}/*')
        self.output_frame_directory = glob.glob(f'{self.conf.OUTPUT_FRAME_DIRECTORY}/*')

        if len(self.input_frame_directory):
            self.frame_width = cv2.imread(self.input_frame_directory[0]).shape[1]

    def get_input_frames(self):
        vid_obj = cv2.VideoCapture(self.conf.INPUT_VIDEO_PATH)
        frame_interval = np.floor((1.0 / self.conf.INPUT_FPS) * 1000)
        success, image = vid_obj.read()
        scale_constant = (self.conf.FRAME_HEIGHT / image.shape[0])
        self.frame_width = int(image.shape[1] * scale_constant)
        image = cv2.resize(image, (self.frame_width, self.conf.FRAME_HEIGHT))
        cv2.imwrite(self.conf.INPUT_FRAME_PATH.format(0), image.astype(np.uint8))

        count = 1
        while success:
            msec_timestamp = count * frame_interval
            vid_obj.set(cv2.CAP_PROP_POS_MSEC, msec_timestamp)
            success, image = vid_obj.read()
            if not success:
                break
            image = cv2.resize(image, (self.frame_width, self.conf.FRAME_HEIGHT))
            cv2.imwrite(self.conf.INPUT_FRAME_PATH.format(count), image.astype(np.uint8))
            count += 1
        self.input_frame_directory = glob.glob(f'{self.conf.INPUT_FRAME_DIRECTORY}/*')

    def get_style_info(self):
        frame_length = len(self.input_frame_directory)
        style_refs = list()
        style_files = sorted(self.style_directory)
        self.t_const = frame_length if self.ref_count == 1 else np.ceil(frame_length / (self.ref_count - 1))

        first_style_ref = cv2.imread(style_files.pop(0))
        first_style_ref = cv2.cvtColor(first_style_ref, cv2.COLOR_BGR2RGB)
        first_style_height, first_style_width, _ = first_style_ref.shape
        style_refs.append(first_style_ref / self.MAX_CHANNEL_INTENSITY)

        for filename in style_files:
            style_ref = cv2.imread(filename)
            style_ref = cv2.cvtColor(style_ref, cv2.COLOR_BGR2RGB)
            style_ref_height, style_ref_width, _ = style_ref.shape
            if style_ref_width != first_style_width or style_ref_height != first_style_height:
                style_ref = cv2.resize(style_ref, (first_style_width, first_style_height))
            style_refs.append(style_ref / self.MAX_CHANNEL_INTENSITY)

        self.transition_style_seq = list()
        for i in range(self.ref_count):
            if self.conf.STYLE_SEQUENCE[i] is None:
                self.transition_style_seq.append(None)
            else:
                self.transition_style_seq.append(style_refs[self.conf.STYLE_SEQUENCE[i]])

    def _trim_img(self, img):
        return img[:self.conf.FRAME_HEIGHT, :self.frame_width]

    def get_output_frames(self, color):
        self.input_frame_directory = glob.glob(f'{self.conf.INPUT_FRAME_DIRECTORY}/*')
        ghost_frame = None
        for count, filename in enumerate(sorted(self.input_frame_directory)):
            content_img = cv2.imread(filename) 
            content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB) / self.MAX_CHANNEL_INTENSITY
            curr_style_img_index = int(count / self.t_const)
            mix_ratio = 1 - ((count % self.t_const) / self.t_const)
            inv_mix_ratio = 1 - mix_ratio

            prev_image = self.transition_style_seq[curr_style_img_index]
            next_image = self.transition_style_seq[curr_style_img_index + 1]
            
            prev_is_content_img = False
            next_is_content_img = False
            if prev_image is None:
                prev_image = content_img
                prev_is_content_img = True
            if next_image is None:
                next_image = content_img
                next_is_content_img = True
            if prev_is_content_img and next_is_content_img:
                temp_ghost_frame = cv2.cvtColor(ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
                cv2.imwrite(self.conf.OUTPUT_FRAME_PATH.format(count), temp_ghost_frame)
                continue
            
            if count > 0:
                content_img = ((1 - self.conf.GHOST_FRAME_TRANSPARENCY) * content_img) + (self.conf.GHOST_FRAME_TRANSPARENCY * ghost_frame)
            content_img = tf.cast(tf.convert_to_tensor(content_img), tf.float32)

            if prev_is_content_img:
                blended_img = next_image
            elif next_is_content_img:
                blended_img = prev_image
            else:
                prev_style = mix_ratio * prev_image
                next_style = inv_mix_ratio * next_image
                blended_img = prev_style + next_style

            blended_img = tf.cast(tf.convert_to_tensor(blended_img), tf.float32)
            expanded_blended_img = tf.constant(tf.expand_dims(blended_img, axis=0))
            expanded_content_img = tf.constant(tf.expand_dims(content_img, axis=0))
            stylized_img = self.hub_module(expanded_content_img, expanded_blended_img).pop()
            stylized_img = tf.squeeze(stylized_img)

            if prev_is_content_img:
                prev_style = mix_ratio * content_img
                next_style = inv_mix_ratio * stylized_img
            if next_is_content_img:
                prev_style = mix_ratio * stylized_img
                next_style = inv_mix_ratio * content_img
            if prev_is_content_img or next_is_content_img:
                stylized_img = self._trim_img(prev_style) + self._trim_img(next_style)

            if color:
                stylized_img = self._color_correct_to_input(content_img, stylized_img)
            
            ghost_frame = np.asarray(self._trim_img(stylized_img))

            temp_ghost_frame = cv2.cvtColor(ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
            cv2.imwrite(self.conf.OUTPUT_FRAME_PATH.format(count), temp_ghost_frame)
        self.output_frame_directory = glob.glob(f'{self.conf.OUTPUT_FRAME_DIRECTORY}/*')

    def _color_correct_to_input(self, content, generated):
        content = np.array((content * self.MAX_CHANNEL_INTENSITY), dtype=np.float32)
        content = cv2.cvtColor(content, cv2.COLOR_BGR2YCR_CB)
        generated = np.array((generated * self.MAX_CHANNEL_INTENSITY), dtype=np.float32)
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2YCR_CB)
        generated = self._trim_img(generated)
        color_corrected = np.zeros(generated.shape, dtype=np.float32)
        color_corrected[:, :, 0] = generated[:, :, 0]
        color_corrected[:, :, 1] = content[:, :, 1]
        color_corrected[:, :, 2] = content[:, :, 2]
        return cv2.cvtColor(color_corrected, cv2.COLOR_YCrCb2BGR) / self.MAX_CHANNEL_INTENSITY


    def create_video(self):
        self.output_frame_directory = glob.glob(f'{self.conf.OUTPUT_FRAME_DIRECTORY}/*')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(self.conf.OUTPUT_VIDEO_PATH, fourcc, self.conf.OUTPUT_FPS, (self.frame_width, self.conf.FRAME_HEIGHT))

        for _, filename in enumerate(sorted(self.output_frame_directory)):
            image = cv2.imread(filename)
            video_writer.write(image)

        video_writer.release()

    def run(self, color):
        self.get_input_frames()
        self.get_style_info()
        self.get_output_frames(color)
        self.create_video()


@app.route('/transfer', methods=['POST'])
def transfer_outcome():
    content = request.files['content']
    style_1 = request.files['style1']
    style_2 = request.files['style2']
    style_3 = request.files['style3']

    if content.filename == '' or style_1.filename == '' or style_2.filename == '' or style_3.filename == '':
        return render_template('index.html', error='Input file(s) not selected.')

    content.save(os.path.join(app.root_path, CONTENT_PATH))
    style_1.save(os.path.join(app.root_path, STYLE_1_PATH))
    style_2.save(os.path.join(app.root_path, STYLE_2_PATH))
    style_3.save(os.path.join(app.root_path, STYLE_3_PATH))

    if request.form.get('preserve_color'):
        StyleFrame().run(True)
    else:
        StyleFrame().run(False)
    
    return render_template('index.html', transfer=True)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', index=True)


if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0')
