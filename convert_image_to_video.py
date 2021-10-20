import os
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

from mlcandy.media.video_reader import VideoReader
from mlcandy.media.video_writer import VideoWriter


def image_to_video(input_path, output_path, original_path):
    image_list = []
    num_images = len(os.listdir(input_path))
    for idx in range(num_images):
        image_name = '%04d.jpg' % idx
        image_path = os.path.join(input_path, image_name)
        image_list.append(image_path)

    original_video = VideoReader(original_path)
    image_example = Image.open(image_list[0])
    height, width = image_example.size[1], image_example.size[0]

    output_video = VideoWriter(
        output_path,
        height,
        width,
        original_video.fps,
        original_video.streams.audio if original_video.has_audio else None
    )


    # output_video = VideoWriter.from_video(original_video, output_path)
    original_video.reset()

    while True:
        try:
            for image_path in tqdm(image_list):
                frame = next(original_video)
                image_array = np.array(Image.open(image_path).convert('RGB'))
                output_video.write(image_array)
        except StopIteration:
            break

    output_video.close()
    original_video.close()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_path', '--input_path',
						default='/root/lib/DiscoFaceGAN/inference/infer_02/align', type=str)
    parser.add_argument('-output_path', '--output_path',
						default='/root/lib/DiscoFaceGAN/inference/infer_02/align.mp4', type=str)
    parser.add_argument('-original_path', '--original_path',
                        default='/root/lib/DiscoFaceGAN/inference/infer_02/original.mp4', type=str)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    image_to_video(args.input_path, args.output_path, args.original_path)
