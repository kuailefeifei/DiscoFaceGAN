import os

import cv2 as cv
import numpy as np
from PIL import Image

from mlcandy.media.video_reader import VideoReader
from mlcandy.media.video_writer import VideoWriter


def concat_video(input_path_grid, output_path, add_tag=True):
    assert input_path_grid[0][0]
    video_grid = []
    for input_paths in input_path_grid:
        video_grid.append(
            [VideoReader(input_path) if input_path else None for input_path in input_paths]
        )
    m = len(video_grid)
    n = len(video_grid[0])
    height = video_grid[0][0].height
    width = video_grid[0][0].width

    output_video = VideoWriter(
        output_path,
        height * m,
        width * n,
        video_grid[0][0].fps,
        video_grid[0][0].streams.audio if video_grid[0][0].has_audio else None
    )
    blank_frame = np.zeros((height, width, 3))
    while True:
        try:
            hconcat_frames = []
            for i, video_list in enumerate(video_grid):
                frames = []
                for j, video in enumerate(video_list):

                    frame = next(video) if video else blank_frame
                    # print(frame.shape)
                    if add_tag:
                        cv.putText(
                            frame,
                            os.path.basename(input_path_grid[i][j]),
                            (0, frame.shape[0] - 5),
                            cv.FONT_HERSHEY_DUPLEX,
                            0.001 * int(max(frame.shape)),
                            (255, 0, 0)
                        )
                    frames.append(frame)
                hconcat_frames.append(np.concatenate(frames, axis=1))
            output_video.write(np.concatenate(hconcat_frames, axis=0))
        except StopIteration:
            break
    for video_list in video_grid:
        for video in video_list:
            if video is not None:
                video.close()
    output_video.close()


def main():
    main_path = '/root/lib/DiscoFaceGAN/inference/infer_02'
    name_1 = 'align.mp4'
    name_2 = 'disco.mp4'
    output_path = os.path.join(main_path, 'compare.mp4')
    video_1 = os.path.join(main_path, name_1)
    video_2 = os.path.join(main_path, name_2)
    input_path_grid = [[video_1, video_2]]
    concat_video(input_path_grid, output_path)


if __name__ == '__main__':
    main()
