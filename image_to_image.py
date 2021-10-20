import os
import argparse
import pickle

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from scipy.io import loadmat,savemat
from DiscoFaceGAN.renderer import face_decoder
from DiscoFaceGAN.training.networks_recon import R_Net
from DiscoFaceGAN.training import misc
from DiscoFaceGAN.preprocess.preprocess_utils import *
import DiscoFaceGAN.dnnlib as dnnlib
import DiscoFaceGAN.dnnlib.tflib as tflib
import DiscoFaceGAN.config as config


model_continue_path = 'training/pretrained_weights/recon_net'
R_net_weights = os.path.join(model_continue_path,'FaceReconModel')
config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'


# define mapping network from z space to lambda space
def CoeffDecoder(z, ch_depth=3, ch_dim=512, coeff_length=128):
    with tf.variable_scope('stage1'):
        with tf.variable_scope('decoder'):
            y = z
            for i in range(ch_depth):
                y = tf.layers.dense(y, ch_dim, tf.nn.relu, name='fc' + str(i))

            x_hat = tf.layers.dense(y, coeff_length, name='x_hat')
            x_hat = tf.stop_gradient(x_hat)

    return x_hat


# restore pre-trained weights
def restore_weights_and_initialize():
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()

    # add batch normalization params into trainable variables
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    var_id_list = [v for v in var_list if 'id' in v.name and 'stage1' in v.name]
    var_exp_list = [v for v in var_list if 'exp' in v.name and 'stage1' in v.name]
    var_gamma_list = [v for v in var_list if 'gamma' in v.name and 'stage1' in v.name]
    var_rot_list = [v for v in var_list if 'rot' in v.name and 'stage1' in v.name]

    saver_id = tf.train.Saver(var_list=var_id_list, max_to_keep=100)
    saver_exp = tf.train.Saver(var_list=var_exp_list, max_to_keep=100)
    saver_gamma = tf.train.Saver(var_list=var_gamma_list, max_to_keep=100)
    saver_rot = tf.train.Saver(var_list=var_rot_list, max_to_keep=100)

    saver_id.restore(tf.get_default_session(), './vae/weights/id/stage1_epoch_395.ckpt')
    saver_exp.restore(tf.get_default_session(), './vae/weights/exp/stage1_epoch_395.ckpt')
    saver_gamma.restore(tf.get_default_session(), './vae/weights/gamma/stage1_epoch_395.ckpt')
    saver_rot.restore(tf.get_default_session(), './vae/weights/rot/stage1_epoch_395.ckpt')


def z_to_lambda_mapping(latents):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        with tf.variable_scope('id'):
            IDcoeff = CoeffDecoder(z=latents[:, :128], coeff_length=160, ch_dim=512, ch_depth=3)
        with tf.variable_scope('exp'):
            EXPcoeff = CoeffDecoder(z=latents[:, 128:128 + 32], coeff_length=64, ch_dim=256, ch_depth=3)
        with tf.variable_scope('gamma'):
            GAMMAcoeff = CoeffDecoder(z=latents[:, 128 + 32:128 + 32 + 16], coeff_length=27, ch_dim=128, ch_depth=3)
        with tf.variable_scope('rot'):
            Rotcoeff = CoeffDecoder(z=latents[:, 128 + 32 + 16:128 + 32 + 16 + 3], coeff_length=3, ch_dim=32,
                                    ch_depth=3)

        INPUTcoeff = tf.concat([IDcoeff, EXPcoeff, Rotcoeff, GAMMAcoeff], axis=1)

        return INPUTcoeff


# generate images using attribute-preserving truncation trick
def truncate_generation(Gs, inputcoeff, rate=0.7, dlatent_average_id=None):
    if dlatent_average_id is None:
        url_pretrained_model_ffhq_average_w_id = 'https://drive.google.com/uc?id=17L6-ENX3NbMsS3MSCshychZETLPtJnbS'
        with dnnlib.util.open_url(url_pretrained_model_ffhq_average_w_id, cache_dir=config.cache_dir) as f:
            dlatent_average_id = np.loadtxt(f)
    dlatent_average_id = np.reshape(dlatent_average_id, [1, 14, 512]).astype(np.float32)
    dlatent_average_id = tf.constant(dlatent_average_id)

    inputcoeff_id = tf.concat([inputcoeff[:, :160], tf.zeros([1, 126])], axis=1)
    dlatent_out = Gs.components.mapping.get_output_for(inputcoeff, None, is_training=False,
                                                       is_validation=True)  # original w space output
    dlatent_out_id = Gs.components.mapping.get_output_for(inputcoeff_id, None, is_training=False, is_validation=True)

    dlatent_out_trun = dlatent_out + (dlatent_average_id - dlatent_out_id) * (1 - rate)
    dlatent_out_final = tf.concat([dlatent_out_trun[:, :8, :], dlatent_out[:, 8:, :]],
                                  axis=1)  # w space latent vector with truncation trick

    fake_images_out = Gs.components.synthesis.get_output_for(dlatent_out_final, randomize_noise=False)
    fake_images_out = tf.clip_by_value((fake_images_out + 1) * 127.5, 0, 255)
    fake_images_out = tf.transpose(fake_images_out, perm=[0, 2, 3, 1])

    return fake_images_out


# calculate average w space latent vector with zero expression, lighting, and pose.
def get_model_and_average_w_id(model_name):
    G, D, Gs = misc.load_pkl(model_name)
    average_w_name = model_name.replace('.pkl', '-average_w_id.txt')
    if not os.path.isfile(average_w_name):
        print('Calculating average w id...\n')
        latents = tf.placeholder(tf.float32, name='latents', shape=[1, 128 + 32 + 16 + 3])
        noise = tf.placeholder(tf.float32, name='noise', shape=[1, 32])
        INPUTcoeff = z_to_lambda_mapping(latents)
        INPUTcoeff_id = INPUTcoeff[:, :160]
        INPUTcoeff_w_noise = tf.concat([INPUTcoeff_id, tf.zeros([1, 64 + 27 + 3]), noise], axis=1)
        dlatent_out = Gs.components.mapping.get_output_for(INPUTcoeff_w_noise, None, is_training=False,
                                                           is_validation=True)
        restore_weights_and_initialize()
        np.random.seed(1)
        average_w_id = []
        for i in range(50000):
            lats = np.random.normal(size=[1, 128 + 32 + 16 + 3])
            noise_ = np.random.normal(size=[1, 32])
            w_out = tflib.run(dlatent_out, {latents: lats, noise: noise_})
            average_w_id.append(w_out)

        average_w_id = np.concatenate(average_w_id, axis=0)
        average_w_id = np.mean(average_w_id, axis=0)
        np.savetxt(average_w_name, average_w_id)
    else:
        average_w_id = np.loadtxt(average_w_name)

    return Gs, average_w_id


def parse_args():
    desc = "Data Preprocess of DisentangledFaceGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--image_path', type=str, default='/root/lib/Deep3DFaceReconstruction/input/000002.jpg', help='Training image path.')
    parser.add_argument('--lm_path', type=str, default='/root/lib/Deep3DFaceReconstruction/input/', help='Deteced landmark path.')
    parser.add_argument('--model', type=str, default=None, help='pkl file name of the generator. If None, use the default pre-trained model.')

    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image_path
    lm_path = args.lm_path
    # lm_path = os.path.join(args.image_path,'lm5p') # detected landmarks for training images should be saved in <image_path>/lm5p subfolder

    # Load BFM09 face model
    if not os.path.isfile('./renderer/BFM face model/BFM_model_front_gan.mat'):
        transferBFM09()

    # Load standard landmarks for alignment
    lm3D = load_lm3d()

    # Build reconstruction model
    with tf.Graph().as_default() as graph:

        images = tf.placeholder(name='input_imgs', shape=[None, 224, 224, 3], dtype=tf.float32)
        Face3D = face_decoder.Face3D()  # analytic 3D face formation process
        coeff = R_Net(images, is_training=False)  # 3D face reconstruction network

        with tf.Session(config=config) as sess:

            var_list = tf.trainable_variables()
            g_list = tf.global_variables()

            # Add batch normalization params into trainable variables
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars

            # Create saver to save and restore weights
            resnet_vars = [v for v in var_list if 'resnet_v1_50' in v.name]
            res_fc = [v for v in var_list if
                      'fc-id' in v.name or 'fc-ex' in v.name or 'fc-tex' in v.name or 'fc-angles' in v.name or 'fc-gamma' in v.name or 'fc-XY' in v.name or 'fc-Z' in v.name or 'fc-f' in v.name]
            resnet_vars += res_fc

            saver = tf.train.Saver(var_list=var_list, max_to_keep=100)
            saver.restore(sess, R_net_weights)

            image_path = args.image_path
            file = image_path.split('/')[-1]

            # load images and landmarks
            image = Image.open(image_path)
            lm = np.loadtxt(os.path.join(lm_path, file.replace('jpg', 'txt')))
            lm = np.reshape(lm, [5, 2])

            # align image for 3d face reconstruction
            align_img, _, _ = Preprocess(image, lm, lm3D)  # 512*512*3 RGB image
            align_img = np.array(align_img)

            align_img_ = align_img[:, :, ::-1]  # RGBtoBGR
            align_img_ = cv2.resize(align_img_,
                                    (224, 224))  # input image to reconstruction network should be 224*224
            align_img_ = np.expand_dims(align_img_, 0)
            coef = sess.run(coeff, feed_dict={images: align_img_})

            # align image for GAN training
            # eliminate translation and rescale face size to proper scale
            rescale_img = crop_n_rescale_face_region(align_img, coef)  # 256*256*3 RGB image
            coef = np.squeeze(coef, 0)

            # save aligned images and extracted coefficients
            # cv2.imwrite(os.path.join(save_path, 'img', file), rescale_img[:, :, ::-1])
            # savemat(os.path.join(save_path, 'coeff', file.replace('.png', '.mat')), {'coeff': coef})

    coef = np.expand_dims(coef, 0)
    # save path for generated images
    save_path = 'test_images'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    resume_pkl = ''

    tflib.init_tf()

    with tf.device('/gpu:0'):

        # Use default pre-trained model
        if args.model is None:
            url_pretrained_model_ffhq = 'https://drive.google.com/uc?id=1nT_cf610q5mxD_jACvV43w4SYBxsPUBq'
            Gs = load_Gs(url_pretrained_model_ffhq)
            average_w_id = None

        else:
            Gs,average_w_id = get_model_and_average_w_id(args.model)
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
        # average_w_id = average w space latent vector with zero expression, lighting, and pose.

        # Print network details.
        Gs.print_layers()

        # Pick latent vector.
        # latents = tf.placeholder(tf.float32, name='latents', shape=[1,128+32+16+3])
        noise = tf.placeholder(tf.float32, name='noise', shape=[1,32])
        # INPUTcoeff = z_to_lambda_mapping(latents)
        INPUTcoeff = tf.placeholder(tf.float32, name='coeff', shape=[1,254])
        INPUTcoeff_w_noise = tf.concat([INPUTcoeff,noise],axis = 1)

        # Generate images
        fake_images_out = truncate_generation(Gs,INPUTcoeff_w_noise,dlatent_average_id=average_w_id)

    # restore_weights_and_initialize()

    np.random.seed(1)

    # lats1 = np.random.normal(size=[1,128+32+16+3])
    noise_ = np.random.normal(size=[1,32])

    fake = tflib.run(fake_images_out, {coeff:coef[:, :254],noise:noise_})
    PIL.Image.fromarray(fake[0].astype(np.uint8), 'RGB').save(os.path.join(save_path,'%03d_%02d.jpg'%(0,0)))


if __name__ == '__main__':
    main()
