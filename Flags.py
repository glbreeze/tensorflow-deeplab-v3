
import tensorflow as tf

flags = tf.app.flags

tf.app.flags.DEFINE_string('port', '', 'kernel')
tf.app.flags.DEFINE_string('mode', '', 'kernel')

flags.DEFINE_string('model_dir', 'model/', 'Directory for the model')

flags.DEFINE_boolean('clean_model_dir', True, 'whether to clean up the model directory')
"""Make sure 'model_checkpoint_path' given in 'checkpoint' file matches """

flags.DEFINE_integer('train_epochs', 26, 'number of training epochs')
"""
'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).'
"""

flags.DEFINE_integer('epochs_per_eval', 1, 'number of training epochs to run between evaluations')

flags.DEFINE_integer('tensorboard_images_max_outputs', 6, 'max number of batch elements to generate for tensorboard')

flags.DEFINE_integer('batch_size', 1, 'batch size')

flags.DEFINE_string('learning_rate_policy', 'poly', 'learning rate policy: poly, piecewise')

flags.DEFINE_integer('max_iter', 30000, 'max iteration for learning rate policy')

flags.DEFINE_string('data_dir', 'dataset/', 'path to the tf records')

flags.DEFINE_string('base_architecture', 'resnet_v2_101', 'base architecture')

flags.DEFINE_string('pre_trained_model', 'ini_checkpoints/resnet_v2_101/resnet_v2_101.ckpt', 'pretrained model')

flags.DEFINE_integer('output_stride', 16, 'output stride')

flags.DEFINE_boolean('freeze_batch_norm', True, 'Freeze batch normalization parameters during the training.')

flags.DEFINE_float('initial_learning_rate', 7e-3, 'Initial learning rate for the optimizer.')

flags.DEFINE_float('end_learning_rate', 1e-6, 'End learning rate for the optimizer.')

flags.DEFINE_integer('initial_global_step', 0, 'Initial global step for controlling learning rate when fine-tuning model.')

flags.DEFINE_float('weight_decay', 2e-4, 'The weight decay to use for regularizing the model.')

flags.DEFINE_boolean('debug', True, 'Whether to use debugger to track down bad values during training.')


#
flags.DEFINE_string('image_dir', 'dataset/VOCdevkit/VOC2012/JPEGImages', 'The directory containing the image data.')

flags.DEFINE_string('output_dir', 'dataset/inference_output', 'Path to the directory to generate inference results')

flags.DEFINE_string('infer_data_list', 'dataset/sample_images_list.txt', 'Path to the file listing the inferring images.')


FLAGS = flags.FLAGS