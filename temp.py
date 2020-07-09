
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string("param_name", "default_val", "description")
tf.app.flags.DEFINE_string("train_data_path", "/home/yongcai/chinese_fenci/train.txt", "training data dir")
tf.app.flags.DEFINE_string("log_dir", "./logs", " the log dir")
tf.app.flags.DEFINE_integer("max_sentence_len", 80, "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_size", 50, "embedding size")

tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
