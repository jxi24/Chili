import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


print(" ".join(tf.sysconfig.get_link_flags()))
