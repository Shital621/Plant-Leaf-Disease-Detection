import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras import backend
import tensorflow as tf
global graph,model,sess
from tensorflow.python.keras.backend import set_session
graph = tf.compat.v1.get_default_graph()
sess=tf.compat.v1.Session()
set_session(sess)



def result_str(img_path):
    test_image = image.load_img(img_path, target_size = (224,224 ))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    with graph.as_default():
        set_session(sess)
        model = load_model('md.h5', compile=False)
        
        result = model.predict(test_image)
    prediction=[]
    class_name=['Healthy', 'septoria', 'stripe_rust']
    prediction=class_name[np.argmax(result[0])]
    return prediction
