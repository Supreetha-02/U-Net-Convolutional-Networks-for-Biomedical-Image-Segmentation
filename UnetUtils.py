import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, Concatenate

class UnetUtils():
    

    
    def __init__(self):
        pass
    
    def contracting_block(self, input_layer, filters, padding, kernel_size = 3):
        

        
        # two 3x3 convolutions (unpadded convolutions), each followed by
        # a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
        # for downsampling.
        conv = Conv2D(filters = filters, 
                      kernel_size = kernel_size, 
                      activation = tf.nn.relu, 
                      padding = padding)(input_layer)

        conv = Conv2D(filters = filters, 
                      kernel_size = kernel_size, 
                      activation = tf.nn.relu, 
                      padding = padding)(conv)

        pool = MaxPooling2D(pool_size = 2, 
                            strides = 2)(conv)

        return conv, pool

    def bottleneck_block(self, input_layer, filters, padding, kernel_size = 3, strides = 1):
        

        
        # two 3x3 convolutions (unpadded convolutions), each followed by
        # a rectified linear unit (ReLU)
        conv = Conv2D(filters = filters, 
                      kernel_size = kernel_size, 
                      padding = padding,
                      strides = strides, 
                      activation = tf.nn.relu)(input_layer)

        conv = Conv2D(filters = filters, 
                      kernel_size = kernel_size, 
                      padding = padding,
                      strides = strides, 
                      activation = tf.nn.relu)(conv)

        return conv

    def expansive_block(self, input_layer, skip_conn_layer, filters, padding, kernel_size = 3, strides = 1):
        

        
  
        transConv = Conv2DTranspose(filters = filters, 
                                    kernel_size = (2, 2),
                                    strides = 2, 
                                    padding = padding)(input_layer)

        if padding == "valid":
            cropped = self.crop_tensor(skip_conn_layer, transConv)
            concat = Concatenate()([transConv, cropped])
        else:
            concat = Concatenate()([transConv, skip_conn_layer])
        
     
        up_conv = Conv2D(filters = filters, 
                         kernel_size = kernel_size, 
                         padding = padding, 
                         activation = tf.nn.relu)(concat)

        up_conv = Conv2D(filters = filters, 
                         kernel_size = kernel_size, 
                         padding = padding, 
                         activation = tf.nn.relu)(up_conv)

        return up_conv
    
    def crop_tensor(self, source_tensor, target_tensor):
        

        
        target_tensor_size = target_tensor.shape[2]
        source_tensor_size = source_tensor.shape[2]
        
        # calculate the delta to ensure correct cropping.
        delta = source_tensor_size - target_tensor_size
        delta = delta // 2
        
        cropped_source = source_tensor[:, delta:source_tensor_size - delta, delta:source_tensor_size - delta, :]
        
        return cropped_source