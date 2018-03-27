import tensorflow as tf 
from tensorflow.example.tutorials.minist import input_data

minist = input_data.read_data_sets('MNIST_data',one_hot =True)
batch_size =100
n_batch =minist.train.num_examples//batch_size




def weight_variable(shape):
	initial =tf.truncated_normal(shape,stddev =0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial =tf.constant(0.1,shape =shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding ='SAME'


x =tf.placeholder(tf.float32,[None,784])
y =tf.placeholder(tf.float32,[None,10]) 
x_image =tf.reshape(x,[-1,28,28,1])#reshape to 2d pic
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

h_conv1 =tf.nn.relu(conv2d(x_image,W_conv1))+b_conv1
h_pool1=max_pool_2x2(h_conv1)

W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2 =tf.nn.relu(conv2d(x_image,W_conv2))+b_conv2
h_pool2=max_pool_2x2(h_conv2)


w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1.7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

prediction =tf.nn.softmax(tf.matmul(h_fc1_drop,))
cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_enropy_with_logits(labels=y,logits=prediction))
train_step=tf.train.AdamOptimizer(le-4).minimize(cross_entropy)
correct_prediction =tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy =tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variable_initializer())
	for epoch in range(21):
		for batch in range(n_batch):
			batch_xs,batch_ys =minist.train.next_batch(batch_size)
			sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
		acc=sess.run(accuracy,feed_dict={x:minist.test.images,y:minist.labels,keep_prob:1.0})
		print("Iter" + str(epoch)+" Testing accuracy = "+str(acc))