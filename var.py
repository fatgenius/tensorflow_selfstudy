import tensorflow as tf

x= tf.Variable([1,2])
a= tf.constant([3,3])

sub =tf.subtract(x,a)
add = tf.add(x,a)

init =tf.global_variables_initializer()
#all var init


#with tf.Session() as sess:
#	sess.run(init)
#	print(sess.run(sub))
#	print(sess.run(add))

state =tf.Variable(0,name='counter') #give a name of var
new_value =tf.add(state,1) #state add 1
update =tf.assign(state,new_value) #assign value
init =tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print(sess.run(state))
	for _ in range(5):
		sess.run(update)
		print(sess.run(state))