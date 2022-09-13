from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D

#### X1 * H1 + X2 * H2

####PADDING = 0 , STRIDE = 1 
#######################################################################################################
# # define input data
# # data = [[0.2, 1, 0],
# # 		[-1, 0, -0.1],
# # 		[0.1, 0, 0.1]]

# data = [[1, 0.5, 0.2],
#         [-1, -0.5, -0.2],
#         [0.1, -0.1, 0]]
# data = asarray(data)
# data = data.reshape(1, 3, 3, 1)


# # create model
# model = Sequential()
# model.add(Conv2D(1, (2,2), input_shape=(3, 3, 1)))
# # summarize model
# # model.summary()


# # define a vertical line detector
# # detector = [[[[1]],[[-0.1]]],
# #             [[[1]],[[-0.1]]]]

# detector = [[[[0.5]],[[0.5]]],
#             [[[-0.5]],[[-0.5]]]]
# weights = [asarray(detector), asarray([0.0])]
# # store the weights in the model
# model.set_weights(weights)


# # apply filter to input data
# yhat = model.predict(data)


# # enumerate rows
# for r in range(yhat.shape[1]):
# 	# print each column in the row
# 	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])

####PADDING = 1, STRIDE = 1
######################################################################################################

# define input data
# data = [[0, 0, 0, 0, 0], 
#       [0, 0.2, 1, 0, 0],
# 		[0, -1, 0, -0.1, 0],
# 		[0, 0.1, 0, 0.1, 0],
#       [0, 0, 0, 0, 0]]

# data = [ [0, 0, 0, 0, 0],
#         [0, 1, 0.5, 0.2, 0],
#         [0, -1, -0.5, -0.2, 0],
#         [0, 0.1, -0.1, 0, 0],
#         [0, 0, 0, 0, 0]]
# data = asarray(data)
# data = data.reshape(1, 5, 5, 1)


# # create model
# model = Sequential()
# model.add(Conv2D(1, (2,2), input_shape=(5, 5, 1)))
# summarize model
# model.summary()


# define a vertical line detector
# detector = [[[[1]],[[-0.1]]],
#             [[[1]],[[-0.1]]]]

# detector = [[[[0.5]],[[0.5]]],
#             [[[-0.5]],[[-0.5]]]]
# weights = [asarray(detector), asarray([0.0])]
# # store the weights in the model
# model.set_weights(weights)


# # apply filter to input data
# yhat = model.predict(data)


# # enumerate rows
# for r in range(yhat.shape[1]):
# 	# print each column in the row
# 	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])

########################################################################################################
#### PADDING  = 1 AND STRIDE = 2 
#### from previous answer
### pick 1st and 3rd in first row 
### pick 1st and 3rd in third row

#######################################################################################################
### PADDING  = 0 AND STRIDE = 2 AND DILATION = 2 
# define input data
data  = [[0, 0, 0, 0, 0], 
      [0, -0.6, 1, 0.1, 0],
		[0,0.7, 0.8, 0.4, 0],
		[0, 0.8, 0.8, -0.5, 0],
      [0, 0, 0, 0, 0]]

# data = [[-0.6, 1, 0.1],
#         [0.7, 0.8, 0.4],
#         [0.8, 0.8, -0.5]]

data = asarray(data)
data = data.reshape(1, 5, 5, 1)


# create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(5, 5, 1)))
# summarize model
model.summary()


# define a vertical line detector
detector = [[[[1.1]],[[0.0]],[[8.2]]],
            [[[0.0]],[[0.0]],[[0.0]]],
            [[[-2.8]],[[0.0]],[[4.7]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)


# apply filter to input data
yhat = model.predict(data)


# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])

#####################################################################################################

## THREE DATA INPUT 
# define input data
# data = [[0.2, 1, 0],
# 		[-1, 0, -0.1],
# 		[0.1, 0, 0.1]]

# data = [[1, 0.5, 0.2],
#         [-1, -0.5, -0.2],
#         [0.1, -0.1, 0]]

# data = [[0.5, -0.5, -0.1],
#         [0, -0.4, 0],
#         [0.5, 0.5, 0.2]]

# data = asarray(data)
# data = data.reshape(1, 3, 3, 1)


# # create model
# model = Sequential()
# model.add(Conv2D(1, (1,1), input_shape=(3, 3, 1)))
# # summarize model
# # model.summary()


# # define a vertical line detector
# # detector = [[[[1]],[[-0.1]]],
# #             [[[1]],[[-0.1]]]]

# detector = [[[[0.5]]]]
# weights = [asarray(detector), asarray([0.0])]
# # store the weights in the model
# model.set_weights(weights)


# # apply filter to input data
# yhat = model.predict(data)


# # enumerate rows
# for r in range(yhat.shape[1]):
# 	# print each column in the row
# 	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])




