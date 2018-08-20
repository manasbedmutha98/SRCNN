import  
import numpy as np
import os
import cv2

from keras import backend as K

class SRCNN():
	def __init__(self):
		self.channels = 3
		self.srcnn_model = self.build_model()
		self.isTesting = False
		if(self.isTesting == True):
			self.predict()
		else:
			self.train()

	def preprocess(self):
		

	def psnr(self, y_true, y_pred):
	    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
	    " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape), str(y_pred.shape))
	    return -10.*np.log10(np.mean(np.square(y_pred - y_true)))


	def build_model(self):
		SRCNN = Sequential()
	    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform',
	                     activation='relu', border_mode='valid', bias=True, input_shape=(32, 32, 1)))
	    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
	                     activation='relu', border_mode='same', bias=True))
	    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
	                     activation='linear', border_mode='valid', bias=True))
	    adam = Adam(lr=0.0003)
	    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
		return SRCNN

	def train(self, self.srcnn_model):
	    model = self.srcnn_model
	    print(model.summary())
	    data, label = pd.read_training_data("./train.h5")
	    val_data, val_label = pd.read_training_data("./test.h5")

	    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
	                                 save_weights_only=False, mode='min')
	    callbacks_list = [checkpoint]

	    model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
	                    callbacks=callbacks_list, shuffle=True, nb_epoch=200, verbose=0)
	    


	def predict(self):
	    model = self.srcnn_model
	    model.load_weights("3051crop_weight_200.h5")
	    IMG_NAME = "/home/mark/Engineer/SR/data/Set14/flowers.bmp"
	    INPUT_NAME = "input2.jpg"
	    OUTPUT_NAME = "pre2.jpg"

	    import cv2
	    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
	    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	    shape = img.shape
	    Y_img = cv2.resize(img[:, :, 0], (shape[1] / 2, shape[0] / 2), cv2.INTER_CUBIC)
	    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
	    img[:, :, 0] = Y_img
	    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
	    cv2.imwrite(INPUT_NAME, img)

	    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
	    Y[0, :, :, 0] = Y_img.astype(float) / 255.
	    pre = model.predict(Y, batch_size=1) * 255.
	    pre[pre[:] > 255] = 255
	    pre[pre[:] < 0] = 0
	    pre = pre.astype(numpy.uint8)
	    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
	    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
	    cv2.imwrite(OUTPUT_NAME, img)

	    # psnr calculation:
	    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
	    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
	    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
	    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
	    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
	    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

	    print "bicubic:"
	    print cv2.PSNR(im1, im2)
	    print "SRCNN:"
	    print cv2.PSNR(im1, im3)


	if __name__ == "__main__":
	train()
	    predict()
