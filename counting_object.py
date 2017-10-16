import cv2
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# switch camera to video streaming
cap		 = cv2.VideoCapture("8.mp4")
# cap = cv2.VideoCapture(1)

a = []
model_dir = ''
bgsMOG = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

label_lines = [line.rstrip() for line 
	in tf.gfile.GFile("labels.txt")]

def create_graph():

	# Creates graph from saved graph_def.pb.
	with tf.gfile.FastGFile(os.path.join(
		model_dir, 'graph.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

# Download and create graph
create_graph()

def detect(frame):
	with tf.Session() as sess:
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		cv2.imwrite("current_frame.jpg",frame)

		image_data = tf.gfile.FastGFile("./current_frame.jpg", 'rb').read()
		predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})

		predictions = np.squeeze(predictions)

		# change n_pred for more predictions
		n_pred=1
		top_k = predictions.argsort()[-n_pred:][::-1]
		for node_id in top_k:
			human_string_n = label_lines[node_id]
			score = predictions[node_id]
		print ('The animal is {} and score is {}'.format(human_string_n, score))
		return human_string_n

if cap:
	while True:
		ret, frame = cap.read()
		if ret:
			fgmask = bgsMOG.apply(frame)
			# To find the contours of the objects
			_, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			# cv2.drawContours(frame,contours,-1,(0,255,0),cv2.cv.CV_FILLED,32)
			try: hierarchy = hierarchy[0]
			except: hierarchy = []
			a = []
			for contour, hier in zip(contours, hierarchy):
				(x, y, w, h) = cv2.boundingRect(contour)

				if w > 80 and h > 80:
					cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 0, 0), 1)
					(x, y, w, h) = cv2.boundingRect(contour)

					x1 = w / 2
					y1 = h / 2
					cx = x + x1
					cy = y + y1
					a.append([cx, cy])
					print(len(a))
					if len(a) > 4:
						if detect(frame) == 'hog':
							print ('Alarm! A group of Hogs was detected. please trigger the trap!')

			cv2.imshow('BGS', fgmask)
			cv2.imshow('Ori+Bounding Box', frame)
			key = cv2.waitKey(100)
			if key == ord('q'):
				break
cap.release()
cv2.destroyAllWindows()