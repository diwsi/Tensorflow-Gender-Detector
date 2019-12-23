# Tensorflow-Gender-Detector
A convolutional neural network implementation sample that detect human genders via Tensorflow and Keras

<b> Training Data </b><br>
I used https://www.kaggle.com/playlist/men-women-classification for training data. Removed useless pictures, cropped from center as square shape and resized to 128x128 pixels.<br><br>
<b>Network Model</b><br>
Tensorflow keras network contains 4 convolutional layers with 3 by 3 filters. Has somoe dropout between dense layers to prevent overfitting.  (trainer.py)<br>
<img src="https://raw.githubusercontent.com/diwsi/diwsi.github.io/master/ao.png" />
<br><b>Training</b><br>
"Sparse Categorical Crossentropy" for loss funtion with adam optimizer over 50 epochs. (trainer.py)
<img src="https://raw.githubusercontent.com/diwsi/diwsi.github.io/master/ao2.PNG" />
<br><b>Prediction</b><br>
After training completed here is the result of test images (predictor.py)
<br><br>
f1.jpg Woman:99% Man:0% <br> 
f2.jpg Woman:99% Man:0% <br>
f3.jpg Woman:99% Man:0% <br>
f4.jpg Woman:99% Man:0% <br>
f5.jpg Woman:99% Man:0% <br>
f6.jpg Woman:100% Man:0% <br>
m1.jpg Woman:43% Man:56% <br>
m2.jpg Woman:0% Man:99% <br>
m3.jpg Woman:0% Man:99% <br>
m4.jpg Woman:28% Man:71% <br>
m5.jpg Woman:2% Man:97% <br>
m6.jpg Woman:99% Man:0% <br>
m7.jpg Woman:0% Man:99% <br>
m8.jpg Woman:0% Man:99% <br>
m9.jpg Woman:0% Man:100% <br>




