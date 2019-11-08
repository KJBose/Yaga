This Project is meant to identify and recognize emotions in a given image.
Plan of Action and flow of activities 

Step1: Image pre processing to convert JPEG images to tensforflow record data
https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py

step2: Feature Engineering
Identify the appropriate features and organise the input for Training set, Validation set and Test set.

step3: Model development
Build YOLO or an LSTM model, using Keras and PyTorch/TensorFlow.

step4: Train the Model with the Training set.

step5: Validate the accuracy and performance, measure precision ad recall.

step6: Fine tune Hyper Parameters.

step7: Test the model with Test set.

step8: Move to Production by hooking up this application to real Yaga data and feed the insights back to Yaga for decision making and Yaga action.




