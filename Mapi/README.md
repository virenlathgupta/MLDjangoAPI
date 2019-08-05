# README

## Star Classifier
**Viren Gupta**

The following Django Application uses a star classifier to produce a trained neural network which can be used to predict whether a star is a pulsar star or not. The app uses conventional neural networks to solve a binary classification problem where the output produced is either a 1 or a 0. 

The app comprises of too specific paths:
1. <server>/App/train 
    - The /App/train path allows a user to instantiate a new neural network where JSON requests are used to specify the parameters such as solver, hidden_layer_sizes and several others. 
2. <server>/App/predict
    -   The /App/predict path takes in data in JSON form along with the model name that was used to instantiate the model in /App/train to predict whether the data in the request belongs to a pulsar star or not. 
