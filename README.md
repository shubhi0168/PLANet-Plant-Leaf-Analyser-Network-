# PLANet - Plant Leaf Analyser Network
It is observed that a lot of crops gets wasted every year due to the spread of diseases (the reasons for this include pests, bacterial or fungal infections). Farmers face heavy losses because of the lack of handy and economical tools that can detect such diseases in their initial stages. Our system can provide a remedy to this problem. Our goal is to develop a web application that recognizes diseases in their initial stages and recommend appropriate pesticides/insecticides. The aim is to build a deployable system for plant disease detection by training Convolutional Neural Networks that detect the specific disease along with the specific species. We are using Plant-Village Dataset for this project. Due to skewed data, we are doing data augmentation using opencv and creating new training data by using Deep Convolutional Generative Adversarial Network(DCGAN). The target leaf shall be segmented out of a live video feed and fed to the models.

Modules used in this project:
- keras
- flask
- django
- opencv
