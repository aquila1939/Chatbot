# Chatbot
This chatbot is limited in its capability of conversing, but I'll build on that in the near future. This chatbot is built to be used in a restaurant that mainly serves pizzas. Take a look at the intents.json file to understand how the chatbot is built to answer some common queries.

WARNINGS:
1. This project uses tensorflow internally and needs python 3.6 specifically to run since python 3.7 has a bug which doesn't let it run      tensorflow.
2. If however you have a python version greater than 3.6, I suggest creating a virtual environment of python 3.6 and installing all the      required packages and modules there.
    To do this go to your directory in command line and type 
    > conda create -n name-of-environment python=3.6
    
    > conda activate name-of-environment
3. Now that you have your virtual environment ready, download all the necessary modules. It is of utmost importance that when you download    tesorflow module, be sure to do it by writing 
    > pip install tensorflow==1.4
   
   This is done because the newer version 2.0 of Tensorflow doesn't work well with the tflearn library.

When it's all done and dusted, run the script whichever way you like.
Do read the script before executing. There are 2 comments in the 2 try blocks in the code. These, when uncommented, raise exceptions which make the except block run. Leave these commented under normal circumstances, however, if you make any changes in the intents file or the script, then make sure to uncomment them to generate errors and necessarily run the except block so that the training and modelling of the new data can happen properly and can be saved in a pickle file again. The try blocks are kept there specifically so that the training and modelling of data doesn't occur unnecessarily every single time that you run the script.
