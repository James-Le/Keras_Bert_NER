# Keras_Bert_NER

This is a Keras framework for sequential sentence classification, recogition and summarization.

The sentence encoding has been moved to Bert-as-Service api(https://github.com/hanxiao/bert-as-service),
which could be faster than traditional embedding because it utilize multiple parallel workers.

To lauch the program, please run "bert-serving-start -model_dir ./cased_L-12_H-768_A-12/ -num_worker=4" 
at first to start the web service for sentence embedding and then python main.py.

