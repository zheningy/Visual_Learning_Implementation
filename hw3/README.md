In this task, implement two different network to predict action sequence from HDMB51 based on ResNet features. You could find dataset from here, ([TrainSet](https://www.dropbox.com/s/y23pdfngf7uu4xn/annotated_train_set.p?dl=0),[TestSet](https://www.dropbox.com/s/2zc1vystx0161cr/randomized_annotated_test_set_no_name_no_num.p?dl=0)). The total size is less than 250MB.

The networks and dataloader are defined in 'network.py'.

Run 'simple_net.py' to train first full connected network and record output of test set.

Run 'rnn.py' to train LSTM network and record output of test set.

