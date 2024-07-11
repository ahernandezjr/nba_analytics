'''
MUCH OF THE FOLLOWING CODE IS NOT ORIGINAL CODE
USED FOR REFERENCING PURPOSES
SOURCE: 'https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130'
'''

# Import df
# df = pd.read_csv(DATA_FILE_NAME)
# df.head(5)


# X, y = df.drop(columns=['Close']), df.Close.values
# X.shape, y.shape



# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# mm = MinMaxScaler()
# ss = StandardScaler()



# X_trans = ss.fit_transform(X)
# y_trans = mm.fit_transform(y.reshape(-1, 1))



# # split a multivariate sequence past, future samples (X and y)
# def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
#     X, y = list(), list() # instantiate X and y
#     for i in range(len(input_sequences)):
#         # find the end of the input, output sequence
#         end_ix = i + n_steps_in
#         out_end_ix = end_ix + n_steps_out - 1
#         # check if we are beyond the dataset
#         if out_end_ix > len(input_sequences): break
#         # gather input and output of the pattern
#         seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
#         X.append(seq_x), y.append(seq_y)
#     return np.array(X), np.array(y)

# X_ss, y_mm = split_sequences(X_trans, y_trans, 100, 50)
# print(X_ss.shape, y_mm.shape)



# assert y_mm[0].all() == y_trans[99:149].squeeze(1).all()

# y_mm[0]


# total_samples = len(X)
# train_test_cutoff = round(0.90 * total_samples)

# X_train = X_ss[:-150]
# X_test = X_ss[-150:]

# y_train = y_mm[:-150]
# y_test = y_mm[-150:] 

# print("Training Shape:", X_train.shape, y_train.shape)
# print("Testing Shape:", X_test.shape, y_test.shape) 



# # convert to pytorch tensors
# X_train_tensors = Variable(torch.Tensor(X_train))
# X_test_tensors = Variable(torch.Tensor(X_test))

# y_train_tensors = Variable(torch.Tensor(y_train))
# y_test_tensors = Variable(torch.Tensor(y_test))



# # reshaping to rows, timestamps, features
# X_train_tensors_final = torch.reshape(X_train_tensors,   
#                                       (X_train_tensors.shape[0], 100, 
#                                        X_train_tensors.shape[2]))
# X_test_tensors_final = torch.reshape(X_test_tensors,  
#                                      (X_test_tensors.shape[0], 100, 
#                                       X_test_tensors.shape[2])) 

# print("Training Shape:", X_train_tensors_final.shape, y_train_tensors.shape)
# print("Testing Shape:", X_test_tensors_final.shape, y_test_tensors.shape) 



# X_check, y_check = split_sequences(X, y.reshape(-1, 1), 100, 50)
# X_check[-1][0:4]



# y_check[-1]
# df.Close.values[-50:]




'''
TEST CODE
'''
    # # Training loop
    # for epoch in tqdm(range(epochs)):
    #     for i, (inputs, targets) in enumerate(dataloader):
    #         # DO NOT Flatten 2d targets to 1d
    #         # targets = targets.view(-1)

    #         # Convert inputs and targets to tensors
    #         print(targets.shape)
    #         inputs = torch.tensor(inputs).float()
    #         targets = torch.tensor(targets).float()

    #         # Forward pass
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)

    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     if (epoch+1) % 100 == 0:
    #         print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


# def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
#     '''
#     Split a multivariate sequence past, future samples (X and y)
    
#     Args:
#         input_sequences (np.array): The input sequences.
#         output_sequence (np.array): The output sequences. 
#         n_steps_in (int): The number of steps in.
#         n_steps_out (int): The number of steps out.
        
#     Returns:
#         np.array: The input sequences.
#         np.array: The output sequences.
#     '''
#     X, y = list(), list() # instantiate X and y
#     for i in range(len(input_sequences)):
#         # find the end of the input, output sequence
#         end_ix = i + n_steps_in
#         out_end_ix = end_ix + n_steps_out - 1
#         # check if we are beyond the dataset
#         if out_end_ix > len(input_sequences): break
#         # gather input and output of the pattern
#         seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
#         X.append(seq_x), y.append(seq_y)
#     return np.array(X), np.array(y)


# # Create NBA dataset
# nba_dataset = NBAPlayerDataset(DATA_FILE_5YEAR_NAME, DATA_FILE_5YEAR_JSON_NAME)

# # Create a training loop for pytorch NBA dataset
# df = pd.read_csv(os.path.join(os.getcwd(), DATASET_DIR, DATA_FILE_NAME))
# X, y = df.drop(columns=['Close']), df.Close.values

# scaler_X = StandardScaler()
# scaler_y = MinMaxScaler()

# X_trans = scaler_X.fit_transform(X)
# y_trans = scaler_y.fit_transform(y.reshape(-1, 1))

# # Assuming split_sequences is a function that you have defined elsewhere
# X_ss, y_mm = split_sequences(X_trans, y_trans, 100, 50)

# total_samples = len(X)
# train_test_cutoff = round(0.90 * total_samples)

# input_size = 4 # number of features
# hidden_size = 2 # number of features in hidden state
# num_layers = 1 # number of stacked lstm layers

# n_epochs = 1000 # 1000 epochs
# learning_rate = 0.001 # 0.001 lr

# df_X_ss = scaler_X.transform(df.drop(columns=['Close'])) # old transformers
# df_y_mm = scaler_y.transform(df.Close.values.reshape(-1, 1)) # old transformers


# def training_loop(n_epochs, lstm, optimiser, loss_fn, X_train, y_train,
#                   X_test, y_test):
#     for epoch in range(n_epochs):
#         lstm.train()
#         outputs = lstm.forward(X_train) # forward pass
#         optimiser.zero_grad() # calculate the gradient, manually setting to 0
#         # obtain the loss function
#         loss = loss_fn(outputs, y_train)
#         loss.backward() # calculates the loss of the loss function
#         optimiser.step() # improve from loss, i.e backprop
#         # test loss
#         lstm.eval()
#         test_preds = lstm(X_test)
#         test_loss = loss_fn(test_preds, y_test)
#         if epoch % 100 == 0:
#             print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, 
#                                                                       loss.item(), 
#                                                                       test_loss.item())) 
            


# import warnings
# warnings.filterwarnings('ignore')

# n_epochs = 1000 # 1000 epochs
# learning_rate = 0.001 # 0.001 lr

# input_size = 4 # number of features
# hidden_size = 2 # number of features in hidden state
# num_layers = 1 # number of stacked lstm layers

# num_classes = 50 # number of output classes 

# lstm = LSTM(num_classes, 
#               input_size, 
#               hidden_size, 
#               num_layers)



# training_loop(n_epochs=n_epochs,
#               lstm=lstm,
#               optimiser=optimiser,
#               loss_fn=loss_fn,
#               X_train=X_train_tensors_final,
#               y_train=y_train_tensors,
#               X_test=X_test_tensors_final,
#               y_test=y_test_tensors)




# df_X_ss = ss.transform(df.drop(columns=['Close'])) # old transformers
# df_y_mm = mm.transform(df.Close.values.reshape(-1, 1)) # old transformers
# # split the sequence
# df_X_ss, df_y_mm = split_sequences(df_X_ss, df_y_mm, 100, 50)
# # converting to tensors
# df_X_ss = Variable(torch.Tensor(df_X_ss))
# df_y_mm = Variable(torch.Tensor(df_y_mm))
# # reshaping the dataset
# df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 100, df_X_ss.shape[2]))

# train_predict = lstm(df_X_ss) # forward pass
# data_predict = train_predict.data.numpy() # numpy conversion
# dataY_plot = df_y_mm.data.numpy()

# data_predict = mm.inverse_transform(data_predict) # reverse transformation
# dataY_plot = mm.inverse_transform(dataY_plot)
# true, preds = [], []
# for i in range(len(dataY_plot)):
#     true.append(dataY_plot[i][0])
# for i in range(len(data_predict)):
#     preds.append(data_predict[i][0])
# plt.figure(figsize=(10,6)) #plotting
# plt.axvline(x=train_test_cutoff, c='r', linestyle='--') # size of the training set

# plt.plot(true, label='Actual Data') # actual plot
# plt.plot(preds, label='Predicted Data') # predicted plot
# plt.title('Time-Series Prediction')
# plt.legend()
# plt.savefig("whole_plot.png", dpi=300)
# plt.show() 