
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
