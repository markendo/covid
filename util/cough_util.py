def extract_labels_from_dir(path):
  '''
  creates data frame of sample names and associated labels and saves it as a csv
  input: path to sample directory
  output: data frame
  '''
  file_list = []
  for file_path in pathlib.Path(path).glob('*/*.wav'):
    dir_path, file_name = os.path.split(file_path)  # /drive/cough/data/neg, neg-cough.wav
    covid_class = os.path.basename(dir_path).split("/")[-1] # neg, pos
    file_list.append({"filename" : file_path, "label" : covid_class })

  return pd.DataFrame(file_list)

def data_split(data, features, test_size=0.30):
  lb = LabelBinarizer()
  labels = lb.fit_transform(data.label)
  lb.get_params(deep=True)
  (x_train, xtest, ytrain, ytest) = train_test_split(features, labels, test_size = test_size,
                                                     stratify = labels, random_state=30)

  return x_train, xtest, ytrain, ytest
