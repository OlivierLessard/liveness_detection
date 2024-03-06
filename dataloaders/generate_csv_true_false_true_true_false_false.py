import torchvision.datasets as dataset
from data_6 import SiameseNetworkDataset

train = True
# ID_type = ['FRA_licence','FRA_new_ID','FRA_old_ID','FRA_passport']
ID_type = ['fake','real']
for type in ID_type:
    if train:
        dir = '../face_db/training/'
        csv_dir = './face_db/training'  + '.csv'
    else:
        dir = '../face_db/validation/'
        csv_dir = './face_db/validation' + '.csv'

    training_dataset = dataset.ImageFolder(root=dir)
    print('hi', len(training_dataset))
    training_pairs = SiameseNetworkDataset(dataset_folder=training_dataset, file_name=csv_dir)
    print(len(training_pairs))