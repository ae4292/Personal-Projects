import streamlit as st
from osgeo import gdal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from rasterio.windows import Window

st.set_page_config(page_title="Crop Prediction Dashboard")

st.markdown("<h1 style='text-align: center; color: black;'>FarmLand Prediction</h1>", unsafe_allow_html=True)

state = st.selectbox("Please Choose a State", ['AL', 'AR', 'AZ','CA','CO','CT','DE','FL',
 'GA','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT',
 'NC','ND','NE', 'NJ','NM', 'NV','NY', 'OH','OK','OR','PA','RI','SC','SD','TN','TX',
 'UT','VA','VT','WA','WI','WV','WY'])
dataset = gdal.Open(f'CropFiles/{state}/clipped.TIF', gdal.GA_ReadOnly)
band = dataset.GetRasterBand(1)
arr = band.ReadAsArray()

px = st.number_input("X Pixel Value",min_value = 0, max_value = len(arr[0])-3000)
py = st.number_input("Y Pixel Value", min_value = 0, max_value = len(arr)-3000)

from osgeo import osr, ogr, gdal


def pixel_to_world(geo_matrix, x, y):
    ul_x = geo_matrix[0]
    ul_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    y_dist = geo_matrix[5]
    _x = x * x_dist + ul_x
    _y = y * y_dist + ul_y
    return _x, _y


def build_transform_inverse(dataset, EPSG):
    source = osr.SpatialReference(wkt=dataset.GetProjection())
    target = osr.SpatialReference()
    target.ImportFromEPSG(EPSG)
    return osr.CoordinateTransformation(source, target)


def find_spatial_coordinate_from_pixel(dataset, transform, x, y):
    world_x, world_y = pixel_to_world(dataset.GetGeoTransform(), x, y)
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(world_x, world_y)
    point.Transform(transform)
    return point.GetX(), point.GetY()

ds = dataset
_t = build_transform_inverse(ds, 4326)
coordinates = find_spatial_coordinate_from_pixel(ds, _t, px, py)
st.markdown(f'>Latitude: {coordinates[0]}') 
st.markdown(f'>Longitude: {coordinates[1]}')
st.markdown(">If the plot to the right is completely purple, then there is no observable farmland in the given area.")

import matplotlib.patches as patches
rect = patches.Rectangle((40, 40), 20, 20, linewidth=1, edgecolor='r', facecolor='none',label = 'selected area')
fig,ax = plt.subplots(1,2, figsize = (10,6))

with rasterio.open(f'CropFiles/{state}/clipped.TIF') as img:
    data = img.read(window=Window(px, py, 3000, 3000))
    show(data, ax =ax[0], zorder = 1)
    data1 = img.read(window=Window(px+1450, py+1450, 100, 100))
    show(data1, ax =ax[1], zorder = 1)
ax[1].set_xlabel("Latitude in Pixels")
ax[1].set_ylabel("Longitude in Pixels")
ax[1].add_patch(rect)
ax[1].legend()
ax[0].scatter(1500,1500, color = 'red', zorder = 2, s = 50, label = 'Point of Interest')
ax[0].legend()
ax[0].set_xlabel("Latitude in Pixels")
ax[0].set_ylabel("Longitude in Pixels")
st.pyplot(fig)

st.markdown(">Please give the models time to run. The Neural Network can take a couple minutes.")
col1, col2 = st.columns(2)
with col1:
    mod = st.button('Click for Classic Classification Models')

if mod:
    print(px)
    pd_data = pd.DataFrame()
    pd_data['x'] = np.zeros(len(data1[0])*len(data1[0][0]))
    pd_data['y'] = np.zeros(len(data1[0])*len(data1[0][0]))
    pd_data['value'] = np.zeros(len(data1[0])*len(data1[0][0]))
    for i in range(0,100):
        for j in range(0,100):
            pd_data.loc[100*i + j, 'x'] = i
            pd_data.loc[100*i + j, 'y'] = j
            pd_data.loc[100*i + j, 'value'] = data1[0][i][j]
    pd_data['crop'] = np.array(pd_data['value'] < 255).astype('int')


    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    i = pd_data[((pd_data['x'] >= 40) & (pd_data['y'] >= 40) & (pd_data['x'] <= 60) & (pd_data['y'] <= 60))].index

    X_train = pd_data.iloc[pd_data.index.difference(i)].drop(['crop','value'], axis = 1)
    y_train = pd_data.iloc[pd_data.index.difference(i)]['crop']

    X_test = pd_data.iloc[i].drop(['crop','value'], axis = 1)
    y_test = pd_data.iloc[i]['crop']


    rfc = RandomForestClassifier()
    knn = KNeighborsClassifier(n_neighbors = 4, algorithm = 'ball_tree', metric = 'l2')
    knn.fit(X_train,y_train)
    rfc.fit(X_train,y_train)

    y_pred_knn = knn.predict(X_test)
    y_pred_rfc = rfc.predict(X_test)
    acc_knn = np.sum(y_test == y_pred_knn)/len(y_test)
    acc_rfc = np.sum(y_test == y_pred_rfc)/len(y_test)

    X_test['predicted_knn'] = y_pred_knn
    X_test['predicted_rfc'] = y_pred_rfc
    X_test = X_test.reset_index()

    data_test = np.zeros((21,21))
    for i in range(len(X_test)):
        x = int(X_test['x'][i]) - 40
        y = int(X_test['y'][i]) - 40
        data_test[x][y] = y_test.reset_index().loc[i,'crop']

    data_knn = np.zeros((21,21))
    for i in range(len(X_test)):
        x = int(X_test['x'][i]) - 40
        y = int(X_test['y'][i]) - 40
        data_knn[x][y] = X_test['predicted_knn'][i]
    
    data_rfc = np.zeros((21,21))
    for i in range(len(X_test)):
        x = int(X_test['x'][i]) - 40
        y = int(X_test['y'][i]) - 40
        data_rfc[x][y] = X_test['predicted_rfc'][i]
    fig,ax = plt.subplots(1,3,figsize=(10,6))
    ax[0].imshow(data_test, cmap='Greys',interpolation='nearest')
    ax[0].set_title('Provided Data')
    ax[1].imshow(data_knn, cmap='Greys',interpolation='nearest')
    ax[1].set_title('K-Nearest Neigbhors')
    ax[2].imshow(data_rfc, cmap='Greys',interpolation='nearest')
    ax[2].set_title('Random Forest Classifier')
    st.pyplot(fig)

with col2:  
    mod1 = st.button('Click for Neural Net Classification Model')
import torch
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
if mod1:
    # Set a random seed for both CPU and GPU
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)  # if using multi-GPU.
    import time

    def train_model(model, loss_func, num_epochs, optimizer, train_loader, x_test, y_test):

        train_loss_log = []
        test_loss_log = []

        # Move model to GPU if CUDA is available
        if torch.cuda.is_available():
            model = model.cuda()
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        tic = time.time()
        for epoch in range(1,num_epochs+1):
            for i, data in enumerate(train_loader):
                x, y = data
            # check if cuda is available
                if torch.cuda.is_available():
                    x , y = x.cuda(), y.cuda()
                # get predicted y value from our current model
                pred_y = model(x)
            # calculate the loss
                loss = loss_func(pred_y,y)
            # Zero the gradient of the optimizer
                optimizer.zero_grad()
            # Backward pass: Compute gradient of the loss with respect to model parameters
                loss.backward()
            # update weights
                optimizer.step()
            # change the model to evaluation mode to calculate the test loss; We will come back to this later after learning Dropout and Batch Normalization
            train_loss_log.append(loss.item())
            model.eval()
            test_pred_y = model(x_test)
            test_loss = loss_func(test_pred_y,y_test)
            test_loss_log.append(test_loss.item())
            # change back to training mode.
            model.train()
            print("Epoch {:2},  Training Loss: {:9.4f},  Test Loss: {:7.4f}".format(epoch, loss.item(), test_loss.item()))
        toc = time.time()
        print("Elapsed Time : {:7.2f}".format(toc-tic))
        return train_loss_log, test_loss_log
    
    pd_data = pd.DataFrame()
    pd_data['x'] = np.zeros(len(data1[0])*len(data1[0][0]))
    pd_data['y'] = np.zeros(len(data1[0])*len(data1[0][0]))
    pd_data['value'] = np.zeros(len(data1[0])*len(data1[0][0]))
    for i in range(0,100):
        for j in range(0,100):
            pd_data.loc[100*i + j, 'x'] = i
            pd_data.loc[100*i + j, 'y'] = j
            pd_data.loc[100*i + j, 'value'] = data1[0][i][j]
        pd_data['crop'] = np.array(pd_data['value'] < 255).astype('int')
    pt_data = pd_data
    pt_data1 = pt_data.drop('value', axis =1)
    first_column = pt_data1.pop('crop')
    pt_data1.insert(0, 'crop', first_column)
    from sklearn.model_selection import train_test_split

    i = pt_data1[((pt_data1['x'] >= 40) & (pt_data1['y'] >= 40) & (pt_data1['x'] <= 60) & (pt_data1['y'] <= 60))].index


    X_train1 = pt_data1.iloc[pd_data.index.difference(i)].drop(['crop'], axis = 1)
    y_train1 = pt_data1.iloc[pd_data.index.difference(i)]['crop']

    x_train, x_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.33, random_state=42)

    x_test = pt_data1.iloc[i].drop(['crop'], axis = 1)
    y_test = pt_data1.iloc[i]['crop']
    
    train_copy = x_train.copy()
    val_copy = x_val.copy()
    test_copy = x_test.copy()
    train_copy.insert(0,'crop',y_train)
    val_copy.insert(0,'crop',y_val)
    test_copy.insert(0,'crop',y_test)

    # Build Datasets and DataLoaders
    from torch.utils.data import DataLoader, Dataset

    # Start your code here. Refer to the previous labs for examples.
    class CustomDataset(Dataset):
        def __init__(self, dataframe):
            self.data = dataframe

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Assuming the DataFrame has two columns: features and target
            x = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float32)
            y = torch.tensor(self.data.iloc[idx,0], dtype=torch.long)
            return x, y

    train_dataset = CustomDataset(train_copy)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = CustomDataset(val_copy)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    test_dataset = CustomDataset(test_copy)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


    x_val = torch.tensor(val_copy.iloc[:, 1:].values, dtype=torch.float32)
    y_val = torch.tensor(val_copy.iloc[:,0].values, dtype=torch.long)
    x_test = torch.tensor(test_copy.iloc[:, 1:].values, dtype=torch.float32)
    y_test = torch.tensor(test_copy.iloc[:,0].values, dtype=torch.long)

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    # Build other models as well.
    class TwoHiddenLayerDropout(nn.Module):
        def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate=0.2):
            super(TwoHiddenLayerDropout, self).__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim1)
            self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after the first linear transformation
            self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
            self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after the second linear transformation
            self.linear3 = nn.Linear(hidden_dim2, 2)  # Assuming binary classification

        def forward(self, x):
            x = self.linear1(x)
            x = F.relu(x)
            x = self.dropout1(x)  # Apply dropout after the first activation
            x = self.linear2(x)
            x = F.relu(x)
            x = self.dropout2(x)  # Apply dropout after the second activation
            x = self.linear3(x)
            return x
    input_dim = train_copy.shape[1] - 1
    hidden_dim1 = 30
    hidden_dim2 = 20
    # Define the model and the optimizer
    two_hidden_layer_dropout = TwoHiddenLayerDropout(input_dim, hidden_dim1, hidden_dim2)
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(two_hidden_layer_dropout.parameters(), lr=learning_rate)
    # Define the loss function: CrossEntropyLoss
    loss_func = nn.CrossEntropyLoss()
    num_epochs = 50
    train_loss_log, test_loss_log = train_model(two_hidden_layer_dropout, loss_func, num_epochs, optimizer, train_loader, x_val, y_val)
    
    i = pd_data[((pd_data['x'] >= 40) & (pd_data['y'] >= 40) & (pd_data['x'] <= 60) & (pd_data['y'] <= 60))].index
    X_test = pd_data.iloc[i].drop(['crop','value'], axis = 1)
    X_test = X_test.reset_index()
    y_pred = two_hidden_layer_dropout(x_test).detach()
    X_test['predicted_nn'] = (y_pred < 0.5).float()[:,0].numpy()
    
    y_test = pd_data.iloc[i]['crop']

    
    data_test = np.zeros((21,21))
    for i in range(len(X_test)):
        x = int(X_test['x'][i]) - 40
        y = int(X_test['y'][i]) - 40
        data_test[x][y] = y_test.reset_index().loc[i,'crop']

    data_nn = np.zeros((21,21))
    for i in range(len(X_test)):
        x = int(X_test['x'][i]) - 40
        y = int(X_test['y'][i]) - 40
        data_nn[x][y] = X_test['predicted_nn'][i]
    fig,ax = plt.subplots(1,2,figsize=(10,6))
    ax[0].imshow(data_test, cmap='Greys',interpolation='nearest')
    ax[0].set_title('Provided Data')
    ax[1].imshow(data_nn, cmap='Greys',interpolation='nearest')
    ax[1].set_title('Neural Network')
    st.pyplot(fig)

