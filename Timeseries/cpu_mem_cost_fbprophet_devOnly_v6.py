from google.cloud import monitoring_v3
import os
import re
import math
import sys


import warnings
warnings.filterwarnings("ignore")

#"ariba-global-stg2-highspeed"
#project = input("Please Enter project Name:")
project = sys.argv[1]
project = project.strip()

HOME_DIR = '/home/jenkins/GCP_CostOptimization/'
os.chdir(HOME_DIR)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="{}{}.json".format(HOME_DIR,project)
client = monitoring_v3.MetricServiceClient()
#import datetime
from datetime import datetime
import time
import json
import pandas as pd
import numpy as np
import threading
from multiprocessing import Process
import logging

client = monitoring_v3.MetricServiceClient()
project_name = client.project_path(project)

from google.cloud.monitoring_v3 import query
from google.cloud import monitoring_v3
from google.cloud.monitoring import enums


from pprint import pprint
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
credentials = GoogleCredentials.get_application_default()
service = discovery.build('compute', 'v1', credentials=credentials)
project = project

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta 
from datetime import date
plt.style.use('fivethirtyeight') # For plots
hostname_zone_df_results = None 
inst_app_dict_results = None

def hostname_zone_df():
    global hostname_zone_df_results
    if hostname_zone_df_results is not None :
        return hostname_zone_df_results
    interval = monitoring_v3.types.TimeInterval()
    now = time.time()
    interval.end_time.seconds = int(now)
    interval.end_time.nanos = int((now - interval.end_time.seconds) * 10**9)
    interval.start_time.seconds = int(now - 1200)
    interval.start_time.nanos = interval.end_time.nanos
    aggregation = monitoring_v3.types.Aggregation()
    aggregation.alignment_period.seconds = 600
    aggregation.per_series_aligner = (monitoring_v3.enums.Aggregation.Aligner.ALIGN_MAX)

    results = client.list_time_series(project_name,
                                      'metric.type = "compute.googleapis.com/instance/cpu/utilization"',
                                      interval,
                                      monitoring_v3.enums.ListTimeSeriesRequest.TimeSeriesView.FULL,aggregation)

    data_dict = {}
    data_dict['Hostname'] = []
    data_dict['Zone'] = []
    data_dict['project_id'] = []
    for ts in results:
        if project != ts.resource.labels['project_id'] :
            continue
        data_dict['Hostname'].append(ts.metric.labels['instance_name'])
        data_dict['Zone'].append(ts.resource.labels['zone'])
        data_dict['project_id'].append(ts.resource.labels['project_id'])
    df = pd.DataFrame(data_dict)
    hostname_zone_df_results = df
    return df


def instance_list():
    #return ['common-au1-ms-ch-vzwq']
    return hostname_zone_df()['Hostname'].tolist()

def chunk(list,size):
    all_list = []
    for i in range(0,len(list),size):
        all_list.append(list[i:i+size])
    return all_list


def instance_app_df1():
    global inst_app_dict_results
    if inst_app_dict_results is not None:
        return inst_app_dict_results
    df = hostname_zone_df()
    instance_app_dict = {}
    instance_app_dict['instance_name'] = []
    instance_app_dict['App_name'] = []
    instance_app_dict['machine_type'] = []
    instance_app_dict['status'] = []
    instance_app_dict['zone'] = []
    instance_app_dict['instance_id'] = []
    instance_app_dict['creationTimestamp'] = []
    instance_app_dict['labels'] = []

    for instance in instance_list():
        request = service.instances().get(project=project, zone=" ".join(df[df['Hostname']==instance]['Zone'].values), instance=instance)
        response = request.execute()
        instance_app_dict['instance_name'].append(instance)
        try:
            instance_app_dict['App_name'].append(response['labels']['app'])
        except KeyError:
            instance_app_dict['App_name'].append("No Data")
            #print("No labels for host",instance)
            
        instance_app_dict['machine_type'].append(response['machineType'][47:])
        instance_app_dict['status'].append(response['status'])
        instance_app_dict['zone'].append(" ".join(df[df['Hostname']==instance]['Zone'].values))
        instance_app_dict['instance_id'].append(response['id'])
        instance_app_dict['creationTimestamp'].append(response['creationTimestamp'])
        try:
            instance_app_dict['labels'].append(response['labels'])
        except KeyError:
            response['labels'] = {}
            instance_app_dict['labels'].append(response['labels'])

    inst_app_dict_results = pd.DataFrame(instance_app_dict)
    return pd.DataFrame(instance_app_dict)


def dataframe_cpu_df(host_name):
    instance_app_df = instance_app_df1()
    from google.cloud.monitoring_v3 import query
    q = query.Query(client,project, 'compute.googleapis.com/instance/cpu/utilization',None,90,0,0)
    try:
        instance_id = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'instance_id'].item()
        q = q.align(enums.Aggregation.Aligner.ALIGN_MAX, minutes=1440)
        dataframe = q.as_dataframe()
        dataframe = q.as_dataframe(label='instance_name')
        dataframe = q.as_dataframe(labels=['zone', 'instance_name'])
        query = q.select_metrics(instance_name=host_name)
        dataframe_cpu = q.as_dataframe()
        dataframe_cpu = q.as_dataframe(labels=['resource_type', 'instance_name'])
    
        dataframe_cpu = pd.Series(dataframe_cpu['gce_instance'][host_name]).to_frame()
        dataframe_cpu.rename(columns={host_name:'cpu'}, inplace=True)
        dataframe_cpu['time'] = dataframe_cpu.index
        dataframe_cpu['time'] = pd.Series(dataframe_cpu["time"]).dt.round("H")
        dataframe_cpu = dataframe_cpu.set_index('time')
        dataframe_cpu['cpu'] = dataframe_cpu['cpu'] * 100
        return dataframe_cpu
    except ValueError:
        print("No data available")
    

def instance_id_func(host_name):
    instance_app_df = instance_app_df1()
    try:
        instance_id = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'instance_id'].item()
        return instance_id
    except ValueError:
        print("No data available")
        
#df.loc[df['B'] == 3, 'A'].iloc[0]

#def instance_id_func(host_name):
    #instance_app_df = instance_app_df1()
    #instance_id = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'instance_id'].iloc[0]
    #return instance_id


def memory_df(host_name):
    instance_id = instance_id_func(host_name)
    from google.cloud.monitoring_v3 import query
    q = query.Query(client,project, 'agent.googleapis.com/memory/percent_used',None,90,0,0)
    q = q.align(enums.Aggregation.Aligner.ALIGN_MAX, minutes=1440)
    dataframe = q.as_dataframe(labels=['resource_type', 'instance_id','state'])
    dataframe_mem = pd.Series(dataframe['gce_instance'][instance_id]['used']).to_frame()
    dataframe_mem['time'] = dataframe.index
    dataframe_mem['time'] = pd.Series(dataframe_mem["time"]).dt.round("H")
    dataframe_mem = dataframe_mem.set_index('time')
    dataframe_mem['instance_id'] = instance_id
    return dataframe_mem
    

def result_df(host_name):
    try:
        dataframe_mem = memory_df(host_name)
        dataframe_cpu = dataframe_cpu_df(host_name)
        result = pd.merge(dataframe_cpu,dataframe_mem[['used','instance_id']],
                          left_on='time',
                          right_on='time',
                          how='left')
        result.rename(columns={host_name:'cpu'}, inplace=True)
        result['instance_name'] = host_name
        return result
    except KeyError:
        print("No memory data available")
        dataframe_cpu = dataframe_cpu_df(host_name)
        return dataframe_cpu
    except TypeError:
        print("NoneType CPU data")
        return dataframe_mem

def display_max_cpu_used(result,host_name,recom_cpu,recom_mem):
    instance_app_df = instance_app_df1()
    zone = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'zone'].item()
    instance_id = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'instance_id'].item()
    machine_type = (instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'machine_type'].item()).split('/')[-1]
    creationTimestamp = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'creationTimestamp'].item()
    labels = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'labels'].item()


    try:
        max_cpu = result['cpu'].max()
    except KeyError:
        max_cpu = -1
    
    size_reduction_factor_cpu = math.floor(math.log((100/(recom_cpu)),2))
    
    print("Max value of CPU is: ",max_cpu)
    print("Zone name is: ",zone)
    print("Instance id is: ",instance_id)
    print("Machine Type is: ",machine_type)
    
    
    
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('compute', 'v1', credentials=credentials)
    request = service.machineTypes().get(project=project, zone=zone, machineType=machine_type)
    response = request.execute()
    matchObj = re.match(r'(\d+)\s+vCPUs,\s+(\d+)\s+(\S+) RAM',response['description'])
    try:
        no_of_cores = matchObj.group(1)
        print("No of cores "+no_of_cores)
        print("Size reduction factor of CPU is: ",size_reduction_factor_cpu)
        print("Recommended core: ",int(no_of_cores)/(2**size_reduction_factor_cpu))
        Recommended_core = int(no_of_cores)/(2**size_reduction_factor_cpu)
    except AttributeError:
        no_of_cores = None
        
    
    #==========json======================================================
    
    new_dict = {}
    new_dict['project'] = []
    new_dict['zone'] = []
    new_dict['instance_name'] = []
    new_dict['instance_id'] = []
    new_dict['recommend'] = {}
    new_dict['current'] = {}
    new_dict['recommend_dev'] = {}
    new_dict['creationTimestamp'] = []
    new_dict['labels'] = {}

    
    new_dict['project'] = project
    new_dict['zone'] = zone
    new_dict['instance_name'] = host_name
    new_dict['instance_id'] = instance_id
    new_dict['machine_type'] = machine_type
    new_dict['creationTimestamp'] = creationTimestamp
    new_dict['labels'] = labels
    new_dict["opt-factor"] =size_reduction_factor_cpu
    
    new_dict['recommend']['cpu'] = round(recom_cpu,2)
    new_dict['recommend']['disk'] = 0
    new_dict['current']['cpu'] = round(max_cpu,2)
    new_dict['current']['disk'] = 0
    new_dict['recommend_dev']['disk'] = ""
    
    if no_of_cores is not None:
        try:
            max_mem = result['used'].max()
            print("Max value of Memory is: ",max_mem)
            #zone = 'us-west2' 
            file_name = r'./pricing'+'/'+zone+'.txt'
            colnames=['Host_Type', 'Cores', 'Memory', 'Price', 'Pre_Empt_Price']
            df_price = pd.read_csv(filepath_or_buffer = file_name, names=colnames, header=None)
            df_price['Pre_Empt_Price'] = df_price['Pre_Empt_Price'].str.replace('$', '')
            df_price['Pre_Empt_Price'] = pd.to_numeric(df_price['Pre_Empt_Price'])
            df_price['Price'] = df_price['Price'].str.replace('$', '')
            df_price['Price'] = pd.to_numeric(df_price['Price'])
            df_price['Memory'] = df_price['Memory'].str.replace('G', '').str.replace('B', '')
            df_price['Memory'] = pd.to_numeric(df_price['Memory'])
            df_price['zone'] = zone
            print('----------------------------------')
            print(recom_mem)
            print('----------------------------------')
            df_price_filtered = df_price[df_price['Host_Type'] == machine_type]
            machine_memory = df_price_filtered.iloc[0]['Memory']
            recom_mem = machine_memory * (recom_mem/100)
            print('----------------------------------')
            print(recom_mem)
            print('----------------------------------')



            Recommended_Memory = recom_mem
            #print("Memory {} {}".format(matchObj.group(2), matchObj.group(3)))
            #print("Recommended Memory: ",Recommended_Memory)
            try:
                recommend_hw_price=findBestHardware('us-west2', Recommended_core, Recommended_Memory,machine_type)
        
                new_dict['cost_savings'] = recommend_hw_price[1]
                new_dict['recommend']['mem'] = round(recom_mem,2)
                new_dict['current']['mem'] = round(max_mem,2)
                new_dict['recommend_dev']['machine_type'] = recommend_hw_price[0]
            except ValueError:
                new_dict['cost_savings'] = "Insufficient data to calculate"
                new_dict['recommend_dev']['machine_type'] = "Insufficient data to calculate"
                print("Not sufficient data to fit the model")
        
        except KeyError:
            print("No memory data for {}".format(host_name))
            Recommended_Memory = -1
            try:
                recommend_hw_price=findBestHardwareCPUOnly('us-west2', Recommended_core,machine_type)
                new_dict['cost_savings'] = recommend_hw_price[1]
                new_dict['recommend']['mem'] = -1
                new_dict['current']['mem'] = -1
                new_dict['recommend_dev']['machine_type'] = recommend_hw_price[0]
            except ValueError:
                new_dict['cost_savings'] = "Insufficient data to calculate"
                new_dict['recommend_dev']['machine_type'] = "Insufficient data to calculate"
                print("Not sufficient data to fit the model")
                
        with open(os.path.join(os.getcwd()+'/images',"{}_{}_{}_{}.json".format(project,zone,host_name,instance_id)),"w") as fp:
            json.dump(new_dict,fp,indent=4)
        upload_blob("gcp_cost_recommendation_bucket",os.path.join(os.getcwd()+'/images',"{}_{}_{}_{}.json".format(project,zone,host_name,instance_id)),
            "{}_{}_{}_{}.json".format(project,zone,host_name,instance_id))
    else:
        print("Machine type is custome")
        try:
            max_mem = result['used'].max()
            new_dict['current']['mem'] = round(max_mem,2)
        except KeyError:
            new_dict['current']['mem'] = -1
        
        new_dict['recommend']['mem'] = "NA"
        new_dict['machine_type'] = str(machine_type) + "(Custom_Type)"
        new_dict['recommend_dev']['machine_type'] = "NA"
        new_dict['cost_savings'] = "NA"
        with open(os.path.join(os.getcwd()+'/images',"{}_{}_{}_{}.json".format(project,zone,host_name,instance_id)),"w") as fp:
            json.dump(new_dict,fp,indent=4)
        upload_blob("gcp_cost_recommendation_bucket",os.path.join(os.getcwd()+'/images',"{}_{}_{}_{}.json".format(project,zone,host_name,instance_id)),
                    "{}_{}_{}_{}.json".format(project,zone,host_name,instance_id))    
    #==========================json creation===============================
    #app_json = json.dumps(new_dict,indent=4)
    #print(app_json)
    
    
def display_cpu_used_no_data(host_name,recom_cpu=-1,recom_mem=-1):
    instance_app_df = instance_app_df1()
    zone = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'zone'].item()
    instance_id = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'instance_id'].item()
    machine_type = (instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'machine_type'].item()).split('/')[-1]
    creationTimestamp = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'creationTimestamp'].item()
    labels = instance_app_df.loc[instance_app_df['instance_name'] == host_name, 'labels'].item()
    
    print("Zone name is: ",zone)
    print("Instance id is: ",instance_id)
    print("Machine Type is: ",machine_type)
    
    #==========json======================================================
    
    new_dict = {}
    new_dict['project'] = []
    new_dict['zone'] = []
    new_dict['instance_name'] = []
    new_dict['instance_id'] = []
    new_dict['recommend'] = {}
    new_dict['current'] = {}
    new_dict['recommend_dev'] = {}
    new_dict['creationTimestamp'] = []
    new_dict['labels'] = {}
    
    new_dict['project'] = project
    new_dict['zone'] = zone
    new_dict['instance_name'] = host_name
    new_dict['instance_id'] = instance_id
    new_dict['machine_type'] = machine_type
    new_dict['creationTimestamp'] = creationTimestamp
    new_dict['labels'] = labels
    
    new_dict['recommend']['cpu'] = "NA"
    new_dict['recommend']['disk'] = 0
    new_dict['current']['cpu'] = -1
    new_dict['current']['disk'] = 0
    new_dict['recommend_dev']['disk'] = ""
    
    new_dict['recommend']['mem'] = "NA"
    new_dict['current']['mem'] = -1
    new_dict['cost_savings'] = "Data not available"
    new_dict['recommend_dev']['machine_type'] = "NA"
                
    with open(os.path.join(os.getcwd()+'/images',"{}_{}_{}_{}.json".format(project,zone,host_name,instance_id)),"w") as fp:
        json.dump(new_dict,fp,indent=4)
    upload_blob("gcp_cost_recommendation_bucket",os.path.join(os.getcwd()+'/images',"{}_{}_{}_{}.json".format(project,zone,host_name,instance_id)),
                "{}_{}_{}_{}.json".format(project,zone,host_name,instance_id))
        
    
def ML_dataset_used(result):
    data = result.sort_index(ascending=True, axis=0)
    data['time'] = data.index
    #data['cpu'] = data['cpu']
    new_data = pd.DataFrame(index=range(0,len(result)),columns=['time','used'])
    for i in range(0,len(data)):
        new_data['time'][i] = data['time'][i]
        #new_data['cpu'][i] = data['cpu'][i]
        new_data['used'][i] = data['used'][i]

    new_data['time'] = pd.to_datetime(new_data.time)
    new_data.index = new_data['time']
    new_data.drop('time', axis=1, inplace=True)
    return new_data

def ML_dataset_cpu(result):
    data = result.sort_index(ascending=True, axis=0)
    data['time'] = data.index
    data['cpu'] = data['cpu']
    #new_data = pd.DataFrame(index=range(0,len(result)),columns=['time','used'])
    new_data = pd.DataFrame(index=range(0,len(result)),columns=['time','cpu'])
    for i in range(0,len(data)):
        new_data['time'][i] = data['time'][i]
        new_data['cpu'][i] = data['cpu'][i]

    new_data['time'] = pd.to_datetime(new_data.time)
    new_data.index = new_data['time']
    new_data.drop('time', axis=1, inplace=True)
    return new_data

def test_dummy(new_data):
    start=new_data.index[-1]
    #date_rng = pd.date_range(start, end='11/01/2019', periods = 168, freq='H')
    #date_rng = pd.date_range(start, periods = 169, freq='H',closed='right')
    date_rng = pd.date_range(start, periods = 9, freq='D',closed='right')
    test = pd.DataFrame(date_rng, columns=['time'])
    test['used'] = 0
    test['time'] = pd.to_datetime(test.time)
    test.index = test['time']
    test.drop('time', axis=1, inplace=True)
    #test = valid.drop(valid.index[0])
    return test
    

def LSTM_modelling(new_data):
    dataset = new_data.values
    train = dataset[0:720,:]
    #test = dataset[600:,:]
    test = test_dummy(new_data)
    #converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1)) #feature_range=(0, 1)
    scaled_data = scaler.fit_transform(train)
    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=2)
    
    #predicting 246 values, using past 60 from the train data
    inputs = new_data[len(new_data) - len(test) - 60:].values
    inputs = inputs.reshape((-1,1))
    inputs  = scaler.transform(inputs)
    
    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    return closing_price

def FBprophet_CPU(dataset,host_name):
    dataset['date'] = pd.to_datetime(dataset.index)
    dataset.set_index('date',inplace=True)
    pred = test_dummy(dataset)
    pred['date'] = pd.to_datetime(pred.index)
    pred.set_index('date',inplace=True)
    pjme = dataset
    split_date= (date.today() - timedelta(days=7)).strftime('%d-%b-%Y')
    pjme_train = pjme.loc[pjme.index <= split_date].copy()
    pjme_test = pjme.loc[pjme.index > split_date].copy()
    pjme_test.rename(columns={'cpu': 'TEST SET'}) \
    .join(pjme_train.rename(columns={'cpu': 'TRAINING SET'}),
          how='outer')
    # Format data for prophet model using ds and y
    pjme_train.reset_index().rename(columns={'date':'ds','cpu':'y'})
    # Setup and train model and fit
    model = Prophet(changepoint_prior_scale=0.95)
    try:
        model.fit(pjme.reset_index().rename(columns={'date':'ds','cpu':'y'}))
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        df_result_prev = pd.DataFrame({'dtime':model.history['ds'].dt.to_pydatetime(), 'y':model.history['y']})
        pjme_test_fcst = model.predict(df=pjme_test.reset_index().rename(columns={'date':'ds'}))
        df_past = pd.DataFrame({'ds':model.history['ds'].dt.to_pydatetime(), 'y': model.history['y']})
        df_future = pd.DataFrame({'ds':forecast['ds'], 'y': forecast['yhat_upper']})
        df_past = df_future.copy() #changed based on feedback from manju
        #df_past = df_past.append(df_future)
        df_past['y'] =  df_past['y'].apply(lambda x: x*1.2)
        pjme_test_fcst = model.predict(df=pjme_test.reset_index().rename(columns={'date':'ds'}))
        pjme_fut_fcst = model.predict(df=pred.reset_index().rename(columns={'date':'ds'}))
        df_past_resample = df_past.copy(deep=True)
        df_past_resample_indx = df_past_resample.set_index('ds')
        data_cols = ['y']
        df_recomm = df_past_resample_indx[data_cols].resample('W').max()
        f, ax = plt.subplots(1)
        plt.plot(model.history['ds'].dt.to_pydatetime(),model.history['y'], color='teal', marker='o', linestyle='solid', linewidth=1, label='Actual Utilization')
        plt.plot(df_recomm.index ,df_recomm['y'], color='orange', marker='o', linestyle='solid', linewidth=1, label='Recommendation' )
        f.set_figheight(15)
        f.set_figwidth(25)
        ax.scatter(pjme_test.index, pjme_test['cpu'], color='yellow')
        ax.legend(loc='upper left', frameon=False)
        #fig = model.plot(pjme_test_fcst, ax=ax)
        #fig1 = model.plot(pjme_fut_fcst, ax=ax)
        #plt.xticks(rotation='vertical')
    
        pjme_test_fcst = pjme_test_fcst.append(pjme_fut_fcst)
        fig = model.plot(pjme_test_fcst, ax=ax)


        df = hostname_zone_df()
        zone = " ".join(df[df['Hostname']==host_name]['Zone'].values)
        instance_id = instance_id_func(host_name)
    
        plt.savefig(os.path.join(os.getcwd()+'/images',"CPU_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id)))
        plt.close()
        upload_blob("gcp_cost_recommendation_bucket",
                    os.path.join(os.getcwd()+'/images',"CPU_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id)),
                    "CPU_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id))
        return df_past['y'].max()
    except ValueError:
        return None
    
def FBprophet_memory(dataset,host_name):
    dataset['date'] = pd.to_datetime(dataset.index)
    dataset.set_index('date',inplace=True)
    pjme = dataset
    pred = test_dummy(dataset)
    pred['date'] = pd.to_datetime(pred.index)
    pred.set_index('date',inplace=True)
    split_date= (date.today() - timedelta(days=7)).strftime('%d-%b-%Y')
    pjme_train = pjme.loc[pjme.index <= split_date].copy()
    pjme_test = pjme.loc[pjme.index > split_date].copy()
    
    pjme_test.rename(columns={'used': 'TEST SET'}) \
    .join(pjme_train.rename(columns={'used': 'TRAINING SET'}),how='outer')

    # Format data for prophet model using ds and y
    pjme_train.reset_index().rename(columns={'date':'ds','used':'y'})

    # Setup and train model and fit
    model = Prophet(changepoint_prior_scale=0.95)
    try:
        model.fit(pjme.reset_index().rename(columns={'date':'ds','used':'y'}))

        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        df_result_prev = pd.DataFrame({'dtime':model.history['ds'].dt.to_pydatetime(), 'y':model.history['y']})
        pjme_test_fcst = model.predict(df=pjme_test.reset_index().rename(columns={'date':'ds'}))
        df_past = pd.DataFrame({'ds':model.history['ds'].dt.to_pydatetime(), 'y': model.history['y']})
        df_future = pd.DataFrame({'ds':forecast['ds'], 'y': forecast['yhat_upper']})
        df_past = df_future.copy() #changed based on feedback from manju
        #df_past = df_past.append(df_future)
        df_past['y'] =  df_past['y'].apply(lambda x: x*1.2)
        pjme_test_fcst = model.predict(df=pjme_test.reset_index().rename(columns={'date':'ds'}))
        pjme_fut_fcst = model.predict(df=pred.reset_index().rename(columns={'date':'ds'}))

        df_past_resample = df_past.copy(deep=True)
        df_past_resample_indx = df_past_resample.set_index('ds')

        data_cols = ['y']
        df_recomm = df_past_resample_indx[data_cols].resample('W').max()
        f, ax = plt.subplots(1)
        plt.plot(model.history['ds'].dt.to_pydatetime(),model.history['y'], color='teal', marker='o', linestyle='solid', linewidth=1, label='Actual Utilization')
        plt.plot(df_recomm.index ,df_recomm['y'], color='orange', marker='o', linestyle='solid', linewidth=1, label='Recommendation' )
        f.set_figheight(5)
        f.set_figwidth(15)
        ax.scatter(pjme_test.index, pjme_test['used'], color='yellow')
        ax.legend(loc='upper left', frameon=False)
        #fig = model.plot(pjme_test_fcst, ax=ax)
        #fig1 = model.plot(pjme_fut_fcst, ax=ax)
        pjme_test_fcst = pjme_test_fcst.append(pjme_fut_fcst)
        fig = model.plot(pjme_test_fcst, ax=ax)

        #plt.xticks(rotation='vertical')
    

        df = hostname_zone_df()
        zone = " ".join(df[df['Hostname']==host_name]['Zone'].values)
        instance_id = instance_id_func(host_name)

    
        plt.savefig(os.path.join(os.getcwd()+'/images',"Memory_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id)))
        plt.close()
        upload_blob("gcp_cost_recommendation_bucket",
                    os.path.join(os.getcwd()+'/images',"Memory_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id)),
                    "Memory_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id))
        #logging.debug("Done with {}".format(host_name))
        return df_past['y'].max()
    except ValueError:
        return None


def images_file():
    path = os.getcwd()
    if not os.path.exists("images"):
        os.mkdir("images")
 

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="ariba-global-stg2-highspeed.json"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    #print('File {} uploaded to {}.'.format(source_file_name,destination_blob_name))
    
    HOME_DIR = '/home/jenkins/GCP_CostOptimization/'
    os.chdir(HOME_DIR)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="{}{}.json".format(HOME_DIR,project)


def findBestHardware(zone_name, core, memory, machine_type):
    colnames=['Host_Type', 'Cores', 'Memory', 'Price', 'Pre_Empt_Price']
    file_name = r'./pricing'+'/'+zone_name+'.txt'
    df_price = pd.read_csv(filepath_or_buffer = file_name, names=colnames, header=None)
    df_price['Pre_Empt_Price'] = df_price['Pre_Empt_Price'].str.replace('$', '')
    df_price['Pre_Empt_Price'] = pd.to_numeric(df_price['Pre_Empt_Price'])
    df_price['Price'] = df_price['Price'].str.replace('$', '')
    df_price['Price'] = pd.to_numeric(df_price['Price'])
    df_price['Memory'] = df_price['Memory'].str.replace('G', '').str.replace('B', '')
    df_price['Memory'] = pd.to_numeric(df_price['Memory'])
    df_price['zone'] = zone_name

    temp  = machine_type.split('-');
    filter_machine_type = temp[0]+'-'+temp[1]

    df_price_filtered = df_price[df_price['Host_Type'].str.startswith(filter_machine_type)]
    df_price = df_price_filtered

    current_dev_price = df_price[(df_price.Host_Type == machine_type)]['Price'].values[0]
    df_price_new = df_price[(df_price.Cores >= core) &  (df_price.Memory >= memory) ]

    le = LabelEncoder()
    le_zone = LabelEncoder()
    cls = le.fit_transform(list(df_price_new["Host_Type"]))
    df_price_new["zone"] = le_zone.fit_transform(list(df_price_new["zone"]))
    df_price_new = df_price_new.drop(columns=['Pre_Empt_Price'])

    X = df_price_new.iloc[:,[1,2,4]].values
    model = KNeighborsClassifier(n_neighbors=1)

    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X)
    model.fit(X_train, cls)

    y_pred = le.inverse_transform(
        model.predict(
            scaler.transform(
                [[core,memory,le_zone.fit_transform(
                    [zone_name]
                )[0]]]
            )
        )
    )[0]
    new_dev_price = df_price[(df_price.Host_Type == y_pred)]['Price'].values[0]
    price_savings = round((current_dev_price - new_dev_price)*24*30,2)
    return (y_pred,price_savings)


##==============Best hardware CPU Only==================================================

def findBestHardwareCPUOnly(zone_name, core, machine_type):
    colnames=['Host_Type', 'Cores', 'Memory', 'Price', 'Pre_Empt_Price']
    file_name = r'./pricing'+'/'+zone_name+'.txt'
    df_price = pd.read_csv(filepath_or_buffer = file_name, names=colnames, header=None)
    df_price['Pre_Empt_Price'] = df_price['Pre_Empt_Price'].str.replace('$', '')
    df_price['Pre_Empt_Price'] = pd.to_numeric(df_price['Pre_Empt_Price'])
    df_price['Price'] = df_price['Price'].str.replace('$', '')
    df_price['Price'] = pd.to_numeric(df_price['Price'])
    df_price['Memory'] = df_price['Memory'].str.replace('G', '').str.replace('B', '')
    df_price['Memory'] = pd.to_numeric(df_price['Memory'])
    df_price['zone'] = zone_name

    temp  = machine_type.split('-');
    filter_machine_type = temp[0]+'-'+temp[1]

    df_price_filtered = df_price[df_price['Host_Type'].str.startswith(filter_machine_type)]
    df_price = df_price_filtered


    current_dev_price = df_price[(df_price.Host_Type == machine_type)]['Price'].values[0]
    df_price_new = df_price[(df_price.Cores >= core)]

    le = LabelEncoder()
    le_zone = LabelEncoder()
    cls = le.fit_transform(list(df_price_new["Host_Type"]))
    df_price_new["zone"] = le_zone.fit_transform(list(df_price_new["zone"]))
    df_price_new = df_price_new.drop(columns=['Pre_Empt_Price'])

    X = df_price_new.iloc[:,[1,4]].values
    model = KNeighborsClassifier(n_neighbors=1)

    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X)
    model.fit(X_train, cls)

    y_pred = le.inverse_transform(
        model.predict(
            scaler.transform(
                [[core,le_zone.fit_transform(
                    [zone_name]
                )[0]]]
            )
        )
    )[0]
    new_dev_price = df_price[(df_price.Host_Type == y_pred)]['Price'].values[0]
    price_savings = round((current_dev_price - new_dev_price)*24*30,2)
    return (y_pred,price_savings)



from google.cloud import storage

import time
timestr = time.strftime("%Y_%m_%d-%H_%M")
import matplotlib.pyplot as plt
#%matplotlib inline

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size


def json_config_plotting(list1):
    for host_name in list1:
        try:
            result = result_df(host_name)
            display_max_cpu_used(result,host_name)
            #==============CPU graph===========================
            new_data = ML_dataset_cpu(result)
            test = test_dummy(new_data)
            closing_price = LSTM_modelling(new_data)
            #train = new_data[:600]
            #test = new_data[600:]
            test['prediction'] = closing_price
            plt.plot(new_data['cpu'],linewidth=2,label="real_value")
            #plt.plot(valid[['used','Predictions']],linewidth=3,label="Prediction")
            plt.plot(test['prediction'],linewidth=2,label="Prediction")
            plt.xlabel('Time')
            plt.ylabel('CPU_Used')
            plt.title('CPU usage and prediction for {}'.format(host_name))
            plt.xticks(rotation='vertical')
            plt.legend(loc='upper left')
            df = hostname_zone_df()
            zone = " ".join(df[df['Hostname']==host_name]['Zone'].values)
            instance_id = instance_id_func(host_name)
            plt.savefig(os.path.join(os.getcwd()+'/images',"CPU_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id)))
            plt.close()
            upload_blob("gcp_cost_recommendation_bucket",
                        os.path.join(os.getcwd()+'/images',"CPU_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id)),
                        "CPU_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id))
        except KeyError:
            print("No Data available for {}".format(host_name))
                                                                                  
            #==============Memory graph===========================
        try:
            new_data = ML_dataset_used(result)
            test = test_dummy(new_data)
            closing_price = LSTM_modelling(new_data)
            test['prediction'] = closing_price
            plt.plot(new_data['used'],linewidth=2,label="real_value")
            plt.plot(test['prediction'],linewidth=2,label="Prediction")
            plt.xlabel('Time')
            plt.ylabel('Memory_Used')
            plt.title('Memory usage and prediction for {}'.format(host_name))
            plt.xticks(rotation='vertical')
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(os.getcwd()+'/images',"Memory_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id)))
            plt.close()
            upload_blob("gcp_cost_recommendation_bucket",
                        os.path.join(os.getcwd()+'/images',"Memory_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id)),
                        "Memory_{}_{}_{}_{}.png".format(project,zone,host_name,instance_id))
            logging.debug("Done with {}".format(host_name))
        except KeyError:
            print("No memory Data available for {}".format(host_name))

    

def json_config_plotting_FBProphet(list1):
    for host_name in list1:
        print("-------------------------------->Processing Data from "+ host_name)
        result = result_df(host_name)
        try:
            result.dropna(how="any", inplace=True)
            try:
                #result = result_df(host_name)
                #==============CPU graph===========================
                dataset = ML_dataset_cpu(result)
                dataset = dataset[dataset['cpu'].notnull()]
                test = test_dummy(dataset)
                recom_cpu = FBprophet_CPU(dataset,host_name)
            except KeyError:
                print("No Data available for {}".format(host_name))
            except IndexError:
                print("No Data available for {}".format(host_name))
                
            
            #==============Memory graph===========================
        
            try:
                dataset = ML_dataset_used(result)
                dataset = dataset[dataset['used'].notnull()]
                test = test_dummy(dataset)
                recom_mem = FBprophet_memory(dataset,host_name)
                if recom_cpu != None and recom_mem != None:
                    display_max_cpu_used(result,host_name,recom_cpu,recom_mem)
            except KeyError:
                if recom_cpu != None:
                    display_max_cpu_used(result,host_name,recom_cpu,recom_mem=-1)
                print("No Data available for {}".format(host_name))
            except IndexError:
                print("No Data available for {}".format(host_name))
        except AttributeError:
            print("No CPU and Memory data available")
            display_cpu_used_no_data(host_name,recom_cpu=-1,recom_mem=-1)
            
    


#list1 = instance_list()
#list1 = ["prodstg-kf02-kafka-z1-0"]
#if __name__=="__main__":
    #logging.basicConfig(filename='app.log', filemode='w',level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
    #images_file()
    #json_config_plotting_FBProphet(list1)


import concurrent.futures
import time
#start = time.perf_counter()

main_list = instance_list()
list_all = chunk(main_list,50)

def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        #list_all = chunk(main_list,50)
        results = executor.map(json_config_plotting_FBProphet,list_all)
    

#finish = time.perf_counter()
#print(f'Finished in {round(finish-start, 2)} second(s)')

if __name__ == '__main__':
    main()
