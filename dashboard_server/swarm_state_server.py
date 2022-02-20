from flask import Flask
from flask import request
import flask
import boto3

app = Flask(__name__)

@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route("/")
def hello_world():
    return "<p> hello, world! </p>"

@app.route("/swarm")
def swarm():
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    swarm_names = []
    for table in list(dynamodb.tables.all()):
        name = table.name
        if name.split('-')[0] != 'Ec2instanceScheduler': #TODO make custom prefix..
            swarm_names.append(name)
    return {'swarm':swarm_names}

@app.route("/devices")
def devices():
    remove_keys = ['DataIndices', 'EvalHistLoss', 'EvalHistMetric', 'TimeStamps']
    return get_all_devices(remove_keys)

@app.route("/swarm-metric")
def swarm_metric():
    hist = get_all_devices_as_hist()
    times, loss_list = get_accs_over_time(hist, 'dummy_key')
    resp = {}
    resp['times'] = times
    resp['accs_list'] = loss_list

    # sample points evenly from all points
    SAMPLE_NUM = 100
    sampled = {}
    sampled['times'] = even_sampling(resp['times'], SAMPLE_NUM)
    sampled['accs_list'] = even_sampling(resp['accs_list'], SAMPLE_NUM)
    return sampled

def even_sampling(orig, num):
    itvl = len(orig)/num
    if itvl < 1:
        return orig
    idx = 0
    lst = []
    while (idx < len(orig)):
        lst.append(orig[int(idx)])
        idx += itvl

    if int(idx) != len(orig) - 1:
        lst.append(orig[-1])    

    return lst

def get_paginator_itr():
    swarm_name = request.args.get('swarm-name')
    client = boto3.client('dynamodb', region_name='us-east-1')
    paginator = client.get_paginator('scan')
    return paginator.paginate(TableName=swarm_name)

def get_all_devices(remove_keys):
    resp = {}
    res = []
    for page in get_paginator_itr():
        for item in page['Items']:
            for key in remove_keys:
                item.pop(key, None)
            res.append(item)
    res = sorted(res, key=lambda x: x['DeviceId']['N'])
    resp['devices'] = res

    return resp

@app.route("/get-all-hist")
def get_all_devices_as_hist():
    # @TODO interpoloate dots when there are too many
    resp = get_all_devices([])
    hist = {}
    hist['dummy_key'] = {}
    for dev in resp['devices']:
        hist['dummy_key'][int(dev['DeviceId']['N'])] = []
        for i in range(len(dev['TimeStamps']['L'])):
            hist['dummy_key'][int(dev['DeviceId']['N'])].append([float(dev['TimeStamps']['L'][i]['N']),
            [float(dev['EvalHistLoss']['L'][i]['N']), float(dev['EvalHistMetric']['L'][i]['N'])]])
        if len(hist['dummy_key'][int(dev['DeviceId']['N'])]) == 0:
            hist['dummy_key'][int(dev['DeviceId']['N'])].append([0, [1, 0]])
    return hist
    
def get_accs_over_time(loaded_hist, key):
    loss_diff_at_time = []
    for k in loaded_hist[key].keys():
        i = 0
        for t, h in loaded_hist[key][k]:
            if i != 0:
                loss_diff_at_time.append((t, loaded_hist[key][k][i][1][1] - loaded_hist[key][k][i-1][1][1]))
            i += 1
    loss_diff_at_time.sort(key=lambda x: x[0])
    # concatenate duplicate time stamps
    ldat_nodup = []
    for lt in loss_diff_at_time:
        if len(ldat_nodup) != 0 and ldat_nodup[-1][0] == lt[0]:
            ldat_nodup[-1] = (ldat_nodup[-1][0], ldat_nodup[-1][1] + lt[1])
        else:
            ldat_nodup.append(lt)
    times = []
    loss_list = []
    times.append(0)
    # get first accuracies
    accum = []
    for c in loaded_hist[key].keys():
        accum.append(loaded_hist[key][c][0][1][1])

    loss_list.append(sum(accum)/len(accum))
    for i in range(1, len(ldat_nodup)):
        times.append(ldat_nodup[i][0])
        loss_list.append(loss_list[i-1] + ldat_nodup[i][1]/len(loaded_hist[key]))

    return times, loss_list

