import csv
import datetime
import importlib
import os
import re
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import psutil
import requests
from huaweiresearchsdk.bridge import BridgeClient
from huaweiresearchsdk.config import BridgeConfig, HttpClientConfig
from huaweiresearchsdk.model import BatchGetFileRequest
from huaweiresearchsdk.model.table import FilterCondition, FilterOperatorType, SearchTableDataRequest


class _RequestsDownloadResponse:
    def __init__(self, response):
        self.data = response.content

    def release_conn(self):
        pass


class _DirectDownloadPool:
    """Use a direct, timeout-aware requests session for SDK attachment URLs."""

    def __init__(self, *args, **kwargs):
        self.session = requests.Session()
        self.session.trust_env = False

    def request(self, method, url, **kwargs):
        try:
            with self.session.request(method, url, timeout=(30, 300)) as response:
                response.raise_for_status()
                return _RequestsDownloadResponse(response)
        finally:
            self.session.close()


# The old SDK uses a bare urllib3 pool for signed attachment URLs. Replace only
# that pool; table queries and request signing still use the original SDK code.
_sdk_service = importlib.import_module("huaweiresearchsdk.service.HiResearchDataService")
# Do not replace PoolManager on the shared urllib3 module used by other code.
_sdk_service.urllib3 = SimpleNamespace(PoolManager=_DirectDownloadPool)


class AttachmentDownloadRequest(BatchGetFileRequest):
    """Adapt SDK 1.0.0's string list to the server's DownLoadListReq list.

    Each object carries the attachment key, source table and source record ID.
    Verified against the live portal and downloads/urls endpoint on 2026-09-16.
    """

    def __init__(self, filepaths, output_path, project_id, table_id, unique_id):
        if not isinstance(table_id, str) or not table_id.strip():
            raise ValueError('附件请求必须提供来源数据表 table_id')
        if not isinstance(unique_id, str) or not unique_id.strip():
            raise ValueError('附件请求必须提供来源记录 uniqueid')
        super().__init__(filepaths, output_path, project_id)
        self.table_id = table_id.strip()
        self.unique_id = unique_id

    def get_file_paths(self):
        return [{'objectKey': path, 'tableId': self.table_id, 'uniqueId': self.unique_id}
                for path in super().get_file_paths()]


def is_permanent_download_error(error):
    response = getattr(error, 'response', None)
    if response is None:
        return False
    code = response.status_code
    # This service can wrap a business-level 400 in an HTTP 500 response.
    try:
        body = response.json()
        if isinstance(body, dict) and 'code' in body:
            business_code = int(body['code'])
            if 400 <= business_code < 500:
                code = business_code
    except (ValueError, TypeError):
        pass
    return 400 <= code < 500 and code not in (408, 429)


def format_download_error(error):
    response = getattr(error, 'response', None)
    if response is None:
        message = '{}: {}'.format(type(error).__name__, error)
    else:
        message = 'HTTP {} {}；服务器响应：{}'.format(
            response.status_code, response.reason, response.text.strip()[:2000])
    # Signed download URLs can occur in transport exceptions and response bodies.
    return re.sub(r'https?://[^\s"<>]+', '<URL omitted>', message)


def monitor_system():
    # 获取CPU使用情况
    cpuper = psutil.cpu_percent()
    # 获取内存使用情况：系统内存大小，使用内存，有效内存，内存使用率
    mem = psutil.virtual_memory()
    # 内存使用率
    memper = mem.percent
    # 获取当前时间
    # now = datetime.datetime.now()
    # ts = now.strftime('%Y-%m-%d %H:%M:%S')
    # line = f'{ts} cpu:{cpuper}%, mem:{memper}%'
    # print(line)


def handle_data_one_run(table_id, columnsName, accessKey, secretKey, yourProjectName,file_name):
    monitor_system()
    start = time.process_time()
    # 任务
    major(table_id, columnsName, accessKey, secretKey, yourProjectName,file_name)
    # end = time.process_time()
    # print("本次运行时间为：%.02f 秒" % (end - start))
    # now = datetime.datetime.now()
    # ts = now.strftime('%Y-%m-%d %H:%M:%S')
    # print("当前本地时间为：{}".format(ts))


class AccessSdk():
    def __init__(self, yourProjectName, yourAccessKey, yourSecretKey):
        self.yourProjectName = yourProjectName
        self.yourAccessKey = yourAccessKey
        self.yourSecretKey = yourSecretKey
        self.accessToken()

    def QueryData(self, tableId, condition, name, sortFields, moreNum):
        self.name = name
        # 构造查询请求
        if len(condition) == 0:
            request = SearchTableDataRequest(tableId, desired_size=3000, sorts=sortFields,
                                             include_fields=name, giveup_when_more_than=moreNum,
                                             project_id=self.project_id)
        else:
            request = SearchTableDataRequest(tableId, filters=condition, desired_size=3000, sorts=sortFields,
                                             include_fields=name, giveup_when_more_than=moreNum,
                                             project_id=self.project_id)
        results = list()

        # 构造回调函数
        def rows_processor(rows, totalCnt):
            # print("totalCnt: ", totalCnt, "len(rows): ", len(rows))
            results.extend(rows)

        # 查询数据结果
        self.bridgeclient.get_bridgedata_provider().query_table_data(request, callback=rows_processor)
        # print(json.dumps(rs))
        if len(results) == 0:
            pass
        else:
            results = self.Collation2(results)
            # print(results)
            # return results
        return results

    def Collation2(self, data):
        resultD = [self.name]
        for i in data:
            # print('----',i)
            appD = []
            for co in self.name:
                if '.' in co:
                    key_list = co.split('.')
                    # print('key_list--',key_list)
                    # print('i.keys()',i.keys())
                    if key_list[0] not in i.keys():
                        appD.append("")
                        continue
                    value = i[key_list[0]]
                    # print(value)
                    for k in range(1, len(key_list)):
                        # print(key_list,k)
                        if key_list[k] in value:
                            value = value[key_list[k]]
                        else:
                            value = ''
                        # print(value)
                    appD.append(value)
                elif co in i.keys():
                    appD.append(i[co])
                else:
                    appD.append("")
            resultD.append(appD)
            # print(resultD)
            # break
        return resultD

    def downloadAudio_Sensoe(self, downloadPath, savePath, table_id, unique_id):
        os.makedirs(savePath, exist_ok=True)
        fileName = os.path.basename(downloadPath)
        filePath = os.path.join(savePath, fileName)
        if os.path.isfile(filePath) and os.path.getsize(filePath) > 0:
            return fileName

        request = AttachmentDownloadRequest(
            filepaths=[downloadPath], output_path=savePath,
            project_id=self.project_id, table_id=table_id, unique_id=unique_id)
        self.bridgeclient.get_bridgedata_provider().batch_download_file(request)
        if not os.path.isfile(filePath) or os.path.getsize(filePath) == 0:
            raise RuntimeError("附件接口返回后未生成有效文件: {}".format(fileName))
        return fileName

    def accessToken(self):
        # 初始化BridgeConfig类
        # bridgeConfig = BridgeConfig(self.yourProjectCode, "product")
        self.bridgeConfig = BridgeConfig("product", self.yourAccessKey, self.yourSecretKey)
        # 连接超时时间，单位s，不设置则默认30s
        connectTimeout = 200
        # 等待接口返回超时时间，单位s，不设置则默认30s
        readTimeout = 200
        # 是否失败重试，默认不重试
        retryOnFail = True
        # 初始化HttpClientConfig类
        self.httpConfig = HttpClientConfig(connectTimeout, readTimeout, retryOnFail)
        self.bridgeclient = BridgeClient(self.bridgeConfig, self.httpConfig)

        projectL = self.bridgeclient.get_bridgedata_provider().list_projects()
        print(projectL)
        projectD = dict()
        for p in projectL:
            projectD.update({p["projectName"]: p})
        self.project_id = projectD[self.yourProjectName]["projectId"]
        print("需要使用的项目id：【{}】".format(self.project_id))

    def re_accessToken(self):

        self.bridgeclient = BridgeClient(self.bridgeConfig, self.httpConfig)

    def __del__(self):
        print("_ " * 20, "我是销毁方法，我被调用了。", "_ " * 20)
        now = datetime.now()
        ts = now.strftime('%Y-%m-%d %H:%M:%S')
        print("下载结束时间：{}".format(ts))


def get_timestamps(start_date, end_date):
    try:
        # 设置间隔天数
        interval = timedelta(days=15)  # 数据少时间间隔可以设置长一点

        # 生成时间戳列表
        timestamps = []
        current_date = start_date
        while current_date <= end_date:
            timestamps.append(int(current_date.timestamp()))
            current_date += interval

        # 打印时间戳列表
        timestamps.append(int(end_date.timestamp()))
        # print(timestamps)
        if timestamps[-1] == timestamps[-2]:
            timestamps.remove(timestamps[-1])

        # print(len(timestamps), timestamps)
        new_timestamps = list()
        for i in range(len(timestamps) - 1):
            new_timestamps.append([timestamps[i], timestamps[i + 1]])
        print('查询区间为：', new_timestamps)
        return new_timestamps
    except:
        print('*'*20+'检查406、407行开始和结束时间是否有冲突 或 未按要求填写')


def get_hourly_timestamps(start, end):
    """
    输入格式：'YYYY-MM-DD'
    输出：UTC时间戳列表（秒级）
    """
    # start = datetime.strptime(start_date, '%Y-%m-%d')
    # end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)  # 包含结束日
    timestamps = []
    current = start
    while current < end:
        # 生成当天24个整点时间戳
        for hour in range(24):
            ts = int((current + timedelta(hours=hour)).timestamp())
            timestamps.append(ts)
        current += timedelta(days=1)
    timestamps.append(int(end.timestamp()))
    new_timestamps = list()
    for i in range(len(timestamps) - 1):
        new_timestamps.append([timestamps[i], timestamps[i + 1]])
    return new_timestamps


def transf_time(path):
    '''
    更新下载数据的时间戳为正常格式
    :param path:
    :return:
    '''
    path0 = os.path.join(os.getcwd(), path)
    # print(path0)
    a = []
    b = 0
    new_name=path.replace('0.csv', '.csv')
    path1 = os.path.join(os.getcwd(), new_name)
    with open(path1, 'w') as f:
        with open(path0, 'r') as ww:
            for i in ww:
                b += 1
                i = i.split(',')
                if 'healthid' in i:
                    for t in i:
                        if t.endswith('time') or t.endswith('Time') or t.endswith('timestamp'):
                            a.append(i.index(t))
                else:
                    for dex in a:
                        # print(i[dex])
                        try:
                            i[dex] = datetime.fromtimestamp(int(i[dex]) // 1000)
                        except:
                            i[dex] = i[dex]
                            # print(i[dex])
                    # print(b,i)
                i1 = ','.join(str(a) for a in i)
                f.write(i1)
    print('保存文件为：',new_name)
    os.remove(path0)


def sleep_data_process(path):
    with open(path, 'r') as f:
        all_data = []
        for line in f:
            if 'healthid' in line:
                if 'sampleData' in line:
                    line_list = line.strip().split(',')
                    all_data.append(line_list)
                else:
                    return
            else:
                line_list = line.strip().split(',', 35)
                first = line_list[0:-1]
                result = line_list[-1]
                last_length = len(result)
                # print(result)
                if last_length >= 32000:
                    result = [result[i:i + 32000] for i in range(0, last_length, 32000)]
                else:
                    result = [result]
                data_line = first + result
                all_data.append(data_line)

        with open(path.replace('.csv','格式处理.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in all_data:
                writer.writerow(row)


def major(table_id, columnsName, accessKey, secretKey, yourProjectName,file_name):
    nowTimeInt = int(time.mktime(time.localtime(time.time())))
    # 项目配置
    print("start1")
    access = AccessSdk(yourProjectName, accessKey, secretKey)
    moreNum = 2000000
    print("start2")
    # 下载超大批量数据表时调用该方法，如ConsecutiveBloodOxygenSaturation
    # timestamps = get_hourly_timestamps(start_date, end_date)

    # 下载总数量不超过1亿数据表时用调用以下方法
    timestamps = get_timestamps(start_date, end_date)

    # 失败重试的时间戳，将上一次下载失败的数据复制到这里，
    # timestamps=[]

    substrings=['_motion_','_apneameasuredata_','_ecg_','_ppg_','_acceleration_','_sensororiginaldata_']
    contains_all = 'sensorData' in columnsName or any(sub in table_id for sub in substrings)
    if  contains_all:
        if not os.path.exists(table_id + '附件'+file_name):
            os.makedirs(table_id + '附件'+file_name)
        get_attachments(access, moreNum, timestamps, table_id, columnsName,file_name)
    else:
        getData(access, moreNum, timestamps, table_id, columnsName,file_name)


def getData(access, moreNum, timestamps, table_id, columnsName,file_name):
    j = 0
    fail_time = list()
    for i in range(len(timestamps)):
        condition = [FilterCondition("recordtime", FilterOperatorType.GREATER_THAN, timestamps[i][0] * 1000),
                     FilterCondition("recordtime", FilterOperatorType.LESS_EQUALS, timestamps[i][1] * 1000)]
        sortFields = [{"name": "uploadtime", "type": "desc"}]
        start_time = datetime.fromtimestamp(timestamps[i][0]).strftime('%Y-%m-%d %H:%M:%S')
        end_time = datetime.fromtimestamp(timestamps[i][1]).strftime('%Y-%m-%d %H:%M:%S')
        try:
            resultData = access.QueryData(table_id, condition, columnsName, sortFields, moreNum)
        except:
            time.sleep(180)
            try:
                print(i + 1, start_time, '———', end_time, '下载失败———重试一次')
                resultData = access.QueryData(table_id, condition, columnsName, sortFields, moreNum)
            except:
                print(i + 1, start_time, '———', end_time, '-' * 80, '重试失败需重新下载')
                fail_time.append(timestamps[i])
                time.sleep(130)
                continue
        # print(resultData)
        if j > 1:
            resultData = resultData[1:]
        if resultData:
            if j <= 1:
                resultData.sort(reverse=True)
            with open(table_id + file_name+'0.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in resultData:
                    j += 1
                    writer.writerow(row)
        print(i + 1, start_time, '———', end_time, '下载数据:', len(resultData), '   累计下载:', j, '条')
        time.sleep(6)
    if fail_time:
        print('下载失败 需要重新下载的列表：')
        print(fail_time)
    try:
        transf_time(table_id + file_name+'0.csv')
    except:
        pass
    # 睡眠数据格式异常处理
    if '_sleepepisode_' in table_id:
        sleep_data_process(table_id + file_name+'.csv')


def get_attachments(access, moreNum, timestamps, table_id, columnsName,file_name):
    # Attachment authorization requires the uniqueid from the SAME source row.
    columnsName = list(columnsName)
    if 'uniqueid' not in columnsName:
        columnsName.append('uniqueid')
    unique_id_index = columnsName.index('uniqueid')
    j = 0
    fail_time = list()
    fail_file = list()
    save_path = table_id + '附件' + file_name
    downloaded = 0
    skipped = 0
    for i in range(len(timestamps)):
        condition = [FilterCondition("recordtime", FilterOperatorType.GREATER_THAN, timestamps[i][0] * 1000),
                     FilterCondition("recordtime", FilterOperatorType.LESS_EQUALS, timestamps[i][1] * 1000)]
        sortFields = [{"name": "uploadtime", "type": "desc"}]
        start_time = datetime.fromtimestamp(timestamps[i][0]).strftime('%Y-%m-%d %H:%M:%S')
        end_time = datetime.fromtimestamp(timestamps[i][1]).strftime('%Y-%m-%d %H:%M:%S')
        try:
            resultData = access.QueryData(table_id, condition, columnsName, sortFields, moreNum)
        except:
            time.sleep(180)
            try:
                print(i + 1, start_time, '———', end_time, '下载失败———重试一次')
                resultData = access.QueryData(table_id, condition, columnsName, sortFields, moreNum)
            except:
                print(i + 1, start_time, '———', end_time, '-' * 80, '重试失败需重新下载')
                fail_time.append(timestamps[i])
                time.sleep(130)
                continue
        # print(resultData)
        if j > 1:
            resultData = resultData[1:]
        if resultData:
            if j <= 1:
                resultData.sort(reverse=True)
            with open(table_id + file_name+'0.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in resultData:
                    j += 1
                    writer.writerow(row)

        attachment_records = []
        for line in resultData:
            attachment_records.extend(
                (str(value), line[unique_id_index]) for value in line
                if str(value).startswith('kitattachments/')
            )

        for kit, unique_id in dict.fromkeys(attachment_records):
            target_path = os.path.join(save_path, os.path.basename(kit))
            if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
                skipped += 1
                continue

            last_error = None
            for attempt in range(1, 4):
                try:
                    access.downloadAudio_Sensoe(kit, save_path, table_id, unique_id)
                    downloaded += 1
                    print('附件进度: 新下载 {}，已存在 {}，失败 {}'.format(
                        downloaded, skipped, len(fail_file)))
                    last_error = None
                    break
                except Exception as error:
                    last_error = error
                    print('附件下载失败（第 {}/3 次）: {}: {}'.format(
                        attempt, os.path.basename(kit), format_download_error(error)))
                    if is_permanent_download_error(error):
                        break
                    if attempt < 3:
                        time.sleep(2 * attempt)

            if last_error is not None:
                fail_file.append([kit, type(last_error).__name__, format_download_error(last_error)])
        print(i + 1, start_time, '———', end_time, '下载数据:', len(resultData), '   累计下载:', j, '条')
        time.sleep(6)
    print('附件保存目录：', save_path)
    print('附件结果: 新下载 {}，已存在 {}，失败 {}'.format(downloaded, skipped, len(fail_file)))
    if fail_time:
        print('下载失败 需要重新下载的列表：')
        print(fail_time)
    if fail_file:
        failure_path = os.path.join(save_path, 'failed_attachments.csv')
        with open(failure_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['attachment', 'error_type', 'error'])
            writer.writerows(fail_file)
        print('失败附件清单：', failure_path)
    try:
        transf_time(table_id + file_name+'0.csv')
    except:
        pass


if __name__ == "__main__":
    now = datetime.now()
    ts = now.strftime('%Y-%m-%d %H:%M:%S')
    print("下载开始时间：{}".format(ts))
    t=now.strftime('%m%d_%H%M')

    # 项目信息与用户鉴权，accessKey与secretKey见research平台右上角我的凭证
    yourProjectName = "第十一届全国大学生生物医学工程创新设计竞赛_智能穿戴与运动健康赛道_赛题二_训练集"
    accessKey = "da6c45e32d9f4b15b8b6c3503b8fe0f7"
    secretKey = "3e99ce87d4edf41b0f05bedbd29451504eface6f29c23388efe1b22d74719250"

    # 需要下载对应元数据Table ID如：t_userbasicinfo_system
    table_id = "t_zsstnnrj_sensororiginaldata_system"

    # 设置下载数据时间范围，根据数据表中recordtime字段进行筛选，如下为下载2022-5-1到2024-11-11的数据
    # 开始时间
    start_date = datetime(2026, 6, 20)
    # 结束时间
    end_date = datetime(2026, 9, 16)

    with open('DataTable.txt', 'r', encoding='utf-8') as all_name:
        i = 0
        for data in all_name:
            if table_id in data:
                columnsName = data.strip().split(',')[1:]
                print(columnsName)
                handle_data_one_run(table_id, columnsName, accessKey, secretKey, yourProjectName,t)
                i+=1
                break
        if i == 0:
            print('DataTable文件中未查找到table_id以及对应字段，请在DataTable中添加')


