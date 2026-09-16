"""Probe one attachment URL request; never print credentials or signed URLs."""
import ast
import csv
import json
import logging
import importlib.util
import argparse
import io
import zipfile
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path

import requests
from huaweiresearchsdk.config import BridgeConfig
from huaweiresearchsdk.config import HttpClientConfig
from huaweiresearchsdk.bridge import BridgeClient
from huaweiresearchsdk.utilities import HttpHelper


def config_from_script():
    source = next(Path(__file__).parent.glob('HuaweiResearch_DataDownlod*.py'))
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    config = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    'yourProjectName', 'accessKey', 'secretKey', 'table_id'
                ):
                    config[target.id] = node.value.value
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-one', action='store_true')
    parser.add_argument('--filename', help='Select a specific attachment from the CSV')
    args = parser.parse_args()
    config = config_from_script()
    with (Path(__file__).parent / '\u4f20\u611f\u5668\u539f\u59cb\u6570\u636e.csv').open(
        encoding='utf-8-sig', newline=''
    ) as stream:
        record = next((row for row in csv.DictReader(stream)
                       if not args.filename or Path(row['sensorData']).name == args.filename), None)
        if record is None:
            raise ValueError('Attachment filename was not found in the CSV')
        key = record['sensorData']
    bridge = BridgeConfig('product', config['accessKey'], config['secretKey'])
    script_path = next(Path(__file__).parent.glob('HuaweiResearch_DataDownlod*.py'))
    spec = importlib.util.spec_from_file_location('download_script', script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    logging.disable(logging.CRITICAL)
    with requests.Session() as session:
        session.trust_env = False
        projects = json.loads(HttpHelper.get(
            session, config['accessKey'], config['secretKey'],
            bridge.get_baseurl() + '/' + bridge.URL_PROJECT_LIST, timeout=(15, 30)
        ))
        project_id = next(p['projectId'] for p in projects
                          if p['projectName'] == config['yourProjectName'])
        request = script.AttachmentDownloadRequest(
            [key], '.', project_id, config['table_id'], record['uniqueid'])
        try:
            if args.download_one:
                output = Path(__file__).parent / 'attachment_probe'
                access = SimpleNamespace(
                    project_id=project_id,
                    bridgeclient=BridgeClient(bridge, HttpClientConfig(30, 60, False)))
                with redirect_stdout(io.StringIO()):
                    filename = script.AccessSdk.downloadAudio_Sensoe(
                        access, key, str(output), config['table_id'], record['uniqueid'])
                target = output / filename
                with zipfile.ZipFile(target) as archive:
                    bad_member = archive.testzip()
                    if bad_member is not None:
                        raise ValueError('ZIP integrity check failed')
                    print('ZIP members:', len(archive.namelist()))
                    print('Member extensions:', sorted({Path(p).suffix for p in archive.namelist()}))
                print('ZIP integrity: OK')
                print('Bytes:', target.stat().st_size)
                print('Saved:', str(target))
                return
            result = json.loads(HttpHelper.post(
                session, config['accessKey'], config['secretKey'],
                bridge.get_baseurl() + '/' + bridge.URL_FILE_DOWNLOAD,
                request.get_file_paths(),
                {'Project-Id': project_id, 'User-Agent': bridge.get_useragent()},
                timeout=(15, 30)
            ))
            print('Response type:', type(result).__name__)
            if isinstance(result, list):
                print('Items:', len(result))
                for item in result:
                    print('Fields:', sorted(item) if isinstance(item, dict) else type(item).__name__)
                    if isinstance(item, dict):
                        print('URL issued:', bool(item.get('url')))
            else:
                print('Response:', json.dumps(result, ensure_ascii=True)[:2000])
        except requests.HTTPError as error:
            print(script.format_download_error(error))


if __name__ == '__main__':
    main()
