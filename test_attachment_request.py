"""Offline integration checks; these do not assert the live server DTO schema."""
import importlib.util
import io
import json
import os
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch, Mock

import requests
import urllib3
from huaweiresearchsdk.config import BridgeConfig, HttpClientConfig
from huaweiresearchsdk.utilities import Consts


original_pool = urllib3.PoolManager
path = next(Path(__file__).parent.glob('HuaweiResearch_DataDownlod*.py'))
spec = importlib.util.spec_from_file_location('download_script', path)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class AttachmentRequestTests(unittest.TestCase):
    def test_download_loop_preserves_record_key_pairs_across_pages(self):
        columns = ['sensorData']
        access = Mock()
        access.QueryData.side_effect = [
            [['sensorData', 'uniqueid'], ['kitattachments/a.zip', 'id-a'],
             ['kitattachments/b.zip', 'id-b']],
            [['sensorData', 'uniqueid'], ['kitattachments/c.zip', 'id-c']]]
        initial_directory = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with patch.object(script.time, 'sleep'), \
                        patch.object(script, 'transf_time'), redirect_stdout(io.StringIO()):
                    script.get_attachments(access, 100, [(1, 2), (2, 3)], 'table', columns, 'test')
                pairs = {(call.args[0], call.args[2], call.args[3])
                         for call in access.downloadAudio_Sensoe.call_args_list}
                self.assertEqual(pairs, {
                    ('kitattachments/a.zip', 'table', 'id-a'),
                    ('kitattachments/b.zip', 'table', 'id-b'),
                    ('kitattachments/c.zip', 'table', 'id-c')})
                self.assertEqual(access.QueryData.call_args.args[2], ['sensorData', 'uniqueid'])
                os.chdir(initial_directory)
        finally:
            os.chdir(initial_directory)
        self.assertEqual(columns, ['sensorData'])

    def test_copied_sensor_table_uses_attachment_downloader(self):
        with patch.object(script, 'AccessSdk'), \
                patch.object(script, 'start_date', None, create=True), \
                patch.object(script, 'end_date', None, create=True), \
                patch.object(script, 'get_timestamps', return_value=[]), \
                patch.object(script.os.path, 'exists', return_value=True), \
                patch.object(script, 'get_attachments') as attachments, \
                patch.object(script, 'getData') as table_only, \
                redirect_stdout(io.StringIO()):
            script.major('t_project_copy_123', ['sensorData'], 'test', 'test', 'test', 'test')
        attachments.assert_called_once()
        table_only.assert_not_called()

    def test_serialized_signed_request_and_zip_written_by_sdk(self):
        key = 'kitattachments/20260724/SensorOriginalData/example.zip'
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('imu.csv', 'x,y,z\n1,2,3\n')
        zip_bytes = buffer.getvalue()
        sent = []

        class ApiSession:
            def post(self, url, data, headers, **kwargs):
                sent.append((url, json.loads(data), headers))
                response = requests.Response()
                response.status_code = 200
                response._content = json.dumps([
                    {'key': key, 'url': 'https://download.example.invalid/signed'}
                ]).encode()
                return response

        class Provider:
            def prepare_request(self):
                return ApiSession()

        class FilePool:
            def request(self, method, url):
                self.data = zip_bytes
                return self

            def release_conn(self):
                pass

        service = script._sdk_service.HiResearchDataService(
            BridgeConfig('product', 'test-access', 'test-secret'),
            HttpClientConfig(1, 1, False), None, Provider()
        )
        with tempfile.TemporaryDirectory() as directory:
            req = script.AttachmentDownloadRequest([key], directory, 'test-project', 'source-table', 'record-1')
            with patch.object(script._sdk_service.urllib3, 'PoolManager', FilePool):
                with redirect_stdout(io.StringIO()):
                    service.batch_download_file(req)
            output = Path(directory) / 'example.zip'
            self.assertEqual(output.read_bytes(), zip_bytes)
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
        self.assertEqual(len(sent), 1)
        expected = [{'objectKey': key, 'tableId': 'source-table', 'uniqueId': 'record-1'}]
        self.assertEqual(sent[0][1], expected)
        self.assertEqual(sent[0][2]['Project-Id'], 'test-project')
        self.assertTrue(sent[0][2][Consts.RESEARCH_AUTHORIZATION])
        self.assertEqual(int(sent[0][2]['content-length']),
                         len(json.dumps(expected).encode()))

    def test_validation_and_shared_urllib3_preserved(self):
        with self.assertRaises(Exception):
            script.AttachmentDownloadRequest(['C:\\bad.zip'], '.', 'project', 'table', 'record-1')
        self.assertIs(urllib3.PoolManager, original_pool)

    def test_table_id_required_and_applied_to_every_object(self):
        for table_id in (None, '', '  '):
            with self.assertRaises(ValueError):
                script.AttachmentDownloadRequest(['a.zip'], '.', 'project', table_id, 'record-1')
        for unique_id in (None, '', '  '):
            with self.assertRaises(ValueError):
                script.AttachmentDownloadRequest(['a.zip'], '.', 'project', 'table', unique_id)
        req = script.AttachmentDownloadRequest(['a.zip', 'b.zip'], '.', 'project', 'copy-table', 'record-1')
        self.assertEqual(req.get_file_paths(), [
            {'objectKey': 'a.zip', 'tableId': 'copy-table', 'uniqueId': 'record-1'},
            {'objectKey': 'b.zip', 'tableId': 'copy-table', 'uniqueId': 'record-1'}])

    def test_business_400_inside_http_500_is_not_retried(self):
        response = requests.Response()
        response.status_code = 500
        response._content = b'{"code":400,"message":"tableId is not exist"}'
        error = requests.HTTPError(response=response)
        self.assertTrue(script.is_permanent_download_error(error))
        response._content = b'{"code":500}'
        self.assertFalse(script.is_permanent_download_error(error))
        response._content = b'{"code":429}'
        self.assertFalse(script.is_permanent_download_error(error))

    def test_http_error_details_and_signed_url_redaction(self):
        response = requests.Response()
        response.status_code = 400
        response.reason = 'Bad Request'
        response._content = b'{"message":"invalid objectKeys"}'
        message = script.format_download_error(requests.HTTPError(response=response))
        self.assertIn('invalid objectKeys', message)
        self.assertIn('400', message)
        message = script.format_download_error(
            requests.ConnectionError('failed https://files.invalid/a?signature=secret')
        )
        self.assertNotIn('signature', message)


if __name__ == '__main__':
    unittest.main()
