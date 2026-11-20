import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile
import requests

from src.acquire_scenarios import SplitReader, contained, download, wait_for_connection, network_request


class ArchiveTests(unittest.TestCase):
    def test_connectivity_checks_use_fifteen_second_intervals(self):
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        with patch('src.acquire_scenarios.requests.head', side_effect=[requests.ConnectionError(), response]) as head, \
             patch('src.acquire_scenarios.time.monotonic', side_effect=[0, 0, 15, 15, 30]), \
             patch('src.acquire_scenarios.time.sleep') as sleep:
            wait_for_connection('test')
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [15, 15])
        self.assertEqual(head.call_count, 2)

    def test_metadata_outage_does_not_give_up_after_five_failures(self):
        response = MagicMock()
        operation = MagicMock(side_effect=[requests.ConnectionError()] * 7 + [response])
        with patch('src.acquire_scenarios.wait_for_connection') as wait:
            self.assertIs(network_request(operation, 'metadata'), response)
        self.assertEqual(wait.call_count, 7)

    def test_stream_interruption_resumes_after_last_written_byte(self):
        def broken_stream(*args, **kwargs):
            yield b'abc'
            raise requests.exceptions.ChunkedEncodingError('offline')
        first = MagicMock(status_code=206, headers={'Content-Range': 'bytes 0-7/8'})
        first.__enter__.return_value = first
        first.iter_content.side_effect = broken_stream
        second = MagicMock(status_code=206, headers={'Content-Range': 'bytes 3-7/8'})
        second.__enter__.return_value = second
        second.iter_content.return_value = [b'defgh']
        with tempfile.TemporaryDirectory() as directory, \
             patch('src.acquire_scenarios.requests.get', side_effect=[first, second]) as get, \
             patch('src.acquire_scenarios.wait_for_connection') as wait:
            path = Path(directory) / 'data'
            download('https://example.test/file', path, 8)
            self.assertEqual(path.read_bytes(), b'abcdefgh')
            self.assertEqual(get.call_args_list[1].kwargs['headers']['Range'], 'bytes=3-7')
            wait.assert_called_once()

    def test_split_zip_reads_across_volume_boundaries_and_checks_crc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = io.BytesIO()
            payload = bytes(range(256)) * 100
            with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as z:
                z.writestr('scenario36/unit1/rgb5/test.jpg', payload)
            raw = data.getvalue()
            paths = []
            for start in range(0, len(raw), 31):
                part = root / str(start)
                part.write_bytes(raw[start:start+31])
                paths.append(part)
            with SplitReader(paths) as reader, zipfile.ZipFile(reader) as z:
                self.assertEqual(z.read(z.namelist()[0]), payload)
                self.assertIsNone(z.testzip())

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                contained(root, '../outside')
            self.assertEqual(contained(root, 'scenario36/file'), root/'scenario36'/'file')

    def test_corrupted_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = io.BytesIO()
            with zipfile.ZipFile(raw, 'w', zipfile.ZIP_STORED) as z:
                z.writestr('sample.txt', b'known payload')
            broken = raw.getvalue().replace(b'known payload', b'wrong payload')
            part = Path(directory) / 'archive.zip.001'
            part.write_bytes(broken)
            with SplitReader([part]) as reader, zipfile.ZipFile(reader) as z:
                with self.assertRaises(zipfile.BadZipFile):
                    z.read('sample.txt')

    def test_download_resumes_exact_byte_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'file.zip.001'
            path.with_name(path.name + '.partial').write_bytes(b'abc')
            response = MagicMock()
            response.status_code = 206
            response.headers = {'Content-Range': 'bytes 3-7/8'}
            response.iter_content.return_value = [b'defgh']
            response.__enter__.return_value = response
            with patch('src.acquire_scenarios.requests.get', return_value=response) as get:
                download('https://example.test/file', path, 8)
            self.assertEqual(path.read_bytes(), b'abcdefgh')
            self.assertEqual(get.call_args.kwargs['headers']['Range'], 'bytes=3-7')
            self.assertFalse(path.with_name(path.name + '.partial').exists())

    def test_ignored_range_does_not_corrupt_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'file.zip.001'
            partial = path.with_name(path.name + '.partial')
            partial.write_bytes(b'abc')
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            response.__enter__.return_value = response
            with patch('src.acquire_scenarios.requests.get', return_value=response):
                with self.assertRaises(ValueError):
                    download('https://example.test/file', path, 8)
            self.assertEqual(partial.read_bytes(), b'abc')
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
