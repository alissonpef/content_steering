import csv
import os
import tempfile

from src.steering.logging_csv import (
    CSV_HEADERS,
    get_unique_log_filename,
    log_data_to_csv,
    setup_csv_logging,
)


def test_csv_logging_flow():
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "test_log.csv")
        setup_csv_logging(filename)
        assert os.path.exists(filename)
        with open(filename) as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers == CSV_HEADERS
        test_data = {
            "timestamp_server": "123456",
            "sim_time_client": 10,
            "client_lat": -23.0,
            "client_lon": -47.0,
        }
        log_data_to_csv(test_data, filename)
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
            assert row[0] == "123456"
            assert row[1] == "10"
            assert row[2] == "-23.0"


def test_get_unique_log_filename():
    with tempfile.TemporaryDirectory() as tempdir:
        base_name = "test"
        suffix = "_run"
        f1 = get_unique_log_filename(base_name, suffix, directory=tempdir)
        assert f1.endswith("test_run_1.csv")
        open(f1, "w").close()
        f2 = get_unique_log_filename(base_name, suffix, directory=tempdir)
        assert f2.endswith("test_run_2.csv")
