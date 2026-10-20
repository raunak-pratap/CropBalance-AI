import numpy as np
import pandas as pd

from fetcher import MandiPriceFetcher


def make_response(rows, arrival_date="15/05/2025"):
    return {
        "markets": [
            {
                "marketName": "Test Mandi",
                "dates": [
                    {
                        "arrivalDate": arrival_date,
                        "data": rows,
                    }
                ],
            }
        ]
    }


def valid_row(**overrides):
    row = {
        "variety": "Common",
        "arrivals": 12.5,
        "minimumPrice": 2000,
        "maximumPrice": 2400,
        "modalPrice": 2200,
    }
    row.update(overrides)
    return row


def test_valid_record_is_parsed():
    fetcher = MandiPriceFetcher()
    df = fetcher._parse_agmarknet_response(
        make_response([valid_row()]), "Punjab"
    )

    assert len(df) == 1
    assert df.iloc[0]["state"] == "Punjab"
    assert df.iloc[0]["market"] == "Test Mandi"
    assert df.iloc[0]["modal_price"] == 2200
    assert pd.Timestamp(df.iloc[0]["date"]) == pd.Timestamp("2025-05-15")


def test_invalid_prices_are_rejected():
    fetcher = MandiPriceFetcher()
    rows = [
        valid_row(minimumPrice=0),
        valid_row(modalPrice=None),
        valid_row(minimumPrice=2300, modalPrice=2200),
        valid_row(maximumPrice=2100, modalPrice=2200),
    ]

    df = fetcher._parse_agmarknet_response(
        make_response(rows), "Punjab"
    )

    assert df.empty


def test_invalid_date_is_skipped():
    fetcher = MandiPriceFetcher()
    df = fetcher._parse_agmarknet_response(
        make_response([valid_row()], arrival_date="not-a-date"),
        "Punjab",
    )

    assert df.empty


def test_missing_arrivals_remain_nan():
    fetcher = MandiPriceFetcher()
    df = fetcher._parse_agmarknet_response(
        make_response([valid_row(arrivals=None)]), "Punjab"
    )

    assert len(df) == 1
    assert np.isnan(df.iloc[0]["arrivals_tonnes"])
