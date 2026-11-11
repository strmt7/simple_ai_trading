"""Exact-order synthetic reads exercise real hidden children and enrolled storage."""

from contextlib import closing
import json
from time import monotonic

import pytest

from simple_ai_trading import binance_recovery_reads as module
from simple_ai_trading import polymarket_http_process as http_module
from simple_ai_trading.api import BinanceAPIError
from simple_ai_trading.binance_close_application import closing_recovery_sha256
from simple_ai_trading.binance_open_intents import OpenIntentError
from simple_ai_trading.runtime_control import RuntimeControlError
from test_binance_close_recovery import _scope, _stage


_SCRIPT = r"""
import hashlib,hmac,json,os,sys,time
from pathlib import Path
from urllib.parse import urlencode,urlsplit
sys.path.insert(0,sys.argv[1])
from simple_ai_trading import polymarket_http_process as engine
from simple_ai_trading import polymarket_sdk_transport as native
if 'RECOVERY_PARENT_SECRET' in os.environ:
    raise RuntimeError('ambient credentials inherited')
source=json.loads(Path(sys.argv[3]).read_text())
mode=sys.argv[4]
def request(self,method,endpoint,*,headers=None,data=None,params=None):
    with Path(sys.argv[5]).open('a',encoding='ascii') as out:
        out.write(method+' '+urlsplit(endpoint).path+'\n')
    if method != 'GET' or data is not None:
        raise RuntimeError('write reached native transport')
    if mode=='stall-order' or mode=='stall-trades' and not endpoint.endswith('/order'):
        time.sleep(60)
    if mode=='exit': os._exit(5)
    if mode=='reject':
        raise engine.PolyApiException(error_msg='PRIVATE-SENTINEL')
    if mode!='policy':
        if headers != {'X-MBX-APIKEY':'offline-placeholder'}:
            raise RuntimeError('key binding differs')
        values={key:value[0] for key,value in params.items()}
        signature=values.pop('signature')
        expected=hmac.new(b'offline-secret',urlencode(values).encode('ascii'),hashlib.sha256).hexdigest()
        if signature != expected or values['recvWindow']!='5000':
            raise RuntimeError('signature binding differs')
    return source['order'] if endpoint.endswith('/order') else source['trades']
native._InlineSdkTransport.request=request
engine._worker(int(sys.argv[2]))
"""


@pytest.fixture
def native_reads(tmp_path, monkeypatch):
    children, markers, settings = [], [], []
    original = http_module.subprocess.Popen

    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(http_module.subprocess, "Popen", launch)
    monkeypatch.setenv("RECOVERY_PARENT_SECRET", "synthetic sentinel")

    def configure(order, trades, mode="normal"):
        payload = tmp_path / f"fixture-{len(settings)}.json"
        payload.write_text(json.dumps(dict(order=order, trades=trades)))
        marker = payload.with_suffix(".calls")
        markers.append(marker)

        def factory(**configuration):
            settings.append(configuration)
            worker = http_module.PolymarketHttpProcess(**configuration)
            command = worker._command
            worker._command = lambda descriptor: [
                *command(descriptor)[:4],
                _SCRIPT,
                *command(descriptor)[5:],
                str(payload),
                mode,
                str(marker),
            ]
            return worker

        monkeypatch.setattr(module, "_http_process", factory)
        return marker

    yield configure, children, markers, settings
    assert all(child.poll() is not None for child in children)


def selection(store, client_id, control, scope=None, **changes):
    scope = scope or _scope()
    result = dict(
        scope=scope,
        client_id=client_id,
        control=control,
        expected_request_sha256=module.closing_collection_sha256(
            store,
            scope=scope,
            client_id=client_id,
        ),
        api_key="offline-placeholder",
        api_secret="offline-secret",
    )
    result.update(changes)
    return result


@pytest.mark.parametrize("product,side", [("spot", "LONG"), ("futures", "SHORT")])
def test_terminal_collection_retains_without_applying_or_rearming(
    tmp_path,
    enrolled_binance_runtime,
    native_reads,
    product,
    side,
):
    store, position, client_id, order, trades = _stage(tmp_path, product, side)
    authority = enrolled_binance_runtime(store, _scope(product), start=False)
    configure, children, _, settings = native_reads
    marker = configure(order, trades)
    before, stopped = store.load_snapshot(strict=True), authority.control.snapshot()
    options = selection(store, client_id, authority.control, _scope(product))
    result = module.collect_stopped_closing_recovery(store, **options)
    assert result.execution.executed_quantity == "0.002"
    assert result.execution.commissions == (("USDT", "0.11"),)
    assert result.futures_realized_pnl == ("-10" if product == "futures" else None)
    assert store.load_snapshot(strict=True) == before == ([position], [])
    assert authority.control.snapshot() == stopped
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    assert len(children) == 1 and settings[0]["read_only"] is True
    assert len(marker.read_text().splitlines()) == 2
    assert all(line.startswith("GET ") for line in marker.read_text().splitlines())
    assert closing_recovery_sha256(store, scope=_scope(product), client_id=client_id)
    assert module.collect_stopped_closing_recovery(store, **options) == result
    assert len(children) == 1  # Cached evidence never refreshes an account.


def test_zero_fill_uses_only_order_read(
    tmp_path, enrolled_binance_runtime, native_reads
):
    store, _, client_id, order, _ = _stage(tmp_path)
    authority = enrolled_binance_runtime(store, _scope(), start=False)
    order.update(status="CANCELED", executedQty="0", cummulativeQuoteQty="0")
    marker = native_reads[0](order, [])
    result = module.collect_stopped_closing_recovery(
        store,
        **selection(store, client_id, authority.control),
    )
    assert result.execution.executed_quantity == "0"
    assert marker.read_text().splitlines() == ["GET /api/v3/order"]


@pytest.mark.parametrize("failure", ["key", "digest", "owner", "enrollment"])
def test_authority_failures_do_not_start_a_child(
    tmp_path,
    enrolled_binance_runtime,
    native_reads,
    failure,
):
    store, _, client_id, order, trades = _stage(tmp_path)
    authority = enrolled_binance_runtime(store, _scope(), start=failure == "owner")
    native_reads[0](order, trades)
    options = selection(store, client_id, authority.control)
    if failure == "key":
        options["api_key"] = "rotated-key"
    if failure == "digest":
        options["expected_request_sha256"] = "0" * 64
    if failure == "enrollment":
        store.opening_intents.path.with_name(
            store.opening_intents.path.name + ".binance_execution.enrollment.json"
        ).unlink()
    with pytest.raises((OpenIntentError, RuntimeControlError)):
        module.collect_stopped_closing_recovery(store, **options)
    assert not native_reads[1]


@pytest.mark.parametrize("mode", ["stall-order", "stall-trades", "exit", "reject"])
def test_faults_reap_and_preserve_unknown_without_response_secrets(
    tmp_path,
    enrolled_binance_runtime,
    native_reads,
    mode,
):
    store, _, client_id, order, trades = _stage(tmp_path)
    authority = enrolled_binance_runtime(store, _scope(), start=False)
    marker = native_reads[0](order, trades, mode)
    options = selection(store, client_id, authority.control, timeout_seconds=3)
    started = monotonic()
    with pytest.raises(BinanceAPIError) as error:
        module.collect_stopped_closing_recovery(store, **options)
    assert monotonic() - started < 7
    assert "PRIVATE-SENTINEL" not in str(error.value)
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    assert authority.control.snapshot().state == "stop_requested"
    with closing(store.opening_intents._connect()) as connection:
        assert not connection.execute("PRAGMA table_info(closing_recovery)").fetchall()
    assert len(marker.read_text().splitlines()) == (2 if mode == "stall-trades" else 1)


@pytest.mark.parametrize("bad", ["live", "missing-trade", "wrong-client", "wrong-cash"])
def test_incomplete_or_conflicting_observations_are_not_retained(
    tmp_path,
    enrolled_binance_runtime,
    native_reads,
    bad,
):
    store, _, client_id, order, trades = _stage(tmp_path)
    authority = enrolled_binance_runtime(store, _scope(), start=False)
    if bad == "live":
        order["status"] = "PARTIALLY_FILLED"
    if bad == "missing-trade":
        trades = []
    if bad == "wrong-client":
        order["clientOrderId"] = "foreign"
    if bad == "wrong-cash":
        trades[0]["quoteQty"] = "109"
    native_reads[0](order, trades)
    with pytest.raises(OpenIntentError):
        module.collect_stopped_closing_recovery(
            store,
            **selection(store, client_id, authority.control),
        )
    with closing(store.opening_intents._connect()) as connection:
        assert not connection.execute("PRAGMA table_info(closing_recovery)").fetchall()


@pytest.mark.parametrize(
    "method,path,data",
    [
        ("POST", "/api/v3/order", None),
        ("DELETE", "/api/v3/order", None),
        ("GET", "/api/v3/account", None),
        ("GET", "/api/v3/order", "body"),
    ],
)
def test_authenticated_child_rejects_writes_and_unlisted_endpoints(
    tmp_path,
    native_reads,
    method,
    path,
    data,
):
    marker = native_reads[0]({}, [], "policy")
    with closing(
        module.BinanceRecoveryReadClient(
            scope=_scope(),
            symbol="BTCUSDT",
            client_id="sait-c-policy",
            api_key="offline-placeholder",
            api_secret="offline-secret",
        )
    ) as client:
        client._http.start(timeout_seconds=5)
        with pytest.raises(Exception):
            client._http.request(
                method,
                _scope().origin + path,
                headers={"X-MBX-APIKEY": "offline-placeholder"},
                data=data,
            )
    assert not marker.exists()


def test_one_order_and_one_bound_trade_page_cannot_be_replayed(tmp_path, native_reads):
    _, _, client_id, order, trades = _stage(tmp_path)
    marker = native_reads[0](order, trades)
    with closing(
        module.BinanceRecoveryReadClient(
            scope=_scope(),
            symbol="BTCUSDT",
            client_id=client_id,
            api_key="offline-placeholder",
            api_secret="offline-secret",
        )
    ) as client:
        with pytest.raises(OpenIntentError):
            client.get_order_trades("BTCUSDT", order_id=123, expected_scope=_scope())
        client.get_order(
            "BTCUSDT", orig_client_order_id=client_id, expected_scope=_scope()
        )
        with pytest.raises(OpenIntentError):
            client.get_order(
                "BTCUSDT", orig_client_order_id=client_id, expected_scope=_scope()
            )
        with pytest.raises(OpenIntentError):
            client.get_order_trades("BTCUSDT", order_id=999, expected_scope=_scope())
        client.get_order_trades("BTCUSDT", order_id=123, expected_scope=_scope())
        with pytest.raises(OpenIntentError):
            client.get_order_trades("BTCUSDT", order_id=123, expected_scope=_scope())
    assert len(marker.read_text().splitlines()) == 2


def test_missing_journal_is_not_created_and_sqlite_error_is_sanitized(tmp_path):
    from simple_ai_trading.positions import PositionsStore

    store = PositionsStore(tmp_path / "absent")
    with pytest.raises(OpenIntentError, match="unavailable"):
        module.closing_collection_sha256(store, scope=_scope(), client_id="sait-c-none")
    assert not store.root.exists()


def test_repeated_native_client_lifetimes_release_resources(
    tmp_path,
    native_reads,
    process_resource_sample,
):
    _, _, client_id, order, trades = _stage(tmp_path)
    configure, children, _, _ = native_reads

    def once():
        configure(order, trades)
        with closing(
            module.BinanceRecoveryReadClient(
                scope=_scope(),
                symbol="BTCUSDT",
                client_id=client_id,
                api_key="offline-placeholder",
                api_secret="offline-secret",
            )
        ) as client:
            client.get_order(
                "BTCUSDT", orig_client_order_id=client_id, expected_scope=_scope()
            )
            client.get_order_trades("BTCUSDT", order_id=123, expected_scope=_scope())
        assert client._secret == b"" and not client._key
        # The fixture normally retains Popen objects to audit teardown. Release
        # only that observation after exit, so its Windows handle does not pollute
        # measurements of production retention. No forced collection is used.
        completed = children.pop()
        assert completed.poll() is not None

    once()
    before = process_resource_sample()
    for _ in range(12):
        once()
    after = process_resource_sample()
    assert all(child.poll() is not None for child in children)
    assert after[0] <= before[0] + 4
    assert after[1] <= before[1]
    assert after[2] <= before[2] + 8 * 1024 * 1024


def test_expired_shared_deadline_does_not_start_a_child(
    tmp_path, native_reads, monkeypatch
):
    _, _, client_id, order, trades = _stage(tmp_path)
    native_reads[0](order, trades)
    with closing(
        module.BinanceRecoveryReadClient(
            scope=_scope(),
            symbol="BTCUSDT",
            client_id=client_id,
            api_key="offline-placeholder",
            api_secret="offline-secret",
        )
    ) as client:
        monkeypatch.setattr(module, "monotonic", lambda: client._deadline + 1)
        with pytest.raises(BinanceAPIError):
            client.get_order(
                "BTCUSDT", orig_client_order_id=client_id, expected_scope=_scope()
            )
    assert not native_reads[1]


def test_cleanup_error_is_static_and_discards_credential_references(
    tmp_path, native_reads
):
    _, _, client_id, order, trades = _stage(tmp_path)
    native_reads[0](order, trades)
    client = module.BinanceRecoveryReadClient(
        scope=_scope(),
        symbol="BTCUSDT",
        client_id=client_id,
        api_key="offline-placeholder",
        api_secret="offline-secret",
    )

    def unavailable():
        raise RuntimeError("PRIVATE-SENTINEL")

    client._http.close = unavailable
    with pytest.raises(OpenIntentError) as error:
        client.close()
    assert "PRIVATE-SENTINEL" not in str(error.value)
    assert client._secret == b"" and not client._key
