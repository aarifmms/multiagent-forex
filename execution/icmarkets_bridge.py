from __future__ import annotations

"""
IC Markets / cTrader Bridge — Raw TLS + Protobuf adapter for macOS/Linux.

Connects to IC Markets via the cTrader Open API using raw TLS sockets
(not WebSocket), ProtoMessage wrapping, and 4-byte big-endian length prefix.

Uses the official ctrader-open-api compiled protobuf definitions for
correct wire format. Falls back to manual protobuf encoding for message
types not covered by the official library.

Setup:
  1. Open a demo account at icmarkets.com
  2. Create a cTrader API app at https://openapi.ctrader.com/apps
  3. Set env vars: ICM_CLIENT_ID, ICM_CLIENT_SECRET, ICM_ACCOUNT_ID
  4. Run with: python main.py --live --broker icmarkets
"""

import asyncio
import logging
import os
import ssl
import struct
import sys
import time

from core.data_store import DataStore, Tick, Candle


# ── cTrader Message Type IDs ─────────────────────────────────────

class ProtoMsg:
    """cTrader Open API message type IDs (payloadType)."""
    # Auth
    OA_APP_AUTH_REQ = 2100
    OA_APP_AUTH_RES = 2101
    OA_ACCOUNT_AUTH_REQ = 2102
    OA_ACCOUNT_AUTH_RES = 2103

    # Account
    OA_ACCOUNT_LIST_REQ = 2106
    OA_ACCOUNT_LIST_RES = 2107
    OA_ACCOUNT_EVENT = 2108
    OA_EXECUTION_EVENT = 2110

    # Symbols & Prices
    OA_SYMBOLS_LIST_REQ = 2112
    OA_SYMBOLS_LIST_RES = 2113
    OA_SUBSCRIBE_SPOTS_REQ = 2114
    OA_SPOT_EVENT = 2115
    OA_SUBSCRIBE_LIVE_TRENDBAR_REQ = 2116
    OA_LIVE_TRENDBAR_EVENT = 2117

    # Trading
    OA_NEW_ORDER_REQ = 2120
    OA_NEW_ORDER_RES = 2121
    OA_CANCEL_ORDER_REQ = 2122
    OA_AMEND_POSITION_SLTP_REQ = 2126
    OA_AMEND_POSITION_SLTP_RES = 2127
    OA_CLOSE_POSITION_REQ = 2128
    OA_POSITION_EVENT = 2129

    # Heartbeat
    HEARTBEAT_EVENT = 1


# ── Manual Protobuf Helpers (for message types not in official lib) ─

def _varint(n: int) -> bytes:
    """Encode integer as protobuf varint."""
    buf = []
    while n > 127:
        buf.append((n & 0x7F) | 0x80)
        n >>= 7
    buf.append(n)
    return bytes(buf)


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode varint. Returns (value, new_position)."""
    result = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            return result, pos
        shift += 7
    return 0, pos


def _encode_field(field_num: int, wire_type: int, value: bytes) -> bytes:
    """Encode a protobuf field: tag = (field_num << 3) | wire_type."""
    tag = (field_num << 3) | wire_type
    return _varint(tag) + value


def _encode_string(field_num: int, s: str) -> bytes:
    """Wire type 2 (length-delimited)."""
    data = s.encode("utf-8")
    return _encode_field(field_num, 2, _varint(len(data)) + data)


def _encode_int64(field_num: int, n: int) -> bytes:
    """Wire type 0 (varint)."""
    return _encode_field(field_num, 0, _varint(n))


def _encode_double(field_num: int, d: float) -> bytes:
    """Wire type 1 (64-bit)."""
    return _encode_field(field_num, 1, struct.pack("<d", d))


def _decode_protobuf(data: bytes) -> dict:
    """Decode a protobuf message into a flat dict {field_num: [values...]}."""
    result: dict[int, list] = {}
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        wire_type = tag & 0x07
        field_num = tag >> 3
        if wire_type == 0:  # varint
            val, pos = _decode_varint(data, pos)
        elif wire_type == 2:  # length-delimited
            length, pos = _decode_varint(data, pos)
            val = data[pos:pos + length]
            pos += length
            try:
                val = val.decode("utf-8")
            except UnicodeDecodeError:
                pass
        elif wire_type == 1:  # 64-bit
            val = struct.unpack("<d", data[pos:pos + 8])[0]
            pos += 8
        elif wire_type == 5:  # 32-bit
            val = struct.unpack("<f", data[pos:pos + 4])[0]
            pos += 4
        else:
            break
        result.setdefault(field_num, []).append(val)
    return result


# ── Manual encoding for messages not in official lib ─────────────

def _build_new_order(symbol_id: int, volume_cents: int, trade_side: int,
                     sl: float = 0, tp: float = 0) -> bytes:
    """Build ProtoOANewOrderReq manually.
    trade_side: 0=BUY, 1=SELL. volume in cents (1000 = 0.01 lot for forex)."""
    body = (
        _encode_int64(1, symbol_id) +
        _encode_int64(2, trade_side) +
        _encode_int64(3, volume_cents)
    )
    if sl:
        body += _encode_double(6, sl)
    if tp:
        body += _encode_double(7, tp)
    return body


def _build_close_position(position_id: int, volume_cents: int) -> bytes:
    """Build ProtoOAClosePositionReq manually."""
    return _encode_int64(1, position_id) + _encode_int64(2, volume_cents)


# ── IC Markets Bridge ────────────────────────────────────────────

class ICMarketsBridge:
    """
    Connects to IC Markets via cTrader Open API over raw TLS + protobuf.

    Protocol (discovered from official ctrader-open-api Twisted client):
      - Raw TLS socket to demo.ctraderapi.com:5035 (or live.ctraderapi.com:5035)
      - Each message: 4-byte big-endian length prefix + ProtoMessage body
      - ProtoMessage { payloadType: uint32, payload: bytes, clientMsgId?: string }
      - Inner messages use standard protobuf encoding

    Provides: live price streaming, account sync, order execution.
    All over a single persistent TLS connection.
    """

    DEMO_HOST = "demo.ctraderapi.com"
    DEMO_PORT = 5035
    LIVE_HOST = "live.ctraderapi.com"
    LIVE_PORT = 5035

    XAUUSD_SYMBOL_ID: int | None = None

    def __init__(self, data_store: DataStore, config: dict):
        self.store = data_store
        self.config = config
        self.logger = logging.getLogger("icmarkets")
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._running = False
        self._account_id: int | None = None
        self._symbol_cache: dict[int, str] = {}
        self._ProtoMessage: type | None = None
        self._pending_orders: dict[str, asyncio.Event] = {}

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        client_id = os.getenv("ICM_CLIENT_ID", "")
        client_secret = os.getenv("ICM_CLIENT_SECRET", "")
        account_id = os.getenv("ICM_ACCOUNT_ID", "")

        if not all([client_id, client_secret, account_id]):
            self.logger.warning(
                "IC Markets credentials not set. Set ICM_CLIENT_ID, ICM_CLIENT_SECRET, "
                "ICM_ACCOUNT_ID env vars. Falling back to Yahoo Finance data."
            )
            return

        self._account_id = int(account_id)
        await self._connect(client_id, client_secret)

    async def stop(self) -> None:
        self._running = False
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None

    # ── Connection & Auth ─────────────────────────────────────

    async def _connect(self, client_id: str, client_secret: str) -> None:
        """TLS connect → app auth → account auth → discover symbols → subscribe."""
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAApplicationAuthReq,
                ProtoOAApplicationAuthRes,
                ProtoOAAccountAuthReq,
                ProtoOASymbolsListReq,
                ProtoOASubscribeSpotsReq,
            )
        except ImportError:
            self.logger.error(
                "ctrader-open-api package required for IC Markets. "
                "Install: pip install ctrader-open-api"
            )
            return

        is_live = self.config.get("system", {}).get("mode") == "live"
        host = self.LIVE_HOST if is_live else self.DEMO_HOST
        port = self.LIVE_PORT if is_live else self.DEMO_PORT

        self.logger.info("Connecting to IC Markets cTrader (%s)...",
                         "LIVE" if is_live else "DEMO")

        try:
            ssl_ctx = ssl.create_default_context()
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, ssl=ssl_ctx, server_hostname=host),
                timeout=15,
            )
        except Exception as e:
            self.logger.error("TLS connection failed: %s", e)
            return

        self.logger.info("TLS connected. Authenticating...")

        # ── Step 1: App Authentication ──
        app_auth = ProtoOAApplicationAuthReq()
        app_auth.clientId = client_id
        app_auth.clientSecret = client_secret
        await self._send_msg(app_auth)

        resp = await self._recv_msg(timeout=10)
        if resp is None:
            self.logger.error("No response to app auth — server closed connection")
            return

        if resp.payloadType == ProtoMsg.OA_APP_AUTH_RES:
            self.logger.info("App authenticated successfully")
        else:
            self.logger.error("App auth unexpected response type: %s", resp.payloadType)
            return

        # ── Step 2: Account Authentication ──
        # cTrader demo: client_id serves as access token for account auth
        acct_auth = ProtoOAAccountAuthReq()
        acct_auth.accessToken = client_id
        acct_auth.ctidTraderAccountId = self._account_id
        await self._send_msg(acct_auth)

        resp = await self._recv_msg(timeout=10)
        if resp is None:
            self.logger.error("No response to account auth")
            return

        if resp.payloadType == ProtoMsg.OA_ACCOUNT_AUTH_RES:
            self.logger.info("Account authenticated: ID=%s", self._account_id)
        else:
            self.logger.error("Account auth failed: response type=%s", resp.payloadType)
            return

        # ── Step 3: Discover XAUUSD symbol ──
        symbols_req = ProtoOASymbolsListReq()
        await self._send_msg(symbols_req)
        resp = await self._recv_msg(timeout=10)
        if resp and resp.payloadType == ProtoMsg.OA_SYMBOLS_LIST_RES:
            self._parse_symbol_list(resp.payload)
        else:
            self.logger.error("Symbol list request failed")
            return

        if not self.XAUUSD_SYMBOL_ID:
            self.logger.error("XAUUSD not found in available symbols")
            return

        # ── Step 4: Subscribe to spot prices ──
        sub = ProtoOASubscribeSpotsReq()
        sub.symbolId.append(self.XAUUSD_SYMBOL_ID)
        await self._send_msg(sub)
        self.logger.info("Subscribed to XAUUSD spot prices (symbol_id=%d)", self.XAUUSD_SYMBOL_ID)

        # ── Step 5: Start streaming ──
        asyncio.create_task(self._read_loop())
        asyncio.create_task(self._account_sync())
        self.logger.info("IC Markets bridge LIVE — streaming XAUUSD prices")

    def _parse_symbol_list(self, payload: bytes) -> None:
        """Extract symbols from ProtoOASymbolsListRes, find XAUUSD."""
        decoded = _decode_protobuf(payload)
        for item in decoded.get(1, []):
            if isinstance(item, bytes):
                inner = _decode_protobuf(item)
                sym_id = inner.get(1, [None])[0]
                name = inner.get(2, [None])[0]
                if isinstance(name, str) and isinstance(sym_id, int):
                    self._symbol_cache[sym_id] = name
                    if name in ("XAUUSD", "XAUUSD.", "GOLD"):
                        self.XAUUSD_SYMBOL_ID = sym_id
                        self.logger.info("Found %s: symbol_id=%d", name, sym_id)

    # ── Wire Protocol ─────────────────────────────────────────

    async def _send_msg(self, msg) -> None:
        """Wrap in ProtoMessage, prefix 4-byte BE length, send."""
        ProtoMessage = self._get_proto_wrapper()

        wrapper = ProtoMessage()
        wrapper.payloadType = msg.payloadType
        wrapper.payload = msg.SerializeToString()
        data = wrapper.SerializeToString()

        prefix = struct.pack(">I", len(data))
        self._writer.write(prefix + data)
        await self._writer.drain()

    async def _send_raw(self, payload_type: int, body: bytes) -> None:
        """Send a manually-encoded message (no compiled protobuf class)."""
        ProtoMessage = self._get_proto_wrapper()

        wrapper = ProtoMessage()
        wrapper.payloadType = payload_type
        wrapper.payload = body
        data = wrapper.SerializeToString()

        prefix = struct.pack(">I", len(data))
        self._writer.write(prefix + data)
        await self._writer.drain()

    async def _recv_msg(self, timeout: float = 30) -> object | None:
        """Read 4-byte BE length prefix, then ProtoMessage body."""
        try:
            header = await asyncio.wait_for(
                self._reader.readexactly(4), timeout=timeout
            )
            length = struct.unpack(">I", header)[0]

            if length > 15_000_000:
                self.logger.warning("Oversized message: %d bytes — skipping", length)
                return None

            data = await asyncio.wait_for(
                self._reader.readexactly(length), timeout=timeout
            )
            ProtoMessage = self._get_proto_wrapper()
            msg = ProtoMessage()
            msg.ParseFromString(data)
            return msg
        except asyncio.TimeoutError:
            return None
        except asyncio.IncompleteReadError:
            self.logger.warning("Connection closed by server")
            return None
        except Exception as e:
            self.logger.error("Receive error: %s", e)
            return None

    def _get_proto_wrapper(self):
        """Lazy-load ProtoMessage from the official library."""
        if self._ProtoMessage is None:
            from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
            self._ProtoMessage = ProtoMessage
        return self._ProtoMessage

    # ── Read Loop ──────────────────────────────────────────────

    async def _read_loop(self) -> None:
        """Continuously read and route messages from cTrader."""
        while self._running and self._reader:
            msg = await self._recv_msg(timeout=30)
            if msg is None:
                continue
            try:
                self._handle_message(msg)
            except Exception as e:
                self.logger.error("Message handling error: %s", e)

    def _handle_message(self, msg) -> None:
        """Route incoming message by payloadType."""
        pt = msg.payloadType

        if pt == ProtoMsg.OA_SPOT_EVENT:
            self._handle_spot_event(msg.payload)
        elif pt == ProtoMsg.OA_EXECUTION_EVENT:
            self._handle_execution_event(msg.payload)
        elif pt == ProtoMsg.OA_ACCOUNT_EVENT:
            self._handle_account_event(msg.payload)
        elif pt == ProtoMsg.OA_POSITION_EVENT:
            self._handle_position_event(msg.payload)
        elif pt in (ProtoMsg.HEARTBEAT_EVENT,):
            pass  # ignore heartbeats, keep connection alive

    # ── Inbound Event Handlers ──────────────────────────────

    def _handle_spot_event(self, payload: bytes) -> None:
        """Parse ProtoOASpotEvent and push tick to DataStore."""
        decoded = _decode_protobuf(payload)
        symbol_id = _first_int(decoded, 1)
        bid = _first_float(decoded, 4)
        ask = _first_float(decoded, 5)

        if bid and ask and bid > 0:
            asyncio.create_task(self._emit_price(float(bid), float(ask)))

    def _handle_execution_event(self, payload: bytes) -> None:
        """Parse ProtoOAExecutionEvent for order fill confirmations."""
        decoded = _decode_protobuf(payload)
        order_id = _first_int(decoded, 1)
        order_type = _first_int(decoded, 2)
        order_status = _first_int(decoded, 3)
        self.logger.info("Execution event: order=%s type=%s status=%s",
                         order_id, order_type, order_status)

    def _handle_account_event(self, payload: bytes) -> None:
        """Parse ProtoOAAccountEvent for balance/equity updates."""
        decoded = _decode_protobuf(payload)
        balance = _first_float(decoded, 3)
        equity = _first_float(decoded, 13)
        margin = _first_float(decoded, 14)
        if balance:
            self.store.balance = float(balance)
        if equity:
            self.store.equity = float(equity)
        if margin:
            self.store.margin = float(margin)

    def _handle_position_event(self, payload: bytes) -> None:
        """Parse ProtoOAPositionEvent for position updates."""
        decoded = _decode_protobuf(payload)
        pos_id = _first_int(decoded, 1)
        volume = _first_float(decoded, 4)
        pnl = _first_float(decoded, 8)
        self.logger.info("Position event: id=%s vol=%s pnl=%s", pos_id, volume, pnl)

    async def _emit_price(self, bid: float, ask: float) -> None:
        """Push a tick to DataStore and aggregate candles."""
        tick = Tick(time=time.time(), bid=round(bid, 2), ask=round(ask, 2))
        await self.store.update_tick(tick)

        mid = round((bid + ask) / 2, 2)
        for tf, secs in [("M1", 60), ("M5", 300), ("M15", 900),
                         ("H1", 3600), ("H4", 14400), ("D1", 86400)]:
            candle_time = int(tick.time // secs) * secs
            candles = await self.store.get_candles(tf, count=1)
            if candles and candles[-1].time == candle_time:
                c = candles[-1]
                c.high = max(c.high, mid)
                c.low = min(c.low, mid)
                c.close = mid
                c.tick_volume += 1
            else:
                await self.store.update_candle(tf, Candle(
                    time=candle_time, open=mid, high=mid, low=mid,
                    close=mid, tick_volume=1,
                ))

    async def _account_sync(self) -> None:
        """Periodically log account state."""
        while self._running:
            self.logger.info(
                "ICM Account | Balance=%.2f Equity=%.2f Positions=%d",
                self.store.balance, self.store.equity, len(self.store.open_positions),
            )
            await asyncio.sleep(30)

    # ── Order Execution ──────────────────────────────────────

    async def place_order(self, direction: str, volume_lots: float,
                          sl: float = 0, tp: float = 0) -> int | None:
        """
        Place a market order on IC Markets.

        direction: 'BUY' or 'SELL'
        volume_lots: e.g. 0.01
        Returns order_id or None on failure.
        """
        if not self._writer or not self.XAUUSD_SYMBOL_ID:
            self.logger.error("Not connected — cannot place order")
            return None

        trade_side = 0 if direction == "BUY" else 1
        volume_cents = int(volume_lots * 100000)

        body = _build_new_order(self.XAUUSD_SYMBOL_ID, volume_cents, trade_side, sl, tp)
        await self._send_raw(ProtoMsg.OA_NEW_ORDER_REQ, body)

        resp = await self._recv_msg(timeout=5)
        if resp and resp.payloadType == ProtoMsg.OA_NEW_ORDER_RES:
            decoded = _decode_protobuf(resp.payload)
            order_id = _first_int(decoded, 1)
            if order_id:
                self.logger.info("Order placed: %s XAUUSD lot=%.2f orderId=%s",
                                 direction, volume_lots, order_id)
                return int(order_id)

        self.logger.error("Order failed: %s", resp)
        return None

    async def close_position(self, position_id: int, volume_lots: float) -> bool:
        """Close a position by position ID."""
        if not self._writer:
            return False

        volume_cents = int(volume_lots * 100000)
        body = _build_close_position(position_id, volume_cents)
        await self._send_raw(ProtoMsg.OA_CLOSE_POSITION_REQ, body)

        resp = await self._recv_msg(timeout=5)
        return resp is not None


# ── Helpers ──────────────────────────────────────────────────────

def _first_int(decoded: dict, field_num: int) -> int | None:
    vals = decoded.get(field_num, [])
    if vals and isinstance(vals[0], int):
        return vals[0]
    return None


def _first_float(decoded: dict, field_num: int) -> float | None:
    vals = decoded.get(field_num, [])
    if vals and isinstance(vals[0], (int, float)):
        return float(vals[0])
    return None
