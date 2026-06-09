#!/usr/bin/env python3
"""Test IC Markets using the EXACT wire format from the official cTrader library."""

from __future__ import annotations

import os
import socket
import ssl
import struct
import sys

CLIENT_ID = os.getenv("ICM_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("ICM_CLIENT_SECRET", "")
ACCOUNT_ID = os.getenv("ICM_ACCOUNT_ID", "")

if not all([CLIENT_ID, CLIENT_SECRET, ACCOUNT_ID]):
    print("ERROR: set ICM_CLIENT_ID, ICM_CLIENT_SECRET, ICM_ACCOUNT_ID")
    sys.exit(1)

# ── Use the official library's protobuf definitions ─────────
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOASymbolsListReq,
    ProtoOASubscribeSpotsReq,
    ProtoOAErrorRes,
    ProtoOAApplicationAuthRes,
    ProtoOAAccountAuthRes,
)


def send_msg(ssock, msg):
    """Send using exact library format: wrap in ProtoMessage, 4-byte BE length."""
    wrapper = ProtoMessage()
    wrapper.payloadType = msg.payloadType
    wrapper.payload = msg.SerializeToString()
    data = wrapper.SerializeToString()

    # Int32StringReceiver: 4-byte big-endian length prefix
    prefix = struct.pack(">I", len(data))
    ssock.sendall(prefix + data)
    return prefix + data


def recv_msg(ssock, timeout=10):
    """Receive using exact library format: 4-byte BE length, then ProtoMessage."""
    ssock.settimeout(timeout)
    try:
        header = ssock.recv(4)
        if len(header) < 4:
            return None
        length = struct.unpack(">I", header)[0]
        if length > 15_000_000:
            print(f"  Oversized message: {length} bytes — skipping")
            return None
        data = b""
        while len(data) < length:
            chunk = ssock.recv(length - len(data))
            if not chunk:
                return None
            data += chunk
        msg = ProtoMessage()
        msg.ParseFromString(data)
        return msg
    except socket.timeout:
        return None
    except Exception as e:
        print(f"  Parse error: {e}")
        return None


def dump_fields(label, pb_msg):
    """Print all fields of a protobuf message for debugging."""
    print(f"  {label}:")
    for f in pb_msg.DESCRIPTOR.fields:
        val = getattr(pb_msg, f.name, None)
        print(f"    {f.name} (field#{f.number}) = {val!r}")


def main():
    host = "demo.ctraderapi.com"
    port = 5035

    print(f"Connecting to {host}:{port} via TLS...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(20)
    sock.connect((host, port))
    ctx = ssl.create_default_context()
    ssock = ctx.wrap_socket(sock, server_hostname=host)
    print("TLS connected. Sending ProtoOAApplicationAuthReq...")

    # ── Step 1: App Authentication ──
    app_auth = ProtoOAApplicationAuthReq()
    app_auth.clientId = CLIENT_ID
    app_auth.clientSecret = CLIENT_SECRET
    print(f"  clientId={CLIENT_ID}")
    raw = send_msg(ssock, app_auth)
    print(f"  Sent {len(raw)} bytes")

    resp = recv_msg(ssock)
    if resp is None:
        print("\nNo response to app auth.")
    elif resp.payloadType == 2142:  # ProtoOAErrorRes
        err = ProtoOAErrorRes()
        err.ParseFromString(resp.payload)
        print(f"\n  APP AUTH ERROR:")
        print(f"    errorCode: {err.errorCode}")
        print(f"    description: {err.description}")
        print(f"    maintenanceEndTimestamp: {err.maintenanceEndTimestamp}")

        # If the error is about invalid credentials, try account auth directly
        # Some cTrader setups don't require app auth first
        if "already authenticated" in str(err.description).lower():
            print("\n  Already authenticated — trying account auth on same connection...")
            # Don't close, proceed to account auth on same connection
        else:
            print("\n  App auth rejected. Trying account auth directly on fresh connection...")
            ssock.close()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(20)
            sock.connect((host, port))
            ssock = ctx.wrap_socket(sock, server_hostname=host)
    elif resp.payloadType == 2101:  # ProtoOAApplicationAuthRes
        auth_res = ProtoOAApplicationAuthRes()
        auth_res.ParseFromString(resp.payload)
        dump_fields("App Auth Response", auth_res)
    else:
        print(f"  Unexpected response type: {resp.payloadType}")
        # Try account auth anyway on same connection
        print("  Proceeding to account auth...")

    # ── Step 2: Account Auth ──
    print("\nSending ProtoOAAccountAuthReq...")
    acct_auth = ProtoOAAccountAuthReq()
    acct_auth.accessToken = CLIENT_ID
    acct_auth.ctidTraderAccountId = int(ACCOUNT_ID)  # correct field name
    print(f"  accessToken={CLIENT_ID}")
    print(f"  ctidTraderAccountId={ACCOUNT_ID}")
    raw = send_msg(ssock, acct_auth)
    print(f"  Sent {len(raw)} bytes")

    resp = recv_msg(ssock)
    if resp is None:
        print("\nNo response to account auth.")
        ssock.close()
        print("Done.")
        return

    print(f"  Response type: {resp.payloadType}")

    if resp.payloadType == 2142:  # ProtoOAErrorRes
        err = ProtoOAErrorRes()
        err.ParseFromString(resp.payload)
        print(f"\n  ACCOUNT AUTH ERROR:")
        print(f"    errorCode: {err.errorCode}")
        print(f"    description: {err.description}")
        ssock.close()
        print("\nDone.")
        return

    if resp.payloadType == 2103:  # ProtoOAAccountAuthRes
        acct_res = ProtoOAAccountAuthRes()
        acct_res.ParseFromString(resp.payload)
        dump_fields("Account Auth Response", acct_res)
        print("\n  *** ACCOUNT AUTH SUCCESS! ***")

        # ── Step 3: Discover XAUUSD ──
        print("\nFetching symbol list...")
        sym_req = ProtoOASymbolsListReq()
        send_msg(ssock, sym_req)
        resp = recv_msg(ssock, timeout=15)
        if resp and resp.payloadType == 2113:  # ProtoOASymbolsListRes
            # Parse nested symbol messages
            from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASymbolsListRes
            sym_res = ProtoOASymbolsListRes()
            sym_res.ParseFromString(resp.payload)
            print(f"  Symbol list fields: {[f.name for f in sym_res.DESCRIPTOR.fields]}")
            # Try to find XAUUSD
            for f in sym_res.DESCRIPTOR.fields:
                val = getattr(sym_res, f.name, None)
                if hasattr(val, '__iter__') and not isinstance(val, str):
                    for item in val:
                        if hasattr(item, 'DESCRIPTOR'):
                            item_fields = {f2.name: getattr(item, f2.name) for f2 in item.DESCRIPTOR.fields}
                            name = item_fields.get('symbolName', item_fields.get('name', ''))
                            sym_id = item_fields.get('symbolId', item_fields.get('id', 0))
                            if 'XAU' in str(name).upper() or 'GOLD' in str(name).upper():
                                print(f"  >>> FOUND: {name} id={sym_id}")
        else:
            print(f"  Symbol list response: type={resp.payloadType if resp else 'None'}")

        # ── Step 4: Subscribe to spots ──
        # We need the symbol ID first — try with known XAUUSD ID or skip
        print("\nSubscribing to XAUUSD spot prices...")
        sub = ProtoOASubscribeSpotsReq()
        # Try common symbol IDs for XAUUSD: 1, 57, 68, 71
        # We'll add what we found above or try a few
        sub.symbolId.append(1)
        send_msg(ssock, sub)

        # Read a few spot events
        print("Listening for spot events (5 seconds)...")
        for i in range(50):
            resp = recv_msg(ssock, timeout=1)
            if resp:
                if resp.payloadType == 2115:  # ProtoOASpotEvent
                    from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASpotEvent
                    spot = ProtoOASpotEvent()
                    spot.ParseFromString(resp.payload)
                    fields = {f.name: getattr(spot, f.name) for f in spot.DESCRIPTOR.fields}
                    if fields.get('bid', 0) > 0:
                        print(f"  SPOT: bid={fields.get('bid')} ask={fields.get('ask')} "
                              f"symbolId={fields.get('symbolId')}")
                elif resp.payloadType in (1,):  # heartbeat
                    pass
                else:
                    print(f"  Msg type={resp.payloadType}")
    else:
        print(f"  Unexpected response. Raw payload[:50]: {resp.payload[:50].hex() if resp.payload else 'empty'}")

    ssock.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
