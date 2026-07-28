from WindPy import w

start_result = w.start()

print("errorcode:", start_result.ErrorCode)
print("connection:", w.isconnected())

error_code, cb = w.wsd(
    "123257.SZ",
    "close,volume,amt",
    "-10TD",
    "",
    "Period=D;Days=Trading",
    usedf=True
)

if error_code != 0:
    raise RuntimeError(f"取数失败：{error_code},{cb}")

error_code_ss, snapshot = w.wss(
    "123257.SZ,123266.SZ",
    "sec_name,close",
    "tradeDate=20260724",
    usedf=True
)

if error_code_ss != 0:
    raise RuntimeError(
        f"截面取数失败：{error_code_ss},{snapshot}"
    )

print(snapshot)

print(cb)