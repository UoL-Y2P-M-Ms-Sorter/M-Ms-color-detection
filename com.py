import serial


def send(port, mode, color):
    data = bytearray()

    data.append(42)

    data.append(mode)
    data.append(color)

    for i in range(4):
        data.append(0)

    data.append(66)

    try:
        port.write(bytes(data))
        return 1
    except Exception:
        return 0
