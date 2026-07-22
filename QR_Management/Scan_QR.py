import cv2
from pyzbar.pyzbar import decode

def scan():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success or frame is None:   # FIX: skip bad/empty frames instead of decoding them
            continue

        for qr in decode(frame):
            data = qr.data.decode("utf-8")
            camera.release()
            cv2.destroyAllWindows()
            return data

        cv2.imshow("Scanned QR", frame)

        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


asset_id = scan()
print(asset_id)
