"""TUCam microscope camera streaming and a live viewer."""
import numpy as np
import cv2
import sys
import os
import threading
from ctypes import *

from backend.repo.hw_access.SDKs.TuCam.TUCam import *


class StreamingTUCam:
    """TUCam camera with thread-safe streaming."""
    def __init__(self, exposure_ms=10.0):
        self.Path = './'
        self.TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)
        TUCAM_Api_Init(pointer(self.TUCAMINIT), 5000)
        if self.TUCAMINIT.uiCamCount == 0:
            raise RuntimeError("No camera found!")
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)
        TUCAM_Dev_Open(pointer(self.TUCAMOPEN))
        if self.TUCAMOPEN.hIdxTUCam == 0:
            raise RuntimeError("Open failure!")
        self.exposure_ms = float(exposure_ms)
        self.auto_exposure = False
        self.set_exposure(exposure_ms)
        self.is_streaming = False
        self.m_frame = TUCAM_FRAME()
        self.m_capmode = TUCAM_CAPTURE_MODES
        self._lock = threading.RLock()

    def set_exposure(self, ms: float):
        """Set a manual exposure (ms); this disables auto exposure."""
        TUCAM_Capa_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value, 0)
        TUCAM_Prop_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDPROP.TUIDP_EXPOSURETM.value, c_double(ms), 0)
        self.exposure_ms = float(ms)
        self.auto_exposure = False

    def set_auto_exposure(self, enabled: bool):
        """Turn auto exposure on or off."""
        TUCAM_Capa_SetValue(self.TUCAMOPEN.hIdxTUCam, TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value,
                            1 if enabled else 0)
        self.auto_exposure = bool(enabled)

    def start_stream(self):
        with self._lock:
            if self.is_streaming:
                return
            self.m_frame.pBuffer = 0
            self.m_frame.ucFormatGet = TUFRM_FORMATS.TUFRM_FMT_USUAl.value
            self.m_frame.uiRsdSize = 1
            TUCAM_Buf_Alloc(self.TUCAMOPEN.hIdxTUCam, pointer(self.m_frame))
            TUCAM_Cap_Start(self.TUCAMOPEN.hIdxTUCam, self.m_capmode.TUCCM_SEQUENCE.value)
            self.is_streaming = True

    def get_latest_frame(self):
        with self._lock:
            if not self.is_streaming:
                return None
            try:
                TUCAM_Buf_WaitForFrame(self.TUCAMOPEN.hIdxTUCam, pointer(self.m_frame), 500)
            except Exception:
                return None
            buf = create_string_buffer(self.m_frame.uiImgSize)
            pointer_data = c_void_p(self.m_frame.pBuffer + self.m_frame.usHeader)
            memmove(buf, pointer_data, self.m_frame.uiImgSize)
            dtype = np.uint8 if self.m_frame.ucElemBytes == 1 else np.uint16
            image_np = np.frombuffer(buf, dtype=dtype)
            if self.m_frame.ucChannels == 3:
                image_np = image_np.reshape((self.m_frame.usHeight, self.m_frame.usWidth, 3))
            else:
                image_np = image_np.reshape((self.m_frame.usHeight, self.m_frame.usWidth))
            return image_np

    def stop_stream(self):
        with self._lock:
            if self.is_streaming:
                TUCAM_Buf_AbortWait(self.TUCAMOPEN.hIdxTUCam)
                TUCAM_Cap_Stop(self.TUCAMOPEN.hIdxTUCam)
                TUCAM_Buf_Release(self.TUCAMOPEN.hIdxTUCam)
                self.is_streaming = False

    def close(self):
        self.stop_stream()
        if self.TUCAMOPEN.hIdxTUCam != 0:
            TUCAM_Dev_Close(self.TUCAMOPEN.hIdxTUCam)
        TUCAM_Api_Uninit()


def main():
    """Run the live camera viewer."""
    print("=== Camera Stream Viewer ===")
    print("ESC: quit")
    print("E: increase exposure (+5ms)")
    print("D: decrease exposure (-5ms)")
    print()

    exposure = 10.0
    camera = StreamingTUCam(exposure_ms=exposure)
    camera.start_stream()

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                continue

            disp = frame.copy()
            if disp.dtype == np.uint16:
                disp = (disp / 256).astype(np.uint8)
            if len(disp.shape) == 2:
                disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

            cv2.putText(disp, f"Exposure: {exposure:.1f}ms", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            h, w = disp.shape[:2]
            if w > 1280:
                disp = cv2.resize(disp, (1280, int(1280 * h / w)))

            cv2.imshow("Camera Stream", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('e') or key == ord('E'):
                exposure += 5.0
                camera.set_exposure(exposure)
                print(f"Exposure: {exposure:.1f}ms")
            elif key == ord('d') or key == ord('D'):
                exposure = max(1.0, exposure - 5.0)
                camera.set_exposure(exposure)
                print(f"Exposure: {exposure:.1f}ms")

    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Stream closed.")


if __name__ == "__main__":
    main()
