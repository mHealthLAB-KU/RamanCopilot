# coding=utf-8
"""Tango stage controller and an interactive test terminal."""

import ctypes
import sys
import re
import time
from ctypes import *
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from backend.common.config.hw import (  # noqa: E402
    STAGE_MAX_X, STAGE_MAX_Y, STAGE_CENTER_X, STAGE_CENTER_Y,
    STAGE_MIN_Z, STAGE_MAX_Z,
)

MAX_SPEED_XY = 5.0  # mm/s, X and Y
MAX_SPEED_Z  = 0.1  # mm/s, Z is slower to protect the objective

MIN_SPEED_MM_S = 0.001  # mm/s


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class TangoController:
    """Wrapper for the Tango stage controller DLL."""

    def __init__(self, dll_path: str = r"C:/Users/user/Desktop/gemma4_raman/backend/repo/hw_access/SDKs/Tango_DLL.dll"):
        self.dll_path = dll_path
        self.dll = None
        self.LSID = c_int()
        self.connected = False
        self.dead = False
        self.dead_reason = None

    def mark_dead(self, reason: str) -> None:
        """Mark this session as unrecoverable and block further DLL calls."""
        self.dead = True
        self.dead_reason = reason
        self.connected = False
        print(f"[STAGE] session marked invalid - {reason}")

    def load_dll(self) -> bool:
        """Load the DLL."""
        try:
            self.dll = cdll.LoadLibrary(self.dll_path)
            print(f"[OK] DLL loaded: {self.dll_path}")
            return True
        except Exception as e:
            print(f"[ERROR] DLL load failed: {e}")
            print("  - Check that Tango_DLL.dll is in the current directory")
            print("  - Check that the Python bitness (32/64) matches the DLL")
            return False

    def create_session(self) -> bool:
        """Create a session ID."""
        if not self.dll:
            print("[ERROR] DLL not loaded")
            return False

        error = self.dll.LSX_CreateLSID(byref(self.LSID))
        if error > 0:
            print(f"[ERROR] session creation failed, error code: {error}")
            return False

        print(f"[OK] session created, LSID: {self.LSID.value}")
        return True

    def connect(self, interface: int = -1, port: str = "", baudrate: int = 57600) -> bool:
        """Connect to the Tango controller."""
        if not self.dll:
            print("[ERROR] DLL not loaded")
            return False

        port_bytes = port.encode("utf-8") if port else b""
        port_param = c_char_p(port_bytes) if port_bytes else b""

        print(f"[INFO] connecting... (interface={interface}, port='{port}')")
        error = self.dll.LSX_ConnectSimple(self.LSID, interface, port_param, baudrate, 0)

        if error > 0:
            print(f"[ERROR] connection failed, error code: {error}")
            self._print_connection_error_help(error)
            return False

        self.connected = True
        print("[OK] Tango controller connected")
        return True

    def _print_connection_error_help(self, error_code: int):
        """Print help for connection errors."""
        print("  Check:")
        print("  1. The Tango controller is powered on")
        print("  2. The USB/PCIe cable is connected")
        print("  3. No other program (e.g. rays-on.exe) is already connected")
        print("  4. The device driver is installed")

    def calibrate(self, axes: int = 3) -> bool:
        """Home the stage to its physical limits and reset to (0, 0)."""
        if not self.connected:
            print("[ERROR] not connected")
            return False
        error = self.dll.LSX_Calibrate(self.LSID, axes)
        if error > 0:
            print(f"[ERROR] calibration failed, error code: {error}")
            return False
        print(f"[OK] stage calibration (homing) done (axes={axes})")
        return True

    def disconnect(self) -> bool:
        """Disconnect."""
        if not self.connected:
            return True
        try:
            error = self.dll.LSX_Disconnect(self.LSID)
            if error > 0:
                print(f"[ERROR] disconnect failed, error code: {error}")
                return False
        except OSError as e:
            print(f"[WARN] DLL exception during disconnect (ignored): {e}")
            return False

        self.connected = False
        print("[OK] disconnected")
        return True

    def free_session(self) -> bool:
        """Free the session ID; return True on success (safe to call repeatedly)."""
        if not self.dll or self.LSID.value <= 0:
            return True
        try:
            self.dll.LSX_FreeLSID(self.LSID)
        except OSError as e:
            print(f"[WARN] DLL exception while freeing the session: {e}")
            return False
        self.LSID.value = 0
        self.connected = False
        print("[OK] session freed")
        return True

    def get_position(self) -> tuple:
        """Return the current position."""
        if self.dead:
            return None
        if not self.connected:
            print("[ERROR] not connected")
            return None

        dx, dy, dz, da = c_double(), c_double(), c_double(), c_double()
        error = self.dll.LSX_GetPos(self.LSID, byref(dx), byref(dy), byref(dz), byref(da))

        if error > 0:
            print(f"[ERROR] position query failed, error code: {error}")
            return None

        return (dx.value, dy.value, dz.value, da.value)

    def get_version(self) -> str:
        """Return the DLL version."""
        if not self.dll:
            return None

        resp = create_string_buffer(256)
        error = self.dll.LSX_GetDLLVersionString(self.LSID, resp, 256)

        if error > 0:
            return None
        return resp.value.decode("ascii")

    def get_firmware_version(self) -> str:
        """Return the firmware version."""
        if not self.connected:
            return None

        inp = c_char_p("?version\r".encode("utf-8"))
        resp = create_string_buffer(256)
        error = self.dll.LSX_SendString(self.LSID, inp, resp, 256, True, 5000)

        if error > 0:
            return None
        return resp.value.decode("ascii").strip()

    def move_absolute(self, x: float, y: float, z: float, a: float = 0, wait: bool = True) -> bool:
        """Move to an absolute position, clipped to the travel range."""
        if self.dead:
            print(f"[ERROR] stage session invalid - move cancelled ({self.dead_reason})")
            return False
        if not self.connected:
            print("[ERROR] not connected")
            return False

        cx = _clip(x, 0.0, STAGE_MAX_X)
        cy = _clip(y, 0.0, STAGE_MAX_Y)
        cz = _clip(z, STAGE_MIN_Z, STAGE_MAX_Z)

        if cx != x or cy != y or cz != z:
            print(f"[WARN] out of range -> clipping: "
                  f"X {x:.4f}→{cx:.4f}  Y {y:.4f}→{cy:.4f}  Z {z:.4f}→{cz:.4f}")

        _cur = self.get_position()
        if _cur is not None:
            _v = self.get_velocity()
            if isinstance(_v, dict) and _v.get("ok"):
                _stalled = [n for n, c, t, s in
                            (("X", _cur[0], cx, _v["x_speed_mm_s"]),
                             ("Y", _cur[1], cy, _v["y_speed_mm_s"]),
                             ("Z", _cur[2], cz, _v["z_speed_mm_s"]))
                            if abs(t - c) > 0.0 and abs(s) < MIN_SPEED_MM_S]
                if _stalled:
                    print(f"[ERROR] move refused - axis with zero speed needs to move: "
                          f"{'/'.join(_stalled)}. No command sent "
                          f"(sending it would make the controller unresponsive).")
                    return False

        error = self.dll.LSX_MoveAbs(self.LSID, c_double(cx), c_double(cy), c_double(cz), c_double(a), wait)
        time.sleep(0.1)
        if error > 0:
            print(f"[ERROR] move failed, error code: {error}")
            return False

        return True

    def move_relative(self, dx: float, dy: float, dz: float, da: float = 0, wait: bool = True) -> bool:
        """Move by a relative offset, clipped to the travel range."""
        if not self.connected:
            print("[ERROR] not connected")
            return False

        pos = self.get_position()
        if pos is None:
            print("[ERROR] cannot read the current position; move cancelled")
            return False

        tx = pos[0] + dx
        ty = pos[1] + dy
        tz = pos[2] + dz

        return self.move_absolute(tx, ty, tz, da, wait)

    def get_velocity(self) -> dict:
        """Return the current velocity in mm/s."""
        if self.dead:
            return {"ok": False, "error": f"session invalid - {self.dead_reason}"}
        if not self.connected:
            return {"ok": False, "error": "not connected"}

        vx = c_double()
        vy = c_double()
        vz = c_double()
        va = c_double()

        error = self.dll.LSX_GetVel(self.LSID, byref(vx), byref(vy), byref(vz), byref(va))

        if error == 0:
            return {
                "ok": True, 
                "x_speed_mm_s": vx.value, 
                "y_speed_mm_s": vy.value, 
                "z_speed_mm_s": vz.value,
                "a_speed_mm_s": va.value
            }
        else:
            return {"ok": False, "error": f"velocity query failed, error code: {error}"}

    def set_velocity(self, vx: float, vy: float, vz: float, va: float) -> bool:
        """Set the velocity in mm/s."""
        if self.dead:
            print(f"[ERROR] stage session invalid - velocity change cancelled ({self.dead_reason})")
            return False
        if not self.connected:
            print("[ERROR] not connected")
            return False

        if abs(vx) > MAX_SPEED_XY:
            print(f"[WARN] x speed {vx} mm/s exceeds the maximum {MAX_SPEED_XY} mm/s; clipping.")
            vx = max(-MAX_SPEED_XY, min(MAX_SPEED_XY, vx))
        if abs(vy) > MAX_SPEED_XY:
            print(f"[WARN] y speed {vy} mm/s exceeds the maximum {MAX_SPEED_XY} mm/s; clipping.")
            vy = max(-MAX_SPEED_XY, min(MAX_SPEED_XY, vy))
        if abs(vz) > MAX_SPEED_Z:
            print(f"[WARN] z speed {vz} mm/s exceeds the maximum {MAX_SPEED_Z} mm/s; clipping.")
            vz = max(-MAX_SPEED_Z, min(MAX_SPEED_Z, vz))

        error = ""
        dx=c_double(vx)
        error1 = self.dll.LSX_SetVelSingleAxis(self.LSID, 1, dx)
        if error1 > 0:
            print(f"[ERROR] x velocity setting failed, error code: {error1}")
            error += f"x error code: {error1}"

        dy=c_double(vy)
        error2 = self.dll.LSX_SetVelSingleAxis(self.LSID, 2, dy)
        if error2 > 0:
            print(f"[ERROR] y velocity setting failed, error code: {error2}")
            error += f"/ y error code: {error2} "

        dz=c_double(vz)
        error3 =   self.dll.LSX_SetVelSingleAxis(self.LSID, 3, dz)
        if error3 > 0:
            print(f"[ERROR] z velocity setting failed, error code: {error3}")
            error += f"/ z error code: {error3} "

        if error:
            print(f"[ERROR] velocity setting failed: {error}")
            return False

        return True

    def send_command(self, command: str) -> str:
        """Send a raw command."""
        if not self.connected:
            print("[ERROR] not connected")
            return None

        if not command.endswith('\r'):
            command += '\r'

        inp = c_char_p(command.encode("utf-8"))
        resp = create_string_buffer(256)
        error = self.dll.LSX_SendString(self.LSID, inp, resp, 256, True, 5000)

        if error > 0:
            print(f"[ERROR] command failed, error code: {error}")
            return None

        return resp.value.decode("ascii").strip()


def print_menu():
    """Print the menu."""
    print("\n" + "=" * 50)
    print("Tango stage test menu")
    print("=" * 50)
    print("1. Read current position")
    print("2. Move to absolute position")
    print("3. Move by relative offset")
    print("4. Set velocity")
    print("5. Send raw command")
    print("6. Show version information")
    print("-" * 50)
    print("7. [test] Keep connection and wait (rays-on.exe test)")
    print("-" * 50)
    print("0. Exit")
    print("=" * 50)


def test_rays_on_conflict(tango: TangoController):
    """Test for a conflict with Rays-ON.exe."""
    print("\n" + "=" * 50)
    print("rays-on.exe conflict test mode")
    print("=" * 50)
    print()
    print("Current state: Python holds the Tango stage connection")
    print()
    print("How to test:")
    print("  1. Start rays-on.exe in this state")
    print("  2. Check how rays-on.exe reacts:")
    print("     - only a stage connection error, spectroscopy still works?")
    print("     - the program does not start at all?")
    print("     - both work normally? (rare)")
    print()
    print("Press Enter to end this test and return to the menu.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n[waiting] press Enter after testing rays-on.exe: ")
            if user_input.lower() == 'q':
                break

            pos = tango.get_position()
            if pos:
                print(f"[connection held] current position: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
            else:
                print("[WARN] the connection may have been lost")

            break

        except KeyboardInterrupt:
            print("\nTest finished")
            break


def interactive_mode(tango: TangoController):
    """Interactive control mode."""
    while True:
        print_menu()
        choice = input("Choice: ").strip()

        if choice == '0':
            print("Exiting...")
            break

        elif choice == '1':
            pos = tango.get_position()
            if pos:
                print(f"\nCurrent position:")
                print(f"  X = {pos[0]:.3f}")
                print(f"  Y = {pos[1]:.3f}")
                print(f"  Z = {pos[2]:.3f}")
                print(f"  A = {pos[3]:.3f}")

        elif choice == '2':
            try:
                print("\nEnter absolute position (mm)")
                x = float(input("  X: "))
                y = float(input("  Y: "))
                z = float(input("  Z: "))

                print(f"\nMoving to ({x}, {y}, {z})...")
                if tango.move_absolute(x, y, z):
                    print("[OK] move complete")
            except ValueError:
                print("[ERROR] enter a number")

        elif choice == '3':
            try:
                print("\nEnter relative offset (mm)")
                dx = float(input("  dX: "))
                dy = float(input("  dY: "))
                dz = float(input("  dZ: "))

                print(f"\nMoving by ({dx}, {dy}, {dz})...")
                if tango.move_relative(dx, dy, dz):
                    print("[OK] move complete")
            except ValueError:
                print("[ERROR] enter a number")

        elif choice == '4':
            try:
                print("\nSet velocity (mm/s)")
                vx = float(input("  X speed: "))
                vy = float(input("  Y speed: "))
                vz = float(input("  Z speed: "))

                if tango.set_velocity(vx, vy, vz, 5.0):
                    print(f"[OK] velocity set: ({vx}, {vy}, {vz})")
            except ValueError:
                print("[ERROR] enter a number")

        elif choice == '5':
            print("\nEnter a Tango command (e.g. ?pos, ?vel, !cal)")
            print("Help: ?=query, !=execute, value=set")
            cmd = input("Command: ").strip()
            if cmd:
                result = tango.send_command(cmd)
                if result is not None:
                    print(f"Response: {result}")

        elif choice == '6':
            dll_ver = tango.get_version()
            fw_ver = tango.get_firmware_version()
            print(f"\nDLL version: {dll_ver}")
            print(f"Firmware version: {fw_ver}")

        elif choice == '7':
            test_rays_on_conflict(tango)

        else:
            print("[ERROR] invalid choice")


def main():
    """Entry point."""
    print("=" * 50)
    print("Tango stage controller test")
    print("=" * 50)

    from backend.repo.hw_access.hao.hardware_manager import STAGE_DLL_PATH
    dll_path = STAGE_DLL_PATH

    tango = TangoController(dll_path)

    if not tango.load_dll():
        input("Press Enter to exit...")
        return 1

    if not tango.create_session():
        input("Press Enter to exit...")
        return 1

    print("\nConnection type:")
    print("  1. Auto-detect (USB/PCIe)")
    print("  2. Specify COM port")
    conn_choice = input("Choice [1]: ").strip() or "1"

    if conn_choice == "2":
        port = input("COM port (e.g. COM5): ").strip()
        if not tango.connect(interface=1, port=port):
            input("Press Enter to exit...")
            return 1
    else:
        if not tango.connect():
            input("Press Enter to exit...")
            return 1

    print("\n" + "-" * 50)
    pos = tango.get_position()
    if pos:
        print(f"Current position: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}, A={pos[3]:.3f}")

    try:
        interactive_mode(tango)
    except KeyboardInterrupt:
        print("\n\nCtrl+C detected, exiting...")
    finally:
        tango.disconnect()
        tango.free_session()

    return 0


if __name__ == "__main__":
    sys.exit(main())
