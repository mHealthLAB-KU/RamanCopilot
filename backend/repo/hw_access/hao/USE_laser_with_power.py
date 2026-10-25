"""Serial control of the Raman laser and ND filter with an interactive terminal."""
import serial
import time
import sys
import math

from backend.common.config.hw import BEAM_SPLITTER_POS, LASER_FILTER_POS, LASER_PORT, MOTORS, ND_FILTER_CAL


def _pos(pulse: int) -> str:
    """Format an absolute controller position as sign + 7 decimal digits."""
    return f"-{int(pulse):07d}"


AX_ND, AX_LASER_FILTER, AX_GRATING, AX_BEAM_SPLITTER = (
    MOTORS[n].axis for n in ("ND_FILTER", "LASER_FILTER", "GRATING", "BEAM_SPLITTER"))
POS_ND_BLOCK   = _pos(ND_FILTER_CAL[-1][1])
POS_BS_IMAGE   = _pos(BEAM_SPLITTER_POS["image"])
POS_BS_SIGNAL  = _pos(BEAM_SPLITTER_POS["signal"])
POS_BS_STANDBY = _pos(MOTORS["BEAM_SPLITTER"].offset)
POS_LASER_FILTER = _pos(LASER_FILTER_POS)
POS_GRATING_STANDBY = "-0570000"


class LaserController:
    """Laser and optics motor controller for the Fastech EziMOTION Plus-R ASCII protocol."""

    AXES = {m.axis: {"name": name, "homing_vel": m.homing_vel, "moving_vel": m.moving_vel}
            for name, m in MOTORS.items()
            if name in ("ND_FILTER", "LASER_FILTER", "GRATING", "BEAM_SPLITTER")}
    AXIS_ORDER = [AX_ND, AX_LASER_FILTER, AX_GRATING, AX_BEAM_SPLITTER]

    FFLAG_ERRORALL        = 0x00000001
    FFLAG_HWPOSILMT       = 0x00000002
    FFLAG_HWNEGALMT       = 0x00000004
    FFLAG_ERRPOSOVERFLOW   = 0x00000080
    FFLAG_ERROVERCURRENT   = 0x00000100
    FFLAG_EMGSTOP         = 0x00010000
    FFLAG_SLOWSTOP        = 0x00020000
    FFLAG_ORIGINRETURNING = 0x00040000
    FFLAG_INPOSITION      = 0x00080000
    FFLAG_SERVOON         = 0x00100000
    FFLAG_ALARMRESET      = 0x00200000
    FFLAG_ORIGINRETOK     = 0x02000000
    FFLAG_MOTIONING       = 0x08000000
    FFLAG_MOTIONPAUSE     = 0x10000000
    FFLAG_MOTIONACCEL     = 0x20000000
    FFLAG_MOTIONDECEL     = 0x40000000
    FFLAG_MOTIONCONST     = 0x80000000

    PARAM_AXISMAXSPEED = 1
    PARAM_ORGSPEED     = 17

    GMMS_HOMING    = 0x001000
    GMMS_ORIGIN_OK = 0x010000
    GMMS_MOVING    = 0x002000
    GMMS_BUSY      = 0x003000

    ND_CAL = ND_FILTER_CAL
    ND_MIN_PCT, ND_MAX_PCT = ND_CAL[-1][0], ND_CAL[0][0]

    def __init__(self, port=LASER_PORT, baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self._power_set = False
        self.is_on = False
        self.power_pct = None
        self.bs_signal = False
        self._connect()
        time.sleep(0.5)

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"✅ [{self.port}] laser controller connected")
        except Exception as e:
            print(f"❌ [{self.port}] connection failed: {e}")
            print("The port is in use or not connected.")

    def _make_packet(self, target_id, cmd, arg=""):
        """Assemble a command packet with its checksum."""
        body = f"{target_id}{cmd}{arg}"
        ascii_sum = sum(ord(c) for c in body)
        checksum = f"{ascii_sum % 256:02x}"
        return f"@{body}{checksum}$".encode('utf-8')

    def _flush_rx(self):
        """Clear the receive buffer."""
        if self.ser and self.ser.in_waiting:
            self.ser.read(self.ser.in_waiting)

    def _read_frames(self, timeout=2.0):
        """Read @...$ frames until a short silence and return them as a list."""
        start = time.time()
        buffer = ""
        frames = []

        while (time.time() - start) < timeout:
            if self.ser.in_waiting:
                chunk = self.ser.read(self.ser.in_waiting).decode(errors='ignore')
                buffer += chunk

                while '@' in buffer and '$' in buffer:
                    at = buffer.index('@')
                    dollar = buffer.index('$', at)
                    frames.append(buffer[at:dollar + 1])
                    buffer = buffer[dollar + 1:]

                if frames:
                    time.sleep(0.03)
                    if not self.ser.in_waiting:
                        return frames
            else:
                if frames:
                    return frames
                time.sleep(0.01)

        return frames

    def _send_command(self, target_id, cmd, arg="", timeout=5.0, retries=2):
        """Send a packet and check the ACK, resetting alarms and retrying on error."""
        if not (self.ser and self.ser.is_open):
            print("⚠️ port not connected")
            return False, None

        expected_key = f"{target_id}{cmd.lower()}"

        for attempt in range(retries + 1):
            self._flush_rx()
            packet = self._make_packet(target_id, cmd, arg)
            print(f"   [TX] {packet.decode()}")
            self.ser.write(packet)

            frames = self._read_frames(timeout=timeout)
            if not frames:
                print(f"   ⚠️ no response [{attempt + 1}/{retries + 1}]")
                if attempt < retries:
                    time.sleep(0.3)
                continue

            for f in frames:
                print(f"   [RX] {f}")

            got_ercd = False
            for f in frames:
                inner = f[1:-1]

                if "ercd" in inner:
                    ercd_idx = inner.index("ercd") + 4
                    ercd_raw = inner[ercd_idx:]
                    ercd_data = ercd_raw[:8] if len(ercd_raw) >= 8 else ercd_raw[:-2]
                    print(f"   ❌ ercd error: 0x{ercd_data} [{attempt + 1}/{retries + 1}]")
                    got_ercd = True
                    break

                inner_lower = inner.lower()
                if expected_key in inner_lower:
                    data_start = inner_lower.index(expected_key) + len(expected_key)
                    data_raw = inner[data_start:]
                    return True, data_raw

            if got_ercd and attempt < retries:
                self._alarm_reset_quick(target_id)
                time.sleep(0.2)
                continue

            if not got_ercd and attempt < retries:
                time.sleep(0.3)

        print(f"   ⚠️ final failure: {cmd}")
        return False, None

    def _alarm_reset_quick(self, target_id):
        """Send an alarm reset (ERCL) to the broadcast address and the given axis."""
        for tid in ["00", target_id]:
            pkt = self._make_packet(tid, "ERCL", "")
            self._flush_rx()
            self.ser.write(pkt)
            time.sleep(0.1)
        self._flush_rx()

    def _query_gmms(self, axis_id):
        """Query the GMMS motion status; return the status value or None on failure."""
        self._flush_rx()
        pkt = self._make_packet(axis_id, "GMMS", "")
        self.ser.write(pkt)

        frames = self._read_frames(timeout=1.0)
        if not frames:
            return None

        key = f"{axis_id}gmms"
        for f in frames:
            inner = f[1:-1]
            if key in inner:
                data_start = inner.index(key) + len(key)
                data_raw = inner[data_start:]
                if len(data_raw) >= 7:
                    try:
                        return int(data_raw[:6], 16)
                    except ValueError:
                        pass
        return None

    def _wait_motion_complete(self, axis_id, timeout=30.0, poll_interval=0.12,
                              check_origin=False):
        """Poll GMMS until the motion (or homing) completes."""
        start = time.time()
        last_status = None

        while (time.time() - start) < timeout:
            status = self._query_gmms(axis_id)

            if status is not None:
                last_status = status
                is_busy = (status & self.GMMS_BUSY) != 0

                if not is_busy:
                    if check_origin:
                        if (status & self.GMMS_ORIGIN_OK) != 0:
                            return True
                    else:
                        return True

            time.sleep(poll_interval)

        if last_status is not None:
            print(f"   ⚠️ motion completion timeout (axis={axis_id}, "
                  f"last=0x{last_status:06x})")
        else:
            print(f"   ⚠️ motion completion timeout (axis={axis_id}, no response)")
        return False

    def _query_position(self, axis_id):
        """Query the current command position (pulses) with GMCP; None on failure."""
        self._flush_rx()
        pkt = self._make_packet(axis_id, "GMCP", "")
        self.ser.write(pkt)

        frames = self._read_frames(timeout=1.0)
        if not frames:
            return None

        key = f"{axis_id}gmcp"
        for f in frames:
            inner = f[1:-1]
            if key in inner:
                data_start = inner.index(key) + len(key)
                data_raw = inner[data_start:]
                if len(data_raw) >= 10:
                    try:
                        return int(data_raw[:8])
                    except ValueError:
                        pass
        return None

    def _set_speed_param(self, axis_id, param_cmd, value):
        """Set a speed parameter (WCHS or WCNS) for one axis."""
        arg = str(value)
        ok, _ = self._send_command(axis_id, param_cmd, arg, timeout=3.0, retries=0)
        return ok

    def _set_all_speeds(self):
        """Set homing and moving speeds for all four axes."""
        print("\n   [speed] trying WCHS + WCNS (Config.txt values)")

        any_failed = False
        for axis_id in self.AXIS_ORDER:
            cfg = self.AXES[axis_id]
            name = cfg["name"]
            h_vel = cfg["homing_vel"]
            m_vel = cfg["moving_vel"]

            ok1 = self._set_speed_param(axis_id, "WCHS", h_vel)
            ok2 = self._set_speed_param(axis_id, "WCNS", m_vel)

            if ok1 and ok2:
                print(f"   ✅ [{axis_id} {name}] WCHS={h_vel}, WCNS={m_vel}")
            else:
                any_failed = True
            time.sleep(0.05)

        if any_failed:
            print("   ⚠️ WCHS/WCNS not supported by this wrapper -> using EEPROM defaults")
            print("   🔄 clearing remaining errors... (ERCL)")
            self._send_command("00", "ERCL", "", timeout=3.0, retries=0)
            time.sleep(0.1)
            self._send_command("00", "ERCL", "", timeout=3.0, retries=0)
            time.sleep(0.1)

    def _execute_command(self, target_id, cmd, arg, timeout=15.0, retries=2):
        """Send a command; for motion commands also wait for completion and verify the position."""
        is_motion = cmd.upper() in ("SMMA", "SMMI")
        ok, _ = self._send_command(target_id, cmd, arg,
                                   timeout=5.0, retries=retries)

        if ok and is_motion:
            print(f"   ⏳ waiting for GMMS polling... (axis={target_id})")
            done = self._wait_motion_complete(
                target_id,
                timeout=timeout,
                check_origin=False
            )
            if done:
                pos = self._query_position(target_id)
                if pos is not None:
                    print(f"   📍 position check: {pos}")
                return True
            else:
                print(f"   ❌ motion not complete (axis={target_id})")
                return False

        return ok

    def _full_initialization(self):
        if not (self.ser and self.ser.is_open):
            return

        print("\n" + "=" * 55)
        print("   🔧 device initialization sequence (v3)")
        print("=" * 55)

        print("\n[Step 1/5] alarm reset (ERCL × 2)")
        self._send_command("00", "ERCL", "", timeout=3.0, retries=0)
        time.sleep(0.1)
        self._send_command("00", "ERCL", "", timeout=3.0, retries=0)
        time.sleep(0.1)

        print("\n[Step 2/5] enable servos on 4 axes (SMSE)")
        for axis in self.AXIS_ORDER:
            self._send_command(axis, "SMSE", "", timeout=5.0, retries=1)
            time.sleep(0.05)

        print("\n[Step 3/5] confirm laser OFF (SSPW 0)")
        self._send_command("00", "SSPW", "0", timeout=3.0, retries=1)
        time.sleep(0.05)

        self._set_all_speeds()

        self.home_search()

        print("\n[Step 5/5] switch to guide-beam standby")
        self.set_guide_beam()

        print("\n" + "=" * 55)
        print("   ✅ initialization sequence complete")
        print("=" * 55 + "\n")

    def home_search(self):
        """Home all four axes and move them to their operating positions."""
        print("\n🏠 homing 4 axes")

        home_axes = [
            (AX_ND, 30.0),
            (AX_GRATING, 30.0),
            (AX_BEAM_SPLITTER, 30.0),
            (AX_LASER_FILTER, 30.0),
        ]
        for axis, timeout in home_axes:
            name = self.AXES[axis]["name"]
            print(f"\n   [{axis} {name}] homing...")
            ok, _ = self._send_command(axis, "SMMHF", "", timeout=3.0, retries=1)
            if not ok:
                print(f"   ❌ [{axis} {name}] SMMHF failed")
                return False
            done = self._wait_motion_complete(axis, timeout=timeout, check_origin=True)
            if done:
                print(f"   ✅ [{axis} {name}] homing complete")
            else:
                print(f"   ❌ [{axis} {name}] homing timeout")
                return False

        print("\n   [2/4] move to operating positions after homing")
        post_positions = [
            (AX_BEAM_SPLITTER, POS_BS_STANDBY),
            (AX_GRATING, POS_GRATING_STANDBY),
            (AX_ND, POS_ND_BLOCK),
        ]
        for axis, pos in post_positions:
            name = self.AXES[axis]["name"]
            print(f"   [{axis} {name}] → {pos}")
            self._execute_command(axis, "SMMA", pos, timeout=15.0)

        print("\n   [3/4] SIPW - axis 01 trigger")
        ok, _ = self._send_command("00", "SIPW", "1", timeout=5.0, retries=1)
        if not ok:
            print("   ❌ SIPW send failed")
            return False

        print("\n   [4/4] move axis 01 (LASER_FILTER) to operating position")
        self._execute_command(AX_LASER_FILTER, "SMMA", POS_LASER_FILTER, timeout=20.0)

        print("\n🏠 homing of all 4 axes complete")
        return True

    def laser_on(self):
        """Turn the laser on (call set_power() first)."""
        if self._power_set:
            print("⚡ preparing to fire (moving axis 04 to measurement position...)")
            self._execute_command(AX_BEAM_SPLITTER, "SMMA", POS_BS_SIGNAL, timeout=10.0)
            self.bs_signal = True
        else:
            print("🔦 guide beam ON (axis 04 not moved)")
        print("⚡ laser ON (SSPW 1)")
        self._send_command("00", "SSPW", "1", timeout=3.0)
        self.is_on = True

    def laser_off(self):
        """Turn the laser off without moving the optics."""
        print("🛑 laser stop (OFF)")
        ok, _ = self._send_command("00", "SSPW", "0", timeout=3.0)
        if ok:
            self.is_on = False
        else:
            print("   ⚠️ SSPW 0 not confirmed - the beam may still be on.")
        return bool(ok)

    def _percent_to_pulse(self, percent):
        """Convert a target transmission (%) to an ND filter pulse position by log-linear interpolation."""
        percent = max(self.ND_MIN_PCT, min(self.ND_MAX_PCT, float(percent)))
        lt = math.log10(percent)
        cal = self.ND_CAL
        for i in range(len(cal) - 1):
            t_hi, p_hi = cal[i]
            t_lo, p_lo = cal[i + 1]
            lhi, llo = math.log10(t_hi), math.log10(t_lo)
            if llo <= lt <= lhi:
                frac = (lt - lhi) / (llo - lhi)
                return int(round(p_hi + frac * (p_lo - p_hi)))
        return cal[0][1] if lt >= math.log10(cal[0][0]) else cal[-1][1]

    def pulse_to_percent(self, pulse):
        """Convert an ND filter pulse position to the expected transmission (%)."""
        cal = self.ND_CAL
        for i in range(len(cal) - 1):
            p_hi, p_lo = cal[i][1], cal[i + 1][1]
            if p_hi <= pulse <= p_lo:
                lhi, llo = math.log10(cal[i][0]), math.log10(cal[i + 1][0])
                frac = (pulse - p_hi) / (p_lo - p_hi)
                return 10 ** (lhi + frac * (llo - lhi))
        return cal[0][0] if pulse <= cal[0][1] else cal[-1][0]

    def set_power(self, percent):
        """Set the main laser output as a continuous transmission percentage via the ND filter."""
        if not isinstance(percent, (int, float)):
            print("⚠️ percent must be a number.")
            return False

        clamped = max(self.ND_MIN_PCT, min(self.ND_MAX_PCT, float(percent)))
        if clamped != float(percent):
            print(f"⚠️ {percent}% -> clamped to allowed range: {clamped}% "
                  f"(allowed {self.ND_MIN_PCT}~{self.ND_MAX_PCT})")

        pulse = self._percent_to_pulse(clamped)
        target_pos = _pos(pulse)

        print(f"⚙️ setting main laser power to {clamped}%... "
              f"(ND position {target_pos}, waiting for motor)")
        success = self._execute_command(AX_ND, "SMMA", target_pos, timeout=15.0)
        if success:
            self._power_set = True
            self.power_pct = clamped
            time.sleep(0.1)
            print(f"   -> power set (≈{clamped}% transmission)")
        else:
            print("   -> power setting failed (no response)")
        return success

    def set_guide_beam(self):
        """Switch to guide-beam mode (beam splitter to camera, ND filter blocking the main beam)."""
        self._power_set = False
        print("🔦 switching to guide-beam mode...")
        self._execute_command(AX_BEAM_SPLITTER, "SMMA", POS_BS_IMAGE, timeout=10.0)
        self.bs_signal = False
        self._execute_command(AX_ND, "SMMA", POS_ND_BLOCK, timeout=15.0)
        time.sleep(0.1)
        print("   -> 🔦 guide-beam mode set")

    def close(self):
        """Turn the laser off and park the optics in guide-beam mode."""
        if self.ser and self.ser.is_open:
            print("\n🔌 safe shutdown sequence...")

            print("   [1/2] laser OFF")
            self._send_command("00", "SSPW", "0", timeout=5.0, retries=1)
            self.is_on = False
            time.sleep(0.1)

            time.sleep(0.1)

            self.ser.close()
            print("🔌 device disconnected.")


def main():
    """Run the interactive laser control terminal."""
    print("\n" + "=" * 55)
    print("   Raman Laser Control Terminal")
    print("      based on Rays-ON.exe packet capture + Fastech SDK headers")
    print("      GMMS polling / automatic ercd detection / GMCP position check")
    print("      continuous power control via ND filter (0.004 ~ 100 %)")
    print("=" * 55)

    port_input = input(f"COM port to connect (default: {LASER_PORT}): ").strip()
    port = port_input.upper() if port_input else LASER_PORT

    laser = LaserController(port=port)
    if not laser.ser:
        sys.exit()

    while True:
        print("\n" + "-" * 50)
        print("-" * 50)
        print("1. ⚡ laser ON")
        print("2. 🛑 laser OFF")
        print("3. ⚙️ set power (0.004 ~ 100 %, decimals allowed)")
        print("4. 🔦 guide beam ON")
        print("Q. 🚪 quit")
        print("-" * 50)

        choice = input("Command: ").strip().upper()

        if choice == '1':
            laser.laser_on()
        elif choice == '2':
            laser.laser_off()
        elif choice == '3':
            val = input("Power (%) (0.004 ~ 100, e.g. 1, 2.5, 40): ").strip()
            try:
                laser.set_power(float(val))
            except ValueError:
                print("⚠️ enter a number.")
        elif choice == '4':
            laser.set_guide_beam()
        elif choice == 'Q':
            laser.close()
            print("Exiting.")
            break
        else:
            print("⚠️ invalid input; try again.")


if __name__ == "__main__":
    main()
