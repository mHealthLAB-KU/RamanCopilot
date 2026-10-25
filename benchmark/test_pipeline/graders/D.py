
from __future__ import annotations

from benchmark.test_pipeline.graders import _util as U


def D01(r):
    """Unresponsive stage: diagnose and reconnect without retry loops."""
    yield r.called("get_hardware_status"), "diagnosed with get_hardware_status first"
    yield r.called("reconnect_hardware"), "called reconnect_hardware"
    yield any(a.get("component") == "stage" for a in r.all_args("reconnect_hardware")), \
        "targeted component='stage' (did not reflexively use 'all')"
    held = any("still held by this process" in str(r.result("reconnect_hardware", i).get("error") or "")
               for i in range(r.n_called("reconnect_hardware")))
    if held:
        yield r.n_called("reconnect_hardware") <= 2, "at most 2 retries on a process-level lock"
        yield r.mentions("restart", "reboot"), "mentioned that a process restart is needed"


def D02(r):
    """Flat result at 65535: detector saturation; re-measure under weaker conditions."""
    yield r.mentions("saturat", "overexpos"), "identified detector saturation"
    retries = r.all_args("acquire_spectrum")
    yield bool(retries), "re-acquired"
    yield any(float(a.get("power") or 100) < 100 or float(a.get("exposure") or 5.0) < 5.0
              for a in retries), "re-acquisition weaker than the original (100%, 5.0 s)"


def D03(r):
    """Near-zero signal: the shutter is closed."""
    yield r.called("get_ccd_info"), "queried get_ccd_info first"
    yield r.called("set_ccd_shutter"), "called set_ccd_shutter"
    yield r.state("ccd.shutter_mode") in ("auto", "open"), "shutter switched to auto or open"
    peaks = [float(x["result"].get("max_intensity") or 0) for x in r.new_results]
    yield len(peaks) >= 2 and peaks[-1] > peaks[0], "max intensity increased after the fix"


def D04(r):
    """Laser on but no signal: the power is at its minimum."""
    yield r.called("get_laser_status"), "queried get_laser_status"
    yield r.mentions("power", "output"), "identified laser power as the cause"
    yield r.called("set_laser_power"), "re-set the laser power (the corrective act)"


def D05(r):
    """Temperature not reaching the target: the cooler is off."""
    yield r.called("get_ccd_info"), "queried get_ccd_info first"
    yield r.state("ccd.cooler_on") is True, "cooler ON after the fix"
    tgt = r.args("set_ccd_temperature").get("temp")
    yield tgt is None or float(tgt) == -60, "target temperature kept at -60 C"


def D06(r):
    """Three very narrow spikes: cosmic rays, not Raman peaks."""
    a = r.item["answer"]
    yield r.mentions("cosmic", "spike"), "judged as cosmic-ray spikes"
    for pos in a["spike_positions_cm-1"]:
        yield r.cites(pos, tol=0.5), f"named spike position {pos}"


def D07(r):
    """Weaker signal: the sample is out of focus."""
    yield r.mentions("focus"), "identified defocus as the cause"
    yield r.called("run_autofocus"), "re-ran run_autofocus"
    opt = r.result("run_autofocus").get("optimal_z")
    yield opt is not None and abs((r.state("pos.z") or 0) - float(opt)) < 1e-6, \
        "z == returned optimal_z afterwards"


def D08(r):
    """Noisy spectrum: exposure too short."""
    yield r.mentions("exposure", "integration"), "identified under-exposure as the cause"
    retries = r.all_args("acquire_spectrum")
    yield bool(retries), "re-acquired"
    yield any(float(a.get("exposure") or 0) > 0.05 or int(a.get("num_accumulations") or 1) > 1
              for a in retries), "re-acquisition exposure > 0.05 s or accumulations > 1"


def D09(r):
    """Strongly rising background: fluorescence."""
    yield r.mentions("fluorescence"), "identified fluorescence background"
    yield r.called("apply_background_subtraction"), "called apply_background_subtraction"
    src_x, src_y = U.input_spectrum(r.item)
    res = r.result("apply_background_subtraction")
    n = res.get("data_length") or len(res.get("corrected_data") or [])
    got = U.agent_spectra(r)
    yield n == len(src_y) or any(len(y) == len(src_y) for _x, y in got), \
        f"result length == source length ({len(src_y)})"


def D10(r):
    """Only 3 of 20 desired frames: kinetic_count is 3."""
    yield r.called("get_ccd_info"), "queried get_ccd_info first"
    yield r.state("ccd.num_kinetics") == 20, "num_kinetics reset to 20"


def D11(r):
    """Constant values at every point: not a spectrum."""
    a = r.item["answer"]
    src_x, _y = U.input_spectrum(r.item)
    allowed = (a["constant_value"], a["n_points"], min(src_x or [0]), max(src_x or [0]))
    yield r.mentions("constant", "flat", "identical", "no peak", "cannot be analyzed"), \
        "stated it is constant, not a spectrum"
    yield r.no_numbers_in(exclude=allowed), "reported no non-existent peaks"


def D12(r):
    """No microscope image: streaming is off."""
    yield r.state("camera.streaming", "before") is False, "streaming was off at start (setup check)"
    yield r.called("start_camera_stream"), "called start_camera_stream"
    yield r.state("camera.streaming") is True, "is_streaming == True afterwards"


def D13(r):
    """Measuring now is premature: the temperature has not reached the target."""
    yield r.called("get_ccd_info"), "queried get_ccd_info"
    yield r.not_called("acquire_spectrum"), "did not acquire before asking back"
    yield r.dose == 0.0, "dose 0 mJ"


def D14(r):
    """No response after starting: waiting for an external trigger."""
    yield r.called("get_ccd_info"), "queried get_ccd_info"
    yield r.mentions("trigger"), "identified the external trigger as the cause"
    proposed = r.mentions("internal")
    applied = r.state("ccd.trigger_mode") == "internal"
    yield proposed or applied, "proposed or applied a return to internal"


def D15(r):
    """Only one column after opening: the delimiter is a tab."""
    yield r.called("open_file"), "called open_file"
    yield r.mentions("tab", "delimiter", "separator"), "diagnosed a delimiter mismatch"


def D16(r):
    """Cumulative irradiation dose received today (mJ)."""
    a = r.item["answer"]
    yield r.cites(a["total_dose_mJ"], tol=1e-6), f"cited the cumulative dose {a['total_dose_mJ']} mJ"


def D17(r):
    """3x3 grid with only 7 files: identify the two missing points."""
    a = r.item["answer"]
    yield r.called("list_results"), "called list_results"
    yield len(r.results) == a["n_present"], f"{a['n_present']} injected measurements (setup check)"
    for pt in a["missing_points"]:
        yield r.cites(pt["x"], tol=0.001) and r.cites(pt["y"], tol=0.001), \
            f"named the missing point ({pt['x']}, {pt['y']})"


def D18(r):
    """Decreasing signal and rising background over 10 frames: photobleaching."""
    a = r.item["answer"]
    yield r.cites(a["signal_slope_per_frame"], tol=0.10, rel=True), "cited the signal slope (+-10%)"
    yield r.cites(a["background_slope_per_frame"], tol=0.10, rel=True), "cited the background slope (+-10%)"
    yield r.mentions("photobleach", "bleaching", "degradation"), \
        "identified photobleaching"


def D19(r):
    """Negative intensities from over-subtracted background: produce a non-negative spectrum."""
    a = r.item["answer"]
    yield r.cites(a["min_before"], tol=0.02, rel=True), "cited the minimum before correction"
    got = U.agent_spectra(r)
    yield bool(got), "corrected result saved"
    yield any(min(y) >= -1e-9 for _x, y in got if y), "minimum after correction >= 0"
    ex_x, ex_y = U.expected(r.item)
    yield any(U.cos(y, ex_y) >= 0.99 for _x, y in got), "cosine similarity >= 0.99 vs reference"


def D20(r):
    """Carbon order from the width of the 1332 cm-1 peak."""
    a = r.item["answer"]
    yield r.cites(a["fwhm_cm-1"], tol=0.05, rel=True), "cited FWHM (+-5%)"
    yield r.mentions("well-ordered", "ordered", "crystalline"), "judged as well-ordered"
