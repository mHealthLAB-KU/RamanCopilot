"""Agent tools for CCD settings."""

from __future__ import annotations

from backend.common.result import fail, ok
from pydantic import Field
from typing import Annotated, Literal, Optional
from backend.repo.hw_access import hw_core as _hw
from backend.service.tools.hw_tools.guards import _ccd_ready, _check_ccd_positive, _serialized
from backend.service.tools.hw_tools.hw_ops import (_apply_acq_mode, _apply_read_mode, _apply_shutter,
                                                   _current_read_mode)


def get_ccd_info() -> dict:
    """Return the current CCD settings and status.

    Returns:
        dict: ok() with temperature, cooler state, acquisition and readout settings and the available gains and speeds, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        t           = _hw._ccd.get_temperature()
        temp_status = _hw._ccd.get_temperature_status()
        cam_status  = _hw._ccd.get_status()
    except Exception as e:
        return fail(str(e))

    def _attr(name):
        return getattr(_hw._ccd, name, None)

    hs_conv = None
    if hasattr(_hw._ccd, 'HSSpeeds_Conventional') and _hw._ccd.HSSpeeds_Conventional:
        hs_conv = _hw._ccd.HSSpeeds_Conventional[0]

    return ok(camera_status=cam_status,
              temperature_C=t,
              temperature_status=temp_status,
              cooler_on=_attr('cooler_on'),
              exposure_time_s=_attr('exposure_time'),
              acquisition_mode=_attr('aq_mode'),
              read_mode=_current_read_mode(),
              read_mode_driver=_attr('ro_mode'),
              trigger_mode=_attr('trigger_mode'),
              shutter_mode=_attr('shutter_mode'),
              num_accumulations=_attr('num_acc'),
              num_kinetics=_attr('num_kin'),
              em_mode=_attr('em_mode'),
              em_gain=_attr('em_gain'),
              output_amp=_attr('output_amp'),
              preamp_gain_index=_attr('preamp_gain_i'),
              preamp_gains_available=_attr('preamp_gains'),
              vs_speeds_us=_attr('VSSpeeds'),
              hs_speeds_conventional_mhz=hs_conv,
              readout_pixels_Nx=_attr('Nx_ro'),
              readout_pixels_Ny=_attr('Ny_ro'),
              detector_Nx=_attr('Nx'),
              detector_Ny=_attr('Ny'))


@_serialized("set_ccd_exposure")
def set_ccd_exposure(
    exposure_time: Annotated[float, Field(description='Exposure time [seconds]. e.g. 0.1, 0.5, 1.0')],
) -> dict:
    """Set the CCD exposure time.

    Returns:
        dict: ok() with the exposure time read back from the CCD, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    if exposure_time <= 0:
        return fail("Exposure time must be greater than 0.")
    try:
        actual = _hw._ccd.set_exposure_time(exposure_time)
        return ok(exposure_time_s=actual)
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_acquisition_mode")
def set_ccd_acquisition_mode(
    mode: Annotated[Literal['single', 'accumulate', 'kinetic', 'run_till_abort'], Field(description='Acquisition mode')],
    num_accumulations: Annotated[Optional[int], Field(ge=1, description='Number of accumulations (used in accumulate/kinetic mode). Omit to keep the value already on the CCD - 0 is not a way to switch accumulation off and is rejected. Note that accumulate mode with 1 accumulation is just a single shot - set this deliberately when you want averaging.')] = None,
    num_kinetics: Annotated[Optional[int], Field(ge=1, description='Total number of frames to acquire (used in kinetic mode). Omit to keep the value already on the CCD.')] = None,
    kinetic_cycle_time: Annotated[Optional[float], Field(gt=0, description='Seconds between kinetic frames (kinetic mode only). Omit to use the minimum interval.')] = None,
) -> dict:
    """Set the CCD acquisition mode.

    Returns:
        dict: ok() with the applied mode and counts, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    err = _check_ccd_positive("num_kinetics", num_kinetics, integer=True)
    if err:
        return err
    try:
        err = _apply_acq_mode(mode, num_accumulations=num_accumulations,
                              kinetic_count=num_kinetics, kinetic_cycle_time=kinetic_cycle_time)
        if err:
            return err
        out = ok(acquisition_mode=mode,
                 num_accumulations=getattr(_hw._ccd, 'num_acc', None),
                 num_kinetics=getattr(_hw._ccd, 'num_kin', None),
                 kinetic_cycle_time_s=getattr(_hw._ccd, 'kinetic_cycle_time', None))
        if mode == 'run_till_abort':
            out["note"] = ("acquire_spectrum cannot run in 'run_till_abort'. Switch back with "
                           "set_ccd_acquisition_mode('single') before measuring, or pass "
                           "acq_mode explicitly to acquire_spectrum.")
        return out
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_read_mode")
def set_ccd_read_mode(
    mode: Annotated[Literal['fvb', 'single_track', 'image'], Field(description='Readout mode')],
    hbin: Annotated[Optional[int], Field(description='Horizontal binning factor. Omit to keep the current value.')] = None,
    single_track_center: Annotated[Optional[int], Field(description='Center row number to read in single_track mode (1-based), e.g. 256. Required the first time you use single_track; omit to reuse the configured track.')] = None,
    single_track_width: Annotated[Optional[int], Field(description='Number of rows to read in single_track mode. Omit to keep the current value.')] = None,
) -> dict:
    """Set the CCD readout mode.

    Returns:
        dict: ok() with the applied readout settings, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        err = _apply_read_mode(mode, hbin=hbin,
                               single_track_center=single_track_center,
                               single_track_width=single_track_width)
        if err:
            return err
        out = ok(read_mode=mode, hbin=_hw._ccd.get_current_hbin(), Nx_ro=_hw._ccd.Nx_ro, Ny_ro=_hw._ccd.Ny_ro)
        if mode == 'image':
            out["note"] = ("acquire_spectrum cannot assemble a 1D spectrum in 'image' mode. "
                           "Call set_ccd_read_mode('fvb') before measuring.")
        return out
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_preamp_gain")
def set_ccd_preamp_gain(
    index: Annotated[int, Field(description='Pre-amplifier gain index (0-based). Typically in the range 0-2')],
) -> dict:
    """Set the pre-amplifier gain.

    Returns:
        dict: ok() with the gain index and value, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        _hw._ccd.set_preamp_gain(index)
        gain_val = _hw._ccd.preamp_gains[index] if _hw._ccd.preamp_gains else None
        return ok(preamp_gain_index=index, gain_value=gain_val)
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_shift_speeds")
def set_ccd_shift_speeds(
    vs_index: Annotated[Optional[int], Field(description='Vertical shift speed index')] = None,
    hs_index: Annotated[Optional[int], Field(description='Horizontal shift speed index')] = None,
) -> dict:
    """Set the vertical and horizontal shift speeds.

    Returns:
        dict: ok() with the applied speeds, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    result = ok()
    try:
        if vs_index is not None:
            _hw._ccd.set_vs_speed(vs_index)
            result["vs_speed_index"] = vs_index
            result["vs_speed_us"]    = (_hw._ccd.VSSpeeds[vs_index]
                                        if vs_index < len(_hw._ccd.VSSpeeds) else None)
        if hs_index is not None:
            _hw._ccd.set_hs_speed_conventional(hs_index)
            result["hs_speed_index"] = hs_index
    except Exception as e:
        return fail(str(e))
    return result


@_serialized("set_ccd_temperature")
def set_ccd_temperature(
    temp: Annotated[int, Field(description='Target temperature [°C]. e.g. -40, -60, -80')],
) -> dict:
    """Set the CCD target temperature.

    Returns:
        dict: ok() with target_temperature_C, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        _hw._ccd.set_temperature(temp)
        return ok(target_temperature_C=temp)
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_cooler")
def set_ccd_cooler(
    on: Annotated[bool, Field(description='true = cooler ON, false = cooler OFF')],
) -> dict:
    """Turn the CCD cooler on or off.

    Returns:
        dict: ok() with the cooler state, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        _hw._ccd.set_cooler(on)
        return ok(cooler="ON" if on else "OFF")
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_shutter")
def set_ccd_shutter(
    mode: Annotated[Literal['auto', 'open', 'close'], Field(description='Shutter mode')],
) -> dict:
    """Set the shutter mode. The setting is kept for later acquisitions.

    Returns:
        dict: ok() with the shutter mode, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        err = _apply_shutter(mode, explicit=True)
        if err:
            return err
        return ok(shutter=mode,
                  note="This setting persists: later acquire_spectrum calls keep it unless you "
                       "pass their own shutter argument. Use 'close' for dark/background frames "
                       "and set it back to 'auto' before normal measurements.")
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_image_flip")
def set_ccd_image_flip(
    hflip: Annotated[bool, Field(description='true = flip horizontally (left-right)')],
    vflip: Annotated[bool, Field(description='true = flip vertically (up-down)')],
) -> dict:
    """Set image flipping. Allowed only in the image readout mode.

    Returns:
        dict: ok() with hflip and vflip, or fail().
    """
    err = _ccd_ready()
    if err:
        return err
    ro = getattr(_hw._ccd, 'ro_mode', None)
    if ro != 'IMG':
        return fail(f"Image flip is only allowed in the 'image' read mode (the CCD is currently in "
                    f"'{ro}'). In 1D spectrum modes (fvb / single_track) flipping would silently "
                    f"misalign the intensity array against the calibrated raman_shift_cm-1 / "
                    f"wavelength_nm axes, and the factory orientation is already set at startup. "
                    f"Call set_ccd_read_mode(mode='image') first if you really need a flipped 2D image.")
    try:
        _hw._ccd.set_image_flip(hflip=hflip, vflip=vflip)
        return ok(hflip=hflip, vflip=vflip, read_mode="image")
    except Exception as e:
        return fail(str(e))
