"""Tool schemas and the dispatch table given to the LLM."""

from __future__ import annotations

from backend.service.tools.hw_tools import acquire_tools as _acq
from backend.service.tools.hw_tools import camera_tools as _cam
from backend.service.tools.hw_tools import ccd_tools as _ccd
from backend.service.tools.hw_tools import laser_tools as _laser
from backend.service.tools.hw_tools import stage_tools as _stage
from backend.service.tools.hw_tools import system_tools as _sys
from backend.service.tools.hw_tools.hw_ops import POWER_RANGE_TXT
from backend.service.tools.session_state import _cache_and_return
from backend.service.tools.non_hw_tools import data_tools as _data
from backend.service.tools.non_hw_tools.file_tools import FILE_TOOLS
from backend.common.result import ok
from backend.service.tools.schema import tool_schema
from backend.service.tools import gating
from backend.service.memory import recall as _recall
from backend.service.memory import recorder as _rec

RAMAN_TOOLS = [
    tool_schema(
        _stage.move_stage,
        'Move the stage to an absolute position (mm). Out-of-range targets are REJECTED with an error rather than clipped, so the reported position is always where the stage actually is. Related: move_stage_relative (same limits, but you give a displacement), move_to_pixel (you give a point on the camera image).',
    ),
    tool_schema(
        _stage.get_stage_position,
        'Read the current stage X, Y, Z position (mm). You do NOT need this right after a move - move_stage / move_stage_relative / move_to_pixel already return the resulting position. get_hardware_status also reports it while diagnosing connection problems.',
    ),
    tool_schema(
        _stage.move_stage_relative,
        'Move the stage by a displacement from its current position (mm). The resulting target is range-checked exactly like move_stage, so a displacement that would leave the travel range is rejected (the error tells you the computed target). Use move_stage when you know the absolute coordinate.',
    ),
    tool_schema(
        _laser.laser_on,
        "Turn the laser on. Which beam actually comes out depends on whether the power is currently applied to the optics: if it is, the MEASUREMENT beam fires; otherwise only the GUIDE beam does (the ND filter is still blocking). The response tells you which one via the 'beam' field - check it, because a guide-beam 'ON' produces no Raman signal. This tool CANNOT arm the measurement beam - use set_laser_power(mw) to arm it without firing, or acquire_spectrum(power=...) which arms it and handles on -> acquire -> off atomically. Use laser_on by itself only for alignment with the guide beam.",
    ),
    tool_schema(
        _laser.laser_off,
        'Stop the laser from firing. The ND filter and the beam splitter stay where they are, so turning it on again emits the same measurement beam - but the camera also stays blind until you call set_guide_beam_mode. You do not need to call this after acquire_spectrum - that tool always turns the laser off (even when it fails) and restores the camera view.',
    ),
    tool_schema(
        _laser.set_laser_power,
        f'Set the laser power, {POWER_RANGE_TXT}, WITHOUT firing the laser. The tool converts it to the ND filter transmission; the result reports power_mw and, for reference, power_pct (percent of full power). This only moves the ND filter to arm the measurement beam - it does not turn the laser on. Use it when the user asks you to set or change the power by itself, or to arm the beam before alignment. For an actual measurement prefer acquire_spectrum(power=...), which applies the power, fires, measures and turns the laser off atomically - chaining set_laser_power + laser_on + laser_off leaves the beam on the sample during your own reasoning time, which can photobleach or damage a biological sample. Out-of-range values are REJECTED, not silently clamped.',
    ),
    tool_schema(
        _laser.get_laser_status,
        'Query the current laser state: whether it is firing (is_on), the applied power (power_mw; power_pct = percent of full power, for reference), and power_armed - whether that power is actually applied to the optics right now. If power_armed is false the ND filter is at the blocking position, so laser_on() would emit only the guide beam; arm it with set_laser_power(mw), or measure with acquire_spectrum(power=...), which applies the power itself. Always read power_armed, not just power_mw - power_mw is the last value that was requested and survives a switch to guide-beam mode, so on its own it will make you think the beam is ready when it is not.',
    ),
    tool_schema(
        _acq.acquire_spectrum,
        "Acquire a Raman spectrum at the current position. Supports three acquisition modes: Single (one shot) / Accumulate (averaged, high SNR) / Kinetic (continuous time series). Automatically handles the laser ON -> power stabilization -> CCD acquisition -> laser OFF flow. IMPORTANT: every parameter is optional and OMITTING one means 'keep whatever the instrument is already set to' (shutter works the same way, except that it opens to 'auto' when nobody has ever set it - the CCD boots with the shutter closed). So set_ccd_exposure / set_ccd_acquisition_mode / set_ccd_read_mode are respected - you may either configure first and then call this with no arguments, or pass the values directly here. The CCD trigger is always internal. Both go through the same code and are equivalent; do not do both 'just in case'. This is also the ONLY way to fire the measurement beam: it applies the power, turns the laser on, acquires, and turns it off again even if the acquisition fails. It also puts the optics back into the guide-beam/camera position afterwards, so the microscope camera can see the sample again - you do NOT need to call set_guide_beam_mode after measuring. The returned exposure_time / laser_power_mw (laser_power_pct = the same power as percent of full, for reference) / num_accumulations are read back from the hardware, so they tell you what was ACTUALLY used. When a calibrator is connected, the raman_shift_cm-1, wavelength_nm, laser_nm fields are included. Kinetic mode returns per-frame data in a frames array. The result always carries saturated (true/false): true means the detector clipped (counts near the 16-bit full well), so peak heights, ratios and SNR from that spectrum are not valid - lower the power or exposure and measure again; a warning field then says the same. max_intensity alone is not a quality measure - a high value can be saturation or a fluorescence background with no Raman peaks on it.",
    ),
    tool_schema(
        _cam.start_camera_stream,
        "Start real-time camera streaming. Streaming must be ON before analyze_microscope_image, preview_grid_scan or run_autofocus can get a frame. The response field already_streaming tells you whether it was ALREADY running before your call - if it was true, someone else (the user's live view) owns the stream, so do not stop it afterwards.",
    ),
    tool_schema(
        _cam.stop_camera_stream,
        'Stop real-time camera streaming. Only call this if YOU started the stream - i.e. start_camera_stream returned already_streaming=false. Stopping a stream the user was watching blanks their live view, and every image tool stops working until it is restarted. When in doubt, leave it running.',
    ),
    tool_schema(
        _ccd.get_ccd_info,
        'Query all current CCD settings and status. Returns temperature, cooling status, exposure time, acquisition mode, readout mode, gain, shift speeds, pixel count, etc. in one call. Use it before and after changing parameters to verify the current state.',
    ),
    tool_schema(
        _ccd.set_ccd_exposure,
        'Set the CCD exposure time (seconds). Larger values give stronger signal but longer measurement time. The value persists - a later acquire_spectrum keeps it unless you pass exposure there. OVERLAP: acquire_spectrum(exposure=...) sets exactly the same thing through the same code. Use this tool when you set the exposure once and then measure several times; pass it to acquire_spectrum when it is a one-off. Never do both for the same measurement.',
    ),
    tool_schema(
        _ccd.set_ccd_acquisition_mode,
        "Set the CCD acquisition mode. 'single': single shot. 'accumulate': num_accumulations shots back to back at the minimum interval; the returned spectrum is their per-shot AVERAGE. 'kinetic': acquire num_kinetics frames continuously. 'run_till_abort': acquire continuously until an abort command (acquire_spectrum cannot use this one - set single/accumulate/kinetic before measuring). The mode and counts persist - acquire_spectrum keeps them unless you pass acq_mode / num_accumulations there. OVERLAP: acquire_spectrum(acq_mode=, num_accumulations=, kinetic_count=) sets the same thing through the same code; use one or the other, not both. This tool is the only way to select 'run_till_abort'. The response reports the counts actually in effect on the hardware, not what you asked for.",
    ),
    tool_schema(
        _ccd.set_ccd_read_mode,
        "Set the CCD readout mode. 'fvb': Full Vertical Binning - sum all rows -> 1D spectrum (used for Raman). 'single_track': read only a specific vertical row. single_track_center is required. 'image': full 2D image or ROI (acquire_spectrum cannot build a 1D spectrum from this - switch back to 'fvb' before measuring). The mode persists - acquire_spectrum keeps it unless you pass read_mode there. OVERLAP: acquire_spectrum(read_mode=, hbin=, single_track_center=, single_track_width=) sets the same thing through the same code, using the same parameter names. Use one or the other, not both. This tool is the only way to select 'image' mode.",
    ),
    tool_schema(
        _ccd.set_ccd_preamp_gain,
        'Set the pre-amplifier gain index. See preamp_gains_available in get_ccd_info() for the available gain list. A larger index gives higher gain, which helps measure weak signals but also increases noise.',
    ),
    tool_schema(
        _ccd.set_ccd_shift_speeds,
        'Set the vertical (VS) and horizontal (HS) pixel shift speeds. VS: a larger index is slower but reduces charge-transfer noise. HS: a larger index is slower but reduces readout noise. You may specify only one of them. See get_ccd_info() for the available speed list.',
    ),
    tool_schema(
        _ccd.set_ccd_temperature,
        'Set the CCD cooling target temperature (°C). Lower temperature reduces dark-current noise. It may take several minutes to reach; check progress with get_ccd_info(). Typical range: -80 to 20°C.',
    ),
    tool_schema(
        _ccd.set_ccd_cooler,
        'Turn the CCD Peltier cooler on (true) or off (false). It must be turned off before shutdown.',
    ),
    tool_schema(
        _ccd.set_ccd_shutter,
        "Set the CCD shutter mode. 'auto'  - open and close automatically during acquisition (normal measurement). 'open'  - force open. 'close' - force closed. Like the other CCD settings this one PERSISTS: a later acquire_spectrum keeps it unless you pass its own shutter argument. So for several dark / background frames set 'close' once here and measure repeatedly; for a single dark frame acquire_spectrum(shutter='close') is enough. Set it back to 'auto' before normal measurements.",
    ),
    tool_schema(
        _ccd.set_ccd_image_flip,
        "Set horizontal/vertical flip of the acquired 2D image. Only valid when the CCD read mode is 'image' - in the 1D spectrum modes (fvb / single_track) flipping would misalign the intensity array against the calibrated wavelength axis, and the correct orientation is already set at startup, so the call is rejected there. You almost never need this for Raman measurements.",
    ),
    tool_schema(
        _stage.get_stage_speed,
        'Query the current stage movement speed (mm/s). Returns the x_speed_mm_s, y_speed_mm_s, z_speed_mm_s fields.',
    ),
    tool_schema(
        _stage.set_stage_speed,
        "Set the stage movement speed. X/Y max 5.0 mm/s, Z max 0.1 mm/s. If a specific axis speed is omitted, its current speed is maintained. Values above the limit are clipped to it; the response reports the speeds that will ACTUALLY be used and lists any clipped axes under 'clipped'.",
    ),
    tool_schema(
        _sys.get_hardware_status,
        "Report which hardware components (stage, laser, ccd, camera) are currently CONNECTED, and for the connected ones whether they actually respond. Read `summary` first. Call this FIRST whenever a hardware tool fails, before trying to fix anything - it tells you whether one device is down or several, which decides what is even worth attempting. It touches nothing and fires no laser, so it is always safe to call. SCOPE: this answers 'is it there and alive'. For the SETTINGS of a working device use the per-device tools instead - get_ccd_info (exposure, mode, temperature), get_laser_status (power and whether it is armed), get_stage_position, get_stage_speed.",
    ),
    tool_schema(
        _sys.reconnect_hardware,
        "Release and re-initialize a hardware component that is unresponsive or stuck. component: 'stage' | 'ccd' | 'camera' | 'laser' | 'all'. Call get_hardware_status first to see what is actually down. Read the returned `errors` text carefully - it distinguishes two very different cases. (a) 'resource is still held by this process': a process-level lock that NO tool can clear. Calling this tool again will not help; the server must be restarted by a human. Do not retry - carry on without that component and say so in your final answer. (b) 're-initialization failed' after a successful release: a device-side problem (power, cable, driver, or another program holding the device). Retrying once is reasonable; beyond that, proceed without the component and state the limitation. Never call this repeatedly in a loop - it cannot fix either case by repetition. Note: reconnecting the 'ccd' restarts its cooling - measurements work right away, but the detector is warmer and noisier until it cools back down.",
    ),
    tool_schema(
        _laser.set_guide_beam_mode,
        'Switch the laser to guide-beam standby state. Moves the beam splitter to the standby position and the ND filter to the main-beam blocking position. Use it for sample alignment and focus checking.',
    ),
    tool_schema(
        _cam.set_camera_exposure,
        'Set the camera (TUCam) exposure time (ms).',
    ),
    tool_schema(
        _cam.set_camera_auto_exposure,
        'Enable (true) or disable (false) camera auto exposure.',
    ),
    tool_schema(
        _cam.analyze_microscope_image,
        "Capture the current view of the TuCam optical microscope camera and pass it to YOU as an image to look at. Use it when visual judgment is needed, e.g. checking sample position, identifying a target, or detecting debris. Streaming must be active. YOU must look at the image, find the target in it, and read off its pixel coordinates yourself - the response does not contain the target's coordinates. Those pixel coordinates are not stage coordinates, so to move there pass them to move_to_pixel. It also saves the view with its stage-coordinate extent, so a later run_analysis gets microscope_image / image_extent injected and can overlay a peak map on the microscope photo. OVERLAP - preview_grid_scan also captures the camera, to show where a planned grid would land; both return the same pixel coordinate system, so a coordinate read from either can be passed to move_to_pixel.",
    ),
    tool_schema(
        _cam.move_to_pixel,
        'Convert pixel coordinates (pixel_x, pixel_y) within the camera image to stage mm coordinates and move there. The stage moves so that whatever is at that pixel ends up under the laser, i.e. inside the red ring drawn on the image (the ring is the laser aim point; it is near but not exactly at the image centre). After checking the target pixel coordinates with analyze_microscope_image, move with this tool.',
    ),
    tool_schema(
        _cam.run_autofocus,
        'Hill-climbing autofocus based on minimizing the guide-beam laser spot area. Computes the spot pixel count (area) via Otsu thresholding on the laser OFF/ON difference image, and moves the stage to the Z position with minimum area (where the laser spot is sharpest). Adaptive hill-climbing auto-adjusts the step size and finally returns to the historical minimum position. It moves Z only, uses the GUIDE beam (not the measurement beam), and leaves the laser off. The search is clipped to the Z travel range; if the response contains z_limit_hits the focus may be physically out of reach, and repeating the call will NOT help - report that instead.',
    ),
    tool_schema(
        _acq.preview_grid_scan,
        "Preview a rows x cols grid mapping WITHOUT moving the stage or firing the laser. Overlays the planned scan points as circles on the current camera view and returns that image so the layout can be visually verified before committing. ORIENTATION (do not confuse the two): rows = number of points stacked VERTICALLY = grid HEIGHT (stage Y axis); cols = number of points side-by-side HORIZONTALLY = grid WIDTH (stage X axis). So rows=3, cols=2 is a TALL grid (3 high x 2 wide) and rows=2, cols=3 is a WIDE grid (2 high x 3 wide) - these are DIFFERENT layouts, never swap them. When the user asks for a grid like 'A x B', decide deliberately which number is the horizontal count (width -> cols) and which is the vertical count (height -> rows), then use this preview image to confirm the drawn orientation matches what they asked. If center_x/center_y are omitted, the current stage position is used as the grid center, and that resolved center is what gets approved - a later run_grid_scan with the centre omitted scans THAT position even if the stage has moved since. Whatever you pass here you must pass IDENTICALLY to run_grid_scan: the approval is matched on the arguments themselves, so omitting the centre here and spelling it out there (or vice versa) is rejected as a mismatch even when both mean the same place. SIZE LIMIT: rows * cols must be <= 400 points; a larger grid is refused here, so agree a smaller size with the user before promising a map. The camera field of view is small, so with wide spacing some points may fall outside the frame; they are still measured, and the response reports how many are in view (n_in_view) along with the exact view size in fov_mm - plan spacing from that returned value rather than from a remembered number.",
    ),
    tool_schema(
        _acq.run_grid_scan,
        "Execute a rows x cols grid mapping: for each point it moves the stage, optionally autofocuses, acquires one spectrum, and auto-saves it (position-tagged). Returns a single compact summary (counts, intensity min/max/mean, and per-point data when 32 points or fewer) instead of one tool message per point - this is the token-efficient way to run a map. ORIENTATION: rows = vertical count (height, stage Y), cols = horizontal count (width, stage X); rows=3,cols=2 is a tall 3x2 grid, rows=2,cols=3 is a wide 2x3 grid - do not swap them. PASS THE APPROVED ARGUMENTS VERBATIM: the approval gate compares the arguments you give, not the physical positions they work out to. If you OMITTED center_x/center_y in the preview you must omit them here too - filling in the numeric coordinates of that same spot is rejected as an 'Approval mismatch'. (An omitted centre runs at the position the approved preview was drawn at, so the scan lands where the user saw it even if the stage has moved in between.) SIZE LIMIT: rows * cols must be <= 400 points. REQUIRES PRIOR HUMAN APPROVAL: do NOT call this in the same turn as preview_grid_scan. Call it ONLY after (1) you showed the user a preview_grid_scan image, (2) you ended that turn, and (3) the user EXPLICITLY approved that exact layout in a later message. If the user has not explicitly approved the previewed grid, do not call this - preview first and wait. The laser is fired at every point. READ THE RESULT BEFORE REPORTING SUCCESS: n_measured can be lower than n_points, and with autofocus='each' the response may carry n_autofocus_failed (those points were measured at whatever Z the stage was at, so a weak signal there is an artefact, not a property of the sample) or n_autofocus_z_limit (the focus is physically out of reach - re-running will not help). If autofocus fails several times in a row the scan STOPS EARLY and comes back with ok=false plus an 'aborted' field; the points measured up to then are still saved, but the grid is incomplete and you must say so rather than reporting the map as done.",
    ),
    tool_schema(
        _data.list_results,
        "Query the list of MEASUREMENTS auto-saved by acquire_spectrum. Returns each item's base (file identifier), session, title, timestamp, and meta (coordinates, etc.). Get the base to pass to combine_spectra / aggregate_spectra_csv / bundle_results here. By default this lists only the measurements from YOUR current session - your own work, not other sessions'. Files live in data/results/<date>/<your session>/. THERE ARE THREE 'list' TOOLS, one per store - choose by what you are looking for: list_results = raw measurements you acquired (this one); list_session_artifacts = files YOU produced (processed spectra from save_result, measurement-point records, figures), each with a path you can open_file; list_uploaded_files = data files the USER attached to the chat.",
    ),
    tool_schema(
        _data.combine_spectra,
        'Combine several saved measurement spectra into a single grid image and render it. Each cell title uses the title auto-generated at save time (scan coordinates, power, exposure) as is. e.g. for a 10x10 scan, arranged by coordinate. If names is omitted, combine everything from that date. This is the ready-made one-call version - it needs no code. run_analysis can also plot the same measurements, but only use it when you need a layout or computation this tool does not give you (overlaid curves, peak maps, custom axes).',
    ),
    tool_schema(
        _data.aggregate_spectra_csv,
        'Build a CSV summarizing several saved measurements, one row per experiment (date, time, title, coordinates, power, exposure, max intensity, total intensity, peak position). Use it to organize multiple experiments into one table. One call, no code. It summarizes ONE ROW PER MEASUREMENT - if you need per-point values computed some other way, or a table of derived quantities, use run_analysis instead.',
    ),
    tool_schema(
        _data.web_search,
        'Search the external web and fetch the top results (title, URL, summary). Use it to find recent/specialist information (literature, recommended parameter values, methodology, etc.) that you cannot answer reliably from your own knowledge. If there is no internet or search is not configured it returns a failure, in which case decide from your own knowledge and say that the web could not be checked.',
    ),
    tool_schema(
        _data.run_analysis,
        "Run 'computation/visualization' Python code on saved measurement data AND on files the user attached to the chat, in a safe sandbox. Use it to handle analyses not provided as tools (baseline correction, peak detection, per-coordinate peak maps/heatmaps, etc.) directly in code. Already injected into the runtime: spectra (list[dict] - each item has base, title, x, y, power_mw, exposure, mode, raman_shift (np.ndarray or None), intensity (np.ndarray)), np (numpy), plt (matplotlib.pyplot). Preprocessing helpers are injected too, so you do not have to re-implement them: ipbsa(y, order=5, max_iterations=100, threshold=0.001) returns (corrected, background) using an iterative-polynomial (IPBSA) baseline fit, and poly_baseline(y, order=5, x=None) returns a single polynomial background fit if you would rather build your own iteration. Prefer ipbsa(...) over hand-writing a polynomial baseline loop - it is the single biggest source of over-long code. A Kinetic measurement carries its time series too: frames is a 2D np.ndarray of shape (n_frames, n_pixels), so frames.mean(axis=0) is the averaged spectrum and frames[:, px] is one pixel's intensity over time. For those items intensity is the frame MEAN (flagged by intensity_is_frame_mean) - use frames, not intensity, when the question is about change over time. Very long runs are cut to the first 200 frames and say so in frames_truncated. This is the ONLY way to analyse a kinetic measurement - open_file only summarises a kinetic file's raw rows and cannot reconstruct the frames. If you pass file_ids, the attached files are parsed and injected as files (list[dict] - each item has file_id, filename, sheet, columns (list[str]), n_rows, and table (dict mapping column name -> np.ndarray for numeric columns, list[str] for text columns)). Inspect a file's structure with open_file first, then use the column names you saw as keys of table. spectra and files can be used together - e.g. overlay an attached reference spectrum on a measured one. If you called analyze_microscope_image first, microscope_image (np.ndarray|None) and image_extent ([xmin,xmax,ymin,ymax] stage mm|None) are also injected - after ax.imshow(microscope_image, extent=image_extent), overlaying peaks at the measurement (x,y) makes a peak map on top of the microscope image. A figure created with plt is auto-saved and shown in the chat. Small numeric results are observed if you print() them. To SAVE a computed spectrum, call the injected hook save_result(filename, intensity, raman_shift=None, wavelength_nm=None, metadata=None) inside your code - it writes data/<filename>.csv at full precision and returns the path, which also comes back in the tool result as saved_files. This is the correct way to persist a processed spectrum (baseline-corrected, spike-removed, normalized, smoothed): do it in the same run_analysis call that computes it. save_result is the ONLY way to write an array - there is no tool that takes an intensity array as an argument. Do NOT print an array in order to re-type it elsewhere: printing thousands of numbers overflows the context window and loses precision. Print only a short summary (how many points, how many spikes removed, where the peaks are). stdout is truncated past 4000 characters. Constraints (safety): no hardware (laser/stage/CCD) control, no network access, no file access other than save_result and plt figures, imports limited to computation libraries such as numpy/scipy/matplotlib/math. A 'measurement' like a 3x3 scan is done first with move_stage + acquire_spectrum, not this tool, and the saved result is analyzed/visualized here. On failure, read error/trace, fix the code, and call again. WHEN NOT TO USE THIS: a ready-made tool already covers some jobs and needs no code - combine_spectra (grid of spectra images), aggregate_spectra_csv (one summary row per measurement), bundle_results (zip for download). Reach for run_analysis when no such tool fits, not by default.",
    ),
    tool_schema(
        _data.bundle_results,
        'Bundle saved measurement files (png/csv/json) into a single zip and provide a download link. Use it when the user wants to download all the results.',
    ),
]


KB_TOOL = tool_schema(
    _data.identify_material,
    "Identify what material a measured spectrum is, by comparing it against the RRUFF reference "
    "library of 6,938 real measured mineral spectra covering 2,272 distinct minerals. Returns "
    "candidate minerals ranked by similarity, each with its chemical formula, elements, crystal "
    "system and classification. Use it after acquiring a spectrum, when the question is what the "
    "sample is, or to check a peak assignment you made by eye. It only reads a spectrum you already "
    "measured - it does not touch the laser or the stage, so it is harmless to the sample and may be "
    "called several times with different algorithms. IT DOES NOT RECOMMEND MEASUREMENT PARAMETERS: "
    "there is no laser power or exposure time anywhere in this library, so choose those yourself. "
    "TWO LIMITS. (1) The library is minerals only, so an organic or engineered sample (polystyrene, "
    "a cell, a battery electrode) has no correct answer in it and the tool will still return its "
    "least-different rows - read score_margin and the candidate's chemistry before believing the top "
    "row. (2) The comparison needs a Raman shift axis; a spectrum stored only as pixel index or "
    "wavelength is rejected.",
)

MEMORY_TOOLS = [
    tool_schema(
        _rec.set_sample_context,
        "Declare what the sample is before you measure it, so that every following measurement record "
        "carries it: the sample name, whether the USER told you or it is YOUR guess, and - if you know them - "
        "the Raman shifts where its characteristic peaks should appear and where that came from. Records keep "
        "this context until you declare another one. The sample name is also the key to memory: the "
        "memory_hint you get after moving or measuring shows past records and learned rules of this sample "
        "name (or of the material identified at that spot), and recent shots of any sample when neither is "
        "known. So give the bare material name (e.g. 'graphene'), the same way every time. The expected bands "
        "are scored on every measurement (band_values: net height, SNR, detected), so give the positions this "
        "sample really shows - a band declared 20 cm-1 away from where it actually sits is scored as not "
        "detected. It does not touch the hardware and does not save any measurement - measurements are "
        "recorded automatically when the laser fires.",
    ),
    tool_schema(
        _recall.query_measurements,
        "Look up past measurement records - every time the laser actually fired, in this or earlier sessions: "
        "the applied power, exposure and frames, where on the stage, what sample it was declared to be, whether "
        "the expected peaks appeared (band_values) and how much the view changed before vs after the shot "
        "(lpips, larger = more change; a raw value that camera noise and focus also move, not a damage verdict). "
        "Use it before choosing parameters for a sample you may have measured "
        "before. It only reads records; it does not touch the hardware.",
    ),
    tool_schema(
        _recall.annotate_measurement,
        "Attach a later finding to a past measurement record without changing the record itself: what the "
        "sample turned out to be (sample_identity), a verdict on the spectrum such as saturated or too weak "
        "(quality), what went wrong (error), or any other remark (note). Use it when the user tells you "
        "something about an earlier measurement, or when you judge one after the fact. Find the record id "
        "with query_measurements first.",
    ),
    tool_schema(
        _recall.compare_measurements,
        "Compare two low-power probes of the SAME spot - one before and one after an irradiation - to decide "
        "whether the irradiation changed the sample. It reads the two saved spectra (no laser) and reports, per "
        "declared band: before/after detection and the after/before ratio with its significance; marker bands "
        "(e.g. hematite) that appeared; new peaks that were not in the before probe; the Si 520 ratio (sample "
        "thinner or gone when Si grows); and LPIPS between the before view and the after view. spectral_change and "
        "reasons summarise it with fixed, stated criteria. Use it for every irreversibility judgement and for "
        "checking that a measurement left a spot unchanged; do not judge by eye from plots alone.",
    ),
]

ALL_TOOLS = RAMAN_TOOLS + FILE_TOOLS + [KB_TOOL] + MEMORY_TOOLS

_GRID_OFF_TEXT = {
    "start_camera_stream": [("analyze_microscope_image, preview_grid_scan or run_autofocus",
                             "analyze_microscope_image or run_autofocus")],
    "analyze_microscope_image": [
        (" OVERLAP - preview_grid_scan also captures the camera, to show where a planned grid would land; "
         "both return the same pixel coordinate system, so a coordinate read from either can be passed to "
         "move_to_pixel.", ""),
    ],
}


_SKILL_TEXT = {
    "open_file": (" A learned-skill file (.md) comes back as its text: open_file('memory:latest.md') is the "
                  "current skill set; memory:v0001.md, memory:v0002.md, ... are earlier versions. If no skill "
                  "exists yet it says so."),
}


def agent_tools() -> list:
    """Return the tool list bound to the model, excluding gated tools."""
    grid_on = gating.enabled("run_grid_scan")
    out = []
    for t in ALL_TOOLS:
        name = t["function"]["name"]
        if not gating.enabled(name):
            continue
        subs = None if grid_on else _GRID_OFF_TEXT.get(name)
        extra = _SKILL_TEXT.get(name) if gating.skills_visible() else None
        if subs or extra:
            desc = t["function"]["description"]
            for a, b in subs or ():
                desc = desc.replace(a, b)
            t = {**t, "function": {**t["function"], "description": desc + (extra or "")}}
        out.append(t)
    return out


TOOL_DISPATCH = {
    "move_stage":               lambda a: _stage.move_stage(**a),
    "get_stage_position":       lambda a: _stage.get_stage_position(),
    "move_stage_relative":      lambda a: _stage.move_stage_relative(**a),
    "get_stage_speed":          lambda a: _stage.get_stage_speed(),
    "set_stage_speed":          lambda a: _stage.set_stage_speed(**a),
    "reconnect_hardware":       lambda a: _sys.reconnect_hardware(**a),
    "get_hardware_status":      lambda a: _sys.get_hardware_status(),
    "laser_on":                 lambda a: _laser.laser_on(),
    "laser_off":                lambda a: _laser.laser_off(),
    "set_laser_power":          lambda a: _laser.set_laser_power(**a),
    "get_laser_status":         lambda a: _laser.get_laser_status(),
    "set_guide_beam_mode":      lambda a: _laser.set_guide_beam_mode(),
    "acquire_spectrum":         lambda a: _cache_and_return(_acq.acquire_spectrum(**a)),
    "start_camera_stream":      lambda a: _cam.start_camera_stream(),
    "stop_camera_stream":       lambda a: _cam.stop_camera_stream(),
    "set_camera_exposure":      lambda a: _cam.set_camera_exposure(**a),
    "set_camera_auto_exposure": lambda a: _cam.set_camera_auto_exposure(**a),
    "analyze_microscope_image": lambda a: _cam.analyze_microscope_image(**a),
    "move_to_pixel":            lambda a: _cam.move_to_pixel(**a),
    "run_autofocus":            lambda a: _cam.run_autofocus(**a),
    "preview_grid_scan":        lambda a: _acq.preview_grid_scan(**a),
    "run_grid_scan":            lambda a: _acq.run_grid_scan(**a),
    "get_ccd_info":             lambda a: _ccd.get_ccd_info(),
    "set_ccd_exposure":         lambda a: _ccd.set_ccd_exposure(**a),
    "set_ccd_acquisition_mode": lambda a: _ccd.set_ccd_acquisition_mode(**a),
    "set_ccd_read_mode":        lambda a: _ccd.set_ccd_read_mode(**a),
    "set_ccd_preamp_gain":      lambda a: _ccd.set_ccd_preamp_gain(**a),
    "set_ccd_shift_speeds":     lambda a: _ccd.set_ccd_shift_speeds(**a),
    "set_ccd_temperature":      lambda a: _ccd.set_ccd_temperature(**a),
    "set_ccd_cooler":           lambda a: _ccd.set_ccd_cooler(**a),
    "set_ccd_shutter":          lambda a: _ccd.set_ccd_shutter(**a),
    "set_ccd_image_flip":       lambda a: _ccd.set_ccd_image_flip(**a),
    "list_results":             lambda a: ok(items=[
        {k: it[k] for k in ("base", "session", "date", "title", "timestamp", "meta")}
        for it in _data.list_results(**a)]),
    "combine_spectra":          lambda a: _data.combine_spectra(**a),
    "aggregate_spectra_csv":    lambda a: _data.aggregate_spectra_csv(**a),
    "bundle_results":           lambda a: _data.bundle_results(**a),
    "run_analysis":             lambda a: _data.run_analysis(**a),
    "web_search":               lambda a: _data.web_search(**a),
}
