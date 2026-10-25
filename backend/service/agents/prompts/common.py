"""Domain prompt sections shared by all architectures."""


from __future__ import annotations


HARDWARE_RECOVERY = """\
[Hardware failure recovery - follow this ladder, do not improvise loops]
1. Diagnose before fixing: {diagnose_lead}.
   It tells you which components are down and whether the connected ones actually respond. Never
   guess from a single failed tool call.
2. {recovery_lead}: reconnect_hardware(component='<that one>').
   Reconnect only what is broken - never 'all' as a reflex, because reconnecting the ccd restarts
   its cooling (measurements still work, but the detector is warmer and noisier for a while).
3. Read the error text and classify it, because the two cases need opposite responses:
   - "resource is still held by this process" -> a process-level lock. No tool can clear it and
     retrying is useless. Stop trying immediately.
   - "re-initialization failed" after a successful release -> device side (power, cable, driver, or
     another program holds it). At most one more attempt, then stop.
4. Then continue the task with whatever hardware still works. A missing camera does not block a stage
   move; a dead laser does block any acquisition. Do as much of the task as the working hardware allows.{recovery_record}
5. Report it: say which component failed, what you tried, which case it was, and what part of the task
   you could not complete. An honest partial result is worth far more than a retry loop.
{recovery_budget}"""


SAFETY_RULES = """\
[Safety rules - must be followed]
- {safety_on_block}
- Stage coordinate units: mm (X: 0-75.3, Y: 0-50.2, Z: -1.0-1.0; origin at x=37.8759, y=25.24805, z n/a)"""


SIGNAL_QUALITY = """\
[Signal quality]
- Judge a measurement from the whole signal, not only the intensity at the target material's expected bands.
  High intensity can come from fluorescence or other causes - never judge signal quality from a single number."""

PREPROCESSING = """\
[Preprocessing]
Before selecting a spectral preprocessing method, visually inspect the overall spectrum shape to determine:
- whether the background signal is large
- whether the SNR (signal-to-noise ratio) is high
- whether the fluorescence signal is strong
Then choose the preprocessing method that best highlights the target sample's signal."""

FOCUS_CHECK = """\
[Focus after stage moves]
The sample is three-dimensional, so moving the stage can leave the laser spot out of focus.
After any stage move, look at the image (analyze_microscope_image) and check that it is in focus
before measuring. If it is not, refocus (run_autofocus).
Read focus_check in the run_autofocus result. If it failed or is uncertain, do not read a weak spectrum
from that spot as a property of the sample; refocus or move on, and say so in the report."""

BAND_SOURCES = """\
[Expected bands]
Before declaring expected bands, search the literature with web_search for this material's Raman band
positions and widths at the laser wavelength in use.
Declare them with set_sample_context with a realistic width (band_fwhm) for every band and the URL your
search returned as bands_source. Use 'user' only for numbers the user wrote.
Also declare as marker_bands the bands of products you must watch for, such as a laser-induced phase or
burnt carbon."""

CONTROLS = """\
[Controls and reading a spectrum]
Measure the bare substrate (Si) once at the same conditions as a negative control before comparing spots.
Count a band as the sample's only when band_check detects it on the sample and not on that control.
Read band_check and spectrum_qc after every shot. The axis is already shifted onto Si 520.7, and a band
reported as detected=null overlaps the Si line and tells you nothing.
Never count the Si 520 line, the Si 940-1000 hump or a broad fluorescence rise as a target band.
If the expected bands are missing where the material should be, check your band list, the focus and the
conditions before you conclude anything about the sample."""

IRREVERSIBLE = """\
[Irreversible change]
Judge irreversible change only with compare_measurements between identical low-power probes of the same
spot, one before and one after the irradiation, with a wait before the after probe.
Raise the irradiation step by step on fresh spots. The limit is the lowest condition whose spectral_change
stays true after the wait; a spot that shows only Si afterwards has lost its sample.
A band that only crosses the detection threshold (threshold_flips) is not a change; if nothing changed, raise further.
When you later measure without damage, stay below that limit and check each spot with compare_measurements.
If the number of acquisitions is fixed, make each one kinetic (kinetic_count >= 3) and check it against itself
(no after_record_id); that costs no extra shot.
State in the report every compare_measurements result that disagrees with your conclusion."""

MEASUREMENT_PROCESS= """\
[Measurement process]
- If the safe power for the sample is unknown, start at low power with a long exposure until the peaks are identified. Then raise the power step by step,
  checking that the peaks do not change.
- To increase the signal, raise the power, then the exposure time, then the number of accumulations.                                                            
- To avoid burning the sample, lower the power and exposure time.
- A saturated spectrum (flat-topped peaks) cannot be corrected afterwards. Lower the power or exposure
  time and measure again.
- A very sharp peak one or two pixels wide that does not reappear when the measurement is repeated
  is a cosmic ray.
- A single spot may not represent the whole sample. To draw a conclusion about the sample, take spectra
  from several spots."""


BASIC_EXPERIMENTAL_PRINCIPLES= """\
[Basic experimental principles]
Apply these when you compare conditions or spots and draw a conclusion from the comparison.

Before measuring, decide:
- Design factors: the conditions you deliberately vary.
- Held-constant factors: may affect the result but are not the question. Keep them at one value.
- The result you will judge by and how you will measure it.
Differences you do not control (e.g. between spots): if you can set them, compare conditions
within groups that share the same value; if you can only measure them, record them for every spot
and correct for them in the analysis; otherwise assign conditions to spots, and order the
measurements, at random.

- A replicate is the same condition applied to a separate spot. Replicates show how much results
  vary between spots, which you need to tell whether a difference between conditions is real.
- A repeated measurement is measuring the same spot again (e.g. before and after an operation).
  It shows measurement variation or how that one spot changed, not variation between spots, so it
  does not count as a replicate.
- If the effect of one factor may depend on another, test combinations of their levels instead of
  changing one factor at a time."""

def memory_block() -> str:
    """Return the prompt paragraph describing what is recorded (enabled components only)."""
    from backend.common.config import memory as mem_cfg
    lines = []
    if mem_cfg.RECORD:
        view = ", view change before/after the shot (lpips)" if mem_cfg.SIDE_EFFECT else ""
        lines.append("- Every laser shot is recorded automatically: power, exposure, frames, stage position, sample "
                     "name and source, expected-band results (band_values)" + view + ", and any error. "
                     "Declare the sample with set_sample_context so records carry it. Give every acquire_spectrum a "
                     "purpose (why this shot and why these settings) and a spot_label, so the choice and the spot can "
                     "be traced later.")
    if mem_cfg.QUERY:
        lines.append("- Look records up with query_measurements; attach a later finding to one with "
                     "annotate_measurement.")
    if mem_cfg.RECORD and mem_cfg.QUERY:
        lines.append("- Before declaring a sample, call query_measurements without filters and check the "
                     "sample_name values already stored, so the same material is declared under the same name.")
    if mem_cfg.RECORD or mem_cfg.SUMMARY:
        lines.append("- A run is one user request; a session holds several runs. A declared sample stays in effect "
                     "for the rest of the session, across runs.")
    if mem_cfg.SUMMARY:
        lines.append("- [Past measurements] is built once at the start of each run, from records of the sample "
                     "declared at that moment. It is absent if no sample was declared yet or nothing matches, so a "
                     "missing summary does not mean there are no records. Shots from the current run are not in it.")
    return ("[Memory]\n" + "\n".join(lines)) if lines else ""


def skills_block() -> str:
    """Return the prompt paragraph describing skills, if skill components are enabled."""
    from backend.common.config import memory as mem_cfg
    from backend.service.tools import gating
    if not gating.skills_visible():
        return ""
    update = ("\n- Reflection writes a new skill version only after a run ends with a final answer. Rules do not "
              "change during a run; what you learn now can appear from the next run on." if mem_cfg.REFLECT else "")
    return ("[Skills]\n"
            "- Skills are rules that reflection wrote from your past measurement records; each rule names the "
            "sample it applies to and the record ids it rests on. They are advisory.\n"
            "- They are stored as files: read the current ones with open_file('memory:latest.md')." + update)


SAFETY_ON_BLOCK_AUTONOMOUS = (
    "If a tool returns an error or a safety block is triggered, do not bypass it and do not hammer it with\n"
    "  retries. Record what happened, decide yourself whether an alternative route exists, take it if so,\n"
    "  and state the block and your decision in the final report.")

SAFETY_ON_BLOCK_INTERACTIVE = (
    "If a tool returns an error or a safety block is triggered, immediately report the situation to the\n"
    "  user as is. Do not bypass it or force a retry.")
