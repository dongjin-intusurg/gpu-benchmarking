#!/usr/bin/env python3
"""Choose the device config for THIS machine, so a run needs no arguments.

Selection, in order:
  1. platform      - /etc/nv_tegra_release present -> jetson, else discrete
  2. device name   - the config's device_name_match must appear in the
                     nvidia-smi name; measuring with another device's config
                     silently corrupts every downstream budget
  3. power mode    - on jetson, the config's required_power_mode must equal the
                     mode nvpmodel reports. A run that measures at a lower
                     mode ships a config declaring it, and picking it by hand is
                     exactly the step that gets forgotten.

Prints the chosen path on stdout; diagnostics go to stderr. Exits 1 with a
message naming what it saw when nothing matches - never guesses.
"""
import glob, json, os, re, subprocess, sys


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def current_mode():
    out = sh("nvpmodel", "-q") or sh("sudo", "-n", "nvpmodel", "-q")
    for line in out.splitlines():
        if "NV Power Mode" in line:
            return line.split(":", 1)[-1].strip()
    return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        search = [sys.argv[1]]
    else:
        # a device config can live in either layout: configs/device_configs/
        # (the repo's shipped location, a sibling of scripts/) or
        # device_configs/ next to the stage scripts (the bundle mirror).
        # DEVICE_CONFIG_DIR overrides. First existing dir with a *.json wins.
        # order: explicit override, then the stage-adjacent dir (device_configs/
        # beside the scripts), then the repo's configs/device_configs. The first
        # existing dir that holds a *.json wins - whichever layout this checkout uses.
        search = [os.environ.get("DEVICE_CONFIG_DIR", ""),
                  os.path.join(here, "..", "device_configs"),
                  os.path.join(here, "..", "..", "configs", "device_configs")]
    d = None
    for cand in search:
        if cand and os.path.isdir(cand) and glob.glob(os.path.join(cand, "*.json")):
            d = cand; break
    if d is None:
        print("no device_configs directory with a *.json found; looked in: "
              + ", ".join(x for x in search if x), file=sys.stderr)
        return 1
    platform = "jetson" if os.path.exists("/etc/nv_tegra_release") else "discrete"
    smi = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").splitlines()
    smi_name = smi[0].strip() if smi else ""
    mode = current_mode() if platform == "jetson" else None

    cands, rejected = [], []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            c = json.load(open(f))
        except Exception as e:
            rejected.append((f, f"unreadable: {e}")); continue
        if (c.get("platform") or "") != platform:
            rejected.append((f, f"platform {c.get('platform')!r} != {platform!r}")); continue
        nm = c.get("device_name_match") or ""
        if nm and nm not in smi_name:
            rejected.append((f, f"name {nm!r} not in {smi_name!r}")); continue
        cands.append((f, c))

    if not cands:
        print(f"no device config matches this machine (platform={platform}, gpu={smi_name!r})",
              file=sys.stderr)
        for f, why in rejected:
            print(f"  {os.path.basename(f)}: {why}", file=sys.stderr)
        return 1

    if platform == "jetson" and mode:
        exact = [(f, c) for f, c in cands
                 if (c.get("required_power_mode") or "").upper() == mode.upper()]
        if len(exact) == 1:
            print(f"device config: {os.path.basename(exact[0][0])}  "
                  f"(gpu={smi_name}, power mode={mode})", file=sys.stderr)
            print(exact[0][0]); return 0
        if len(exact) > 1:
            print(f"several configs declare required_power_mode {mode!r}: "
                  + ", ".join(os.path.basename(f) for f, _ in exact)
                  + " - set DEVICE_CFG explicitly", file=sys.stderr)
            return 1
        print(f"no config declares required_power_mode {mode!r}; candidates: "
              + ", ".join(f"{os.path.basename(f)}={c.get('required_power_mode')}" for f, c in cands),
              file=sys.stderr)
        print("  measuring at a mode no config describes would adjudicate clock drift against "
              "the wrong targets - add a config for this mode or switch modes", file=sys.stderr)
        return 1

    if len(cands) == 1:
        print(f"device config: {os.path.basename(cands[0][0])}  (gpu={smi_name})", file=sys.stderr)
        print(cands[0][0]); return 0
    print("several configs match this device: "
          + ", ".join(os.path.basename(f) for f, _ in cands)
          + " - set DEVICE_CFG explicitly", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
