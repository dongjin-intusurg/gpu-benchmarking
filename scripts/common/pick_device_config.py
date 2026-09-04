#!/usr/bin/env python3
"""Choose the device config for THIS machine, so a run needs no arguments.

Selection: platform (/etc/nv_tegra_release -> jetson, else discrete), then the config's
device_name_match must appear in the nvidia-smi name, then on jetson required_power_mode must
equal the nvpmodel mode. Prints the chosen path on stdout; diagnostics go to stderr. Exits 1
naming what it saw when nothing matches uniquely - never guesses.
"""
import glob
import json
import os
import subprocess
import sys


def command_stdout(*argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def current_power_mode():
    output = command_stdout("nvpmodel", "-q") or command_stdout("sudo", "-n", "nvpmodel", "-q")
    for line in output.splitlines():
        if "NV Power Mode" in line:
            return line.split(":", 1)[-1].strip()
    return None


def gpu_name():
    lines = command_stdout("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").splitlines()
    return lines[0].strip() if lines else ""


def search_dirs():
    """Explicit argument, else DEVICE_CONFIG_DIR, the stage-adjacent device_configs/, then the
    repo's configs/device_configs - whichever layout this checkout uses."""
    if len(sys.argv) > 1:
        return [sys.argv[1]]
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.environ.get("DEVICE_CONFIG_DIR", ""),
            os.path.join(here, "..", "device_configs"),
            os.path.join(here, "..", "..", "configs", "device_configs")]


def first_dir_with_configs(candidates):
    for directory in candidates:
        if directory and os.path.isdir(directory) and glob.glob(os.path.join(directory, "*.json")):
            return directory
    return None


def matching_configs(config_dir, platform, smi_name):
    """(matching [(path, config)], rejected [(path, reason)]) over every *.json in the dir."""
    matching, rejected = [], []
    for path in sorted(glob.glob(os.path.join(config_dir, "*.json"))):
        try:
            config = json.load(open(path))
        except Exception as error:
            rejected.append((path, f"unreadable: {error}"))
            continue
        if (config.get("platform") or "") != platform:
            rejected.append((path, f"platform {config.get('platform')!r} != {platform!r}"))
            continue
        name_match = config.get("device_name_match") or ""
        if name_match and name_match not in smi_name:
            rejected.append((path, f"name {name_match!r} not in {smi_name!r}"))
            continue
        matching.append((path, config))
    return matching, rejected


def basenames(entries):
    return ", ".join(os.path.basename(path) for path, _ in entries)


def choose_by_power_mode(candidates, mode, smi_name):
    """Measuring at a mode no config describes would adjudicate clock drift against the wrong
    targets, so the mode must match exactly one config."""
    exact = [(path, config) for path, config in candidates
             if (config.get("required_power_mode") or "").upper() == mode.upper()]
    if len(exact) == 1:
        path = exact[0][0]
        print(f"device config: {os.path.basename(path)}  (gpu={smi_name}, power mode={mode})",
              file=sys.stderr)
        print(path)
        return 0
    if len(exact) > 1:
        print(f"several configs declare required_power_mode {mode!r}: {basenames(exact)}"
              " - set DEVICE_CFG explicitly", file=sys.stderr)
        return 1
    declared = ", ".join(f"{os.path.basename(path)}={config.get('required_power_mode')}"
                         for path, config in candidates)
    print(f"no config declares required_power_mode {mode!r}; candidates: {declared}", file=sys.stderr)
    print("  measuring at a mode no config describes would adjudicate clock drift against "
          "the wrong targets - add a config for this mode or switch modes", file=sys.stderr)
    return 1


def main():
    search = search_dirs()
    config_dir = first_dir_with_configs(search)
    if config_dir is None:
        print("no device_configs directory with a *.json found; looked in: "
              + ", ".join(directory for directory in search if directory), file=sys.stderr)
        return 1

    platform = "jetson" if os.path.exists("/etc/nv_tegra_release") else "discrete"
    smi_name = gpu_name()
    mode = current_power_mode() if platform == "jetson" else None

    candidates, rejected = matching_configs(config_dir, platform, smi_name)
    if not candidates:
        print(f"no device config matches this machine (platform={platform}, gpu={smi_name!r})",
              file=sys.stderr)
        for path, reason in rejected:
            print(f"  {os.path.basename(path)}: {reason}", file=sys.stderr)
        return 1

    if platform == "jetson" and mode:
        return choose_by_power_mode(candidates, mode, smi_name)

    if len(candidates) == 1:
        path = candidates[0][0]
        print(f"device config: {os.path.basename(path)}  (gpu={smi_name})", file=sys.stderr)
        print(path)
        return 0
    print(f"several configs match this device: {basenames(candidates)} - set DEVICE_CFG explicitly",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
