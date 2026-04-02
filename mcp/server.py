"""
EvicPress MCP Server

Lets Claude run commands on Machine A (vLLM/LMCache) and Machine B (EvicPress)
over SSH. IP addresses are set dynamically each session via set_machine_ips.

Transport: stdio (register in claude_desktop_config.json or .claude/mcp.json)
"""

import os
import yaml
import paramiko
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import FastMCP

# ──────────────────────────────────────────────────────────────────
# Server init
# ──────────────────────────────────────────────────────────────────

mcp = FastMCP("evicpress_mcp")

# In-memory state: IPs set per session
_ips: dict[str, Optional[str]] = {"machine_a": None, "machine_b": None}

# Load SSH config once at startup
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────
# SSH helper
# ──────────────────────────────────────────────────────────────────

def _ssh_run(ip: str, machine_key: str, command: str) -> dict:
    """
    Open an SSH connection, run `command`, return stdout/stderr/exit_code.
    Raises ValueError if credentials are missing.
    """
    cfg = _load_config()
    ssh_cfg = cfg["ssh"][machine_key]
    timeouts = cfg.get("timeouts", {})

    key_path = os.path.expanduser(ssh_cfg["key_path"])
    if not os.path.exists(key_path):
        raise FileNotFoundError(
            f"SSH key not found at '{key_path}'. "
            f"Update mcp/config.yaml with the correct key_path."
        )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=ip,
            port=ssh_cfg.get("port", 22),
            username=ssh_cfg["username"],
            key_filename=key_path,
            timeout=timeouts.get("connect", 10),
        )

        _, stdout, stderr = client.exec_command(
            command, timeout=timeouts.get("command", 60)
        )
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")

        return {"stdout": out, "stderr": err, "exit_code": exit_code, "ip": ip}
    finally:
        client.close()


def _format_result(result: dict) -> str:
    lines = [f"[{result['ip']}] exit_code={result['exit_code']}"]
    if result["stdout"]:
        lines.append("--- stdout ---")
        lines.append(result["stdout"].rstrip())
    if result["stderr"]:
        lines.append("--- stderr ---")
        lines.append(result["stderr"].rstrip())
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────

@mcp.tool(
    name="set_machine_ips",
    annotations={
        "title": "Set Machine IP Addresses",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def set_machine_ips(machine_a_ip: str, machine_b_ip: str) -> str:
    """
    Set the current IP addresses for Machine A and Machine B.
    Call this at the start of each session or whenever the IPs change.

    machine_a_ip: IP address of Machine A (vLLM / LMCache node).
    machine_b_ip: IP address of Machine B (EvicPress storage node).
    """
    _ips["machine_a"] = machine_a_ip.strip()
    _ips["machine_b"] = machine_b_ip.strip()
    return (
        f"IPs updated.\n"
        f"  Machine A (vLLM/LMCache): {_ips['machine_a']}\n"
        f"  Machine B (EvicPress):    {_ips['machine_b']}"
    )


@mcp.tool(
    name="get_machine_ips",
    annotations={
        "title": "Get Current Machine IPs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def get_machine_ips() -> str:
    """Return the currently configured IP addresses for both machines."""
    a = _ips["machine_a"] or "(not set)"
    b = _ips["machine_b"] or "(not set)"
    return f"Machine A (vLLM/LMCache): {a}\nMachine B (EvicPress):    {b}"


@mcp.tool(
    name="run_on_machine_a",
    annotations={
        "title": "Run Command on Machine A",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
def run_on_machine_a(command: str) -> str:
    """
    SSH into Machine A (vLLM / LMCache node) and run a shell command.
    Returns stdout, stderr, and exit code.

    Requires set_machine_ips to have been called first.
    command: The shell command to execute on Machine A.
    """
    ip = _ips["machine_a"]
    if not ip:
        return (
            "Machine A IP is not set. "
            "Call set_machine_ips(machine_a_ip=..., machine_b_ip=...) first."
        )
    try:
        result = _ssh_run(ip, "machine_a", command)
        return _format_result(result)
    except Exception as e:
        return f"Error connecting to Machine A ({ip}): {e}"


@mcp.tool(
    name="run_on_machine_b",
    annotations={
        "title": "Run Command on Machine B",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
def run_on_machine_b(command: str) -> str:
    """
    SSH into Machine B (EvicPress storage node) and run a shell command.
    Returns stdout, stderr, and exit code.

    Requires set_machine_ips to have been called first.
    command: The shell command to execute on Machine B.
    """
    ip = _ips["machine_b"]
    if not ip:
        return (
            "Machine B IP is not set. "
            "Call set_machine_ips(machine_a_ip=..., machine_b_ip=...) first."
        )
    try:
        result = _ssh_run(ip, "machine_b", command)
        return _format_result(result)
    except Exception as e:
        return f"Error connecting to Machine B ({ip}): {e}"


@mcp.tool(
    name="check_connection",
    annotations={
        "title": "Check SSH Connectivity",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
def check_connection(machine: str) -> str:
    """
    Test SSH connectivity to a machine by running `hostname && uptime`.

    machine: Either "machine_a" or "machine_b".
    """
    if machine not in ("machine_a", "machine_b"):
        return "machine must be 'machine_a' or 'machine_b'."

    ip = _ips[machine]
    if not ip:
        return f"{machine} IP is not set. Call set_machine_ips first."

    label = "Machine A (vLLM/LMCache)" if machine == "machine_a" else "Machine B (EvicPress)"
    try:
        result = _ssh_run(ip, machine, "hostname && uptime")
        return f"[{label}] connection OK\n" + _format_result(result)
    except Exception as e:
        return f"[{label}] connection FAILED ({ip}): {e}"


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
