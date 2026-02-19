"""
nexus_launcher.py — Smart launcher for Nexus Shadow-Quant.

One entry point for everything:
  [1] 🚀 Trade Mode  — installs full deps, starts backend + frontend
  [2] 🧠 Train Mode  — installs training deps only, launches trainer
  [3] ❌ Exit

Premium terminal UI powered by the 'rich' library.
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════
ROOT = Path(__file__).parent.resolve()
DESKTOP = ROOT / "desktop"
BACKEND = DESKTOP / "python_backend"
TRAINING_KIT = BACKEND / "training_kit"
VENV_DIR = BACKEND / "venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
NODE_MODULES = DESKTOP / "node_modules"

# ═══════════════════════════════════════════════════════════════════
# BOOTSTRAP: ensure 'rich' is available
# ═══════════════════════════════════════════════════════════════════
def _ensure_rich():
    """Install rich if missing (needed for the premium UI)."""
    try:
        import rich  # noqa: F401
    except ImportError:
        print("\n  Installing terminal UI library (rich)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "rich", "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("  Done!\n")

_ensure_rich()

# Now safe to import rich
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.rule import Rule
from rich import box

console = Console()

# ═══════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════

BRAND_COLOR = "bright_cyan"
ACCENT = "bright_magenta"
SUCCESS = "bright_green"
WARN = "bright_yellow"
ERR = "bright_red"

# ASCII banner
BANNER = r"""
 ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
 ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
 ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
 ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
 ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
        S H A D O W - Q U A N T   v7.0.0
"""


def show_banner():
    console.print(f"\n[{BRAND_COLOR}]{BANNER}[/]", highlight=False)
    console.print(
        f"        [{ACCENT}]Jamba Hybrid SSM • Multi-Model Architecture[/]\n",
        justify="center",
    )


def show_menu():
    table = Table(
        box=box.HEAVY_EDGE,
        border_style=BRAND_COLOR,
        show_header=False,
        padding=(0, 3),
        min_width=52,
    )
    table.add_column("Option", style="bold white", width=6)
    table.add_column("Mode", style="bold white", width=20)
    table.add_column("Description", style="dim white")

    table.add_row("[1]", "🚀  Trade Mode", "Start the full dashboard")
    table.add_row("[2]", "🧠  Train Mode", "Train Jamba models (CLI)")
    table.add_row("[3]", "⚗️   Train Kit", "Web-based training UI")
    table.add_row("[4]", "❌  Exit", "Quit launcher")

    console.print(Panel(table, title=f"[bold {BRAND_COLOR}]Choose Your Mode[/]",
                        border_style=BRAND_COLOR, padding=(1, 2)))


def show_arch_menu():
    """Show architecture picker for training."""
    table = Table(box=box.ROUNDED, border_style=ACCENT, show_header=True, padding=(0, 2))
    table.add_column("#", style="bold white", width=4)
    table.add_column("Model", style="bold white", width=18)
    table.add_column("Params", style="bright_green", width=10)
    table.add_column("Notes", style="dim white")

    table.add_row("[1]", "SmallJamba", "4.4M", "Default • ultra-low VRAM")
    table.add_row("[2]", "LiteJamba ⚗️", "~12M", "Experimental • 2021-2026 only")
    table.add_row("[3]", "MediumJamba", "~28M", "Higher capacity")
    table.add_row("[4]", "LargeJamba 🔥", "~60M", "Maximum • needs 12+ GB VRAM")

    console.print(Panel(table, title=f"[bold {ACCENT}]Select Architecture[/]",
                        border_style=ACCENT, padding=(1, 2)))


# ═══════════════════════════════════════════════════════════════════
# SYSTEM CHECKS
# ═══════════════════════════════════════════════════════════════════

def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if ok:
        console.print(f"  ✅  Python {ver_str}", style=SUCCESS)
    else:
        console.print(f"  ❌  Python {ver_str} — need 3.10+", style=ERR)
    return ok


def check_node():
    node = shutil.which("node")
    if not node:
        console.print("  ❌  Node.js not found", style=ERR)
        console.print("      Install from: [link]https://nodejs.org[/link]", style="dim")
        return False
    try:
        ver = subprocess.check_output([node, "--version"], text=True).strip()
        console.print(f"  ✅  Node.js {ver}", style=SUCCESS)
        return True
    except Exception:
        console.print("  ❌  Node.js check failed", style=ERR)
        return False


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(f"  ✅  GPU: {name} ({vram:.1f} GB VRAM)", style=SUCCESS)
            return True
        else:
            console.print("  ⚠️   No CUDA GPU (will use CPU — slower)", style=WARN)
            return False
    except ImportError:
        console.print("  ⚠️   PyTorch not installed yet", style=WARN)
        return False
    except Exception as e:
        console.print(f"  ⚠️   GPU check error: {e}", style=WARN)
        return False


def run_checks(need_node=False):
    console.print(Rule(f"[{BRAND_COLOR}]System Check[/]"))
    py_ok = check_python()
    if need_node:
        node_ok = check_node()
    else:
        node_ok = True
    check_gpu()
    console.print()
    if not py_ok or not node_ok:
        console.print(f"  [{ERR}]Fix the issues above before continuing.[/]")
        input("\n  Press Enter to go back...")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
# DEPENDENCY INSTALLATION
# ═══════════════════════════════════════════════════════════════════

def run_pip_install(requirements_path: Path, label: str):
    """Install Python dependencies from a requirements file."""
    if not requirements_path.exists():
        console.print(f"  ⚠️  {requirements_path} not found — skipping", style=WARN)
        return True

    console.print(f"\n  📦  Installing {label}...", style="bold white")
    console.print(f"      ({requirements_path})", style="dim")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path), "-q"],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode == 0:
            console.print(f"  ✅  {label} installed!", style=SUCCESS)
            return True
        else:
            console.print(f"  ❌  {label} failed:", style=ERR)
            for line in proc.stderr.strip().split("\n")[-5:]:
                console.print(f"      {line}", style="dim red")
            return False
    except subprocess.TimeoutExpired:
        console.print(f"  ❌  {label} timed out (10 min)", style=ERR)
        return False


def run_npm_install():
    """Install Node.js dependencies."""
    if NODE_MODULES.exists() and any(NODE_MODULES.iterdir()):
        console.print("  ✅  node_modules already exists — skipping npm install", style=SUCCESS)
        return True

    console.print("\n  📦  Installing frontend dependencies (npm)...", style="bold white")
    try:
        proc = subprocess.run(
            ["npm", "install"],
            cwd=str(DESKTOP), capture_output=True, text=True,
            timeout=300, shell=True,
        )
        if proc.returncode == 0:
            console.print("  ✅  npm install complete!", style=SUCCESS)
            return True
        else:
            console.print("  ❌  npm install failed:", style=ERR)
            for line in proc.stderr.strip().split("\n")[-5:]:
                console.print(f"      {line}", style="dim red")
            return False
    except subprocess.TimeoutExpired:
        console.print("  ❌  npm install timed out (5 min)", style=ERR)
        return False


def setup_env():
    """Create .env file if it doesn't exist."""
    env_file = BACKEND / ".env"
    if env_file.exists():
        return
    example = ROOT / ".env.example"
    if example.exists():
        shutil.copy2(example, env_file)
        console.print("  📝  Created .env from template — edit to add API keys", style=WARN)
    else:
        env_file.write_text("OPENAI_API_KEY=your_key_here\n")
        console.print("  📝  Created blank .env — edit to add API keys", style=WARN)


# ═══════════════════════════════════════════════════════════════════
# MODE: TRADE
# ═══════════════════════════════════════════════════════════════════

def mode_trade():
    console.clear()
    show_banner()
    console.print(Panel("🚀  [bold]Trade Mode[/] — Full Dashboard",
                        border_style=BRAND_COLOR, padding=(0, 2)))

    if not run_checks(need_node=True):
        return

    # Install deps
    console.print(Rule(f"[{BRAND_COLOR}]Dependencies[/]"))
    req_file = BACKEND / "requirements.txt"
    if not req_file.exists():
        req_file = ROOT / "requirements.txt"
    run_pip_install(req_file, "Python backend dependencies")
    run_npm_install()
    setup_env()

    # Launch services
    console.print(Rule(f"[{BRAND_COLOR}]Launching[/]"))

    # Start backend
    console.print("\n  🔧  Starting Python backend (api_server.py)...", style="bold white")
    backend_proc = subprocess.Popen(
        [sys.executable, "api_server.py"],
        cwd=str(BACKEND),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    # Wait for backend
    with Progress(
        SpinnerColumn(style=BRAND_COLOR),
        TextColumn("[bold white]Waiting for backend to initialize..."),
        BarColumn(bar_width=30, style=BRAND_COLOR, complete_style=SUCCESS),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("boot", total=100)
        for i in range(100):
            time.sleep(0.08)
            progress.update(task, advance=1)

    # Start frontend
    console.print("  🖥️   Starting frontend (Vite dev server)...", style="bold white")
    frontend_proc = subprocess.Popen(
        ["npx", "vite", "--port", "5173", "--open"],
        cwd=str(DESKTOP),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        shell=True,
    )

    time.sleep(2)

    console.print()
    console.print(Panel.fit(
        "[bold bright_green]NEXUS SHADOW-QUANT IS RUNNING![/]\n\n"
        "  🖥️   Frontend:  [link]http://localhost:5173[/link]\n"
        "  🔧  Backend:   [link]http://localhost:8420[/link]\n\n"
        f"  [{WARN}]Press Enter to STOP both servers.[/]",
        border_style=SUCCESS,
        padding=(1, 3),
    ))

    input()

    # Cleanup
    console.print("\n  🛑  Shutting down...", style="bold white")
    try:
        backend_proc.terminate()
    except Exception:
        pass
    try:
        frontend_proc.terminate()
    except Exception:
        pass
    # Kill orphan processes
    os.system('taskkill /F /FI "WINDOWTITLE eq Nexus*" >nul 2>&1')
    console.print("  ✅  All servers stopped. Goodbye!\n", style=SUCCESS)


# ═══════════════════════════════════════════════════════════════════
# MODE: TRAIN (CLI)
# ═══════════════════════════════════════════════════════════════════

def mode_train():
    console.clear()
    show_banner()
    console.print(Panel("🧠  [bold]Train Mode[/] — Jamba Model Training (CLI)",
                        border_style=ACCENT, padding=(0, 2)))

    if not run_checks(need_node=False):
        return

    # Install deps
    console.print(Rule(f"[{ACCENT}]Dependencies[/]"))
    req_file = BACKEND / "requirements.txt"
    if not req_file.exists():
        req_file = ROOT / "requirements.txt"
    run_pip_install(req_file, "Training dependencies")

    # Architecture selection
    console.print(Rule(f"[{ACCENT}]Architecture[/]"))
    show_arch_menu()

    arch_map = {"1": "small", "2": "lite", "3": "medium", "4": "large"}
    choice = Prompt.ask(
        f"  [{ACCENT}]Select model[/]",
        choices=["1", "2", "3", "4"],
        default="1",
    )
    arch = arch_map[choice]
    arch_labels = {"small": "SmallJamba", "lite": "LiteJamba ⚗️", "medium": "MediumJamba", "large": "LargeJamba 🔥"}

    console.print(f"\n  Selected: [bold]{arch_labels[arch]}[/] (--arch {arch})", style=ACCENT)

    # Training options
    console.print(Rule(f"[{ACCENT}]Training Options[/]"))

    epochs = Prompt.ask(f"  [{ACCENT}]Epochs[/]", default="25")
    lr = Prompt.ask(f"  [{ACCENT}]Learning rate[/]", default="1e-4")
    stride = Prompt.ask(f"  [{ACCENT}]Window stride[/]", default="5")
    quick = Prompt.ask(f"  [{ACCENT}]Quick test mode? (100K rows, 2 epochs)[/]", choices=["y", "n"], default="n")
    skip_dl = Prompt.ask(f"  [{ACCENT}]Skip data download? (if you already have data)[/]", choices=["y", "n"], default="y")

    # Set CUDA memory allocator for reduced fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Build command
    cmd = [sys.executable, "train_mamba.py", "--arch", arch, "--epochs", epochs,
           "--lr", lr, "--stride", stride]
    if quick == "y":
        cmd.append("--quick")
    if skip_dl == "y":
        cmd.append("--skip-download")

    cmd_str = " ".join(cmd)
    console.print(f"\n  [dim]$ {cmd_str}[/]")
    console.print()

    go = Prompt.ask(f"  [{SUCCESS}]Start training?[/]", choices=["y", "n"], default="y")
    if go != "y":
        console.print("  Cancelled.\n", style="dim")
        return

    # Run training
    console.print(Rule(f"[{ACCENT}]Training[/]"))
    console.print(f"\n  🚀  Launching {arch_labels[arch]} training...\n", style="bold white")

    try:
        subprocess.run(cmd, cwd=str(BACKEND))
    except KeyboardInterrupt:
        console.print("\n  ⏹️   Training interrupted by user.", style=WARN)

    console.print()
    input("  Press Enter to go back to the menu...")


# ═══════════════════════════════════════════════════════════════════
# MODE: TRAINING KIT (Web UI)
# ═══════════════════════════════════════════════════════════════════

def mode_training_kit():
    console.clear()
    show_banner()
    console.print(Panel("⚗️   [bold]Training Kit[/] — Web-Based Training Dashboard",
                        border_style=ACCENT, padding=(0, 2)))

    if not run_checks(need_node=False):
        return

    # Install deps
    console.print(Rule(f"[{ACCENT}]Dependencies[/]"))
    tk_req = TRAINING_KIT / "requirements.txt"
    run_pip_install(tk_req, "Training Kit dependencies")

    # Launch
    console.print(Rule(f"[{ACCENT}]Launching[/]"))
    console.print("\n  🌐  Starting Training Kit on [link]http://localhost:5555[/link]", style="bold white")
    console.print("  Press Ctrl+C to stop.\n", style="dim")

    try:
        subprocess.run(
            [sys.executable, "train_server.py"],
            cwd=str(TRAINING_KIT),
        )
    except KeyboardInterrupt:
        console.print("\n  ⏹️   Training Kit stopped.", style=WARN)

    console.print()
    input("  Press Enter to go back to the menu...")


# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    while True:
        try:
            console.clear()
        except Exception:
            os.system('cls')
        show_banner()
        show_menu()

        try:
            choice = Prompt.ask(
                f"\n  [{BRAND_COLOR}]Enter choice[/]",
                choices=["1", "2", "3", "4"],
                default="1",
            )
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n  [{BRAND_COLOR}]Goodbye! 👋[/]\n")
            break

        try:
            if choice == "1":
                mode_trade()
            elif choice == "2":
                mode_train()
            elif choice == "3":
                mode_training_kit()
            elif choice == "4":
                console.print(f"\n  [{BRAND_COLOR}]Goodbye! 👋[/]\n")
                break
        except Exception as e:
            console.print(f"\n  [bright_red]Error: {e}[/]\n")
            input("  Press Enter to return to menu...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(f"\n\n  [{BRAND_COLOR}]Interrupted. Goodbye! 👋[/]\n")
