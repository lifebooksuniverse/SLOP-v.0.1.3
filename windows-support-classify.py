#!/usr/bin/env python3
# pyright: reportMissingImports=false
import os
import json
import shutil
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

# Platform detection and terminal handling
IS_WINDOWS = os.name == 'nt'
if not IS_WINDOWS:
    print("\nThis script is Windows-only.")
    raise SystemExit("\nUse the classify-v0.0.3 script instead.\n")

import msvcrt

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
from transformers import AutoProcessor

psutil: Any
rawpy: Any

try:
    import psutil  # type: ignore[import-not-found]
except ImportError:
    psutil = None

try:
    import rawpy  # type: ignore[import-not-found]
except ImportError:
    rawpy = None

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.live import Live
from rich.prompt import Prompt, IntPrompt
from rich.text import Text

console = Console()

RAW_EXTENSIONS = (
    '.3fr', '.arw', '.bay', '.cr2', '.cr3', '.crw', '.dcr', '.dng', '.erf', '.fff',
    '.iiq', '.kdc', '.mdc', '.mef', '.mos', '.mrw', '.nef', '.nrw', '.orf', '.pef',
    '.raf', '.raw', '.rw2', '.rwl', '.sr2', '.srf', '.srw', '.x3f'
)
STANDARD_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff')
SUPPORTED_EXTENSIONS = STANDARD_EXTENSIONS + RAW_EXTENSIONS
ERROR_FOLDER_NAME = "error"


def get_nvidia_smi_path() -> Optional[str]:
    cmd = shutil.which("nvidia-smi")
    if cmd:
        return cmd
    if IS_WINDOWS:
        possible_paths = [
            Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "nvidia-smi.exe",
            Path("C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe")
        ]
        for p in possible_paths:
            if p.exists():
                return str(p)
    return None


def has_nvidia_gpu() -> bool:
    smi_path = get_nvidia_smi_path()
    if not smi_path:
        return False
    try:
        result = subprocess.run([smi_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0 and "NVIDIA" in result.stdout
    except Exception:
        return False


def check_onnx_gpu_installed() -> bool:
    return "CUDAExecutionProvider" in ort.get_available_providers()


def load_image(img_path: Path) -> Optional[Image.Image]:
    try:
        ext = img_path.suffix.lower()
        if ext in RAW_EXTENSIONS:
            if rawpy is None:
                return None
            with rawpy.imread(str(img_path)) as raw:
                try:
                    rgb_array = raw.postprocess(
                        half_size=True,
                        use_camera_wb=True,
                        no_auto_bright=True,
                        output_color=rawpy.ColorSpace.sRGB,
                        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    )
                except Exception:
                    return None
                if rgb_array is None or rgb_array.size == 0:
                    return None
                return Image.fromarray(np.asarray(rgb_array)).convert("RGB")

        with Image.open(img_path) as image:
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        return None


def move_to_error_folder(src_path: Path, output_dir: Path) -> Path:
    error_dir = output_dir / ERROR_FOLDER_NAME
    error_dir.mkdir(parents=True, exist_ok=True)
    dest_path = error_dir / src_path.name
    counter = 1
    while dest_path.exists():
        dest_path = error_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
        counter += 1
    shutil.move(str(src_path), str(dest_path))
    return dest_path


def move_to_not_photos_folder(src_path: Path, output_dir: Path) -> Path:
    not_photos_dir = output_dir / "not_photos"
    not_photos_dir.mkdir(parents=True, exist_ok=True)
    dest_path = not_photos_dir / src_path.name
    counter = 1
    while dest_path.exists():
        dest_path = not_photos_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
        counter += 1
    shutil.move(str(src_path), str(dest_path))
    return dest_path


def safe_home_dir() -> Path:
    return Path.home().expanduser().resolve()


def is_safe_under_home(path: Path) -> bool:
    home_dir = safe_home_dir()
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        return False
    try:
        resolved.relative_to(home_dir)
        return True
    except ValueError:
        return False


def get_key_nonblocking() -> str:
    if not msvcrt.kbhit():
        return ""

    ch = msvcrt.getch()
    if ch in (b'\x00', b'\xe0'):
        msvcrt.getch()
        return ""
    try:
        return ch.decode('utf-8', errors='ignore')
    except Exception:
        return ""


def get_key_blocking() -> str:
    ch = msvcrt.getch()
    if ch in (b'\x00', b'\xe0'):
        msvcrt.getch()
        return ""
    try:
        return ch.decode('utf-8', errors='ignore')
    except Exception:
        return ""


def read_key_sequence() -> str:
    ch = msvcrt.getch()
    if ch in (b'\x00', b'\xe0'):
        sc = msvcrt.getch()
        if sc == b'H':
            return '\x1b[A'  # UP
        elif sc == b'P':
            return '\x1b[B'  # DOWN
        elif sc == b'M':
            return '\x1b[C'  # RIGHT
        elif sc == b'K':
            return '\x1b[D'  # LEFT
        return ""
    if ch == b'\x1b':
        return '\x1b'
    try:
        return ch.decode('utf-8', errors='ignore')
    except Exception:
        return ""


def load_classifications_log(log_path: Path) -> Dict[str, str]:
    classified = {}
    if not log_path.exists():
        return classified
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                classified[parts[1]] = parts[2]
    return classified


def append_classification_log(log_path: Path, idx: int, filename: str, category: str, accuracy: Optional[float] = None):
    file_exists = log_path.exists()
    with open(log_path, "a", encoding="utf-8") as f:
        if not file_exists or log_path.stat().st_size == 0:
            f.write("# | file name | category | accuracy |\n")
        if accuracy is None:
            f.write(f"{idx} | {filename} | {category}\n")
        else:
            f.write(f"{idx} | {filename} | {category} | {accuracy:.6f}\n")


def clear_classification_log(log_path: Path):
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")


def parse_category_prompts(raw_input: str) -> List[str]:
    prompts = []
    for part in raw_input.replace("\n", ",").split(","):
        cleaned = part.strip()
        if cleaned:
            prompts.append(cleaned)
    return prompts


def select_directory(start_dir: Path, title: str = "Select destination folder") -> Optional[Path]:
    home_dir = safe_home_dir()
    base_dir = start_dir.expanduser() if start_dir.exists() else home_dir
    if not base_dir.exists() or not is_safe_under_home(base_dir):
        base_dir = home_dir

    current_dir = base_dir.resolve()
    selection_index = 0
    while True:
        if not current_dir.exists() or not is_safe_under_home(current_dir):
            current_dir = home_dir

        folder_entries: List[Tuple[str, Path]] = []
        if current_dir != current_dir.parent and is_safe_under_home(current_dir.parent):
            folder_entries.append(("..", current_dir.parent))
        folder_entries.extend(sorted(
            [(p.name, p) for p in current_dir.iterdir() if p.is_dir() and not p.name.startswith(".") and is_safe_under_home(p)],
            key=lambda item: item[0].lower(),
        ))

        if not folder_entries:
            folder_entries = [(".", current_dir)]

        if selection_index >= len(folder_entries):
            selection_index = len(folder_entries) - 1
        if selection_index < 0:
            selection_index = 0

        selected_name, selected_path = folder_entries[selection_index]

        console.clear()
        console.print(Panel(
            f"[bold cyan]{title}[/bold cyan]\n[dim]Current:[/dim] [bold]{current_dir}[/bold]",
            border_style="cyan"
        ))

        lines = []
        for idx, (name, path) in enumerate(folder_entries):
            if idx == selection_index:
                lines.append(f"[bold green]> {name}[/bold green]")
            else:
                lines.append(f"[white]  {name}[/white]")

        console.print(Panel("\n".join(lines), title="Folders", border_style="cyan"))
        console.print(f"\n[bold yellow]Selected:[/bold yellow] {selected_path}")
        console.print("[dim]Arrow keys move • Right enters • Left goes up • Enter chooses • '/' jumps to a custom path • Esc cancels[/dim]")

        key = read_key_sequence()

        if key in ("\x1b", "\x03"):
            return None

        if key == '/':
            raw_path = Prompt.ask("Enter an absolute or relative folder path to open", default=str(current_dir), show_default=True)
            if not raw_path.strip():
                continue
            candidate = Path(raw_path.strip()).expanduser()
            if not is_safe_under_home(candidate):
                console.print("[red]Only folders inside your home directory are allowed.[/red]")
                Prompt.ask("Press Enter to continue")
                continue
            if not candidate.exists():
                if Prompt.ask(f"Directory '{candidate}' does not exist. Create it?", choices=["y", "n"], default="y") == "y":
                    candidate.mkdir(parents=True, exist_ok=True)
                else:
                    console.print("[yellow]Path not used.[/yellow]")
                    Prompt.ask("Press Enter to continue")
                    continue
            if candidate.exists() and candidate.is_dir():
                return candidate
            console.print("[red]That path is not a directory.[/red]")
            Prompt.ask("Press Enter to continue")
            continue

        if key in ('\r', '\n'):
            if selected_name == "..":
                return current_dir.parent if current_dir.parent.exists() else current_dir
            return selected_path

        if key == '\x1b[A':
            selection_index = max(0, selection_index - 1)
        elif key == '\x1b[B':
            selection_index = min(len(folder_entries) - 1, selection_index + 1)
        elif key == '\x1b[C':
            if selected_name == "..":
                current_dir = current_dir.parent if current_dir.parent.exists() else current_dir
                selection_index = 0
            else:
                current_dir = selected_path
                selection_index = 0
        elif key == '\x1b[D':
            if current_dir != current_dir.parent:
                current_dir = current_dir.parent
                selection_index = 0


def load_config(config_path: Path) -> Dict[str, Any]:
    default_categories = [
        "a photo of a sunset",
        "a portrait of a person",
        "a document or receipt",
        "a photo of a cat or dog",
        "a landscape nature scene",
        "a wild bird or animal",
        "A sepia photo",
        "A black and white photo with nature",
    ]

    default_config: Dict[str, Any] = {
        "categories": default_categories,
        "source_dir": str(Path.home() / "photos"),
        "output_dir": str(Path.home() / "photos" / "sorted"),
        "use_gpu": False,
        "threads": os.cpu_count() or 4,
        "batch_size": 4,
    }

    if not config_path.exists():
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            return default_config

        categories = loaded.get("categories")
        if not isinstance(categories, list) or not categories:
            categories = default_categories

        source_dir = loaded.get("source_dir")
        if not source_dir:
            source_dir = str(Path.home() / "photos")

        source_path = Path(str(source_dir)).expanduser()
        if not is_safe_under_home(source_path):
            source_path = safe_home_dir() / "photos"

        output_dir = loaded.get("output_dir")
        if not output_dir:
            output_dir = str(source_path / "sorted")
        else:
            output_path = Path(str(output_dir)).expanduser()
            if not is_safe_under_home(output_path):
                output_dir = str(source_path / "sorted")

        return {
            "categories": [str(item).strip() for item in categories if str(item).strip()],
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "use_gpu": bool(loaded.get("use_gpu", False)),
            "threads": max(1, int(loaded.get("threads", os.cpu_count() or 4))),
            "batch_size": max(1, int(loaded.get("batch_size", 4))),
        }
    except Exception:
        return default_config


def save_config(config_path: Path, categories: List[str], source_dir: Path, output_dir: Path, use_gpu: bool, threads: int, batch_size: int):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    safe_source = source_dir.expanduser().resolve()
    safe_output = output_dir.expanduser().resolve()
    if not is_safe_under_home(safe_source):
        safe_source = safe_home_dir() / "photos"
    if not is_safe_under_home(safe_output):
        safe_output = safe_source / "sorted"
    payload = {
        "categories": categories,
        "source_dir": str(safe_source),
        "output_dir": str(safe_output),
        "use_gpu": use_gpu,
        "threads": max(1, threads),
        "batch_size": max(1, batch_size),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def prompt_for_source_dir(source_dir: Path) -> Path:
    source_choice = select_directory(source_dir, "Select input photo folder")
    if source_choice is not None:
        source_dir = source_choice

    source_dir = source_dir.expanduser().resolve()
    if not is_safe_under_home(source_dir):
        source_dir = safe_home_dir() / "photos"

    if not source_dir.exists():
        if Prompt.ask(f"Directory '{source_dir}' does not exist. Create it?", choices=["y", "n"], default="y") == "y":
            source_dir.mkdir(parents=True, exist_ok=True)

    return source_dir


def derive_output_dir(source_dir: Path) -> Path:
    source_dir = source_dir.expanduser().resolve()
    if not is_safe_under_home(source_dir):
        source_dir = safe_home_dir() / "photos"
    return source_dir / "sorted"


def get_vram_usage_mb() -> Tuple[float, float]:
    smi_path = get_nvidia_smi_path()
    if not smi_path:
        return 0.0, 0.0

    try:
        result = subprocess.run(
            [smi_path, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0, 0.0

        first_line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 2:
            return 0.0, 0.0

        used_mb = float(parts[0])
        total_mb = float(parts[1])
        return used_mb, total_mb
    except Exception:
        return 0.0, 0.0


def make_usage_bar(used: float, total: float, width: int = 20) -> str:
    if total <= 0:
        return "░" * width

    pct = max(0.0, min(1.0, used / total))
    filled = max(0, min(width, int(round(pct * width))))
    return ("█" * filled) + ("░" * (width - filled))


def normalize_category_folder_name(category: str) -> str:
    cleaned = category.strip()
    cleaned = cleaned.replace("a photo of a ", "")
    cleaned = cleaned.replace("a ", "")
    cleaned = cleaned.replace(" or ", "_")
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "uncategorized"


def summarize_sort_results(folder_counts: Dict[str, int], elapsed_seconds: float, avg_confidence: float, broken_files: int = 0) -> str:
    lines: List[str] = [
        f"Time taken: {elapsed_seconds:.2f} seconds",
        f"Average confidence: {avg_confidence:.2%}",
    ]

    if broken_files > 0:
        lines.append(f"{broken_files} broken images were moved to error folder.")

    for folder_name in sorted(folder_counts.keys()):
        count = folder_counts[folder_name]
        if count > 0:
            lines.append(f"{count} photos were moved to {folder_name} folder.")

    return "\n".join(lines)


def fast_resort_from_log(source_dir: Path, output_dir: Path, log_path: Path):
    classified_map = load_classifications_log(log_path)
    if not classified_map:
        console.print("[bold red]No records found in classifications.txt to fast-sort from![/bold red]")
        Prompt.ask("Press Enter to return")
        return

    console.clear()
    console.print(Panel(
        f"[bold cyan]S.L.O.P Fast Re-Sort Mode[/bold cyan]\n"
        f"Loaded [green]{len(classified_map)}[/green] entries from ledger log.\n"
        f"Source: {source_dir} | Destination: {output_dir}",
        border_style="cyan"
    ))
    
    action_type = Prompt.ask("Choose disk operation", choices=["copy", "move"], default="copy")
    confirm1 = Prompt.ask(f"[bold red]Are you sure you want to batch {action_type} files using log data?[/bold red]", choices=["y", "n"], default="n")
    if confirm1 != "y":
        return

    console.print("\n[cyan]Executing fast batch operation...[/cyan]")
    success_count = 0
    missing_count = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(bar_width=40), TaskProgressColumn()) as progress:
        task = progress.add_task("[green]Processing files...", total=len(classified_map))
        
        for filename, matched_cat in classified_map.items():
            src_file = source_dir / filename
            if not src_file.exists():
                missing_count += 1
                progress.advance(task)
                continue

            clean_folder_name = matched_cat.replace("a photo of a ", "").replace("a ", "").replace(" or ", "_").replace(" ", "_")
            dest_dir = output_dir / clean_folder_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / filename

            if not dest_file.exists():
                if action_type == "copy":
                    shutil.copy(src_file, dest_file)
                else:
                    shutil.move(src_file, dest_file)
            success_count += 1
            progress.advance(task)

    console.print(f"\n[bold green]Fast Re-Sort Complete![/bold green] Successfully processed [cyan]{success_count}[/cyan] files. (Skipped missing source files: {missing_count})")
    Prompt.ask("Press Enter to return to main menu")


def redo(source_dir: Path, output_dir: Path, log_path: Path) -> int:
    if not output_dir.exists():
        return 0

    moved_count = 0
    for file_path in sorted(output_dir.rglob("*"), key=lambda p: str(p)):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            dest_path = source_dir / file_path.name
            counter = 1
            while dest_path.exists():
                dest_path = source_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
                counter += 1
            shutil.move(str(file_path), str(dest_path))
            moved_count += 1

    for root, dirs, files in os.walk(output_dir, topdown=False):
        current_dir = Path(root)
        try:
            current_dir.rmdir()
        except OSError:
            pass

    clear_classification_log(log_path)
    return moved_count


def low_confwarn(avg_confidence: float, processed_count: int):
    if processed_count == 0:
        return

    if avg_confidence <= 0.35:
        console.print(Panel(
            f"[bold yellow]Low confidence warning[/bold yellow]\n"
            f"Average confidence across processed photos: [bold red]{avg_confidence:.2%}[/bold red]\n"
            f"This is below the recommended 35% threshold.\n\n"
            f"What this usually means:\n"
            f"• Your categories may be [cyan]too broad[/cyan] or [cyan]too specific[/cyan]\n"
            f"• Some prompts may overlap and confuse the model\n"
            f"• The model may be picking a weaker match because the labels don't fit the images well\n\n"
            f"Helpful fixes:\n"
            f"• Merge very similar categories into one label\n"
            f"• Remove prompts that are overly niche or redundant\n"
            f"• Use more obvious, concrete labels like 'a cat or dog' instead of several near-duplicates\n\n"
            f"SigLIP scores are relative across the prompt set, so a low score usually means the category list needs tuning rather than a crash or software bug.",
            border_style="yellow"
        ))


class photosorter:
    def __init__(self, model_dir: str, use_gpu: bool = False, threads: int = 4):
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        
        opts = ort.SessionOptions()
        if not use_gpu:
            opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        
        self.session = ort.InferenceSession(
            str(Path(model_dir) / "model.onnx"),
            opts,
            providers=providers
        )
        
    @staticmethod
    def _normalize_probs(logits: np.ndarray) -> Tuple[np.ndarray, int, float]:
        logits = np.asarray(logits, dtype=np.float32)
        if logits.size == 0:
            return logits, -1, 0.0

        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits)
        best_idx = int(np.argmax(probs))
        return probs, best_idx, float(probs[best_idx])

    def classify_batch(self, img_paths: List[Path], categories: List[str]) -> List[Tuple[Path, str, float, str]]:
        images = []
        valid_paths = []

        for p in img_paths:
            img = load_image(p)
            if img is not None:
                images.append(img)
                valid_paths.append(p)

        if not images:
            return [(p, "ERROR", 0.0, "error") for p in img_paths]

        tokenized = self.processor(
            text=categories,
            images=images,
            padding="max_length",
            max_length=64,
            return_tensors="np"
        )

        pixel_values = np.asarray(tokenized["pixel_values"], dtype=np.float32)
        input_ids = np.asarray(tokenized["input_ids"], dtype=np.int64)
        ort_inputs: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

        outputs = self.session.run(None, ort_inputs)
        logits = np.asarray(outputs[0], dtype=np.float32)

        if logits.ndim == 1:
            logits = np.expand_dims(logits, axis=0)

        valid_results = {}
        for i, path in enumerate(valid_paths):
            img_logits = logits[i]
            probs, best_idx, confidence = self._normalize_probs(img_logits)
            valid_results[path] = (path, categories[best_idx], confidence, "sorted")

        results = []
        for p in img_paths:
            if p in valid_results:
                results.append(valid_results[p])
            else:
                results.append((p, "ERROR", 0.0, "error"))

        return results


class SLOPTUI:
    def __init__(self, categories: List[str], total_files: int, initial_sorted: int, use_gpu: bool, threads: int, batch_size: int):
        self.categories = categories
        self.total_files = total_files
        self.use_gpu = use_gpu
        self.threads = threads
        self.batch_size = batch_size
        self.sorted_count = initial_sorted
        self.status_msg = "Sorting"
        self.history = []
        self.process: Any = psutil.Process(os.getpid()) if psutil is not None else None
        
    def make_header(self) -> Panel:
        if self.process is not None:
            mem_info = self.process.memory_info()
            ram_used_mb = mem_info.rss / (1024 * 1024)
        else:
            ram_used_mb = 0.0

        if psutil is not None:
            sys_ram: Any = psutil.virtual_memory()
            ram_total_mb = sys_ram.total / (1024 * 1024)
            ram_pct = sys_ram.percent
        else:
            ram_total_mb = 0.0
            ram_pct = 0.0

        ram_bar = make_usage_bar(ram_used_mb, ram_total_mb)

        vram_used_mb, vram_total_mb = get_vram_usage_mb()
        vram_bar = "N/A"
        if vram_total_mb > 0:
            vram_bar = make_usage_bar(vram_used_mb, vram_total_mb)

        device_str = "[bold green]GPU (CUDA)[/bold green]" if self.use_gpu else f"[bold yellow]CPU ({self.threads} Threads)[/bold yellow]"

        stats_text = (
            f"[bold cyan]S.L.O.P v0.1.3[/bold cyan] [dim]| Simple Local Organizer for Photos[/dim]\n"
            f"[dim font]Device: {device_str} | Batch: {self.batch_size} | "
            f"RAM {ram_bar} [green]{ram_used_mb:.1f} MB[/green]/[yellow]{ram_total_mb:.1f} MB[/yellow] ({ram_pct:.0f}%) | "
            f"VRAM {vram_bar} [green]{vram_used_mb:.1f} MB[/green]/[yellow]{vram_total_mb:.1f} MB[/yellow] | "
            f"Progress: [bold]{self.sorted_count}/{self.total_files}[/bold] Sorted | State: [bold magenta]{self.status_msg}[/bold magenta][/dim font]\n"
            f"[dim font underline]Hotkeys:[/dim font underline] [bold white]\\[p][/bold white] Pause/Resume  |  [bold white]\\[q][/bold white] Stop Sorting"
        )
        return Panel(stats_text, border_style="cyan")
        
    def make_table(self) -> Table:
        table = Table(title="Recent Classifications", expand=True)
        table.add_column("File", style="white", overflow="ellipsis")
        table.add_column("Matched Category", style="magenta")
        table.add_column("Confidence", style="green")
        table.add_column("Status", style="yellow")
        
        for item in self.history[-6:]:
            status_style = "red" if item['status'] == "error" else "yellow"
            table.add_row(
                item['file'],
                item['category'],
                f"{item['conf']:.2%}",
                Text(item['status'], style=status_style)
            )
        return table

    def build_layout(self, progress: Progress) -> Layout:
        layout = Layout()
        layout.split(
            Layout(self.make_header(), name="header", size=5),
            Layout(progress, name="progress", size=3),
            Layout(self.make_table(), name="table")
        )
        return layout


def edit_config_menu(categories: List[str], use_gpu: bool, threads: int, batch_size: int, source_dir: Path, output_dir: Path, log_path: Path, config_path: Path) -> Tuple[List[str], bool, int, int, Path, Path, Path]:
    max_cpus = os.cpu_count() or 4
    gpu_available = has_nvidia_gpu()
    gpu_runtime_installed = check_onnx_gpu_installed()
    
    while True:
        console.clear()
        console.print(Panel("[bold cyan]S.L.O.P Configuration Editor[/bold cyan]", border_style="cyan"))
        
        if use_gpu and not gpu_available:
            use_gpu = False
            
        console.print("\n[bold white]Target Categories:[/bold white]")
        for idx, cat in enumerate(categories, 1):
            console.print(f"  [cyan]{idx}.[/cyan] {cat}")
            
        console.print("\n[bold white]Hardware Execution Settings:[/bold white]")
        device_label = "[green]GPU (CUDA)[/green]" if use_gpu else f"[yellow]CPU ({threads}/{max_cpus} Threads)[/yellow]"
        console.print(f"  [bold]Current Device:[/bold] {device_label}")
        console.print(f"  [bold]Batch Size:[/bold] {batch_size}")
        console.print(f"  [bold]Source Folder:[/bold] {source_dir}")
        console.print(f"  [bold]Output Folder:[/bold] {output_dir}")
        console.print(f"  [bold]Classification Log:[/bold] {log_path}")
        
        console.print("\n[dim]Options:[/dim]")
        console.print("  [bold green]1[/bold green] Add a category prompt")
        console.print("  [bold red]2[/bold red] Remove a category prompt")
        console.print("  [bold yellow]3[/bold yellow] Clear all categories")
        console.print("  [bold magenta]4[/bold magenta] Toggle CPU/GPU Mode")
        console.print("  [bold cyan]5[/bold cyan] Adjust Batch Size")
        if not use_gpu:
            console.print("  [bold blue]6[/bold blue] Adjust CPU Thread Count")
        console.print("  [bold white]7[/bold white] Clear classification log")
        console.print("  [bold blue]8[/bold blue] Change source photo directory")
        console.print("  [bold white]9[/bold white] Done editing and return")

        valid_choices = ["1", "2", "3", "4", "5", "7", "8", "9"]
        if not use_gpu:
            valid_choices.insert(5, "6")

        choice = Prompt.ask("\nSelect action", choices=valid_choices, default="9")

        if choice == "1":
            new_cats = parse_category_prompts(Prompt.ask("Enter new category prompt(s), separated by commas"))
            if not new_cats:
                continue
            for new_cat in new_cats:
                if new_cat and new_cat.lower() not in {existing.lower() for existing in categories}:
                    categories.append(new_cat)
        elif choice == "2":
            if not categories:
                console.print("[bold red]There are no categories to remove.[/bold red]")
                Prompt.ask("Press Enter to continue")
                continue
            for idx, cat in enumerate(categories, 1):
                console.print(f"  [cyan]{idx}.[/cyan] {cat}")

            raw_remove = Prompt.ask("Enter category number(s) to remove, separated by commas")
            remove_indexes = []
            for token in raw_remove.replace("\n", ",").split(","):
                try:
                    idx = int(token.strip())
                    if 1 <= idx <= len(categories):
                        remove_indexes.append(idx)
                except ValueError:
                    continue

            if not remove_indexes:
                console.print("[yellow]No valid category numbers were entered.[/yellow]")
                Prompt.ask("Press Enter to continue")
                continue

            removed_items = []
            for idx in sorted(set(remove_indexes), reverse=True):
                removed_items.append(categories.pop(idx - 1))

            console.print(f"[yellow]Removed:[/yellow] {', '.join(removed_items)}")
            Prompt.ask("Press Enter to continue")
        elif choice == "3":
            if Prompt.ask("Clear all categories?", choices=["y", "n"], default="n") == "y":
                categories.clear()
        elif choice == "4":
            if gpu_available and gpu_runtime_installed:
                use_gpu = not use_gpu
            else:
                console.print("[bold red]NVIDIA GPU or onnxruntime-gpu runtime not detected![/bold red]")
                Prompt.ask("Press Enter to continue")
        elif choice == "5":
            if use_gpu:
                console.print("[bold green]Recommended values for GPU: 16-64[/bold green] | Memory is the main factor for GPU batches. |")
            else:
                console.print("[bold yellow]Recommended values for CPU: 1-8[/bold yellow] | For CPU, smaller batches are safer and more stable. |")
            batch_size = IntPrompt.ask("Enter batch size (e.g., 1, 4, 8, 16)", default=batch_size)
            batch_size = max(1, batch_size)
        elif choice == "6":
            threads = IntPrompt.ask(f"Enter thread count (1 - {max_cpus})", default=threads)
            threads = max(1, min(max_cpus, threads))
        elif choice == "7":
            if Prompt.ask("Clear the classification log file?", choices=["y", "n"], default="n") == "y":
                clear_classification_log(log_path)
                console.print(f"[yellow]Cleared:[/yellow] {log_path}")
                Prompt.ask("Press Enter to continue")
        elif choice == "8":
            new_source = select_directory(source_dir, "Select photo source directory")
            if new_source is None:
                console.print("[yellow]Source directory unchanged.[/yellow]")
                Prompt.ask("Press Enter to continue")
                continue
            if not new_source.exists():
                if Prompt.ask(f"Directory '{new_source}' does not exist. Create it?", choices=["y", "n"], default="y") == "y":
                    new_source.mkdir(parents=True, exist_ok=True)
                else:
                    console.print("[yellow]Source directory unchanged.[/yellow]")
                    Prompt.ask("Press Enter to continue")
                    continue
            source_dir = new_source
            console.print(f"[green]Source photo directory set to:[/green] {source_dir}")
            Prompt.ask("Press Enter to continue")
        elif choice == "9":
            if not categories:
                console.print("[bold red]You must have at least one category to proceed![/bold red]")
                Prompt.ask("Press Enter to continue")
                continue
            output_dir = derive_output_dir(source_dir)
            save_config(config_path, categories, source_dir, output_dir, use_gpu, threads, batch_size)
            break

    return categories, use_gpu, threads, batch_size, source_dir, output_dir, log_path


CONFIG_FILE = Path("./config.txt")


def main():
    MODEL_DIR = "./siglip_onnx"
    saved_config = load_config(CONFIG_FILE)
    categories = saved_config["categories"]
    source_dir = Path(saved_config["source_dir"]).expanduser()
    output_dir = Path(saved_config["output_dir"]).expanduser()
    use_gpu = bool(saved_config["use_gpu"])
    threads = max(1, int(saved_config["threads"]))
    batch_size = max(1, int(saved_config["batch_size"]))
    log_file = output_dir / "classifications.txt"

    if not is_safe_under_home(source_dir):
        source_dir = safe_home_dir() / "photos"
    if not source_dir.exists():
        source_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold yellow]Created '{source_dir}'. Drop photos in there and re-run.[/bold yellow]")
        save_config(CONFIG_FILE, categories, source_dir, output_dir, use_gpu, threads, batch_size)
        return

    source_dir = prompt_for_source_dir(source_dir)
    output_dir = derive_output_dir(source_dir)
    OUTPUT_DIR = output_dir
    log_file = OUTPUT_DIR / "classifications.txt"
    save_config(CONFIG_FILE, categories, source_dir, output_dir, use_gpu, threads, batch_size)

    image_files = [p for p in source_dir.iterdir() if not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    image_files.sort(key=lambda p: p.name)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    not_photo_count = 0
    for entry in sorted(source_dir.iterdir(), key=lambda p: p.name):
        if entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in SUPPORTED_EXTENSIONS:
            move_to_not_photos_folder(entry, OUTPUT_DIR)
            not_photo_count += 1

    image_files = [p for p in source_dir.iterdir() if not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    image_files.sort(key=lambda p: p.name)

    if not_photo_count > 0:
        console.print(f"[yellow]Moved {not_photo_count} non-photo file(s) to [/yellow][bold cyan]{OUTPUT_DIR / 'not_photos'}[/bold cyan]")
        Prompt.ask("Press Enter to continue")

    while True:
        existing_log = load_classifications_log(log_file)
        already_sorted_count = len(existing_log)
        unprocessed_files = [p for p in image_files if p.name not in existing_log]

        console.clear()
        cat_table = Table(title="Target Categories", show_header=False, box=None)
        for cat in categories:
            cat_table.add_row(f" [magenta]•[/magenta] {cat}")

        hw_info = "[green]GPU (CUDA)[/green]" if use_gpu else f"[yellow]CPU ({threads} Threads)[/yellow]"
        resume_msg = f" (Resuming from file #{already_sorted_count + 1})" if already_sorted_count > 0 else ""

        console.print(Panel(
            "[bold cyan]S.L.O.P v0.1.3[/bold cyan] [dim]| Simple Local Organizer for Photos[/dim]\n"
            f"Found [bold green]{len(image_files)}[/bold green] total image(s) in [bold cyan]{source_dir}[/bold cyan]\n"
            f"Already Sorted in Log: [yellow]{already_sorted_count}[/yellow] | Remaining Queue: [green]{len(unprocessed_files)}[/green]{resume_msg}\n"
            f"Hardware: {hw_info} | Batch Size: {batch_size}",
            border_style="cyan"
        ))
        console.print(cat_table)
        console.print(
            "\n[bold white]Actions:[/bold white]\n"
            "  [green]Enter[/green] -> Start sorting\n"
            "  [yellow]'e'[/yellow]   -> Edit config\n"
            "  [magenta]'m'[/magenta]   -> Fast re-sort/layout from existing classifications.txt\n"
            "  [cyan]'r'[/cyan]   -> Refresh source folder\n"
            "  [bold yellow]'d'[/bold yellow]   -> Redo sorting: pull all renamed photos back into the source folder\n"
            "  [bold red]'q'[/bold red]   -> Quit"
        )

        key = get_key_blocking()
        if key in ('\r', '\n'):
            if not unprocessed_files:
                console.print("[bold green]All images in folder have already been classified![/bold green]")
                Prompt.ask("Press Enter to return to the main menu")
                continue

            console.print("\n[dim]Starting the model[/dim]")
            engine = photosorter(MODEL_DIR, use_gpu=use_gpu, threads=threads)
            tui = SLOPTUI(categories, total_files=len(image_files), initial_sorted=already_sorted_count, use_gpu=use_gpu, threads=threads, batch_size=batch_size)

            session_confidence_total = 0.0
            session_confidence_count = 0
            session_images_processed = 0
            category_move_counts: Dict[str, int] = {}
            sort_started_at = time.perf_counter()

            progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(bar_width=40), TaskProgressColumn())
            task_id = progress.add_task("[bold cyan]Sorting photos...", total=len(image_files), completed=already_sorted_count)

            try:
                with Live(tui.build_layout(progress), refresh_per_second=8) as live:
                    for i in range(0, len(unprocessed_files), batch_size):
                        chunk = unprocessed_files[i:i+batch_size]

                        key = get_key_nonblocking()
                        if key.lower() == 'q':
                            tui.status_msg = "CONFIRM EXIT"
                            live.update(tui.build_layout(progress))
                            live.stop()
                            console.print()
                            if Prompt.ask("[bold red]Are you sure you want to stop sorting and exit?[/bold red]", choices=["y", "n"], default="n") == "y":
                                if Prompt.ask("[bold red]Double Check: Exit completely? (Progress is saved)[/bold red]", choices=["y", "n"], default="n") == "y":
                                    console.print("[yellow]Operation terminated by user.[/yellow]")
                                    return
                            tui.status_msg = "Sorting"
                            live.start()

                        if key.lower() == 'p':
                            tui.status_msg = "PAUSED"
                            live.update(tui.build_layout(progress))
                            while True:
                                pk = get_key_nonblocking()
                                if pk.lower() == 'p':
                                    tui.status_msg = "Sorting"
                                    break
                                elif pk.lower() == 'q':
                                    key = 'q'
                                    break
                                time.sleep(0.1)
                            if key.lower() == 'q':
                                continue

                        results = engine.classify_batch(chunk, categories)

                        for valid_path, matched_cat, conf, status in results:
                            session_images_processed += 1
                            if status == "error":
                                error_dest = move_to_error_folder(valid_path, OUTPUT_DIR)
                                append_classification_log(log_file, already_sorted_count + 1, valid_path.name, "ERROR", 0.0)
                                already_sorted_count += 1
                                tui.sorted_count += 1

                                tui.history.append({
                                    'file': valid_path.name,
                                    'category': "ERROR",
                                    'conf': 0.0,
                                    'status': 'error'
                                })
                                tui.history = tui.history[-10:]

                                progress.update(task_id, advance=1, description=f"[red]Error: {error_dest.name}")
                                live.update(tui.build_layout(progress))
                                continue

                            actual_category = matched_cat if conf >= 0.15 else "low_confidence"
                            clean_folder_name = normalize_category_folder_name(actual_category)
                            dest_dir = OUTPUT_DIR / clean_folder_name
                            dest_dir.mkdir(parents=True, exist_ok=True)

                            dest_path = dest_dir / valid_path.name
                            if dest_path.exists():
                                counter = 1
                                while dest_path.exists():
                                    dest_path = dest_dir / f"{valid_path.stem}_{counter}{valid_path.suffix}"
                                    counter += 1

                            shutil.move(str(valid_path), str(dest_path))
                            category_move_counts[clean_folder_name] = category_move_counts.get(clean_folder_name, 0) + 1

                            append_classification_log(log_file, already_sorted_count + 1, dest_path.name, actual_category, conf)
                            already_sorted_count += 1
                            tui.sorted_count += 1
                            session_confidence_total += conf
                            session_confidence_count += 1

                            tui.history.append({
                                'file': valid_path.name,
                                'category': actual_category,
                                'conf': conf,
                                'status': 'sorted'
                            })
                            tui.history = tui.history[-10:]

                            progress.update(task_id, advance=1, description=f"[cyan]Processing: {valid_path.name}")
                            live.update(tui.build_layout(progress))

                    elapsed_seconds = time.perf_counter() - sort_started_at if session_images_processed > 0 else 0.0

                    if session_confidence_count > 0:
                        avg_confidence = session_confidence_total / session_confidence_count
                        low_confwarn(avg_confidence, session_confidence_count)
                    else:
                        avg_confidence = 0.0

                    broken_files = 0
                    if (OUTPUT_DIR / ERROR_FOLDER_NAME).exists():
                        for error_file in (OUTPUT_DIR / ERROR_FOLDER_NAME).rglob("*"):
                            if error_file.is_file():
                                broken_files += 1

                    total_processed = session_images_processed
                    total_sorted = session_confidence_count
                    summary_lines = summarize_sort_results(category_move_counts, elapsed_seconds, avg_confidence, broken_files)
                    console.print(Panel(
                        f"[bold cyan]Sorting complete[/bold cyan]\n"
                        f"[green]Processed:[/green] {total_processed} image(s)\n"
                        f"[green]Sorted successfully:[/green] {total_sorted} image(s)\n"
                        f"[green]Errors:[/green] {broken_files} image(s)\n\n"
                        f"{summary_lines}",
                        border_style="cyan"
                    ))

                    if broken_files > 0:
                        console.print(f"[bold red]Broken image(s):[/bold red] {broken_files} file(s) in {OUTPUT_DIR / ERROR_FOLDER_NAME}")

                    console.print("\n[dim]Press Enter to return to the main menu.[/dim]")
                    Prompt.ask("Press Enter to continue")
            finally:
                pass

            continue
        elif key.lower() == 'e':
            categories, use_gpu, threads, batch_size, source_dir, output_dir, log_file = edit_config_menu(categories, use_gpu, threads, batch_size, source_dir, output_dir, log_file, CONFIG_FILE)
            OUTPUT_DIR = output_dir
            log_file = OUTPUT_DIR / "classifications.txt"
            save_config(CONFIG_FILE, categories, source_dir, output_dir, use_gpu, threads, batch_size)
            image_files = [p for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
            image_files.sort(key=lambda p: p.name)
        elif key.lower() == 'm':
            fast_resort_from_log(source_dir, OUTPUT_DIR, log_file)
        elif key.lower() == 'r':
            image_files = [p for p in source_dir.iterdir() if not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTENSIONS]
            image_files.sort(key=lambda p: p.name)
            save_config(CONFIG_FILE, categories, source_dir, output_dir, use_gpu, threads, batch_size)
            console.print(f"[green]Refreshed source folder:[/green] {source_dir} | {len(image_files)} image(s) found")
            Prompt.ask("Press Enter to continue")
        elif key.lower() == 'd':
            redo_count = 0
            if OUTPUT_DIR.exists():
                for file_path in OUTPUT_DIR.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        redo_count += 1

            confirm_redo = Prompt.ask(f"[bold red]Redo sorting? This will move {redo_count} photo(s) from sorted_photos back to the source folder and clear the log.[/bold red]", choices=["y", "n"], default="n")
            if confirm_redo == "y":
                double_confirm = Prompt.ask("[bold red]Final confirmation: continue with redo?[/bold red]", choices=["y", "n"], default="n")
                if double_confirm == "y":
                    moved_count = redo(source_dir, OUTPUT_DIR, log_file)
                    image_files = [p for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
                    image_files.sort(key=lambda p: p.name)
                    console.print(f"[yellow]Redo complete:[/yellow] {moved_count} image(s) returned to {source_dir}")
                    Prompt.ask("Press Enter to continue")
        elif key.lower() == 'q':
            console.print("[yellow]Terminate[/yellow]")
            return


if __name__ == "__main__":
    main()