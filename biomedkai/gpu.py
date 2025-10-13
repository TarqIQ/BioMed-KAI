# biomedkai/gpu.py
import subprocess, sys, shutil, platform, re
import typer

app = typer.Typer()

def _run(cmd):
    return subprocess.call(cmd, shell=(platform.system()=="Windows"))

def _guess_flavor():
    # crude but robust: prefer CUDA if nvidia-smi is present
    if shutil.which("nvidia-smi"):
        # try to parse CUDA version from nvidia-smi
        out = subprocess.check_output(["nvidia-smi"]).decode("utf-8", errors="ignore")
        m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", out)
        if m:
            major, minor = m.groups()
            if major == "12":
                if minor in {"4","3","2","1"}:
                    return f"cu12{minor}"
                return "cu124"  # default to 12.4 if ambiguous
        return "cu124"
    # macOS Metal?
    if platform.system() == "Darwin":
        return "metal"
    # AMD ROCm (heuristic; users can override)
    return "rocm"

@app.command()
def install(flavor: str = typer.Option(None, help="cu121|cu122|cu124|rocm|metal"),
            version: str = typer.Option("0.2.90", help="llama-cpp-python version")):
    """Install a GPU-enabled llama-cpp-python wheel from the official wheel index."""
    if flavor is None:
        flavor = _guess_flavor()
        typer.echo(f"[biomedkai] Detected flavor: {flavor}")
    index = f"https://abetlen.github.io/llama-cpp-python/whl/{flavor}"
    cmd = f'"{sys.executable}" -m pip install --upgrade --no-cache-dir --extra-index-url {index} "llama-cpp-python=={version}"'
    typer.echo(f"[biomedkai] Installing GPU wheel:\n  {cmd}")
    rc = _run(cmd)
    if rc != 0:
        raise typer.Exit(code=rc)
    typer.echo("[biomedkai] Done. Run `biomedkai gpu doctor` to verify.")

@app.command()
def doctor():
    """Verify that llama-cpp is GPU-capable and offloads layers."""
    try:
        from llama_cpp import llama_print_system_info, Llama
        try:
            info = llama_print_system_info().decode()
        except Exception:
            info = "(system info unavailable)"
        print(info)
        llm = Llama(model_path="H:/workspace/NEXIS/src/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", n_gpu_layers=1, n_ctx=32)  # model_path None just to test ctor in some builds
        print("[biomedkai] llama-cpp import OK. GPU offload likely supported.")
    except Exception as e:
        print("[biomedkai] GPU check failed:", e)
        raise
