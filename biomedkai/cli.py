import json, pathlib, typer
from typing import Optional
from .orchestrator import run_query

app = typer.Typer(help="BioMed-KAI CLI")

@app.command()
def run(query: str,
        language: str = typer.Option("en", help="Source language"),
        config: Optional[str] = typer.Option(None, help="Path to YAML config"),
        agent: Optional[str] = typer.Option(None, "--agent", help="Force a specific agent by name"),
        out: str = typer.Option("out.json", help="Output JSON file")):
    res = run_query(query=query, language=language, cfg_path=config, force_agent=agent)
    pathlib.Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    typer.echo(f"Wrote {out}")
