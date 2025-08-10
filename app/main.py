import io
import os
import shutil
import tarfile
import tempfile
import zipfile
import subprocess
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="LaTeX Compiler API", version="1.0.0")

# -------- Utilities --------


def _extract_archive(archive_path: Path, dest_dir: Path):
    """Extract .zip or .tar(.gz/.bz2/.xz) into dest_dir."""
    suffixes = "".join(archive_path.suffixes)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif suffixes.endswith(
        (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar")
    ):
        mode = "r"
        if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
            mode = "r:gz"
        elif suffixes.endswith(".tar.bz2") or suffixes.endswith(".tbz2"):
            mode = "r:bz2"
        elif suffixes.endswith(".tar.xz") or suffixes.endswith(".txz"):
            mode = "r:xz"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(
            "Unsupported archive format. Please upload a .zip or .tar.gz/.bz2/.xz"
        )


def _guess_main_tex(project_dir: Path) -> Optional[Path]:
    """Try to find an entrypoint .tex if not provided."""
    # Priority: main.tex, thesis.tex, report.tex, else first *.tex found (alphabetically)
    candidates = ["main.tex", "thesis.tex", "tesis.tex", "report.tex", "paper.tex"]
    for c in candidates:
        p = project_dir / c
        if p.exists():
            return p
    tex_files = sorted(project_dir.rglob("*.tex"))
    # Exclude common template fragment names
    tex_files = [t for t in tex_files if t.name not in {"preamble.tex"}]
    return tex_files[0] if tex_files else None


def _build_cmd(
    engine: Literal["xelatex", "lualatex", "pdflatex"],
    force_biber: bool,
    main_file: Path,
) -> list[str]:
    """Construct latexmk command without -jobname to avoid biber/jobname issues."""
    cmd = ["latexmk", "-interaction=nonstopmode", "-halt-on-error"]
    if engine == "xelatex":
        cmd += ["-xelatex"]
    elif engine == "lualatex":
        cmd += ["-lualatex"]
    else:
        cmd += ["-pdf"]  # pdflatex
    if force_biber:
        cmd += ["-use-biber"]
    cmd += [str(main_file)]
    return cmd


def _run(cmd: list[str], cwd: Path, timeout_sec: int) -> tuple[int, str]:
    """Run a subprocess and capture combined stdout/stderr."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        return 124, out + "\n[ERROR] Timeout exceeded"


def _zip_with_logs(project_dir: Path, pdf_path: Path, build_log: str) -> bytes:
    """Create a zip containing the output PDF, main.log (if exists), and latexmk output."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if pdf_path.exists():
            zf.write(pdf_path, arcname=pdf_path.name)
        # Include main.log if present
        log = next(project_dir.glob("*.log"), None)
        if log and log.exists():
            zf.write(log, arcname=log.name)
        # Include latexmk combined output as build_output.txt
        zf.writestr("build_output.txt", build_log)
    buf.seek(0)
    return buf.read()


# -------- Models --------


class Health(BaseModel):
    ok: bool = True


# -------- Routes --------


@app.get("/health", response_model=Health)
def health():
    return Health(ok=True)


@app.post("/compile")
async def compile_latex(
    project: UploadFile = File(..., description="ZIP or TAR of LaTeX project"),
    main: Optional[str] = Form(
        None, description="Main TeX file, e.g. main.tex. If omitted, guessed."
    ),
    engine: Literal["xelatex", "lualatex", "pdflatex"] = Form("xelatex"),
    use_biber: Literal["auto", "true", "false"] = Form("auto"),
    timeout_sec: int = Form(300),
    return_type: Literal["pdf", "zip"] = Form("pdf"),
    jobname: Optional[str] = Form(None, description="Optional PDF jobname"),
):
    """
    Compile a LaTeX project using latexmk, returning a PDF or a ZIP with PDF + logs.
    """
    # Create isolated temp workspace
    workdir = Path(tempfile.mkdtemp(prefix="latex-compile-"))
    try:
        # Save upload
        archive_path = workdir / f"upload{Path(project.filename).suffix}"
        with open(archive_path, "wb") as f:
            content = await project.read()
            f.write(content)

        # Extract
        project_dir = workdir / "src"
        project_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive_path, project_dir)

        # Determine main file
        main_file = project_dir / main if main else _guess_main_tex(project_dir)
        if not main_file or not main_file.exists():
            raise HTTPException(
                status_code=400,
                detail="Cannot find main .tex file. Provide 'main' form field.",
            )

        # Build command
        force_biber = use_biber == "true"
        cmd = _build_cmd(engine=engine, force_biber=force_biber, main_file=main_file)

        # Security/sanity: do not allow shell-escape escalation
        # latexmk passes through; ensure restricted shell-escape only (XeLaTeX already defaults to restricted).
        env = os.environ.copy()
        env["max_print_line"] = "1000"

        # Run
        code, output = _run(cmd, cwd=project_dir, timeout_sec=timeout_sec)

        # Find produced PDF (jobname or main basename)
        pdf_name = f"{jobname}.pdf" if jobname else f"{main_file.stem}.pdf"
        pdf_path = project_dir / pdf_name

        if code == 0 and pdf_path.exists():
            if return_type == "pdf":
                return StreamingResponse(
                    open(pdf_path, "rb"),
                    media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{pdf_name}"'},
                )
            else:
                payload = _zip_with_logs(project_dir, pdf_path, output)
                return StreamingResponse(
                    io.BytesIO(payload),
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f'attachment; filename="{pdf_name.rsplit(".",1)[0]}-bundle.zip"'
                    },
                )
        else:
            # Try to surface latex/biber logs
            log_text = output
            log_file = next(project_dir.glob("*.log"), None)
            if log_file and log_file.exists():
                try:
                    log_text += "\n\n===== main.log =====\n" + log_file.read_text(
                        errors="ignore"
                    )
                except Exception:
                    pass
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Compilation failed",
                    "return_code": code,
                    "engine": engine,
                    "used_biber": use_biber,
                    "message": log_text[:500000],  # cap size
                },
            )
    finally:
        # Clean workspace
        shutil.rmtree(workdir, ignore_errors=True)
